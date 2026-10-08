"""Live smoke test: run the REAL support agent (real Gemini) on a handful of representative NON-golden VirginTrains openers.

    export GEMINI_API_KEY=...          # PowerShell: $env:GEMINI_API_KEY="..."
    python scripts/smoke_test_agent.py --limit 12 --seed 42

This is a diagnostic, not the evaluation: no accuracy or score is computed and nothing is written to disk. Messages are opening
customer messages from `dev_calibration` (golden / reserve / excluded cases are never used and every case is checked before any
model call). Historical resolution type, candidate intent etc. are printed for inspection only; they are weak metadata, NOT human
ground truth, and are never sent to the model. `--select-only` prints the selected cases without calling Gemini (no key needed).

Exit codes: 0 = ran (even if some scenarios were unavailable), 1 = at least one case failed, 2 = setup problem (e.g. no API key).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from common.logging_utils import configure_logging
from evaluation.smoke_report import (
    SetupFailure,
    changed_files,
    fingerprint_files,
    format_case_result,
    format_summary,
    run_case,
    scrub_secret,
    summarize_results,
)
from evaluation.smoke_selection import (
    SmokeSafetyError,
    build_safety_context,
    load_pool,
    load_safety_frame,
    select_smoke_cases,
)
from evaluation.splits import LeakageError, verify_no_leakage
from models.base import ModelError
from models.gemini import api_key_from_env

ROOT = _bootstrap.REPO_ROOT
KEY_HELP = (
    "SETUP FAILURE: GEMINI_API_KEY is not set, so the live smoke test cannot run (nothing was sent anywhere and no mock was used).\n"
    "  Create a key at https://aistudio.google.com/apikey, then either put it in the git-ignored .env file\n"
    "  (copy .env.example to .env and fill in GEMINI_API_KEY=...) or set it for this shell:\n"
    "    PowerShell:  $env:GEMINI_API_KEY = \"your-key\"\n"
    "    bash/zsh:    export GEMINI_API_KEY=\"your-key\"\n"
    "  and re-run:  python scripts/smoke_test_agent.py --limit 12 --seed 42\n"
    "  (`--select-only` shows which cases would be sent, without a key.)"
)


def protected_files(processed: Path) -> list[Path]:
    """Files this script must never change (hashed before and after the run)."""
    c = ROOT / "configs"
    golden = ROOT / "data" / "golden"
    return [
        c / "virgintrains_intents.yaml", c / "virgintrains_taxonomy_decisions.yaml", c / "virgintrains_cluster_labels.yaml", c / "support_agent.yaml",
        processed / "taxonomy_calibration.csv", processed / "splits" / "virgintrains_split_assignments.csv", processed / "virgintrains_resolution_memory.parquet",
        golden / "virgintrains_golden_candidates.csv", golden / "virgintrains_golden_candidates.parquet",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=12, help="Number of cases to run (default 12 = one per scenario; more adds EXTRA cases).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the deterministic tie-break between equally good candidates.")
    parser.add_argument("--select-only", action="store_true", help="Print the selected cases and safety checks; make no model calls.")
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--config", type=Path, default=None, help="Support-agent config (default: configs/support_agent.yaml).")
    parser.add_argument("--pause-seconds", type=float, default=0.0, help="Wait between cases to stay under a per-minute rate limit (e.g. 20 on the free tier).")
    parser.add_argument("--env-file", type=Path, default=_bootstrap.DEFAULT_ENV_FILE, help="Git-ignored file with GEMINI_API_KEY=... (default: .env).")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    configure_logging(args.log_level)
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.pause_seconds < 0:
        parser.error("--pause-seconds must be >= 0")
    _bootstrap.load_env_file(args.env_file)

    key = api_key_from_env()
    if key is None and not args.select_only:
        print(KEY_HELP, file=sys.stderr)
        return 2

    processed = args.processed_dir
    protected = protected_files(processed)
    before = fingerprint_files(protected)

    # ---- data selection and leakage safety (no model involved) ----
    try:
        frame = load_safety_frame(processed)
        verify_no_leakage(frame)  # existing split-level checks: golden vs train/dev customers, conversations, tweets, openers
        safety = build_safety_context(frame)
        pool = load_pool(processed, ROOT / "configs" / "virgintrains_cluster_labels.yaml")
        selection = select_smoke_cases(pool, safety, limit=args.limit, seed=args.seed)
    except (LeakageError, SmokeSafetyError) as exc:
        print(f"LEAKAGE-SAFETY FAILURE: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"SETUP FAILURE: missing input file: {exc}", file=sys.stderr)
        return 2

    total = len(selection.cases)
    print(f"Selected {total} of {selection.requested} requested case(s) from dev_calibration (seed {args.seed}). "
          f"Golden, reserve and excluded cases: never used. All selected cases passed the leakage checks.")
    for name in selection.unavailable:
        print(f"SCENARIO UNAVAILABLE: {name}")
    if args.select_only:
        for i, case in enumerate(selection.cases, 1):
            print(f"{i:02d}. {case.scenario:28s} {case.case_id:14s} {case.why_selected}\n      {case.message[:160]!r}")
        return 0

    # ---- the real agent: real Gemini client + real retriever + real policy ----
    from agent.support_agent import SupportAgent

    try:
        agent = SupportAgent.from_config(args.config, processed_dir=processed)
    except (ModelError, FileNotFoundError) as exc:
        print(f"SETUP FAILURE: {scrub_secret(str(exc), [key])}", file=sys.stderr)
        return 2
    print(f"Model: {getattr(getattr(agent.classifier, 'model', None), 'name', 'unknown')}   (live Gemini; the API key is never printed)\n")

    taxonomy_intents = list(agent.classifier.allowed_intents)  # candidate intents + fallback, from the registry
    corpus_ids = set(agent.retriever.corpus["case_id"])  # type: ignore[attr-defined]
    outcomes = []
    try:
        for i, case in enumerate(selection.cases, 1):
            if i > 1 and args.pause_seconds > 0:
                time.sleep(args.pause_seconds)
            outcome = run_case(
                agent, case, allowed_intents=taxonomy_intents, corpus_case_ids=corpus_ids, split_by_case=safety.split_by_case,
                policy_settings=agent.config.policy, secrets=[key],
            )
            outcomes.append(outcome)
            print(format_case_result(i, total, outcome), flush=True)
    except SetupFailure as exc:
        print(f"SETUP FAILURE: {exc}", file=sys.stderr)
        return 2

    summary = summarize_results(outcomes, selection)
    print("\n" + format_summary(summary))
    touched = changed_files(before, fingerprint_files(protected))
    if touched:
        print(f"\nINTEGRITY FAILURE: protected files changed during the run: {touched}", file=sys.stderr)
        return 1
    print("\nProtected files (taxonomy, calibration, split assignments, resolution memory, golden candidates): unchanged. No output files written.")
    return 1 if any(o.failed for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
