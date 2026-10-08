"""End-to-end agent evaluation on a stratified ~50-case slice of the golden set.

    python scripts/evaluate_agent.py                         # Groq gpt-oss-120b (needs GROQ_API_KEY in .env)
    python scripts/evaluate_agent.py --provider gemini       # same agent on Gemini
    python scripts/evaluate_agent.py --select-only           # print the selected cases; no model calls, no key

Runs the unchanged SupportAgent (classification -> retrieval -> policy -> generation -> grounding) on each case's opening
message and scores routing against the human `gold_should_escalate` label. Every model call is cached in
reports/agent_eval/model_cache.jsonl, so a rerun makes no new calls and an interrupted run (rate limit) resumes.

Outputs in reports/agent_eval/: virgintrains_agent_evaluation.md and .json, case_results.jsonl, judge_inputs.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from common.logging_utils import configure_logging
from evaluation.agent_eval import (
    DEFAULT_SAMPLE,
    MIN_PER_INTENT,
    SAMPLE_SEED,
    CachingModel,
    agent_metrics,
    golden_agent_frame,
    judge_items,
    render_markdown,
    run_agent_cases,
    select_stratified,
)
from evaluation.golden_eval import GoldenLabelStore, GoldVocabulary
from evaluation.intent_eval import BLIND_HUMAN
from evaluation.labeling_store import LabelStoreError
from evaluation.smoke_report import changed_files, fingerprint_files
from taxonomy.registry import intent_names

ROOT = _bootstrap.REPO_ROOT


def build_model(provider: str, config):
    m = config.model
    if provider == "groq":
        from models.groq import GroqModel

        return GroqModel(max_retries=m.max_retries)
    from models.gemini import GeminiModel

    return GeminiModel(model_name=m.name, timeout_seconds=m.timeout_seconds, max_retries=m.max_retries, response_schema=m.response_schema)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", type=Path, default=ROOT / "data" / "golden")
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--config", type=Path, default=None, help="Support-agent config (default: configs/support_agent.yaml).")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "agent_eval")
    parser.add_argument("--provider", choices=("groq", "gemini"), default="groq")
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE, help="Number of golden cases (default 50).")
    parser.add_argument("--min-per-intent", type=int, default=MIN_PER_INTENT)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--env-file", type=Path, default=_bootstrap.DEFAULT_ENV_FILE)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    configure_logging(args.log_level)
    _bootstrap.load_env_file(args.env_file)

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["taxonomy"]
    intents = intent_names(registry)
    splits = args.processed_dir / "splits"
    assignments_path = splits / "virgintrains_split_assignments.csv"
    try:
        store = GoldenLabelStore(
            args.golden_dir, GoldVocabulary.from_registry(registry), split_manifest_path=splits / "virgintrains_split_manifest.json",
            assignments_path=assignments_path, registry_path=args.registry,
        )
    except LabelStoreError as exc:
        print(f"Refusing to evaluate: {exc}", file=sys.stderr)
        return 2
    cases = select_stratified(golden_agent_frame(store), intents, n=args.n, min_per_intent=args.min_per_intent, seed=args.seed)
    print(f"Selected {len(cases)} golden cases ({int(cases[BLIND_HUMAN].sum())} blind-human).")
    print(cases.groupby(["gold_intent", "gold_should_escalate"]).size().unstack(fill_value=0).to_string())
    if args.select_only:
        return 0

    from agent.config import load_config
    from agent.support_agent import SupportAgent
    from models.base import ModelConfigError
    from retrieval import ResolutionRetriever, SentenceTransformerEncoder

    config = load_config(args.config)
    protected = [
        args.registry, ROOT / "configs" / "support_agent.yaml", assignments_path, args.processed_dir / "virgintrains_resolution_memory.parquet",
        args.golden_dir / "virgintrains_golden_v1.csv", args.golden_dir / "virgintrains_golden_v1_label_audit.jsonl",
    ]
    before = fingerprint_files(protected)
    try:
        inner = build_model(args.provider, config)
    except ModelConfigError as exc:
        print(f"SETUP FAILURE: {exc}", file=sys.stderr)
        return 2
    report_dir = args.report_dir
    model = CachingModel(inner, report_dir / "model_cache.jsonl")
    memory = args.processed_dir / "virgintrains_resolution_memory.parquet"
    retriever = ResolutionRetriever.from_files(memory, assignments_path, encoder=SentenceTransformerEncoder(), cache_dir=args.processed_dir / "cache")
    agent = SupportAgent.from_parts(model, retriever, registry, config)
    split_by_case = dict(pd.read_csv(assignments_path, usecols=["case_id", "split"]).itertuples(index=False, name=None))
    corpus_ids = set(retriever.corpus["case_id"])  # type: ignore[attr-defined]

    def progress(i: int, total: int, rec: dict) -> None:
        print(f"[{i:02d}/{total}] {rec['case_id']:14s} gold={rec['gold_should_escalate']:3s} {rec['policy_action']:11s} -> {rec['final_action']:11s} "
              f"gen={rec['generation']:13s} grounding={rec['grounding']}", flush=True)

    try:
        records, stopped = run_agent_cases(agent, cases, model, allowed_intents=intents, corpus_case_ids=corpus_ids, split_by_case=split_by_case, progress=progress)
    except ModelConfigError as exc:
        print(f"SETUP FAILURE: {exc}", file=sys.stderr)
        return 2

    blind = [r for r in records if r["blind_human"]]
    results = {
        "meta": {
            "provider": args.provider, "model": model.name, "seed": args.seed, "selected": len(cases), "scored": len(records),
            "blind_human_selected": int(cases[BLIND_HUMAN].sum()), "stopped": stopped, "cache_hits": model.hits, "cache_misses": model.misses,
        },
        "subsets": {"all sampled": agent_metrics(records), "blind human only": agent_metrics(blind)},
        "records": records,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "virgintrains_agent_evaluation.md").write_text(render_markdown(results), encoding="utf-8")
    (report_dir / "virgintrains_agent_evaluation.json").write_text(json.dumps({k: v for k, v in results.items() if k != "records"}, indent=2), encoding="utf-8")
    with open(report_dir / "case_results.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    with open(report_dir / "judge_inputs.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(j, ensure_ascii=False) + "\n" for j in judge_items(records))

    m = results["subsets"]["all sampled"]
    print(f"\nScored {len(records)}/{len(cases)}. Model calls: {model.hits} cached, {model.misses} new." + (f"\n{stopped}" if stopped else ""))
    for key in ("escalation_coverage", "safe_automation_rate", "false_auto_handle_rate", "escalation_precision", "escalation_recall"):
        v = m[key]
        print(f"  {key:24s} {'n/a' if v is None else f'{100 * v:.1f}%'}")
    print(f"  generation produced      {m['generation']['produced']}/{m['generation']['attempted']}   grounding pass {m['grounding']['pass']}/{m['grounding']['run']}")
    touched = changed_files(before, fingerprint_files(protected))
    if touched:
        print(f"\nINTEGRITY FAILURE: protected files changed during the run: {touched}", file=sys.stderr)
        return 1
    print(f"\nReport: {report_dir / 'virgintrains_agent_evaluation.md'}")
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
