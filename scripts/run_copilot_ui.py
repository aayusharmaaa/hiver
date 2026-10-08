"""Support Copilot demo UI over the 50-case end-to-end evaluation slice.

    python scripts/run_copilot_ui.py              # http://127.0.0.1:8770/
    python scripts/run_copilot_ui.py --no-live    # cached results only, no API key needed

Tickets with a cached result from scripts/evaluate_agent.py show it (marked "cached"). Other tickets can be run live with the
real agent on Groq when GROQ_API_KEY is set; live calls are cached in data/processed/cache/copilot_model_cache.jsonl, so the
evaluation artifacts are never modified. No judge scores are shown.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from agent.config import load_config
from common.logging_utils import configure_logging
from evaluation.agent_eval import MIN_PER_INTENT, SAMPLE_SEED, DEFAULT_SAMPLE, CachingModel, case_record, golden_agent_frame, select_stratified
from evaluation.golden_eval import GoldenLabelStore, GoldVocabulary
from evaluation.labeling_store import LabelStoreError
from evaluation.smoke_report import check_result_invariants
from taxonomy.registry import intent_names
from ui.copilot import CopilotApp, build_tickets, load_records, serve

ROOT = _bootstrap.REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", type=Path, default=ROOT / "data" / "golden")
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "agent_eval" / "case_results.jsonl")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-live", action="store_true", help="Never call a model; show cached results only.")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--env-file", type=Path, default=_bootstrap.DEFAULT_ENV_FILE)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    configure_logging(args.log_level)
    _bootstrap.load_env_file(args.env_file)

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["taxonomy"]
    intents = intent_names(registry)
    splits = args.processed_dir / "splits"
    assignments = splits / "virgintrains_split_assignments.csv"
    try:
        store = GoldenLabelStore(
            args.golden_dir, GoldVocabulary.from_registry(registry), split_manifest_path=splits / "virgintrains_split_manifest.json",
            assignments_path=assignments, registry_path=args.registry,
        )
    except LabelStoreError as exc:
        print(f"Refusing: {exc}", file=sys.stderr)
        return 2
    cases = select_stratified(golden_agent_frame(store), intents, n=DEFAULT_SAMPLE, min_per_intent=MIN_PER_INTENT, seed=SAMPLE_SEED)
    extra = store.verified_frame()[["case_id", "conversation", "first_timestamp", "gold_confidence"]]
    cases = cases.drop(columns=[c for c in extra.columns if c != "case_id" and c in cases.columns]).merge(extra, on="case_id", how="left")
    records = load_records(args.results)
    config = load_config()
    p = config.policy

    live_runner = None
    if not args.no_live and os.environ.get("GROQ_API_KEY"):
        state: dict = {}
        build_lock = threading.Lock()

        def agent_parts():
            with build_lock:
                if not state:
                    from agent.support_agent import SupportAgent
                    from models.groq import GroqModel
                    from retrieval import ResolutionRetriever, SentenceTransformerEncoder

                    model = CachingModel(GroqModel(max_retries=config.model.max_retries), args.processed_dir / "cache" / "copilot_model_cache.jsonl")
                    retriever = ResolutionRetriever.from_files(
                        args.processed_dir / "virgintrains_resolution_memory.parquet", assignments,
                        encoder=SentenceTransformerEncoder(), cache_dir=args.processed_dir / "cache",
                    )
                    state.update(
                        model=model, agent=SupportAgent.from_parts(model, retriever, registry, config),
                        corpus=set(retriever.corpus["case_id"]),  # type: ignore[attr-defined]
                        splits=dict(pd.read_csv(assignments, usecols=["case_id", "split"]).itertuples(index=False, name=None)),
                    )
            return state

        rows = {c.case_id: c for c in cases.itertuples(index=False)}

        def run_live(case_id: str) -> dict:
            s = agent_parts()
            case = rows[case_id]
            before = len(s["model"].runtime_errors)
            result = s["agent"].handle(message=case.text, conversation_context=None)
            record = case_record(case, result, check_result_invariants(result, intents, s["corpus"], s["splits"]))
            if len(s["model"].runtime_errors) > before:
                record["model_error"] = s["model"].runtime_errors[-1]
            return record

        live_runner = run_live

    meta = {
        "brand": "VirginTrains", "cached_results": len(records), "slice_size": len(cases), "model": "openai/gpt-oss-120b (Groq)",
        "thresholds": {"min_top_similarity": p.min_top_similarity, "min_usable_similarity": p.min_usable_similarity,
                       "min_usable_evidence": p.min_usable_evidence, "min_classifier_confidence": p.min_classifier_confidence},
    }
    app = CopilotApp(build_tickets(cases, records), meta, live_runner)
    print(f"{len(cases)} tickets; {len(records)} with cached agent results; live mode {'on' if live_runner else 'off'}.")
    serve(app, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
