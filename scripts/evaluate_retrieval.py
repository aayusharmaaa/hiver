"""Proxy retrieval evaluation: BM25 vs embeddings vs hybrid vs hybrid + candidate intent.

    python scripts/evaluate_retrieval.py

Queries are non-golden dev_calibration cases; the corpus is the train_retrieval resolution memory; golden and reserve cases
are never touched. Relevance is a *proxy* built from candidate intent + rule-derived resolution type (not human ground truth).
Outputs: reports/virgintrains_retrieval_evaluation.md and .json (the console shows the comparison table and examples).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from common.logging_utils import configure_logging
from evaluation.retrieval import render_console, render_markdown, run_evaluation, select_queries
from evaluation.splits import DEV, GOLDEN
from ingestion.resolution_memory import build_memory_records
from retrieval import ResolutionRetriever, SentenceTransformerEncoder
from taxonomy.registry import cluster_intent_map, load_labels

logger = logging.getLogger("evaluate_retrieval")


def main() -> None:
    root = _bootstrap.REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--cluster-labels", type=Path, default=root / "configs" / "virgintrains_cluster_labels.yaml")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    parser.add_argument("--sem-weight", type=float, default=None, help="Override the tuned semantic weight (0-1).")
    parser.add_argument("--intent-weight", type=float, default=None, help="Override the tuned intent bonus.")
    parser.add_argument("--max-queries", type=int, default=None, help="Use only the first N eligible queries (quick runs).")
    parser.add_argument("--examples", type=int, default=6, help="Good and bad examples to show.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip the index-text ablation (saves one embedding pass).")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    configure_logging(args.log_level)

    processed = args.processed_dir
    assignments = pd.read_csv(processed / "splits" / "virgintrains_split_assignments.csv")
    memory_path = processed / "virgintrains_resolution_memory.parquet"
    if not memory_path.exists():
        raise SystemExit(f"{memory_path} not found. Run: python scripts/build_resolution_memory.py")
    cache_dir = processed / "cache"
    encoder = SentenceTransformerEncoder()
    retriever = ResolutionRetriever.from_files(memory_path, processed / "splits" / "virgintrains_split_assignments.csv", encoder=encoder, cache_dir=cache_dir)
    alt = None
    if not args.skip_ablation:
        alt = ResolutionRetriever.from_files(
            memory_path, processed / "splits" / "virgintrains_split_assignments.csv", encoder=encoder, cache_dir=cache_dir,
            index_fields=("customer_problem", "historical_response"),
        )

    dev = pd.read_parquet(processed / "splits" / "virgintrains_dev_calibration.parquet")
    split_of = dict(zip(assignments["case_id"], assignments["split"]))
    assert set(dev["case_id"]) == {c for c, s in split_of.items() if s == DEV}, "dev file does not match the split assignments"
    golden_ids = {c for c, s in split_of.items() if s == GOLDEN}
    assert not (set(dev["case_id"]) & golden_ids), "golden cases must never be evaluation queries"
    preview = json.loads((processed / "virgintrains_cluster_preview.json").read_text(encoding="utf-8"))
    cmap = cluster_intent_map(preview, load_labels(args.cluster_labels))
    records = build_memory_records(
        dev, {cid: spec.get("final_intent") for cid, spec in cmap.items()}, split_of=split_of, cluster_of=dict(zip(dev["case_id"], dev["cluster_id"]))
    )
    queries = select_queries(records)
    if args.max_queries:
        queries = queries.head(args.max_queries)
    logger.info("%d dev queries, %d corpus episodes", len(queries), len(retriever.corpus))

    results = run_evaluation(retriever, queries, alt_retriever=alt, sem_weight=args.sem_weight, intent_weight=args.intent_weight, n_examples=args.examples)
    results["meta"]["dev_cases_total"] = int(len(dev))
    results["meta"]["query_selection"] = "strong evidence + non-fallback candidate intent"

    args.report_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = args.report_dir / "virgintrains_retrieval_evaluation.md", args.report_dir / "virgintrains_retrieval_evaluation.json"
    md_path.write_text(render_markdown(results), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(render_console(results))
    print(f"\nFull report: {md_path}\nMetrics JSON: {json_path}")


if __name__ == "__main__":
    main()
