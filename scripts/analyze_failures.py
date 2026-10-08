"""Failure analysis from existing evaluation artifacts. No model calls, no API key, a few seconds.

    python scripts/analyze_failures.py

Reads the golden pack (integrity-checked), the cached LLM intent predictions, the retrieval proxy evaluation and the
end-to-end agent run (partial runs are labelled as such). Writes reports/failure_analysis.md and .json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from agent.config import load_config
from common.logging_utils import configure_logging
from evaluation.agent_eval import DEFAULT_SAMPLE
from evaluation.failure_analysis import analyse, golden_frame, load_jsonl, load_llm_predictions, render_markdown
from evaluation.golden_eval import AUDIT_LOG, CANDIDATES_PARQUET, GoldenLabelStore, GoldVocabulary, label_provenance
from evaluation.golden_review import candidate_intents
from evaluation.labeling_store import LabelStoreError
from taxonomy.registry import load_labels

ROOT = _bootstrap.REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", type=Path, default=ROOT / "data" / "golden")
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--cluster-labels", type=Path, default=ROOT / "configs" / "virgintrains_cluster_labels.yaml")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    configure_logging(args.log_level)

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["taxonomy"]
    splits = args.processed_dir / "splits"
    try:
        store = GoldenLabelStore(
            args.golden_dir, GoldVocabulary.from_registry(registry), split_manifest_path=splits / "virgintrains_split_manifest.json",
            assignments_path=splits / "virgintrains_split_assignments.csv", registry_path=args.registry,
        )
    except LabelStoreError as exc:
        print(f"Refusing: {exc}", file=sys.stderr)
        return 2
    candidates = pd.read_parquet(args.golden_dir / CANDIDATES_PARQUET, columns=["case_id", "cluster_id", "other_agent_turn_count", "dm_redirect"])
    preview = json.loads((args.processed_dir / "virgintrains_cluster_preview.json").read_text(encoding="utf-8"))
    golden = golden_frame(
        store.verified_frame(), label_provenance(args.golden_dir / AUDIT_LOG),
        candidate_intents(candidates, preview, load_labels(args.cluster_labels)), candidates,
    )

    preds = load_llm_predictions(args.reports_dir / "intent_eval" / "llm_predictions.jsonl")
    retrieval = json.loads((args.reports_dir / "virgintrains_retrieval_evaluation.json").read_text(encoding="utf-8"))
    records = load_jsonl(args.reports_dir / "agent_eval" / "case_results.jsonl")
    modes = analyse(golden, preds, retrieval, records, load_config().policy, expected_agent_cases=DEFAULT_SAMPLE)
    meta = {
        "golden_cases": len(golden), "blind_cases": int(golden["blind"].sum()), "llm_predictions": len(preds),
        "llm_model": next(iter(preds.values()))["model"] if preds else "none", "agent_cases": len(records), "agent_expected": DEFAULT_SAMPLE,
    }
    out = args.reports_dir / "failure_analysis.md"
    out.write_text(render_markdown(modes, meta), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"meta": meta, "modes": modes}, indent=2, ensure_ascii=False), encoding="utf-8")
    for m in modes:
        print(f"{m['title'][:55]:55s} {m['count']!s:>4} / {m['denominator']!s:<4} [{m['layer']}]")
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
