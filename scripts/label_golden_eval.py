"""Blind local labeling tool for the 250-case golden evaluation set.

    python scripts/label_golden_eval.py            # opens http://127.0.0.1:8766/
    python scripts/label_golden_eval.py --check    # verify the files and print progress, no server

Shows one case at a time (full conversation, oldest first) with the provisional candidate taxonomy as a reference, and saves
gold_intent, gold_should_escalate, gold_resolution_type, gold_confidence and human_notes into data/golden/virgintrains_golden_v1.csv
on every Save. It resumes where you stopped.

It never shows a candidate intent, cluster, historical resolution label, retrieved evidence, policy decision, draft reply or
grounding result, and never pre-fills a label. Before every read and write it re-verifies the frozen sample (candidates sha256,
split membership, case ids, conversations) and refuses labels edited outside the tool.
Audit trail: data/golden/virgintrains_golden_v1_label_audit.jsonl. Rolling backup: virgintrains_golden_v1.csv.bak.
"""

from __future__ import annotations

import argparse
import getpass
import logging
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from common.logging_utils import configure_logging
from evaluation.golden_eval import GoldenLabelStore, GoldVocabulary, build_reference, label_provenance
from evaluation.golden_labeling_ui import serve_golden
from evaluation.labeling_store import LabelStoreError
from taxonomy.registry import load_labels

logger = logging.getLogger("label_golden_eval")


def main() -> None:
    root = _bootstrap.REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", type=Path, default=root / "data" / "golden")
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--registry", type=Path, default=root / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--cluster-labels", type=Path, default=root / "configs" / "virgintrains_cluster_labels.yaml")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--labeler", default=getpass.getuser(), help="Name recorded in the audit log.")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify the golden pack and print progress; do not start the tool.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["taxonomy"]
    splits = args.processed_dir / "splits"
    try:
        store = GoldenLabelStore(
            args.golden_dir,
            GoldVocabulary.from_registry(registry),
            split_manifest_path=splits / "virgintrains_split_manifest.json",
            assignments_path=splits / "virgintrains_split_assignments.csv",
            registry_path=args.registry,
            labeler=args.labeler,
        )
    except LabelStoreError as exc:
        raise SystemExit(f"Refusing to start: {exc}")

    counts = store.counts()
    logger.info("Golden pack verified: %d / %d labelled, %d partial, %d unlabelled", counts["labelled"], counts["total"], counts["partial"], counts["unlabelled"])
    sources = Counter(label_provenance(store.audit_path).values())
    logger.info("Label provenance: %s", dict(sorted(sources.items())) or "none yet")
    if args.check:
        return
    reference = build_reference(registry, load_labels(args.cluster_labels).get("confusable_notes", []))
    serve_golden(store, reference, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
