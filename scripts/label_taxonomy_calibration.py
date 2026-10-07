"""Local labeling tool for the 200 taxonomy-calibration cases.

    python scripts/label_taxonomy_calibration.py            # opens http://127.0.0.1:8765/
    python scripts/label_taxonomy_calibration.py --check    # verify the files and print progress, no server

What it does: shows one case at a time (conversation first, system suggestion only on request) and saves your five human_*
fields into data/processed/taxonomy_calibration.csv on every Save. It resumes where you stopped.

What it will not do: write any other column (case ids, tweet ids and sampling metadata are fingerprint-checked before every
write), touch golden / train / dev data, fill in a label for you, or run after the taxonomy has been frozen.
Audit trail: data/processed/taxonomy_calibration_label_audit.jsonl. Rolling backup: taxonomy_calibration.csv.bak.
"""

from __future__ import annotations

import argparse
import getpass
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from common.logging_utils import configure_logging
from evaluation.labeling_store import LabelStore, LabelStoreError, vocabulary_from_registry
from evaluation.labeling_support import build_reference
from evaluation.labeling_ui import serve
from taxonomy.registry import load_labels

logger = logging.getLogger("label_taxonomy_calibration")


def main() -> None:
    root = _bootstrap.REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--registry", type=Path, default=root / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--cluster-labels", type=Path, default=root / "configs" / "virgintrains_cluster_labels.yaml")
    parser.add_argument("--frozen", type=Path, default=root / "configs" / "virgintrains_taxonomy_v1.yaml")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--labeler", default=getpass.getuser(), help="Name recorded in the audit log.")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify the calibration files and print progress; do not start the tool.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.frozen.exists():
        raise SystemExit(f"{args.frozen} exists: the taxonomy is frozen, so calibration labels must not change.")
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["taxonomy"]
    assignments = args.processed_dir / "splits" / "virgintrains_split_assignments.csv"
    try:
        store = LabelStore(
            args.processed_dir / "taxonomy_calibration.csv",
            args.processed_dir / "taxonomy_calibration_manifest.json",
            vocabulary_from_registry(registry),
            assignments_path=assignments if assignments.exists() else None,
            labeler=args.labeler,
        )
    except LabelStoreError as exc:
        raise SystemExit(f"Refusing to start: {exc}")

    counts = store.counts()
    logger.info("Calibration file verified: %d cases, %d labelled, %d partial, %d unlabelled", counts["total"], counts["labelled"], counts["partial"], counts["unlabelled"])
    if args.check:
        return

    guide_path = args.processed_dir / "taxonomy_calibration_guide.md"
    guide_md = guide_path.read_text(encoding="utf-8") if guide_path.exists() else "# Guide not found\n\nRun `python scripts/build_taxonomy_calibration.py --guide-only`."
    reference = build_reference(registry, load_labels(args.cluster_labels).get("confusable_notes", []))
    serve(store, reference, guide_md, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
