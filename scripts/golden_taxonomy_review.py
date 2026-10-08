"""Write reports/golden_taxonomy_review.md once all 250 golden cases are human-labelled.

    python scripts/golden_taxonomy_review.py

Refuses (exit 2) while any case is unlabelled or partial. The golden pack is integrity-checked first. The report compares the
candidate taxonomy with the human labels to inform a later, human-made freeze decision. It does not change the taxonomy and
computes no agent metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from common.logging_utils import configure_logging
from evaluation.golden_eval import CANDIDATES_PARQUET, GoldenLabelStore, GoldVocabulary
from evaluation.golden_review import ReviewNotReadyError, build_review_markdown, candidate_intents, review_frame
from evaluation.labeling_store import LabelStoreError
from taxonomy.registry import load_labels

logger = logging.getLogger("golden_taxonomy_review")


def main() -> None:
    root = _bootstrap.REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", type=Path, default=root / "data" / "golden")
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--registry", type=Path, default=root / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--cluster-labels", type=Path, default=root / "configs" / "virgintrains_cluster_labels.yaml")
    parser.add_argument("--out", type=Path, default=root / "reports" / "golden_taxonomy_review.md")
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
        )
    except LabelStoreError as exc:
        raise SystemExit(f"Refusing: {exc}")

    golden = pd.read_parquet(args.golden_dir / CANDIDATES_PARQUET, columns=["case_id", "cluster_id"])
    preview = json.loads((args.processed_dir / "virgintrains_cluster_preview.json").read_text(encoding="utf-8"))
    try:
        df = review_frame(store.verified_frame(), candidate_intents(golden, preview, load_labels(args.cluster_labels)))
    except ReviewNotReadyError as exc:
        logger.error("Not written: %s", exc)
        raise SystemExit(2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_review_markdown(df, registry_status=registry.get("status")), encoding="utf-8")
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
