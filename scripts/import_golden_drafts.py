"""Apply AI-assistant draft labels to UNLABELLED golden cases, recorded as drafts for human review.

    python scripts/import_golden_drafts.py --drafts data/golden/virgintrains_golden_v1_assistant_drafts.csv

Every draft goes through GoldenLabelStore (validated, integrity-checked, atomic) and is written to the audit log with
source=assistant_draft. A case that already has a label is never touched. Drafts are NOT human labels: each one must be
opened in scripts/label_golden_eval.py and saved (confirmed) or corrected; `--check` there reports how many remain unreviewed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from common.logging_utils import configure_logging
from evaluation.golden_eval import ASSISTANT_DRAFT, GOLD_COLUMNS, GoldenLabelStore, GoldVocabulary, case_status
from evaluation.labeling_store import LabelStoreError, LabelValidationError

logger = logging.getLogger("import_golden_drafts")


def main() -> None:
    root = _bootstrap.REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--golden-dir", type=Path, default=root / "data" / "golden")
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--registry", type=Path, default=root / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--drafted-by", default="ai-assistant", help="Name recorded in the audit log for these drafts.")
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
            labeler=args.drafted_by,
        )
    except LabelStoreError as exc:
        raise SystemExit(f"Refusing: {exc}")

    drafts = pd.read_csv(args.drafts, dtype=str, keep_default_na=False)
    missing = [c for c in ["case_id", *GOLD_COLUMNS] if c not in drafts]
    if missing:
        raise SystemExit(f"draft file is missing columns {missing}")
    if drafts["case_id"].duplicated().any():
        raise SystemExit("draft file has duplicate case ids")

    current = store.verified_frame().set_index("case_id")
    unknown = sorted(set(drafts["case_id"]) - set(current.index))
    if unknown:
        raise SystemExit(f"{len(unknown)} draft case id(s) are not golden cases, e.g. {unknown[0]}")
    applied = skipped = 0
    for row in drafts.itertuples(index=False):
        if case_status(current.loc[row.case_id]) != "unlabelled":
            skipped += 1
            continue
        try:
            store.save_label(row.case_id, {c: getattr(row, c) for c in GOLD_COLUMNS}, source=ASSISTANT_DRAFT)
        except LabelValidationError as exc:
            raise SystemExit(f"{row.case_id}: invalid draft {exc.errors}; nothing after this case was applied")
        applied += 1
    logger.info("Applied %d assistant drafts; skipped %d cases that already had a label. Review them in scripts/label_golden_eval.py.", applied, skipped)


if __name__ == "__main__":
    main()
