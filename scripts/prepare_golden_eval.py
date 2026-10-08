"""Prepare the frozen 250 golden candidates for blind human labeling.

    python scripts/prepare_golden_eval.py

Output:
    data/golden/virgintrains_golden_v1.csv            case_id, labeling_order, first_customer_message, conversation,
                                                      first_timestamp, source_tweet_ids, conversation_id, then blank
                                                      gold_intent, gold_should_escalate, gold_resolution_type, gold_confidence, human_notes
    data/golden/virgintrains_golden_v1.manifest.json  fingerprints, allowed values, definitions, leakage checks

Written only if (otherwise non-zero exit, nothing written):
  * the candidates parquet sha256 equals the one in the split manifest (sampling unchanged);
  * there are exactly 250 unique cases, identical to the golden_eval split membership;
  * no case is in train_retrieval / dev_calibration, and no customer, conversation, group, source/context tweet or opener
    is shared with them;
  * no label has been entered yet (the pack is never rebuilt once labeling starts).
No candidate intent, cluster, historical resolution label or agent output is written to the pack.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

from common.logging_utils import configure_logging
from evaluation.golden_eval import EXPECTED_CASES, GoldenIntegrityError, prepare_golden_pack
from evaluation.labeling_store import LabelStoreError

logger = logging.getLogger("prepare_golden_eval")


def main() -> None:
    root = _bootstrap.REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", type=Path, default=root / "data" / "golden")
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--registry", type=Path, default=root / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    try:
        frame, manifest = prepare_golden_pack(args.golden_dir, args.processed_dir, args.registry, EXPECTED_CASES)
    except (GoldenIntegrityError, LabelStoreError) as exc:
        logger.error("Not written: %s", exc)
        raise SystemExit(1)
    logger.info("Integrity checks passed (all zero): %s", manifest["leakage_checks_all_zero"])
    logger.info(
        "Wrote %s (%d cases, human columns blank) and %s; candidates sha256 %s, taxonomy reference status %s",
        args.golden_dir / "virgintrains_golden_v1.csv",
        len(frame),
        "virgintrains_golden_v1.manifest.json",
        manifest["golden_candidates_parquet_sha256"][:12],
        manifest["taxonomy_reference"]["status"],
    )


if __name__ == "__main__":
    main()
