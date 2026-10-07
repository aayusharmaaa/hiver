"""Prepare the 250 existing golden candidates for human labelling. Runs ONLY after the taxonomy is frozen.

Output:
    data/golden/virgintrains_golden_v1.csv        case_id, first_customer_message, conversation, gold_* (blank), labeling_notes
                                                  (+ provenance columns: source_tweet_ids, conversation_id, golden_stratum_weight)
    data/golden/virgintrains_golden_v1.meta.json  which frozen taxonomy this file was prepared against

Gates (non-zero exit, nothing written):
  * configs/virgintrains_taxonomy_v1.yaml exists and passes integrity verification;
  * the golden candidate set is identical to the one recorded at freeze time (sampling was not altered);
  * the output file does not already contain gold labels (never overwritten once labelling has started).
The candidate intent or any system suggestion is deliberately NOT shown, so the labeller works independently.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from common.logging_utils import configure_logging
from evaluation.inspection import conversation_text
from evaluation.taxonomy_calibration import ESCALATION_SIGNALS, HUMAN_RESOLUTION_TYPES, HUMAN_RESOLVED_VALUES
from taxonomy.finalize import verify_frozen
from taxonomy.registry import case_ids_sha256, sha256_file

logger = logging.getLogger("prepare_golden_labeling")
GOLD_COLUMNS = ["gold_intent", "gold_resolution_type", "gold_resolved", "gold_escalation_signal", "labeling_notes"]


def fail(msg: str) -> None:
    logger.error(msg)
    raise SystemExit(1)


def main() -> None:
    root = _bootstrap.REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", type=Path, default=root / "data" / "golden")
    parser.add_argument("--frozen", type=Path, default=root / "configs" / "virgintrains_taxonomy_v1.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    if not args.frozen.exists():
        fail(f"{args.frozen} does not exist. The taxonomy must be human-calibrated and frozen (scripts/finalize_taxonomy.py) before the golden set is prepared for labelling.")
    frozen = yaml.safe_load(args.frozen.read_text(encoding="utf-8"))
    problems = verify_frozen(frozen)
    if problems:
        fail("Frozen taxonomy failed verification: " + "; ".join(problems))

    parquet = args.golden_dir / "virgintrains_golden_candidates.parquet"
    golden = pd.read_parquet(parquet)
    if case_ids_sha256(golden["case_id"]) != frozen["metadata"]["golden_case_ids_sha256"] or sha256_file(parquet) != frozen["metadata"]["golden_candidates_parquet_sha256"]:
        fail("The golden candidates differ from those recorded when the taxonomy was frozen. Their sampling must not be altered.")

    out = args.golden_dir / "virgintrains_golden_v1.csv"
    if out.exists():
        existing = pd.read_csv(out, dtype=str, keep_default_na=False)
        if any((existing[c].str.strip() != "").any() for c in GOLD_COLUMNS if c in existing):
            fail(f"{out} already contains gold labels; refusing to overwrite.")

    frame = pd.DataFrame(
        {
            "case_id": golden["case_id"],
            "first_customer_message": golden["opening_message"].fillna(""),
            "conversation": [conversation_text(list(t)) for t in golden["full_turns"]],
        }
    )
    for col in GOLD_COLUMNS:
        frame[col] = ""
    frame["source_tweet_ids"] = [json.dumps([int(x) for x in ids]) for ids in golden["source_tweet_ids"]]
    frame["conversation_id"] = golden["conversation_id"].to_numpy()
    frame["golden_stratum_weight"] = golden["golden_stratum_weight"].round(4).to_numpy()
    frame.to_csv(out, index=False, encoding="utf-8")

    meta = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_cases": len(frame),
        "taxonomy_version": frozen["metadata"]["taxonomy_version"],
        "taxonomy_content_sha256": frozen["metadata"]["content_sha256"],
        "taxonomy_date_created": frozen["metadata"]["date_created"],
        "golden_case_ids_sha256": frozen["metadata"]["golden_case_ids_sha256"],
        "allowed_gold_intent": frozen["labels"],
        "allowed_gold_resolution_type": list(HUMAN_RESOLUTION_TYPES),
        "allowed_gold_resolved": list(HUMAN_RESOLVED_VALUES),
        "allowed_gold_escalation_signal": list(ESCALATION_SIGNALS),
        "note": "Golden evaluation set. Never used for retrieval, prompts, thresholds, or taxonomy changes.",
    }
    (args.golden_dir / "virgintrains_golden_v1.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d cases, gold_* blank) against taxonomy %s", out, len(frame), meta["taxonomy_content_sha256"][:12])


if __name__ == "__main__":
    main()
