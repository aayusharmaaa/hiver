"""Build data/processed/virgintrains_resolution_memory.parquet from the existing VirginTrains support cases.

    python scripts/build_resolution_memory.py

Only `train_retrieval` cases are written. Golden (`golden_eval`), the held-back reserve and dev cases are refused, so the
memory can never leak evaluation or calibration data into retrieval. The taxonomy is read as a *candidate* signal only.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from common.logging_utils import configure_logging
from evaluation.splits import TRAIN
from ingestion.resolution_memory import build_memory_records
from taxonomy.registry import cluster_intent_map, load_labels, sha256_file

logger = logging.getLogger("build_resolution_memory")
MEMORY_NAME = "virgintrains_resolution_memory.parquet"


def main() -> None:
    root = _bootstrap.REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--cluster-labels", type=Path, default=root / "configs" / "virgintrains_cluster_labels.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    processed = args.processed_dir
    assignments = pd.read_csv(processed / "splits" / "virgintrains_split_assignments.csv")
    train_ids = set(assignments.loc[assignments["split"] == TRAIN, "case_id"])
    cases = pd.read_parquet(processed / "virgintrains_cases.parquet")
    cases = cases[cases["case_id"].isin(train_ids)].reset_index(drop=True)
    clusters = pd.read_parquet(processed / "virgintrains_case_clusters.parquet", columns=["case_id", "cluster_id"])
    preview = json.loads((processed / "virgintrains_cluster_preview.json").read_text(encoding="utf-8"))
    cmap = cluster_intent_map(preview, load_labels(args.cluster_labels))
    intent_of_cluster = {cid: spec.get("final_intent") for cid, spec in cmap.items()}

    memory = build_memory_records(
        cases,
        intent_of_cluster,
        split_of=dict(zip(assignments["case_id"], assignments["split"])),
        cluster_of=dict(zip(clusters["case_id"], clusters["cluster_id"])),
    )
    assert set(memory["split"]) == {TRAIN} and len(memory) == len(train_ids), "memory must contain exactly the train_retrieval cases"
    out = processed / MEMORY_NAME
    memory.to_parquet(out, index=False)

    primary = memory[memory["in_primary_corpus"]]
    logger.info("Wrote %s: %d episodes (%d in the primary retrieval corpus)", out, len(memory), len(primary))
    logger.info("Evidence quality: %s", memory["evidence_quality"].value_counts().to_dict())
    logger.info("Exclusions: %s", memory.loc[~memory["in_primary_corpus"], "exclusion_reason"].value_counts().to_dict())
    logger.info("Primary corpus by intent: %s", primary["intent"].value_counts().to_dict())
    logger.info("Primary corpus by resolution type: %s", primary["resolution_type"].value_counts().to_dict())
    (processed / "virgintrains_resolution_memory_manifest.json").write_text(
        json.dumps(
            {
                "file": MEMORY_NAME,
                "file_sha256": sha256_file(out),
                "split": TRAIN,
                "n_episodes": int(len(memory)),
                "n_primary_corpus": int(len(primary)),
                "evidence_quality": memory["evidence_quality"].value_counts().to_dict(),
                "exclusion_reasons": memory.loc[~memory["in_primary_corpus"], "exclusion_reason"].value_counts().to_dict(),
                "intent_source": "candidate_taxonomy_not_ground_truth",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
