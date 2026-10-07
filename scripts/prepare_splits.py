"""Create leakage-safe splits and the inspection sample.

Outputs:
    data/processed/splits/virgintrains_train_retrieval.parquet   retrieval corpus / prompt examples
    data/processed/splits/virgintrains_dev_calibration.parquet   thresholds / calibration
    data/processed/splits/virgintrains_split_assignments.csv     every case -> split + reason
    data/processed/splits/virgintrains_split_manifest.json       config, counts, leakage checks, usage rules
    data/golden/virgintrains_golden_candidates.{parquet,csv}     golden eval candidates (unlabelled)
    data/processed/virgintrains_case_sample.csv                  diverse inspection sample (train+dev only)
    reports/virgintrains_case_inspection.md

Run after discover_intents.py. Golden candidates are never used for sampling examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from common.logging_utils import configure_logging
from evaluation.inspection import conversation_text, render_case_markdown
from evaluation.sampling import add_coverage_columns, diversity_sample
from evaluation.splits import DEV, EXCLUDED, GOLDEN, RESERVE, TRAIN, SplitConfig, finalize_splits, provisional_split, verify_no_leakage

logger = logging.getLogger("prepare_splits")

USAGE_RULES = {
    TRAIN: "Retrieval corpus, prompt/few-shot examples, taxonomy examples.",
    DEV: "Threshold tuning, calibration, prompt selection. Never used as retrieval corpus for golden evaluation.",
    GOLDEN: "Final reporting only. Never used for retrieval, prompt examples, or threshold tuning.",
    RESERVE: "Held back with the golden pool. Not used for retrieval, prompts, or inspection.",
    EXCLUDED: "Removed from train/dev because it duplicates or shares context with a golden case.",
}
LABEL_COLUMNS = ["human_intent", "human_should_auto_handle", "human_escalation_reason", "human_reply_acceptable", "human_notes"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cluster_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {int(v["cluster_id_at_review"]): v["name"] for v in spec.get("clusters", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--golden-dir", type=Path, default=_bootstrap.REPO_ROOT / "data" / "golden")
    parser.add_argument("--reports-dir", type=Path, default=_bootstrap.REPO_ROOT / "reports")
    parser.add_argument("--labels", type=Path, default=_bootstrap.REPO_ROOT / "configs" / "virgintrains_cluster_labels.yaml")
    parser.add_argument("--golden-size", type=int, default=SplitConfig.golden_size)
    parser.add_argument("--sample-size", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    cfg = SplitConfig(seed=args.seed, golden_size=args.golden_size)

    cases = pd.read_parquet(args.processed_dir / "virgintrains_cases.parquet")
    clusters = pd.read_parquet(args.processed_dir / "virgintrains_case_clusters.parquet").set_index("case_id")["cluster_id"]
    assigned = finalize_splits(cases, provisional_split(cases, cfg), clusters, cfg)
    checks = verify_no_leakage(assigned)
    logger.info("Leakage checks (all must be 0): %s", checks)

    split_dir = args.processed_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    args.golden_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    keep = [c for c in assigned.columns if c not in ("provisional_split",)]
    files: dict[str, Path] = {}
    for name, split in (("train_retrieval", TRAIN), ("dev_calibration", DEV)):
        path = split_dir / f"virgintrains_{name}.parquet"
        assigned.loc[assigned["split"] == split, keep].to_parquet(path, index=False, compression="zstd")
        files[name] = path
    golden = assigned.loc[assigned["split"] == GOLDEN, keep].copy()
    golden_path = args.golden_dir / "virgintrains_golden_candidates.parquet"
    golden.to_parquet(golden_path, index=False, compression="zstd")
    files["golden_candidates"] = golden_path
    golden_csv = golden[["case_id", "cluster_id", "resolution_type", "resolved", "dm_redirect", "length_bucket", "time_period", "golden_stratum_weight", "opening_message"]].copy()
    golden_csv["conversation"] = [conversation_text(list(t)) for t in golden["full_turns"]]
    for col in LABEL_COLUMNS:
        golden_csv[col] = ""
    golden_csv.to_csv(args.golden_dir / "virgintrains_golden_candidates.csv", index=False, encoding="utf-8")

    assigned[["case_id", "split", "split_reason", "group_id", "customer_id", "conversation_id", "cluster_id"]].to_csv(
        split_dir / "virgintrains_split_assignments.csv", index=False
    )

    manifest = {
        "config": cfg.to_dict(),
        "counts": assigned["split"].value_counts().to_dict(),
        "golden_cluster_counts": golden["cluster_id"].value_counts().sort_index().to_dict(),
        "leakage_checks_all_zero": checks,
        "usage_rules": USAGE_RULES,
        "file_sha256": {k: sha256(v) for k, v in files.items()},
        "notes": [
            "Groups = cases linked by shared customer or conversation; groups are hashed to splits.",
            "Intent clusters were fit on train+dev openers only; golden-pool cases were assigned by nearest centroid.",
            "Golden cases are stratified (not natural prevalence); use golden_stratum_weight for prevalence-weighted metrics.",
            "Golden candidates are unlabelled; fill the human_* columns in the CSV.",
        ],
    }
    (split_dir / "virgintrains_split_manifest.json").write_text(json.dumps(manifest, indent=2, default=int), encoding="utf-8")

    population = assigned[assigned["split"].isin([TRAIN, DEV])]
    dims = ["length_bucket", "resolved", "resolution_type", "dm_redirect", "time_period", "cluster_id"]
    picked = diversity_sample(population, dims, args.sample_size, args.seed)
    sample = population.loc[picked].sort_values("first_timestamp")
    sample_csv = sample[["case_id", "split", "cluster_id", "length_bucket", "time_period", "turn_count", "resolved", "resolution_type", "dm_redirect",
                         "first_timestamp", "customer_id", "conversation_id", "opening_message", "resolution_summary"]].copy()
    sample_csv["conversation"] = [conversation_text(list(t)) for t in sample["full_turns"]]
    sample_csv["source_tweet_ids"] = [json.dumps([int(x) for x in ids]) for ids in sample["source_tweet_ids"]]
    sample_csv.to_csv(args.processed_dir / "virgintrains_case_sample.csv", index=False, encoding="utf-8")
    for dim in dims:
        logger.info("Sample coverage %s: %s", dim, sample[dim].astype(str).value_counts().to_dict())

    names = load_cluster_names(args.labels)
    header = [
        "# VirginTrains case inspection sample",
        "",
        f"{len(sample)} cases drawn by coverage sampling over {', '.join(dims)} from train + dev only "
        "(golden-pool cases are never shown). Rare levels are over-represented on purpose; this is not a prevalence-faithful sample.",
        "",
        "---",
        "",
    ]
    body = [render_case_markdown(r, names.get(int(r["cluster_id"]))) for _, r in sample.iterrows()]
    (args.reports_dir / "virgintrains_case_inspection.md").write_text("\n".join(header + body), encoding="utf-8")
    logger.info("Wrote %d-case sample, inspection markdown, and splits %s", len(sample), manifest["counts"])


if __name__ == "__main__":
    main()
