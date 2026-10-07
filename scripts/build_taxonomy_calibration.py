"""Draw the taxonomy calibration sample from the golden-pool reserve and write the labeling pack.

Outputs:
    data/processed/taxonomy_calibration.csv            one row per case; human_* columns are EMPTY
    data/processed/taxonomy_calibration_guide.md       reviewer guide only (also shown in the labeling tool)
    data/processed/taxonomy_calibration_labeling.md    guide + every case, readable offline (shuffled order)
    data/processed/taxonomy_calibration_manifest.json  config, allocation, exclusions, safety checks, source-column fingerprint

Run after prepare_splits.py and generate_taxonomy.py. Refuses to overwrite a CSV that already contains
human labels, and refuses to run once configs/virgintrains_taxonomy_v1.yaml exists (the sample is frozen).
Only golden_pool_reserve cases are ever sampled; golden cases are used only to *exclude* neighbours.

`--guide-only` rewrites just the two markdown files from the EXISTING CSV (safe after labelling has started): it never
touches the CSV or the manifest. Guide examples come from train_retrieval cases only (enforced).
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
from evaluation.calibration_guide import render_guidelines, render_labeling_markdown
from evaluation.labeling_examples import load_guide_examples
from evaluation.sampling import add_coverage_columns
from evaluation.splits import GOLDEN, RESERVE
from evaluation.taxonomy_calibration import (
    DEFAULT_QUOTAS,
    HUMAN_COLUMNS,
    CalibrationConfig,
    assert_calibration_safe,
    calibration_frame,
    exclusion_reasons,
    sample_calibration,
    source_columns_sha256,
)
from taxonomy.registry import case_ids_sha256, cluster_intent_map, load_labels, sha256_text

logger = logging.getLogger("build_taxonomy_calibration")

CASE_COLUMNS = [
    "case_id", "opening_message", "full_turns", "resolution_type", "resolved", "dm_redirect", "resolution_outcome", "resolution_summary",
    "turn_count", "first_timestamp", "is_continuation", "starts_with_customer", "has_brand_reply",
    "source_tweet_ids", "context_tweet_ids",
]


def has_human_labels(path: Path) -> bool:
    if not path.exists():
        return False
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return any(c in df and (df[c].str.strip() != "").any() for c in HUMAN_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--labels", type=Path, default=_bootstrap.REPO_ROOT / "configs" / "virgintrains_cluster_labels.yaml")
    parser.add_argument("--registry", type=Path, default=_bootstrap.REPO_ROOT / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--examples", type=Path, default=_bootstrap.REPO_ROOT / "configs" / "virgintrains_labeling_examples.yaml")
    parser.add_argument("--frozen", type=Path, default=_bootstrap.REPO_ROOT / "configs" / "virgintrains_taxonomy_v1.yaml")
    parser.add_argument("--size", type=int, default=CalibrationConfig.target_size)
    parser.add_argument("--seed", type=int, default=CalibrationConfig.seed)
    parser.add_argument("--guide-only", action="store_true", help="Rewrite only the markdown guide/pack from the existing CSV.")
    parser.add_argument("--force", action="store_true", help="Overwrite even if the CSV already has human labels (destroys them).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    out_csv = args.processed_dir / "taxonomy_calibration.csv"
    out_md = args.processed_dir / "taxonomy_calibration_labeling.md"
    out_guide = args.processed_dir / "taxonomy_calibration_guide.md"
    out_manifest = args.processed_dir / "taxonomy_calibration_manifest.json"
    if args.frozen.exists() and not args.guide_only:
        raise SystemExit(f"{args.frozen} exists: the taxonomy is frozen, so the calibration sample must not change.")
    if not args.guide_only and has_human_labels(out_csv) and not args.force:
        raise SystemExit(f"{out_csv} already contains human labels. Refusing to overwrite (use --guide-only to refresh the guide, or --force to discard them).")

    cfg = CalibrationConfig(seed=args.seed, target_size=args.size)
    cases = pd.read_parquet(args.processed_dir / "virgintrains_cases.parquet", columns=CASE_COLUMNS + ["customer_id", "conversation_id"])
    assignments = pd.read_csv(args.processed_dir / "splits" / "virgintrains_split_assignments.csv")
    clusters = pd.read_parquet(args.processed_dir / "virgintrains_case_clusters.parquet").drop(columns=["cluster_id"])
    clusters_full = pd.read_parquet(args.processed_dir / "virgintrains_case_clusters.parquet", columns=["case_id", "cluster_id"])
    preview = json.loads((args.processed_dir / "virgintrains_cluster_preview.json").read_text(encoding="utf-8"))
    labels = load_labels(args.labels)
    cmap = cluster_intent_map(preview, labels)
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["taxonomy"]
    golden_ids = assignments.loc[assignments["split"] == GOLDEN, "case_id"]

    if args.guide_only:
        frame = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
        write_guides(frame, cases, assignments, clusters_full, cmap, registry, labels, args, out_md, out_guide, calibration_ids=set(frame["case_id"]), golden_ids=set(golden_ids))
        logger.info("Guide-only refresh: %s and %s rewritten; CSV and manifest untouched.", out_guide, out_md)
        return

    golden = cases[cases["case_id"].isin(golden_ids)].merge(assignments[["case_id", "group_id"]], on="case_id", suffixes=("", "_a"))
    reserve = (
        assignments.loc[assignments["split"] == RESERVE, ["case_id", "split", "group_id"]]
        .merge(cases, on="case_id")
        .merge(clusters_full, on="case_id")
        .merge(clusters, on="case_id", how="left")
    )
    reserve["cluster_name"] = reserve["cluster_id"].map(lambda c: cmap.get(int(c), {}).get("cluster_name"))
    reserve["candidate_intent"] = reserve["cluster_id"].map(lambda c: cmap.get(int(c), {}).get("final_intent"))
    reserve["runner_up_cluster_name"] = reserve["second_cluster_id"].map(lambda c: cmap.get(int(c), {}).get("cluster_name") if pd.notna(c) else None)
    reserve["runner_up_intent"] = reserve["second_cluster_id"].map(lambda c: cmap.get(int(c), {}).get("final_intent") if pd.notna(c) else None)
    reserve["margin"] = reserve["margin"].fillna(reserve["margin"].median())
    reserve = add_coverage_columns(reserve)

    reasons = exclusion_reasons(reserve, golden)
    reasons[(reasons == "") & reserve["candidate_intent"].isna()] = "no_candidate_intent"
    logger.info("Reserve %d cases; exclusions: %s", len(reserve), reasons[reasons != ""].value_counts().to_dict())
    pool = reserve[reasons == ""]

    sample = sample_calibration(pool, cfg)
    checks = assert_calibration_safe(sample, assignments)
    frame = calibration_frame(sample)
    frame["labeling_order"] = frame["case_id"].map(lambda c: int(sha256_text(f"{cfg.seed}:{c}")[:12], 16)).rank(method="first").astype(int)
    frame = frame.sort_values("labeling_order").reset_index(drop=True)
    cols = ["case_id", "labeling_order"] + [c for c in frame.columns if c not in ("case_id", "labeling_order", *HUMAN_COLUMNS)] + HUMAN_COLUMNS
    frame = frame[cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False, encoding="utf-8")
    on_disk = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
    write_guides(on_disk, cases, assignments, clusters_full, cmap, registry, labels, args, out_md, out_guide, calibration_ids=set(sample["case_id"]), golden_ids=set(golden_ids))

    manifest = {
        "config": {"seed": cfg.seed, "target_size": cfg.target_size, "low_margin_fraction": cfg.low_margin_fraction, "low_margin_quantile": cfg.low_margin_quantile},
        "source_split": RESERVE,
        "reserve_cases": int(len(reserve)),
        "exclusions": {k: int(v) for k, v in reasons[reasons != ""].value_counts().items()},
        "eligible_cases": int(len(pool)),
        "eligible_by_candidate_intent": {k: int(v) for k, v in pool["candidate_intent"].value_counts().items()},
        "quotas": DEFAULT_QUOTAS,
        "selected_by_candidate_intent": {k: int(v) for k, v in sample["candidate_intent"].value_counts().items()},
        "selection_reason": {k: int(v) for k, v in sample["selection_reason"].value_counts().items()},
        "n_selected": int(len(sample)),
        "leakage_checks_all_zero": checks,
        "calibration_case_ids_sha256": case_ids_sha256(sample["case_id"]),
        "source_columns_sha256": source_columns_sha256(on_disk),
        "golden_case_ids_sha256": case_ids_sha256(golden_ids),
        "notes": [
            "Sample is stratified to over-represent confusable intents; use stratum_weight for within-intent prevalence only.",
            "Reserve cases that share a group, customer, conversation, tweet, or opening text with any golden case were excluded.",
            "human_* columns are empty. Do not pre-fill them with model output.",
            "source_columns_sha256 fingerprints every non-human column; the labeling tool refuses to run if it changes.",
        ],
    }
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Selected %d cases: %s", len(sample), manifest["selected_by_candidate_intent"])
    logger.info("Wrote %s, %s, %s, %s", out_csv, out_guide, out_md, out_manifest)


def write_guides(frame, cases, assignments, clusters_full, cmap, registry, labels, args, out_md, out_guide, calibration_ids, golden_ids) -> None:
    """Write the guide and the offline pack. Worked examples are train cases only; calibration/golden ids are forbidden."""
    split_of = assignments.set_index("case_id")["split"]
    wanted = {c["case_id"] for s in yaml.safe_load(args.examples.read_text(encoding="utf-8"))["sections"] for c in s["cases"]}
    candidate_of = {r.case_id: (cmap.get(int(r.cluster_id)) or {}).get("final_intent") for r in clusters_full[clusters_full["case_id"].isin(wanted)].itertuples()}
    examples = load_guide_examples(args.examples, cases[cases["case_id"].isin(wanted)], split_of, calibration_ids | golden_ids, candidate_of)
    notes = labels.get("confusable_notes", [])
    by_case = cases[cases["case_id"].isin(set(frame["case_id"]))].set_index("case_id")["full_turns"]
    turns = {cid: list(by_case[cid]) for cid in frame["case_id"]}
    out_guide.write_text(render_guidelines(registry, notes, len(frame), examples), encoding="utf-8")
    out_md.write_text(render_labeling_markdown(frame, turns, registry, notes, examples), encoding="utf-8")


if __name__ == "__main__":
    main()
