"""Finalize and freeze the taxonomy from human calibration labels plus the reviewer's decisions.

Gates (the script exits non-zero and writes nothing if any fails):
  1. configs/virgintrains_taxonomy_v1.yaml must not exist yet (a frozen taxonomy is immutable).
  2. The registry must still be the candidate (not already HUMAN_CALIBRATED).
  3. The golden case set must be byte-for-byte what it was when the calibration sample was drawn.
  4. The calibration CSV must contain exactly the sampled case ids, with >= 95% of rows fully labelled and no invalid labels.
  5. configs/virgintrains_taxonomy_decisions.yaml must be signed by a named reviewer and be internally valid.
  6. The resulting registry must have 8-12 intents + a fallback, every required field, and examples only from train_retrieval.

Writes:
  configs/virgintrains_intents.yaml        status HUMAN_CALIBRATED (CANDIDATE marker removed)
  configs/virgintrains_taxonomy_v1.yaml    frozen copy with metadata and a content hash
  reports/taxonomy_calibration_report.md   comparison report for the labels used

Use --dry-run to run every check and print the result without writing anything.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml
from compare_taxonomy import render

from common.logging_utils import configure_logging
from evaluation.splits import GOLDEN
from evaluation.taxonomy_comparison import Thresholds, compare, load_labelled
from taxonomy.finalize import DecisionError, apply_decisions, build_frozen, finalize_registry, validate_registry, verify_frozen
from taxonomy.registry import STATUS_CALIBRATED, case_ids_sha256, intent_names, registry_status, sha256_file

logger = logging.getLogger("finalize_taxonomy")


def fail(msg: str) -> None:
    logger.error(msg)
    raise SystemExit(1)


def main() -> None:
    root = _bootstrap.REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--golden-dir", type=Path, default=root / "data" / "golden")
    parser.add_argument("--registry", type=Path, default=root / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--frozen", type=Path, default=root / "configs" / "virgintrains_taxonomy_v1.yaml")
    parser.add_argument("--decisions", type=Path, default=root / "configs" / "virgintrains_taxonomy_decisions.yaml")
    parser.add_argument("--reports-dir", type=Path, default=root / "reports")
    parser.add_argument("--min-labelled-fraction", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.frozen.exists():
        fail(f"{args.frozen} already exists. A frozen taxonomy is immutable; create a new version deliberately if you really need one.")
    if registry_status(args.registry) == STATUS_CALIBRATED:
        fail(f"{args.registry} is already HUMAN_CALIBRATED.")

    calib_csv = args.processed_dir / "taxonomy_calibration.csv"
    manifest = json.loads((args.processed_dir / "taxonomy_calibration_manifest.json").read_text(encoding="utf-8"))
    assignments = pd.read_csv(args.processed_dir / "splits" / "virgintrains_split_assignments.csv")
    golden_ids = assignments.loc[assignments["split"] == GOLDEN, "case_id"]
    if case_ids_sha256(golden_ids) != manifest["golden_case_ids_sha256"]:
        fail("The golden case set changed since the calibration sample was drawn. Re-draw the calibration sample from a consistent split.")

    raw = pd.read_csv(calib_csv, dtype=str, keep_default_na=False)
    if case_ids_sha256(raw["case_id"]) != manifest["calibration_case_ids_sha256"]:
        fail("The calibration CSV no longer contains exactly the sampled case ids (rows added or removed).")
    candidate = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["taxonomy"]
    names = intent_names(candidate)
    fallback = candidate["fallback"]["name"]
    load = load_labelled(raw, names, fallback)
    if load.problems:
        fail("Invalid human labels:\n  " + "\n  ".join(load.problems[:25]))
    fraction = load.n_complete / load.n_rows if load.n_rows else 0.0
    if fraction < args.min_labelled_fraction:
        fail(f"Only {load.n_complete}/{load.n_rows} calibration cases ({100 * fraction:.0f}%) are fully labelled; need >= {100 * args.min_labelled_fraction:.0f}%. "
             "Labels are never auto-filled.")

    if not args.decisions.exists():
        fail(f"{args.decisions} not found. Write your merge/split decisions after reading reports/taxonomy_calibration_report.md.")
    decisions = yaml.safe_load(args.decisions.read_text(encoding="utf-8")) or {}
    try:
        decided = apply_decisions(candidate, decisions)
    except DecisionError as exc:
        fail(f"Invalid decisions: {exc}")

    train_ids = set(pd.read_parquet(args.processed_dir / "splits" / "virgintrains_train_retrieval.parquet", columns=["case_id"])["case_id"])
    forbidden = set(golden_ids) | set(raw["case_id"])
    errors = validate_registry(decided, train_ids, forbidden)
    if errors:
        fail("Final registry is invalid:\n  " + "\n  ".join(errors[:30]))

    labelled = load.frame.copy()
    labelled["stratum_weight"] = pd.to_numeric(labelled["stratum_weight"], errors="coerce").fillna(1.0)
    th = Thresholds()
    result = compare(labelled, names, fallback, th)
    report_md = render(result, load, th, fallback)

    calibration = {
        "set": "golden_pool_reserve",
        "calibration_set_size": load.n_rows,
        "labelled_cases": load.n_complete,
        "candidate_vs_human_agreement": round(result["overall"]["agreement"], 3),
        "calibration_case_ids_sha256": manifest["calibration_case_ids_sha256"],
        "calibration_labels_sha256": sha256_file(calib_csv),
        "decisions_sha256": sha256_file(args.decisions),
        "report": "reports/taxonomy_calibration_report.md",
    }
    registry = finalize_registry(decided, decisions, calibration)
    frozen = build_frozen(
        registry,
        {
            "calibration_set_size": load.n_rows,
            "labelled_cases": load.n_complete,
            "golden_case_ids_sha256": manifest["golden_case_ids_sha256"],
            "golden_candidates_parquet_sha256": sha256_file(args.golden_dir / "virgintrains_golden_candidates.parquet"),
            "calibration_labels_sha256": calibration["calibration_labels_sha256"],
            "decisions_sha256": calibration["decisions_sha256"],
            "reviewed_by": decisions["reviewed_by"],
            "reviewed_on": decisions["reviewed_on"],
        },
    )
    problems = verify_frozen(frozen)
    if problems:
        fail("Frozen taxonomy failed verification: " + "; ".join(problems))
    logger.info("Final taxonomy: %d intents + fallback %s", len(decided["intents"]), fallback)
    for it in decided["intents"]:
        logger.info("  %-34s n=%d", it["name"], it["n_cases"])
    if args.dry_run:
        logger.info("Dry run: nothing written.")
        return

    header = "# VirginTrains intent registry. HUMAN_CALIBRATED on the golden-pool reserve set. Generated by scripts/finalize_taxonomy.py.\n"
    args.registry.write_text(header + yaml.safe_dump(registry, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    args.frozen.write_text(
        "# FROZEN taxonomy v1. Do not edit: metadata.content_sha256 is verified by tests and by the classifier loader.\n"
        + yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True, width=110),
        encoding="utf-8",
    )
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "taxonomy_calibration_report.md").write_text(report_md, encoding="utf-8")
    logger.info("Wrote %s and %s (content_sha256 %s)", args.registry, args.frozen, frozen["metadata"]["content_sha256"][:12])


if __name__ == "__main__":
    main()
