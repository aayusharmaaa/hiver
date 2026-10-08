"""Leakage and freeze-order guarantees, checked against the real generated artifacts (skipped if the pipeline has not been run)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from evaluation.splits import DEV, GOLDEN, RESERVE, TRAIN
from evaluation.taxonomy_calibration import HUMAN_COLUMNS
from taxonomy.finalize import verify_frozen
from taxonomy.registry import case_ids_sha256, cluster_intent_map, load_labels, registry_status, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SPLITS = PROCESSED / "splits"
GOLDEN_DIR = ROOT / "data" / "golden"
CONFIGS = ROOT / "configs"

FILES = {
    "assignments": SPLITS / "virgintrains_split_assignments.csv",
    "train": SPLITS / "virgintrains_train_retrieval.parquet",
    "dev": SPLITS / "virgintrains_dev_calibration.parquet",
    "split_manifest": SPLITS / "virgintrains_split_manifest.json",
    "golden": GOLDEN_DIR / "virgintrains_golden_candidates.parquet",
    "cases": PROCESSED / "virgintrains_cases.parquet",
    "calibration": PROCESSED / "taxonomy_calibration.csv",
    "calibration_manifest": PROCESSED / "taxonomy_calibration_manifest.json",
    "preview": PROCESSED / "virgintrains_cluster_preview.json",
    "labels": CONFIGS / "virgintrains_cluster_labels.yaml",
    "registry": CONFIGS / "virgintrains_intents.yaml",
}
FROZEN = CONFIGS / "virgintrains_taxonomy_v1.yaml"
GOLDEN_V1 = GOLDEN_DIR / "virgintrains_golden_v1.csv"

pytestmark = pytest.mark.skipif(not all(p.exists() for p in FILES.values()), reason="pipeline artifacts not generated")


@pytest.fixture(scope="module")
def assignments() -> pd.DataFrame:
    return pd.read_csv(FILES["assignments"])


@pytest.fixture(scope="module")
def golden_ids(assignments) -> set[str]:
    return set(assignments.loc[assignments["split"] == GOLDEN, "case_id"])


@pytest.fixture(scope="module")
def golden() -> pd.DataFrame:
    return pd.read_parquet(FILES["golden"])


@pytest.fixture(scope="module")
def calibration() -> pd.DataFrame:
    return pd.read_csv(FILES["calibration"], dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def cases() -> pd.DataFrame:
    return pd.read_parquet(FILES["cases"], columns=["case_id", "source_tweet_ids", "full_turns"])


def test_no_golden_case_in_train_retrieval(golden_ids) -> None:
    train = set(pd.read_parquet(FILES["train"], columns=["case_id"])["case_id"])
    assert golden_ids and not (golden_ids & train)


def test_no_golden_case_in_dev_calibration(golden_ids) -> None:
    dev = set(pd.read_parquet(FILES["dev"], columns=["case_id"])["case_id"])
    assert not (golden_ids & dev)


def test_no_golden_customer_group_or_conversation_in_train_or_dev(assignments) -> None:
    g = assignments[assignments["split"] == GOLDEN]
    td = assignments[assignments["split"].isin([TRAIN, DEV])]
    for col in ("customer_id", "group_id", "conversation_id"):
        assert not (set(g[col]) & set(td[col])), col


def test_golden_candidates_are_exactly_the_original_sampling(golden, golden_ids) -> None:
    manifest = json.loads(FILES["split_manifest"].read_text(encoding="utf-8"))
    assert set(golden["case_id"]) == golden_ids and len(golden) == manifest["counts"][GOLDEN] == 250
    assert sha256_file(FILES["golden"]) == manifest["file_sha256"]["golden_candidates"], "golden candidates were modified after sampling"
    assert sha256_file(FILES["train"]) == manifest["file_sha256"]["train_retrieval"]
    assert sha256_file(FILES["dev"]) == manifest["file_sha256"]["dev_calibration"]


def test_no_golden_case_is_used_by_taxonomy_calibration(calibration, golden_ids, assignments) -> None:
    ids = set(calibration["case_id"])
    assert ids and not (ids & golden_ids)
    splits = assignments.set_index("case_id").loc[sorted(ids), "split"]
    assert set(splits) == {RESERVE}, "calibration must draw only from golden_pool_reserve"
    golden_rows = assignments[assignments["split"] == GOLDEN]
    cal_rows = assignments[assignments["case_id"].isin(ids)]
    for col in ("customer_id", "group_id", "conversation_id"):
        assert not (set(golden_rows[col]) & set(cal_rows[col])), f"calibration shares {col} with golden"


def test_calibration_manifest_matches_current_golden_and_sample(calibration, golden_ids) -> None:
    manifest = json.loads(FILES["calibration_manifest"].read_text(encoding="utf-8"))
    assert manifest["golden_case_ids_sha256"] == case_ids_sha256(golden_ids)
    assert manifest["calibration_case_ids_sha256"] == case_ids_sha256(calibration["case_id"])
    assert manifest["source_split"] == RESERVE and all(v == 0 for v in manifest["leakage_checks_all_zero"].values())


def test_calibration_human_labels_are_never_prefilled_by_the_pipeline(calibration) -> None:
    """Until a person labels the sheet, every human_* cell is blank. Once labelled, the manifest still records no labels were generated."""
    manifest = json.loads(FILES["calibration_manifest"].read_text(encoding="utf-8"))
    assert "Do not pre-fill" in " ".join(manifest["notes"])
    if not (calibration[HUMAN_COLUMNS[:4]] != "").any(axis=None):
        assert (calibration[HUMAN_COLUMNS] == "").all(axis=None)


def test_calibration_sample_contains_only_machine_columns_plus_blank_human_columns(calibration) -> None:
    for col in ("case_id", "first_customer_message", "conversation", "candidate_intent", "candidate_cluster_id", "auto_resolution_type", "auto_resolved", "source_tweet_ids"):
        assert col in calibration
    assert list(calibration.columns[-5:]) == HUMAN_COLUMNS


def test_every_golden_case_maps_to_exactly_one_final_intent_or_fallback(golden) -> None:
    preview = json.loads(FILES["preview"].read_text(encoding="utf-8"))
    cmap = cluster_intent_map(preview, load_labels(FILES["labels"]))
    registry = yaml.safe_load(FILES["registry"].read_text(encoding="utf-8"))["taxonomy"]
    valid = {i["name"] for i in registry["intents"]} | {registry["fallback"]["name"]}
    renames: dict[str, str] = {}
    if FROZEN.exists():
        frozen = yaml.safe_load(FROZEN.read_text(encoding="utf-8"))
        valid = set(frozen["labels"])
        renames = frozen["taxonomy"].get("renames_applied", {})
    assert golden["cluster_id"].notna().all()
    assert golden["case_id"].is_unique, "a golden case must appear exactly once"
    for case_id, cid in zip(golden["case_id"], golden["cluster_id"]):
        entry = cmap.get(int(cid))
        assert entry and entry.get("final_intent"), f"{case_id}: cluster {cid} has no intent"
        intent = renames.get(entry["final_intent"], entry["final_intent"])
        assert intent in valid, f"{case_id}: {intent} is not a registry intent or the fallback"


def test_all_case_ids_retain_source_provenance(cases, assignments, golden, calibration) -> None:
    assert cases["case_id"].is_unique
    assert set(assignments["case_id"]) <= set(cases["case_id"]), "every split case traces to a source case"
    assert (cases["source_tweet_ids"].map(len) > 0).all()
    for row in cases.itertuples():
        src = {int(x) for x in row.source_tweet_ids}
        assert row.case_id.startswith("case_")
        for turn in row.full_turns:
            assert int(turn["tweet_id"]) in src and turn.get("source_row") is not None
    for label, frame in (("golden", golden), ("calibration", calibration)):
        assert set(frame["case_id"]) <= set(cases["case_id"]), label
    by_id = cases.set_index("case_id")["source_tweet_ids"]
    for cid, ids in zip(golden["case_id"], golden["source_tweet_ids"]):
        assert [int(x) for x in ids] == [int(x) for x in by_id[cid]], "golden provenance diverges from the case table"
    for cid, ids in zip(calibration["case_id"], calibration["source_tweet_ids"]):
        assert [int(x) for x in json.loads(ids)] == [int(x) for x in by_id[cid]], "calibration provenance diverges from the case table"


def test_taxonomy_is_frozen_before_golden_evaluation_and_stays_frozen(golden_ids) -> None:
    status = registry_status(FILES["registry"])
    if not FROZEN.exists():
        # The golden pack may exist before a freeze: it is labelled against the CANDIDATE taxonomy (scripts/prepare_golden_eval.py).
        if GOLDEN_V1.exists():
            pack = json.loads((GOLDEN_DIR / "virgintrains_golden_v1.manifest.json").read_text(encoding="utf-8"))
            assert pack["taxonomy_reference"]["status"] == "CANDIDATE_NOT_GROUND_TRUTH"
            assert pack["golden_case_ids_sha256"] == case_ids_sha256(golden_ids)
        assert not (GOLDEN_DIR / "virgintrains_golden_v1.meta.json").exists(), "the freeze-gated golden script ran without a frozen taxonomy"
        assert status != "HUMAN_CALIBRATED", "a calibrated registry without a frozen v1 means the freeze was skipped"
        return
    frozen = yaml.safe_load(FROZEN.read_text(encoding="utf-8"))
    assert verify_frozen(frozen) == [], "frozen taxonomy was modified"
    assert status == "HUMAN_CALIBRATED"
    meta = frozen["metadata"]
    assert meta["golden_case_ids_sha256"] == case_ids_sha256(golden_ids), "golden set changed after freezing"
    assert meta["source_split"] == RESERVE
    side_path = GOLDEN_DIR / "virgintrains_golden_v1.meta.json"
    if GOLDEN_V1.exists() and side_path.exists():
        side = json.loads(side_path.read_text(encoding="utf-8"))
        assert side["taxonomy_content_sha256"] == meta["content_sha256"], "golden_v1.csv was prepared against a different taxonomy"
        assert side["golden_case_ids_sha256"] == meta["golden_case_ids_sha256"]


def test_golden_v1_when_present_covers_exactly_the_golden_cases_and_uses_only_allowed_labels(golden_ids) -> None:
    if not GOLDEN_V1.exists():
        pytest.skip("golden_v1.csv not prepared yet")
    df = pd.read_csv(GOLDEN_V1, dtype=str, keep_default_na=False)
    assert set(df["case_id"]) == golden_ids and df["case_id"].is_unique and len(df) == 250
    pack = json.loads((GOLDEN_DIR / "virgintrains_golden_v1.manifest.json").read_text(encoding="utf-8"))
    filled = df.loc[df["gold_intent"].str.strip() != "", "gold_intent"]
    assert all(v in pack["allowed"]["intents"] or v.startswith("NEW:") for v in filled), "gold_intent must be a candidate intent or NEW:<name>"
    assert list(df.columns[-5:]) == ["gold_intent", "gold_should_escalate", "gold_resolution_type", "gold_confidence", "human_notes"]
