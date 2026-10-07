from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from taxonomy_fixtures import FALLBACK, INTENTS, candidate_taxonomy, write_world

from taxonomy.finalize import (
    DecisionError,
    apply_decisions,
    build_frozen,
    finalize_registry,
    validate_decisions,
    validate_registry,
    verify_frozen,
)
from taxonomy.registry import STATUS_CALIBRATED, registry_status

ROOT = Path(__file__).resolve().parents[1]
SIGNED = {"reviewed_by": "Reviewer", "reviewed_on": "2026-10-08"}


def tax() -> dict:
    return copy.deepcopy(candidate_taxonomy()["taxonomy"])


def test_unsigned_or_malformed_decisions_are_rejected() -> None:
    names = set(INTENTS) | {FALLBACK}
    assert any("reviewed_by" in e for e in validate_decisions({"reviewed_on": "2026-10-08"}, names))
    assert any("reviewed_on" in e for e in validate_decisions({"reviewed_by": "A", "reviewed_on": "yesterday"}, names))
    errors = validate_decisions({**SIGNED, "merges": [{"into": "x", "from": ["ghost"]}], "renames": {"ghost": "y"}, "edits": {"a": {"name": "z"}}, "new_intents": [{"name": "n"}]}, names)
    text = " | ".join(errors)
    assert "definition" in text and "ghost" in text and "non-editable" in text and "missing" in text
    with pytest.raises(DecisionError):
        apply_decisions(tax(), {})


def test_empty_decisions_keep_the_candidate_intents_but_mark_them_calibrated() -> None:
    out = apply_decisions(tax(), SIGNED)
    assert {i["name"] for i in out["intents"]} == set(INTENTS)
    assert all(i["status"] == "human_calibrated" for i in out["intents"])
    assert out["fallback"]["status"] == "human_calibrated"


def test_merge_combines_counts_examples_and_rewrites_confusables() -> None:
    decisions = {**SIGNED, "merges": [{"into": "service_status_delay_enquiry", "from": ["journey_disruption_complaint"], "definition": "Delay  and disruption  enquiries."}]}
    out = apply_decisions(tax(), decisions)
    names = [i["name"] for i in out["intents"]]
    assert "journey_disruption_complaint" not in names and len(names) == len(INTENTS) - 1
    merged = next(i for i in out["intents"] if i["name"] == "service_status_delay_enquiry")
    assert merged["definition"] == "Delay and disruption enquiries."
    assert merged["n_cases"] == 100 + 99
    assert {c["case_id"] for c in merged["positive_examples"]} == {"train_1", "train_2"}
    assert out["renames_applied"]["journey_disruption_complaint"] == "service_status_delay_enquiry"
    assert abs(sum(merged["resolution_types"].values()) - 1.0) < 0.05
    assert merged["safe_to_auto_handle"] == "NEEDS_REVIEW"
    for i in out["intents"]:
        assert all(c["intent"] != "journey_disruption_complaint" and c["intent"] != i["name"] for c in i["known_confusable_intents"])


def test_rename_drop_new_and_edit() -> None:
    decisions = {
        **SIGNED,
        "renames": {"onboard_wifi_issue": "wifi_issue"},
        "drop_to_fallback": ["first_class_catering_issue"],
        "new_intents": [{"name": "lost_property_request", "definition": "Lost items.", "inclusion_criteria": ["a"], "exclusion_criteria": ["b"]}],
        "edits": {"seat_reservation_issue": {"definition": "Seat things."}, "first_class_catering_issue": {"definition": "now fallback text"}},
    }
    out = apply_decisions(tax(), decisions)
    names = [i["name"] for i in out["intents"]]
    assert "wifi_issue" in names and "onboard_wifi_issue" not in names
    assert "first_class_catering_issue" not in names and out["fallback"]["n_cases"] == 20 + 92
    new = next(i for i in out["intents"] if i["name"] == "lost_property_request")
    assert new["status"] == "human_proposed" and new["safe_to_auto_handle"] == "NEEDS_REVIEW" and new["n_cases"] == 0
    assert next(i for i in out["intents"] if i["name"] == "seat_reservation_issue")["definition"] == "Seat things."
    assert out["fallback"]["definition"] == "now fallback text"


def test_edit_of_a_vanished_intent_fails_loudly() -> None:
    with pytest.raises(DecisionError):
        apply_decisions(tax(), {**SIGNED, "edits": {"ghost": {"definition": "x"}}})


def test_validate_registry_checks_count_fields_provenance() -> None:
    ok = apply_decisions(tax(), SIGNED)
    assert validate_registry(ok, {"train_1", "train_2"}, {"gold_1"}) == []

    too_few = copy.deepcopy(ok)
    too_few["intents"] = too_few["intents"][:5]
    assert any("expected 8-12" in e for e in validate_registry(too_few))

    missing = copy.deepcopy(ok)
    missing["intents"][0]["definition"] = ""
    assert any("missing definition" in e for e in validate_registry(missing))

    leaky = copy.deepcopy(ok)
    leaky["intents"][0]["positive_examples"] = [{"text": "x", "case_id": "gold_1"}]
    errs = validate_registry(leaky, {"train_1", "train_2"}, {"gold_1"})
    assert any("not a train_retrieval case" in e for e in errs) and any("golden/calibration" in e for e in errs)

    dangling = copy.deepcopy(ok)
    dangling["intents"][0]["known_confusable_intents"] = [{"intent": "nonexistent", "evidence": "", "distinguishing_note": ""}]
    assert any("unknown intent" in e for e in validate_registry(dangling))

    dup = copy.deepcopy(ok)
    dup["intents"][1]["name"] = dup["intents"][0]["name"]
    assert any("duplicate" in e for e in validate_registry(dup))


def _frozen() -> dict:
    registry = finalize_registry(apply_decisions(tax(), SIGNED), SIGNED, {"set": "golden_pool_reserve"})
    meta = {"calibration_set_size": 200, "golden_case_ids_sha256": "g" * 64, "date_created": "2026-10-08"}
    return build_frozen(registry, meta)


def test_final_registry_drops_candidate_marker_and_documents_calibration() -> None:
    registry = finalize_registry(apply_decisions(tax(), SIGNED), SIGNED, {"set": "golden_pool_reserve", "calibration_set_size": 200})
    inner = registry["taxonomy"]
    assert inner["status"] == STATUS_CALIBRATED and "warning" not in inner
    assert "CANDIDATE_NOT_GROUND_TRUTH" not in yaml.safe_dump(registry)
    assert "human-calibrated" in inner["calibration"]["statement"] and "golden_pool_reserve" in inner["calibration"]["statement"]


def test_frozen_taxonomy_metadata_and_integrity() -> None:
    frozen = _frozen()
    meta = frozen["metadata"]
    for key in ("taxonomy_version", "calibration_set_size", "date_created", "source_split"):
        assert key in meta
    assert meta["taxonomy_version"] == "v1" and meta["source_split"] == "golden_pool_reserve" and meta["frozen"] is True
    assert frozen["labels"][-1] == FALLBACK == frozen["fallback"] and len(frozen["labels"]) == len(INTENTS) + 1
    assert verify_frozen(frozen) == []
    assert verify_frozen(yaml.safe_load(yaml.safe_dump(frozen))) == [], "must survive a YAML round-trip"


def test_any_edit_to_a_frozen_taxonomy_is_detected() -> None:
    tampered = _frozen()
    tampered["taxonomy"]["intents"][0]["definition"] = "quietly changed after seeing golden results"
    assert any("edited" in e for e in verify_frozen(tampered))
    relabeled = _frozen()
    relabeled["labels"] = relabeled["labels"][::-1]
    assert verify_frozen(relabeled)
    candidate = _frozen()
    candidate["taxonomy"]["status"] = "CANDIDATE_NOT_GROUND_TRUTH"
    assert any("CANDIDATE" in e for e in verify_frozen(candidate))


def run(script: str, world: dict, *extra: str) -> subprocess.CompletedProcess:
    args = {
        "finalize_taxonomy.py": ["--processed-dir", world["processed"], "--golden-dir", world["golden"], "--registry", world["registry"], "--frozen", world["frozen"], "--decisions", world["decisions"], "--reports-dir", world["reports"]],
        "prepare_golden_labeling.py": ["--golden-dir", world["golden"], "--frozen", world["frozen"]],
    }[script]
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args), *extra], capture_output=True, text=True, cwd=ROOT)


def test_end_to_end_label_finalize_freeze_prepare_golden(tmp_path) -> None:
    w = write_world(tmp_path, labelled=True)
    assert registry_status(w["registry"]) != STATUS_CALIBRATED

    dry = run("finalize_taxonomy.py", w, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert not w["frozen"].exists() and registry_status(w["registry"]) != STATUS_CALIBRATED, "dry run must not write"

    early = run("prepare_golden_labeling.py", w)
    assert early.returncode == 1 and not (w["golden"] / "virgintrains_golden_v1.csv").exists(), "golden prep before freeze must refuse"

    done = run("finalize_taxonomy.py", w)
    assert done.returncode == 0, done.stderr
    assert registry_status(w["registry"]) == STATUS_CALIBRATED
    frozen = yaml.safe_load(w["frozen"].read_text(encoding="utf-8"))
    assert verify_frozen(frozen) == [] and frozen["metadata"]["calibration_set_size"] == 40 and frozen["metadata"]["labelled_cases"] == 40
    assert (w["reports"] / "taxonomy_calibration_report.md").exists()

    again = run("finalize_taxonomy.py", w)
    assert again.returncode == 1, "a frozen taxonomy must never be re-frozen"

    prep = run("prepare_golden_labeling.py", w)
    assert prep.returncode == 0, prep.stderr
    out = pd.read_csv(w["golden"] / "virgintrains_golden_v1.csv", dtype=str, keep_default_na=False)
    assert list(out.columns[:8]) == ["case_id", "first_customer_message", "conversation", "gold_intent", "gold_resolution_type", "gold_resolved", "gold_escalation_signal", "labeling_notes"]
    assert len(out) == 10 and (out[["gold_intent", "gold_resolution_type", "gold_resolved", "gold_escalation_signal", "labeling_notes"]] == "").all().all()
    assert out["case_id"].tolist() == [f"gold_{i}" for i in range(10)], "golden order/sampling must be unchanged"
    assert "candidate_intent" not in out.columns, "no system suggestion is shown to the golden labeller"

    out.loc[0, "gold_intent"] = INTENTS[0]
    out.to_csv(w["golden"] / "virgintrains_golden_v1.csv", index=False)
    assert run("prepare_golden_labeling.py", w).returncode == 1, "must not overwrite gold labels"


def test_finalize_refuses_without_labels_or_signature_or_after_golden_change(tmp_path) -> None:
    unlabelled = write_world(tmp_path / "a", labelled=False)
    r = run("finalize_taxonomy.py", unlabelled)
    assert r.returncode == 1 and "fully labelled" in r.stderr and not unlabelled["frozen"].exists()

    unsigned = write_world(tmp_path / "b")
    unsigned["decisions"].write_text(yaml.safe_dump({"reviewed_by": "", "reviewed_on": ""}), encoding="utf-8")
    r = run("finalize_taxonomy.py", unsigned)
    assert r.returncode == 1 and "reviewed_by" in r.stderr and not unsigned["frozen"].exists()

    changed = write_world(tmp_path / "c")
    split = changed["processed"] / "splits" / "virgintrains_split_assignments.csv"
    df = pd.read_csv(split)
    df.loc[df["case_id"] == "gold_0", "split"] = "train_retrieval"
    df.to_csv(split, index=False)
    r = run("finalize_taxonomy.py", changed)
    assert r.returncode == 1 and "golden case set changed" in r.stderr and not changed["frozen"].exists()

    partial = write_world(tmp_path / "d")
    csv = partial["processed"] / "taxonomy_calibration.csv"
    frame = pd.read_csv(csv, dtype=str, keep_default_na=False)
    frame.loc[:5, "human_intent"] = ""
    frame.loc[:5, ["human_resolution_type", "human_resolved", "human_escalation_signal"]] = ""
    frame.to_csv(csv, index=False)
    assert "fully labelled" in run("finalize_taxonomy.py", partial).stderr, "under 95% labelled must refuse"


def test_golden_prep_detects_altered_golden_after_freeze(tmp_path) -> None:
    w = write_world(tmp_path)
    assert run("finalize_taxonomy.py", w).returncode == 0
    parquet = w["golden"] / "virgintrains_golden_candidates.parquet"
    df = pd.read_parquet(parquet)
    df.iloc[:9].to_parquet(parquet)
    r = run("prepare_golden_labeling.py", w)
    assert r.returncode == 1 and "must not be altered" in r.stderr


def test_golden_prep_rejects_tampered_frozen_taxonomy(tmp_path) -> None:
    w = write_world(tmp_path)
    assert run("finalize_taxonomy.py", w).returncode == 0
    frozen = yaml.safe_load(w["frozen"].read_text(encoding="utf-8"))
    frozen["taxonomy"]["intents"][0]["definition"] = "edited"
    w["frozen"].write_text(yaml.safe_dump(frozen), encoding="utf-8")
    r = run("prepare_golden_labeling.py", w)
    assert r.returncode == 1 and "verification" in r.stderr
