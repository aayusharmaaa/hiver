from __future__ import annotations

import pandas as pd
import pytest

from evaluation.taxonomy_calibration import HUMAN_COLUMNS
from evaluation.taxonomy_comparison import Thresholds, cohen_kappa, compare, load_labelled, wilson

FB = "unclear_or_media_only"
NAMES = ["status", "journey", "praise", "chitchat", "ticket", FB]


def frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (candidate, human[, human_resolution_type, auto_resolution_type, human_notes])"""
    data = []
    for i, r in enumerate(rows):
        cand, hum = r[0], r[1]
        hres = r[2] if len(r) > 2 else "information_provided"
        ares = r[3] if len(r) > 3 else "information_provided"
        notes = r[4] if len(r) > 4 else ""
        data.append(
            {
                "case_id": f"c{i}", "candidate_intent": cand, "first_customer_message": f"msg {i}", "stratum_weight": 2.0,
                "auto_resolution_type": ares, "auto_resolved": "False",
                "human_intent": hum, "human_resolution_type": hres, "human_resolved": "no", "human_escalation_signal": "none", "human_notes": notes,
            }
        )
    return pd.DataFrame(data)


def labelled(rows: list[tuple]):
    load = load_labelled(frame(rows), NAMES, FB)
    assert not load.problems, load.problems
    return load.frame


def test_unlabelled_rows_produce_no_results_and_are_not_invented() -> None:
    blank = frame([("status", "")] * 5)
    for col in HUMAN_COLUMNS:
        blank[col] = ""
    load = load_labelled(blank, NAMES, FB)
    assert load.n_complete == 0 and load.n_rows == 5 and load.problems == []
    assert compare(load.frame, NAMES, FB) == {"n_labelled": 0}


def test_validation_flags_partial_unknown_duplicate_and_bad_vocab() -> None:
    df = frame([("status", "status"), ("status", "made_up_intent"), ("journey", "journey"), ("journey", "journey")])
    df.loc[0, "human_resolved"] = ""
    df.loc[2, "human_resolution_type"] = "nonsense"
    df.loc[3, "case_id"] = "c2"
    problems = " | ".join(load_labelled(df, NAMES, FB).problems)
    assert "partially labelled" in problems and "unknown human_intent" in problems and "unknown human_resolution_type" in problems and "duplicate case_id" in problems


def test_labels_are_normalised_notes_case_preserved_and_fallback_aliases() -> None:
    df = frame([("status", "  Status "), ("chitchat", "UNCLEAR", "information_provided", "unresolved", "Customer Said Hi")])
    load = load_labelled(df, NAMES, FB)
    assert not load.problems
    assert load.frame.loc[0, "human_intent"] == "status"
    assert load.frame.loc[1, "human_intent"] == FB
    assert load.frame.loc[1, "human_notes"] == "Customer Said Hi"


def test_new_intent_proposals_are_accepted_and_reported() -> None:
    rows = [("status", "new:lost_property")] * 3 + [("status", "status")] * 5
    result = compare(labelled(rows), NAMES, FB)
    assert result["proposed_new_intents"][0]["name"] == "lost_property" and result["proposed_new_intents"][0]["n"] == 3


def test_confusion_matrix_purity_and_recall() -> None:
    rows = [("status", "status")] * 8 + [("status", "journey")] * 2 + [("journey", "journey")] * 6 + [("journey", "status")] * 4
    result = compare(labelled(rows), NAMES, FB)
    m = result["confusion_matrix"]
    assert m.loc["status", "status"] == 8 and m.loc["status", "journey"] == 2 and m.loc["journey", "status"] == 4
    assert (m.loc["praise"] == 0).all(), "intents with no candidate cases still appear"
    purity = {p["candidate_intent"]: p for p in result["purity"]}
    assert purity["status"]["precision"] == pytest.approx(0.8) and purity["journey"]["precision"] == pytest.approx(0.6)
    lo, hi = purity["status"]["precision_ci95"]
    assert lo < 0.8 < hi
    recall = {r["human_intent"]: r for r in result["recall"]}
    assert recall["status"]["recall"] == pytest.approx(8 / 12) and recall["status"]["candidate_mix"] == {"status": 8, "journey": 4}
    assert result["overall"]["agreement"] == pytest.approx(14 / 20)


def test_systematic_disagreements_respect_min_pair() -> None:
    rows = [("status", "status")] * 6 + [("status", "journey")] * 3 + [("status", "praise")] * 1
    result = compare(labelled(rows), NAMES, FB, Thresholds(min_pair=3))
    pairs = {(d["candidate_intent"], d["human_intent"]): d for d in result["systematic_disagreements"]}
    assert ("status", "journey") in pairs and ("status", "praise") not in pairs
    assert pairs[("status", "journey")]["examples"][0]["case_id"]


def test_merge_requires_mutual_confusion_not_one_way_leakage() -> None:
    mutual = [("status", "status")] * 6 + [("status", "journey")] * 4 + [("journey", "journey")] * 6 + [("journey", "status")] * 4
    result = compare(labelled(mutual), NAMES, FB)
    assert [m["intents"] for m in result["merge_candidates"]] == [["journey", "status"]]
    assert result["merge_candidates"][0]["mutual_confusion_rate"] == pytest.approx(0.4)

    one_way = [("status", "status")] * 6 + [("status", "journey")] * 6 + [("journey", "journey")] * 10
    assert compare(labelled(one_way), NAMES, FB)["merge_candidates"] == []


def test_merge_reading_flags_different_resolution_handling() -> None:
    rows = [("status", "status", "information_provided")] * 6 + [("status", "journey", "information_provided")] * 4
    rows += [("journey", "journey", "compensation")] * 6 + [("journey", "status", "information_provided")] * 4
    m = compare(labelled(rows), NAMES, FB)["merge_candidates"][0]
    assert m["human_resolution_type_overlap"] < 0.6 and "keep separate" in m["reading"]


def test_split_candidate_needs_low_purity_and_a_strong_second_intent() -> None:
    mixed = [("status", "status")] * 6 + [("status", "journey")] * 5 + [("status", "praise")] * 1
    result = compare(labelled(mixed), NAMES, FB)
    assert [s["candidate_intent"] for s in result["split_candidates"]] == ["status"]

    one_noisy_tail = [("status", "status")] * 8 + [("status", "journey")] * 1 + [("status", "praise")] * 1 + [("status", "ticket")] * 1
    assert compare(labelled(one_noisy_tail), NAMES, FB)["split_candidates"] == []


def test_low_support_intents_are_flagged_and_not_split() -> None:
    rows = [("status", "journey")] * 2 + [("status", "praise")] * 2
    result = compare(labelled(rows), NAMES, FB)
    assert result["purity"][0]["low_support"] is True and result["split_candidates"] == []


def test_fallback_quality() -> None:
    rows = [(FB, FB)] * 3 + [(FB, "status")] * 2 + [("status", FB)] * 1 + [("status", "status")] * 4
    fb = compare(labelled(rows), NAMES, FB)["fallback_quality"]
    assert fb["precision"] == pytest.approx(3 / 5) and fb["recall"] == pytest.approx(3 / 4)
    assert fb["candidate_fallback_cases_humans_assigned_real_intent"] == {"status": 2}
    assert fb["human_fallback_cases_candidate_missed"] == {"status": 1}


def test_fallback_without_cases_does_not_crash() -> None:
    fb = compare(labelled([("status", "status")] * 3), NAMES, FB)["fallback_quality"]
    assert fb["precision"] is None and fb["recall"] is None


def test_resolution_type_and_resolved_agreement() -> None:
    rows = [("status", "status", "information_provided", "information_provided")] * 3 + [("status", "status", "refund", "unresolved")]
    result = compare(labelled(rows), NAMES, FB)
    assert result["resolution_type"]["agreement"] == pytest.approx(0.75)
    assert result["resolved"]["human_unclear"] == 0 and result["resolved"]["agreement"] == pytest.approx(1.0)


def test_helpers() -> None:
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert cohen_kappa(pd.Series(list("aabb")), pd.Series(list("aabb"))) == pytest.approx(1.0)
    assert cohen_kappa(pd.Series(list("aabb")), pd.Series(list("abab"))) == pytest.approx(0.0)
