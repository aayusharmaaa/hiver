"""Intent-classification harness: data safety, baselines, cached Gemini predictions, metrics and subsets.

Synthetic data only; the Gemini classifier is a fake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from test_golden_eval import GOOD, open_store, prepare, write_golden_world

from evaluation.golden_eval import ASSISTANT_DRAFT
from evaluation.intent_eval import (
    ALL_REVIEWED,
    BLIND_HUMAN,
    IntentEvalError,
    MajorityBaseline,
    assert_no_golden,
    evaluate_systems,
    fit_tfidf_logreg,
    gemini_predictions,
    golden_eval_frame,
    intent_metrics,
    normalise_text,
    render_markdown,
    weak_labelled_split,
)
from evaluation.splits import GOLDEN

INTENTS = ["a_intent", "b_intent", "unclear_or_media_only"]


def test_normalise_text_masks_urls_and_handles() -> None:
    assert normalise_text("@VirginTrains My WIFI https://t.co/x  is down") == "handletoken my wifi urltoken is down"


def test_majority_baseline_predicts_the_most_common_label_with_a_stable_tie_break() -> None:
    assert MajorityBaseline().fit([""] * 3, ["b", "a", "b"]).predict(["x", "y"]) == ["b", "b"]
    assert MajorityBaseline().fit([""] * 2, ["b", "a"]).label_ == "a"
    with pytest.raises(IntentEvalError):
        MajorityBaseline().fit([], [])


def test_fitting_data_refuses_golden_cases_and_other_splits() -> None:
    frame = pd.DataFrame({"case_id": ["c1", "c2"], "split": ["train_retrieval", "train_retrieval"]})
    assert_no_golden(frame, {"g1"}, expected_split="train_retrieval")
    with pytest.raises(IntentEvalError, match="golden"):
        assert_no_golden(frame, {"c2"}, expected_split="train_retrieval")
    with pytest.raises(IntentEvalError, match="other splits"):
        assert_no_golden(frame.assign(split=["train_retrieval", "dev_calibration"]), set(), expected_split="train_retrieval")
    with pytest.raises(IntentEvalError, match="only use"):
        weak_labelled_split("unused", "unused", GOLDEN)


def test_tfidf_logreg_selects_c_on_dev_and_learns_the_training_signal() -> None:
    train = pd.DataFrame({"text": ["wifi broken"] * 6 + ["refund please"] * 6, "label": ["wifi"] * 6 + ["refund"] * 6})
    dev = pd.DataFrame({"text": ["wifi broken again", "refund please now"], "label": ["wifi", "refund"]})
    sel = fit_tfidf_logreg(train, dev, grid=(0.1, 1.0))
    assert sel.c in (0.1, 1.0) and set(sel.dev_macro_f1) == {0.1, 1.0}
    assert list(sel.model.predict(["the wifi is broken", "a refund please"])) == ["wifi", "refund"]


def test_metrics_cover_accuracy_macro_f1_per_intent_and_confusion() -> None:
    gold = ["a_intent", "a_intent", "b_intent", "NEW:thing"]
    pred = ["a_intent", "b_intent", "b_intent", "a_intent"]
    m = intent_metrics(gold, pred, INTENTS)
    assert m["accuracy"] == 0.5
    assert m["macro_f1_labels"] == ["a_intent", "b_intent", "NEW:thing"]
    assert m["per_intent"]["a_intent"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    assert m["per_intent"]["b_intent"]["f1"] == pytest.approx(2 / 3)
    assert m["per_intent"]["NEW:thing"]["f1"] == 0.0
    assert m["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    assert m["confusion"]["labels"] == ["a_intent", "b_intent", "unclear_or_media_only", "NEW:thing"]
    assert m["confusion"]["matrix"][0] == [1, 1, 0, 0]
    assert m["confusion"]["matrix"][3] == [1, 0, 0, 0]


def test_golden_frame_splits_blind_human_from_reviewed_drafts(tmp_path: Path) -> None:
    world = write_golden_world(tmp_path)
    prepare(world)
    store = open_store(world)
    human, confirmed, corrected, unreviewed = world["ids"][:4]
    store.save_label(human, GOOD)
    for cid in (confirmed, corrected, unreviewed):
        store.save_label(cid, GOOD, source=ASSISTANT_DRAFT)
    store.save_label(confirmed, GOOD)
    store.save_label(corrected, {**GOOD, "gold_intent": "chitchat_non_support"})
    frame = golden_eval_frame(open_store(world)).set_index("case_id")
    assert frame[BLIND_HUMAN].sum() == 1 and frame.loc[human, BLIND_HUMAN]
    assert set(frame.index[frame[ALL_REVIEWED]]) == {human, confirmed, corrected}
    assert frame.loc[corrected, "gold_intent"] == "chitchat_non_support"
    assert frame.loc[human, "text"].startswith("opening message")


def test_systems_are_scored_per_subset_with_coverage() -> None:
    golden = pd.DataFrame(
        {"case_id": ["c1", "c2", "c3"], "gold_intent": ["a_intent", "b_intent", "a_intent"], BLIND_HUMAN: [True, True, False], ALL_REVIEWED: [True, True, True]}
    )
    preds = {"full": {"c1": "a_intent", "c2": "a_intent", "c3": "a_intent"}, "partial": {"c1": "a_intent"}}
    res = evaluate_systems(golden, preds, INTENTS)
    assert res[BLIND_HUMAN]["full"]["n"] == 2 and res[ALL_REVIEWED]["full"]["n"] == 3
    assert res[ALL_REVIEWED]["full"]["accuracy"] == pytest.approx(2 / 3)
    assert res[ALL_REVIEWED]["partial"]["coverage"] == {"predicted": 1, "subset_size": 3}
    md = render_markdown({"meta": {"train_cases": 10, "dev_cases": 5, "majority_label": "a_intent", "tfidf_c": 1.0, "tfidf_dev_macro_f1": {"1.0": 0.5},
                                    "gemini_model": "fake", "gemini_stopped": None, "gemini_unparseable": 0, "provenance": "human: 2"}, "subsets": res})
    assert "Blind human labels" in md and "All reviewed labels" in md and "confusion matrix" in md and "| partial | 1 / 3 |" in md
    skipped = render_markdown({"meta": {"train_cases": 10, "dev_cases": 5, "majority_label": "a_intent", "tfidf_c": 1.0, "tfidf_dev_macro_f1": {"1.0": 0.5},
                                         "gemini_model": None, "provenance": "human: 2"}, "subsets": res})
    assert "Gemini: not run" in skipped


class OutputBroken(Exception):
    pass


class QuotaHit(Exception):
    pass


class FakeClassifier:
    def __init__(self, answers: dict[str, object]):
        self.answers, self.calls = answers, []

    def build_prompt(self, message: str, context: str | None) -> str:
        return f"classify: {message}"

    def classify(self, message: str, context: str | None):
        self.calls.append(message)
        answer = self.answers[message]
        if isinstance(answer, Exception):
            raise answer
        return type("C", (), {"intent": answer, "confidence": 0.9})()


def run(classifier: FakeClassifier, cases: pd.DataFrame, cache: Path, sleeps: list[float] | None = None):
    return gemini_predictions(
        classifier, cases, cache, model_name="fake-1", fallback_intent="unclear_or_media_only",
        output_errors=(OutputBroken,), stop_errors=(QuotaHit,), pause_seconds=2.0, sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


def test_gemini_predictions_are_cached_and_unparseable_output_becomes_the_fallback(tmp_path: Path) -> None:
    cases = pd.DataFrame({"case_id": ["c1", "c2"], "text": ["hi", "bad"]})
    cache, sleeps = tmp_path / "preds.jsonl", []
    fake = FakeClassifier({"hi": "a_intent", "bad": OutputBroken("not json")})
    preds, stopped = run(fake, cases, cache, sleeps)
    assert preds == {"c1": "a_intent", "c2": "unclear_or_media_only"} and stopped is None
    assert sleeps == [2.0]
    records = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines()]
    assert records[1]["error"].startswith("unparseable output")
    again = FakeClassifier({})
    assert run(again, cases, cache)[0] == preds and again.calls == []


def test_gemini_run_stops_on_quota_and_resumes_from_the_cache(tmp_path: Path) -> None:
    cases = pd.DataFrame({"case_id": ["c1", "c2", "c3"], "text": ["one", "two", "three"]})
    cache = tmp_path / "preds.jsonl"
    preds, stopped = run(FakeClassifier({"one": "a_intent", "two": QuotaHit("429"), "three": "b_intent"}), cases, cache)
    assert preds == {"c1": "a_intent"} and "stopped after 1" in stopped
    resumed = FakeClassifier({"two": "b_intent", "three": "b_intent"})
    preds, stopped = run(resumed, cases, cache)
    assert stopped is None and preds == {"c1": "a_intent", "c2": "b_intent", "c3": "b_intent"}
    assert resumed.calls == ["two", "three"]
