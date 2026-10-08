"""End-to-end agent harness: stratified golden slice, cached model calls, stage detection and routing metrics.

The real SupportAgent runs with a scripted model and a fake retriever; no network, no key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from agent_fixtures import GOOD_CLASSIFICATION, TAXONOMY, FakeRetriever, ScriptedModel, agent_config

from agent.schemas import AUTO_HANDLE, ESCALATE, AgentResult, Classification, Decision
from agent.support_agent import SupportAgent
from evaluation.agent_eval import (
    GEN_DECLINED,
    GEN_ERROR,
    GEN_NOT_ATTEMPTED,
    GEN_PRODUCED,
    GROUND_FAIL,
    GROUND_NOT_RUN,
    GROUND_PASS,
    GROUND_UNAVAILABLE,
    AgentEvalError,
    CachingModel,
    agent_metrics,
    judge_items,
    render_markdown,
    run_agent_cases,
    select_stratified,
    stage_outcomes,
)
from evaluation.intent_eval import ALL_REVIEWED, BLIND_HUMAN
from models.base import ModelOutputError, ModelRuntimeError

INTENTS = ["onboard_wifi_issue", "ticket_booking_query", "service_status_delay_enquiry", "unclear_or_media_only"]
CORPUS = {f"case_{i}" for i in range(1, 6)}
SPLITS = {c: "train_retrieval" for c in CORPUS}


def golden_frame() -> pd.DataFrame:
    rows = []
    for i in range(40):
        intent = INTENTS[i % 3] if i < 36 else "NEW:lost_property"
        rows.append({
            "case_id": f"g{i:02d}", "labeling_order": i + 1, "text": f"message {i}", "gold_intent": intent,
            "gold_should_escalate": "yes" if i % 4 == 0 else "no", "gold_resolution_type": "information_provided",
            "provenance": "human" if i < 20 else "assistant_draft_confirmed", BLIND_HUMAN: i < 20, ALL_REVIEWED: i != 5,
        })
    return pd.DataFrame(rows)


def test_selection_is_stratified_deterministic_and_skips_unreviewed_and_new_intents() -> None:
    golden = golden_frame()
    picked = select_stratified(golden, INTENTS, n=12, min_per_intent=3, seed=1)
    assert len(picked) == 12 and picked["case_id"].is_unique
    assert "g05" not in set(picked["case_id"]) and not picked["gold_intent"].str.startswith("NEW:").any()
    assert picked["gold_intent"].value_counts().min() >= 3
    for _, rows in picked.groupby("gold_intent"):
        assert set(rows["gold_should_escalate"]) == {"yes", "no"}
    assert picked["case_id"].tolist() == select_stratified(golden, INTENTS, n=12, min_per_intent=3, seed=1)["case_id"].tolist()
    assert picked[BLIND_HUMAN].mean() > golden[BLIND_HUMAN].mean()
    with pytest.raises(AgentEvalError):
        select_stratified(golden, INTENTS, n=500)


class Inner:
    name = "inner-1"

    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), 0

    def generate_json(self, prompt, *, system=None, temperature=0.0, max_output_tokens=1024, schema=None):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    def generate_text(self, prompt, **kw):
        raise AssertionError


def test_caching_model_replays_outputs_and_output_errors_but_never_runtime_errors(tmp_path: Path) -> None:
    cache = tmp_path / "cache.jsonl"
    m = CachingModel(Inner({"a": 1}, ModelOutputError("bad json"), ModelRuntimeError("429")), cache)
    assert m.generate_json("p1", system="s") == {"a": 1}
    with pytest.raises(ModelOutputError):
        m.generate_json("p2", system="s")
    with pytest.raises(ModelRuntimeError):
        m.generate_json("p3", system="s")
    assert m.runtime_errors == ["429"] and len(cache.read_text(encoding="utf-8").splitlines()) == 2
    replay = CachingModel(Inner({"a": 2}), cache)
    assert replay.generate_json("p1", system="s") == {"a": 1}
    with pytest.raises(ModelOutputError, match="bad json"):
        replay.generate_json("p2", system="s")
    assert replay.inner.calls == 0 and replay.hits == 2
    assert replay.generate_json("p1", system="other") == {"a": 2} and replay.inner.calls == 1


def run_one(model_kwargs: dict, gold_escalate: str = "no", tmp_path: Path | None = None, results=None):
    scripted = ScriptedModel(**model_kwargs)
    cache = (tmp_path or Path(".")) / "cache.jsonl"
    model = CachingModel(scripted, cache)
    agent = SupportAgent.from_parts(model, FakeRetriever(results), TAXONOMY, agent_config())
    cases = pd.DataFrame([{
        "case_id": "g1", "labeling_order": 1, "text": "The wifi on my train keeps dropping and I cannot log in", "gold_intent": "onboard_wifi_issue",
        "gold_should_escalate": gold_escalate, "gold_resolution_type": "troubleshooting", "provenance": "human", BLIND_HUMAN: True, ALL_REVIEWED: True,
    }])
    return run_agent_cases(agent, cases, model, allowed_intents=INTENTS, corpus_case_ids=CORPUS, split_by_case=SPLITS), scripted


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, (AUTO_HANDLE, GEN_PRODUCED, GROUND_PASS, AUTO_HANDLE)),
        ({"generator": {"reply": None}}, (AUTO_HANDLE, GEN_DECLINED, GROUND_NOT_RUN, ESCALATE)),
        ({"generator": [ModelOutputError("x"), ModelOutputError("x")]}, (AUTO_HANDLE, GEN_ERROR, GROUND_NOT_RUN, ESCALATE)),
        ({"verifier": {"grounded": False, "confidence": 0.9, "unsupported_claims": ["made up"]}}, (AUTO_HANDLE, GEN_PRODUCED, GROUND_FAIL, ESCALATE)),
        ({"verifier": ModelOutputError("garbled")}, (AUTO_HANDLE, GEN_PRODUCED, GROUND_UNAVAILABLE, ESCALATE)),
        ({"classifier": {**GOOD_CLASSIFICATION, "intent": "service_status_delay_enquiry"}}, (ESCALATE, GEN_NOT_ATTEMPTED, GROUND_NOT_RUN, ESCALATE)),
    ],
)
def test_every_pipeline_stage_is_recorded_from_the_real_agent(tmp_path: Path, kwargs, expected) -> None:
    (records, stopped), _ = run_one(kwargs, tmp_path=tmp_path)
    assert stopped is None
    r = records[0]
    assert (r["policy_action"], r["generation"], r["grounding"], r["final_action"]) == expected
    assert r["invariant_errors"] == []


def test_a_runtime_failure_stops_the_run_and_a_rerun_uses_the_cache(tmp_path: Path) -> None:
    (records, stopped), _ = run_one({"verifier": ModelRuntimeError("rate limit")}, tmp_path=tmp_path)
    assert records == [] and "rate limit" in stopped
    (records, stopped), scripted = run_one({"classifier": AssertionError("must come from cache"), "generator": AssertionError("cached")}, tmp_path=tmp_path)
    assert stopped is None and records[0]["final_action"] == AUTO_HANDLE
    assert scripted.count("classifier") == 0 and scripted.count("generator") == 0 and scripted.count("verifier") == 1


def record(final: str, gold: str, generation: str = GEN_NOT_ATTEMPTED, grounding: str = GROUND_NOT_RUN, reply=None, draft=None) -> dict:
    return {
        "case_id": f"c{final}{gold}{generation}", "provenance": "human", "message": "m", "gold_intent": "a", "pred_intent": "a",
        "gold_should_escalate": gold, "gold_resolution_type": "refund", "policy_action": AUTO_HANDLE if generation != GEN_NOT_ATTEMPTED else ESCALATE,
        "generation": generation, "grounding": grounding, "final_action": final, "reasons": ["intent 'x' is never auto-handled: why"],
        "reply": reply, "internal_draft_reply": draft, "unsupported_claims": [], "evidence": [], "classification_failed": False, "invariant_errors": [],
        "labeling_order": 1,
    }


def test_routing_metrics_follow_their_definitions() -> None:
    records = [
        record(AUTO_HANDLE, "no", GEN_PRODUCED, GROUND_PASS, reply="Sorry, please see https://x.y for help?"),
        record(AUTO_HANDLE, "yes", GEN_PRODUCED, GROUND_PASS, reply="Please DM us your booking reference."),
        record(ESCALATE, "yes"),
        record(ESCALATE, "yes", GEN_PRODUCED, GROUND_FAIL, draft="We will refund you."),
        record(ESCALATE, "no"),
    ]
    m = agent_metrics(records)
    assert m["escalation_coverage"] == 3 / 5 and m["safe_automation_rate"] == 1 / 5
    assert m["false_auto_handle_rate"] == 1 / 2
    assert m["escalation_precision"] == 2 / 3 and m["escalation_recall"] == 2 / 3
    assert m["generation"] == {"attempted": 3, "produced": 3, "declined": 0, "error": 0, "success_rate": 1.0}
    assert m["grounding"]["pass"] == 2 and m["grounding"]["fail"] == 1 and m["grounding"]["pass_rate"] == 2 / 3
    assert m["reply_quality"]["all_drafts"]["count"] == 3 and m["reply_quality"]["sent_replies"]["count"] == 2
    assert m["reply_quality"]["sent_replies"]["share_with_link"] == 0.5 and m["reply_quality"]["sent_replies"]["share_mentioning_dm"] == 0.5
    assert m["escalation_reasons"] == {"intent '…' is never auto-handled": 2, "grounding fail": 1}
    empty = agent_metrics([])
    assert empty["escalation_coverage"] is None and empty["false_auto_handle_rate"] is None


def test_judge_items_cover_every_draft_with_blank_judge_fields_and_report_renders() -> None:
    records = [record(AUTO_HANDLE, "no", GEN_PRODUCED, GROUND_PASS, reply="ok"), record(ESCALATE, "yes", GEN_PRODUCED, GROUND_FAIL, draft="bad"), record(ESCALATE, "no")]
    items = judge_items(records)
    assert [i["sent_to_customer"] for i in items] == [True, False]
    assert set(items[0]["judge"]) == {"grounded", "answers_question", "no_invented_commitments", "tone", "overall", "notes"}
    assert all(v is None for k, v in items[0]["judge"].items() if k != "notes")
    json.dumps(items)
    md = render_markdown({
        "meta": {"provider": "fake", "model": "m", "seed": 1, "selected": 3, "scored": 3, "blind_human_selected": 3, "stopped": None, "cache_hits": 0, "cache_misses": 3},
        "subsets": {"all sampled": agent_metrics(records), "blind human only": agent_metrics(records[:1])}, "records": records,
    })
    for text in ("escalation coverage", "safe automation rate", "false auto-handle rate", "escalation precision", "Grounding:", "judge_inputs.jsonl", "Per case"):
        assert text in md


def test_stage_outcomes_reads_classification_failures() -> None:
    result = AgentResult(classification=Classification(intent="unclear_or_media_only", confidence=0.0), decision=Decision(action=ESCALATE, confidence=1.0, reasons=["classification failed: x"]))
    assert stage_outcomes(result)["classification_failed"] is True
