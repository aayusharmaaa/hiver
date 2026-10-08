import json
from types import SimpleNamespace

import pandas as pd

from evaluation.failure_analysis import (
    agent_observed,
    analyse,
    excerpt,
    golden_frame,
    hard_rule_gaps,
    intent_errors,
    load_llm_predictions,
    oracle_hard_reasons,
    policy_over_escalation,
    render_markdown,
    retrieval_misses,
)

SETTINGS = SimpleNamespace(
    escalate_intents={"service_status_delay_enquiry": "live data", "unclear_or_media_only": "unclear"},
    sensitive_patterns=[r"\b(wheelchair|refund\w*)\b"],
    min_message_words=3,
)

ROWS = [
    # case, order, message, gold intent, should escalate, notes
    ("c1", 1, "@VirginTrains is the 17:30 to Euston running on time today?", "service_status_delay_enquiry", "no", ""),
    ("c2", 2, "@VirginTrains lovely trip, staff helped with my wheelchair", "praise_positive_feedback", "no", ""),
    ("c3", 3, "@VirginTrains can I change my booking for tomorrow please", "ticket_booking_query", "yes", "needs account"),
    ("c4", 4, "@VirginTrains wifi keeps dropping on the train", "onboard_wifi_issue", "no", ""),
    ("c5", 101, "@VirginTrains heating broken on the 13.00", "journey_disruption_complaint", "no", "Heating on another operator's service."),
    ("c6", 102, "@VirginTrains https://t.co/x", "unclear_or_media_only", "yes", ""),
]


def golden():
    pack = pd.DataFrame(
        [{"case_id": c, "labeling_order": o, "first_customer_message": m, "gold_intent": g, "gold_should_escalate": e, "gold_resolution_type": "information_provided",
          "gold_confidence": "high", "human_notes": n} for c, o, m, g, e, n in ROWS]
    )
    provenance = {c: ("human" if o <= 100 else "assistant_draft_confirmed") for c, o, *_ in ROWS}
    candidate = {c: g for c, _, _, g, *_ in ROWS} | {"c1": "journey_disruption_complaint"}
    extra = pd.DataFrame({"case_id": [r[0] for r in ROWS], "other_agent_turn_count": [0, 0, 0, 0, 0, 1], "dm_redirect": [False] * 6})
    return golden_frame(pack, provenance, candidate, extra)


PREDS = {
    "c1": {"intent": "journey_disruption_complaint", "confidence": 0.95},
    "c2": {"intent": "praise_positive_feedback", "confidence": 0.97},
    "c3": {"intent": "ticket_booking_query", "confidence": 0.9},
    "c4": {"intent": "service_status_delay_enquiry", "confidence": 0.6},
    "c5": {"intent": "journey_disruption_complaint", "confidence": 0.9},
    "c6": {"intent": "chitchat_non_support", "confidence": 0.9},
}


def record(case_id, gold_esc, final, *, generation="not_attempted", grounding="not_run", reasons=()):
    return {"case_id": case_id, "message": f"msg {case_id}", "gold_intent": "x", "pred_intent": "x", "gold_should_escalate": gold_esc, "final_action": final,
            "generation": generation, "grounding": grounding, "reasons": list(reasons), "unsupported_claims": ["made up time"] if grounding == "fail" else []}


def test_golden_frame_marks_only_human_provenance_as_blind():
    g = golden()
    assert g.loc[g.blind, "case_id"].tolist() == ["c1", "c2", "c3", "c4"]


def test_intent_errors_use_blind_cases_only_and_flag_confident_errors():
    errors, boundary = intent_errors(golden(), PREDS)
    assert (errors["count"], errors["denominator"]) == (2, 4)
    assert [e["case_id"] for e in errors["examples"]] == ["c1"]
    assert "journey_disruption_complaint / service_status_delay_enquiry" in boundary["observed"]
    assert "disagrees with the final gold intent on 1 of 6" in boundary["observed"]


def test_oracle_reasons_and_policy_over_escalation():
    assert oracle_hard_reasons("is it late", "service_status_delay_enquiry", SETTINGS) == ["intent rule (service_status_delay_enquiry)"]
    assert oracle_hard_reasons("hi", "praise_positive_feedback", SETTINGS) == ["low-information message"]
    m = policy_over_escalation(golden(), SETTINGS)
    assert (m["count"], m["denominator"]) == (2, 3)
    assert {e["case_id"] for e in m["examples"]} == {"c1", "c2"}
    assert "sensitive wording (wheelchair)" in m["examples"][0]["detail"]


def test_hard_rule_gaps_and_observed_unsafe_auto_handles():
    m = hard_rule_gaps(golden(), SETTINGS, [record("c3", "yes", "AUTO_HANDLE", generation="produced", grounding="pass")])
    assert (m["count"], m["denominator"]) == (1, 1) and m["examples"][0]["case_id"] == "c3"
    assert "1 unsafe auto-handles out of 1" in m["observed"]


def test_agent_observed_counts_and_partial_scope():
    recs = [
        record("a", "no", "ESCALATE", reasons=["weak retrieval: best similarity 0.4 < 0.60"]),
        record("b", "no", "ESCALATE", generation="produced", grounding="fail"),
        record("c", "no", "AUTO_HANDLE", generation="produced", grounding="pass"),
        record("d", "yes", "ESCALATE", generation="declined"),
    ]
    over, gen, ground, weak = agent_observed(recs, expected=50)
    assert "4 of 50 cases (PARTIAL)" in over["scope"]
    assert (over["count"], over["denominator"]) == (2, 3)
    assert (gen["count"], gen["denominator"]) == (1, 3)
    assert (ground["count"], ground["denominator"]) == (1, 2) and "made up time" in ground["examples"][0]["detail"]
    assert weak["count"] == 1
    assert "PARTIAL" not in agent_observed(recs, expected=4)[0]["scope"]


def test_retrieval_misses_reads_the_proxy_report():
    retrieval = {"meta": {"n_test": 10}, "failures": {"n_queries_without_hit_in_top5": 3, "kinds": {"same_intent_but_other_resolution_type": 2}},
                 "examples": {"bad": [{"case_id": "q1", "query": "refund?", "intent": "x", "resolution_type": "refund", "first_relevant_rank": 9,
                                       "results": [{"intent": "x", "resolution_type": "information_provided"}]}]}}
    m = retrieval_misses(retrieval)
    assert (m["count"], m["denominator"]) == (3, 10) and "heuristic" in m["scope"]
    assert "rank 9" in m["examples"][0]["detail"]


def test_load_predictions_skips_errors(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [{"case_id": "a", "intent": "x", "error": None}, {"case_id": "b", "intent": None, "error": "bad"}]), encoding="utf-8")
    assert set(load_llm_predictions(p)) == {"a"}
    assert load_llm_predictions(tmp_path / "missing.jsonl") == {}


def test_full_report_renders_every_mode_with_real_ids_only():
    retrieval = {"meta": {"n_test": 1}, "failures": {}, "examples": {}}
    modes = analyse(golden(), PREDS, retrieval, [], SETTINGS, expected_agent_cases=50)
    assert len(modes) == 11
    md = render_markdown(modes, {"golden_cases": 6, "blind_cases": 4, "llm_predictions": 6, "llm_model": "m", "agent_cases": 0, "agent_expected": 50})
    assert "partial; agent-run counts are not final" in md
    assert "No examples in the data available so far" in md
    ids = {r[0] for r in ROWS} | {"q1"}
    cited = {line.split("|")[1].strip() for line in md.splitlines() if line.startswith("| c") and not line.startswith("| case |")}
    assert cited and cited <= ids
    other = next(m for m in modes if m["key"] == "other_operator")
    assert other["count"] == 2 and [e["case_id"] for e in other["examples"]] == ["c5"]


def test_excerpt_truncates_and_flattens():
    assert excerpt("a\n b", 10) == "a b"
    assert excerpt("x" * 20, 10).endswith("…") and len(excerpt("x" * 20, 10)) == 10
