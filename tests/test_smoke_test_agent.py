"""Smoke-test workflow tests: selection, leakage safety, formatting, invariants. No Gemini, no network; `agent.handle` is mocked."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agent.schemas import AUTO_HANDLE, ESCALATE, AgentResult, Classification, Decision, Evidence, GroundingResult
from evaluation import smoke_report as R
from evaluation import smoke_selection as S
from evaluation.splits import DEV, EXCLUDED, GOLDEN, RESERVE, TRAIN
from models.base import ModelConfigError, ModelRuntimeError

REPO = Path(__file__).resolve().parents[1]
INTENTS = ["service_status_delay_enquiry", "ticket_booking_query", "seat_reservation_issue", "delay_repay_refund_claim", "onboard_wifi_issue",
           "first_class_catering_issue", "customer_service_complaint", "praise_positive_feedback", "journey_disruption_complaint", "chitchat_non_support", "unclear_or_media_only"]

MESSAGES = {  # one genuine-looking opener per scenario
    "service_status": ("@VirginTrains is the 18:30 to Manchester running on time today?", "service_status_delay_enquiry", {}),
    "ticket_booking": ("@VirginTrains can I book a ticket for tomorrow's train from London to Leeds?", "ticket_booking_query", {}),
    "seat_reservation": ("@VirginTrains my seat reservation was not showing on the train and someone was in my seat", "seat_reservation_issue", {}),
    "delay_repay_or_refund": ("@VirginTrains how do I claim a refund for my delayed train yesterday evening?", "delay_repay_refund_claim", {}),
    "onboard_wifi": ("@VirginTrains the wifi on this train is not working at all today", "onboard_wifi_issue", {}),
    "first_class_catering": ("@VirginTrains there was no food in first class on the 9:00 from Euston today", "first_class_catering_issue", {}),
    "customer_service_complaint": ("@VirginTrains the staff on the 11:38 were very rude to me and the guard was unhelpful to everyone", "customer_service_complaint", {}),
    "praise": ("@VirginTrains thanks for a great journey the staff were brilliant and helpful today", "praise_positive_feedback", {}),
    "short_low_information": ("@VirginTrains refund?", "delay_repay_refund_claim", {}),
    "multi_intent": ("@VirginTrains I bought tickets for my train and it was delayed by two hours so I want a refund too for this whole journey please", "journey_disruption_complaint", {}),
    "ambiguous_boundary_case": ("@VirginTrains I am sat on your train and it feels very warm in this carriage", "journey_disruption_complaint", {"boundary_intent": "service_status_delay_enquiry", "margin": 0.001}),
    "likely_escalation": ("@VirginTrains I made a complaint three weeks ago and still no refund or reply from you at all", "journey_disruption_complaint", {"escalation_signal": "customer_relations_or_formal_route"}),
}


def pool_row(case_id: str, message: str, intent: str | None, split: str = DEV, **kw) -> dict:
    row = {"case_id": case_id, "split": split, "message": message, "intent": intent, "boundary_intent": None, "margin": 0.4, "resolution_type": "information_provided",
           "resolved": True, "dm_redirect": False, "escalation_signal": "none", "evidence_quality": "strong", "customer_id": f"cust_{case_id}",
           "conversation_id": f"conv_{case_id}", "group_id": case_id, "tweet_ids": [abs(hash(case_id)) % 10**6, 7_000_000 + len(case_id)],
           "starts_with_customer": True, "has_brand_reply": True, "is_continuation": False}
    row.update(kw)
    return row


def filler(n: int) -> list[dict]:
    return [pool_row(f"filler_{i}", f"@VirginTrains hello could you help me with something number {i} please", "chitchat_non_support") for i in range(n)]


def make_pool(skip: tuple[str, ...] = (), extra: list[dict] | None = None, fillers: int = 10) -> pd.DataFrame:
    rows = [pool_row(f"case_{name}", msg, intent, **kw) for name, (msg, intent, kw) in MESSAGES.items() if name not in skip]
    return pd.DataFrame(rows + filler(fillers) + (extra or []))


def make_safety(pool: pd.DataFrame, golden_rows: list[dict] | None = None, splits: dict[str, str] | None = None) -> S.SafetyContext:
    rows = [{"case_id": r.case_id, "split": splits.get(r.case_id, r.split) if splits else r.split, "customer_id": r.customer_id, "conversation_id": r.conversation_id,
             "group_id": r.group_id, "source_tweet_ids": list(r.tweet_ids), "context_tweet_ids": [], "opening_message": r.message} for r in pool.itertuples()]
    for g in golden_rows or []:
        rows.append({"case_id": g["case_id"], "split": GOLDEN, "customer_id": g.get("customer_id", "g_cust"), "conversation_id": g.get("conversation_id", "g_conv"),
                     "group_id": g.get("group_id", "g_grp"), "source_tweet_ids": g.get("tweets", [9_999_999]), "context_tweet_ids": [], "opening_message": g.get("message", "golden text only")})
    return S.build_safety_context(pd.DataFrame(rows))


def select(pool=None, **kw):
    pool = make_pool() if pool is None else pool
    return S.select_smoke_cases(pool, make_safety(pool), **kw)


# ---------------------------------------------------------------- selection
class TestSelection:
    def test_every_scenario_is_attempted_and_found(self):
        sel = select(limit=12)
        assert [c.scenario for c in sel.cases] == list(S.SCENARIO_NAMES) and sel.unavailable == []
        assert {c.case_id for c in sel.cases} == {f"case_{n}" for n in MESSAGES}
        assert all("filler" not in c.case_id for c in sel.cases)

    def test_scenarios_pick_the_intended_kind_of_message(self):
        by = {c.scenario: c for c in select().cases}
        assert by["short_low_information"].message.endswith("refund?")
        assert by["ambiguous_boundary_case"].boundary_intent == "service_status_delay_enquiry"
        assert by["likely_escalation"].escalation_signal == "customer_relations_or_formal_route"
        assert "claim a refund" in by["delay_repay_or_refund"].message

    def test_only_allowed_splits_and_never_train_queries(self):
        train_bait = pool_row("train_wifi", "@VirginTrains the wifi on this train is not working at all and very slow", "onboard_wifi_issue", split=TRAIN, margin=0.4)
        pool = make_pool(extra=[train_bait])
        sel = select(pool)
        assert {c.split for c in sel.cases} <= S.ALLOWED_SPLITS == {TRAIN, DEV} and {c.split for c in sel.cases} == {DEV}
        assert "train_wifi" not in {c.case_id for c in sel.cases}  # train cases are in the retrieval corpus: they would retrieve themselves

    @pytest.mark.parametrize("bad_split", [GOLDEN, RESERVE, EXCLUDED, None])
    def test_a_disallowed_or_missing_split_in_the_pool_fails_loudly(self, bad_split):
        pool = make_pool(extra=[pool_row("bad", "@VirginTrains the wifi is not working on this train today", "onboard_wifi_issue", split=bad_split)])
        with pytest.raises(S.SmokeSafetyError, match="disallowed splits"):
            S.select_smoke_cases(pool, make_safety(pool), limit=12)

    def test_deterministic_for_a_fixed_seed(self):
        a, b = select(seed=42), select(seed=42)
        assert [(c.scenario, c.case_id) for c in a.cases] == [(c.scenario, c.case_id) for c in b.cases]
        shuffled = make_pool().sample(frac=1, random_state=3).reset_index(drop=True)
        assert [c.case_id for c in select(shuffled, seed=42).cases] == [c.case_id for c in a.cases]  # independent of row order

    def test_seed_only_breaks_ties(self):
        twins = [pool_row(f"twin_{i}", "@VirginTrains the wifi on this train is not working at all today", "onboard_wifi_issue") for i in range(6)]
        pool = make_pool(skip=("onboard_wifi",), extra=twins)
        picks = {next(c.case_id for c in select(pool, seed=s).cases if c.scenario == "onboard_wifi") for s in range(12)}
        assert len(picks) > 1  # different seeds break the exact tie differently (same seed: see the determinism test)

    def test_limit_is_respected_and_keeps_diversity(self):
        for limit in (1, 4, 7, 12):
            sel = select(limit=limit)
            assert len(sel.cases) == limit
        four = [c.scenario for c in select(limit=4).cases]
        assert set(four) == set(S.PRIORITY[:4]) and "short_low_information" in four and "multi_intent" in four

    def test_limit_above_scenarios_adds_labelled_extras_without_replacing_coverage(self):
        sel = select(limit=15)
        assert len(sel.cases) == 15 and [c.scenario for c in sel.cases[:12]] == list(S.SCENARIO_NAMES)
        assert [c.scenario for c in sel.cases[12:]] == [S.EXTRA] * 3
        assert len({c.case_id for c in sel.cases}) == 15

    def test_limit_never_exceeds_available_cases(self):
        pool = make_pool(fillers=2)
        assert len(select(pool, limit=50).cases) == 12 + 2

    def test_missing_scenario_is_reported_not_replaced(self):
        pool = make_pool(skip=("onboard_wifi", "praise"))
        sel = select(pool, limit=12)
        assert set(sel.unavailable) == {"onboard_wifi", "praise"}
        assert len(sel.cases) == 10 and "onboard_wifi" not in {c.scenario for c in sel.cases}
        assert not any(c.scenario == S.EXTRA for c in sel.cases)  # no silent substitution

    def test_invalid_limit(self):
        with pytest.raises(ValueError):
            select(limit=0)

    def test_message_contains_only_the_opening_text(self):
        for c in select().cases:
            assert c.message == MESSAGES[c.scenario][0]


# ---------------------------------------------------------------- safety
class TestSafety:
    def case(self, **kw) -> S.SmokeCase:
        base = dict(case_id="c1", split=DEV, scenario="x", message="@VirginTrains hello there friend", why_selected="t", candidate_intent=None, resolution_type="other",
                    resolved=False, dm_redirect=False, escalation_signal="none", customer_id="u1", conversation_id="v1", group_id="g1", tweet_ids=(1, 2))
        base.update(kw)
        return S.SmokeCase(**base)

    def ctx(self, **kw) -> S.SafetyContext:
        base = dict(split_by_case={"c1": DEV, "gold": GOLDEN, "res": RESERVE, "exc": EXCLUDED}, golden_customers=frozenset({"gu"}), golden_conversations=frozenset({"gv"}),
                    golden_groups=frozenset({"gg"}), golden_tweets=frozenset({99}), golden_openers=frozenset({S.opener_key("duplicate of a golden opener")}))
        base.update(kw)
        return S.SafetyContext(**base)

    def test_a_clean_dev_or_train_case_passes(self):
        S.validate_case_safety(self.case(), self.ctx())
        S.validate_case_safety(self.case(case_id="t1", split=TRAIN), self.ctx(split_by_case={"t1": TRAIN}))

    @pytest.mark.parametrize("case_id, split", [("gold", GOLDEN), ("res", RESERVE), ("exc", EXCLUDED)])
    def test_golden_reserve_and_excluded_are_rejected(self, case_id, split):
        with pytest.raises(S.SmokeSafetyError, match=split):
            S.validate_case_safety(self.case(case_id=case_id, split=split), self.ctx())

    def test_unknown_case_or_missing_split_is_rejected(self):
        with pytest.raises(S.SmokeSafetyError, match="no split metadata"):
            S.validate_case_safety(self.case(case_id="nope"), self.ctx())
        with pytest.raises(S.SmokeSafetyError, match="no split metadata"):
            S.validate_case_safety(self.case(), self.ctx(split_by_case={"c1": float("nan")}))

    def test_case_split_must_match_the_assignments(self):
        with pytest.raises(S.SmokeSafetyError, match="assignments say"):
            S.validate_case_safety(self.case(split=TRAIN), self.ctx())

    @pytest.mark.parametrize("kw, what", [({"customer_id": "gu"}, "customer"), ({"conversation_id": "gv"}, "conversation"), ({"group_id": "gg"}, "group"),
                                          ({"tweet_ids": (5, 99)}, "tweets"), ({"message": "Duplicate of a golden opener!"}, "opening text")])
    def test_golden_provenance_overlap_is_rejected(self, kw, what):
        with pytest.raises(S.SmokeSafetyError, match=what):
            S.validate_case_safety(self.case(**kw), self.ctx())

    def test_empty_message_is_rejected(self):
        with pytest.raises(S.SmokeSafetyError, match="empty"):
            S.validate_case_safety(self.case(message="  "), self.ctx())

    def test_selection_rejects_a_case_that_shares_a_golden_customer(self):
        pool = make_pool()
        safety = make_safety(pool, golden_rows=[{"case_id": "g1", "customer_id": "cust_case_onboard_wifi"}])
        with pytest.raises(S.SmokeSafetyError, match="customer"):
            S.select_smoke_cases(pool, safety, limit=12)

    def test_build_safety_context_collects_golden_provenance(self):
        pool = make_pool()
        ctx = make_safety(pool, golden_rows=[{"case_id": "g1", "customer_id": "gc", "tweets": [5, 6]}])
        assert "gc" in ctx.golden_customers and {5, 6} <= ctx.golden_tweets and ctx.split_by_case["g1"] == GOLDEN


# ---------------------------------------------------------------- results, formatting, invariants
def evidence(i=1, split_ok=True, tweets=("11", "12"), rtype="self_service", intent="onboard_wifi_issue") -> Evidence:
    return Evidence(case_id=f"train_{i}", score=1.05, customer_problem="the wifi is broken on my train " * 8, historical_response="Please reconnect.", resolution_summary="Agent shared the help page.",
                    resolution_type=rtype, source_tweet_ids=list(tweets), semantic_similarity=0.71, intent=intent)


def auto_result(**kw) -> AgentResult:
    base = dict(classification=Classification(intent="onboard_wifi_issue", confidence=0.9, rationale="wifi"), evidence=[evidence(i) for i in (1, 2, 3)],
                decision=Decision(action=AUTO_HANDLE, confidence=0.8, reasons=["all checks passed"]), reply="Please reconnect to the wifi.", grounding=GroundingResult(grounded=True, confidence=0.95))
    base.update(kw)
    return AgentResult(**base)


def escalate_result(**kw) -> AgentResult:
    base = dict(classification=Classification(intent="onboard_wifi_issue", confidence=0.6, multi_intent=True), evidence=[evidence(1)],
                decision=Decision(action=ESCALATE, confidence=1.0, reasons=["message appears to contain more than one request (multi-intent)"]))
    base.update(kw)
    return AgentResult(**base)


CORPUS = {f"train_{i}" for i in range(1, 6)}
SPLITS = {c: TRAIN for c in CORPUS}


def smoke_case(**kw) -> S.SmokeCase:
    base = dict(case_id="case_q", split=DEV, scenario="onboard_wifi", message="@VirginTrains the wifi is not working on my train", why_selected="test", candidate_intent="onboard_wifi_issue",
                resolution_type="self_service", resolved=True, dm_redirect=False, escalation_signal="none")
    base.update(kw)
    return S.SmokeCase(**base)


class FakeAgent:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, []

    def handle(self, message, conversation_context=None):
        self.calls.append({"message": message, "conversation_context": conversation_context})
        if self.exc:
            raise self.exc
        return self.result


def run(agent, case=None):
    return R.run_case(agent, case or smoke_case(), allowed_intents=INTENTS, corpus_case_ids=CORPUS, split_by_case=SPLITS)


class TestRunAndFormat:
    def test_agent_receives_only_the_opening_message_with_keyword_arguments(self):
        agent = FakeAgent(auto_result())
        case = smoke_case(resolution_type="UNIQUE_HISTORICAL_TYPE", candidate_intent="unique_candidate_intent")
        run(agent, case)
        assert agent.calls == [{"message": case.message, "conversation_context": None}]

    def test_formats_auto_handle(self):
        text = R.format_case_result(1, 12, run(FakeAgent(auto_result())))
        for needle in ("CASE 01 / 12", "CASE ID:\ncase_q", "SOURCE SPLIT:\ndev_calibration", "SCENARIO:\nonboard_wifi", "Historical metadata is weak/candidate reference only and is NOT human ground truth.",
                       "PREDICTED INTENT:\nonboard_wifi_issue", "CLASSIFIER CONFIDENCE:\n0.90", "MULTI-INTENT:\nfalse", "LOW-INFORMATION:\nfalse", "TOP RETRIEVED EVIDENCE:",
                       "1. train_1 | score 1.050 | similarity 0.71 | self_service | onboard_wifi_issue", "DECISION:\nAUTO_HANDLE", "DRAFT REPLY:\nPlease reconnect to the wifi.",
                       "GROUNDING:\nPASS", "UNSUPPORTED CLAIMS:\nnone", "END CASE"):
            assert needle in text
        assert text.count("\n1. ") == 1 and "\n4. " not in text  # only the top 3 evidence items are shown

    def test_formats_escalate_with_no_reply(self):
        text = R.format_case_result(2, 12, run(FakeAgent(escalate_result())))
        assert "DECISION:\nESCALATE" in text and "DRAFT REPLY:\n<none>" in text and "GROUNDING:\nNOT RUN" in text
        assert "- message appears to contain more than one request" in text and "multi-intent escalation" in text and "MULTI-INTENT:\ntrue" in text

    def test_grounding_failure_is_displayed(self):
        result = auto_result(decision=Decision(action=ESCALATE, confidence=0.9, reasons=["grounding verification failed: invents a refund"]), reply=None,
                             grounding=GroundingResult(grounded=False, confidence=0.9, unsupported_claims=["promises a refund"]), internal_draft_reply="We will refund you.")
        outcome = run(FakeAgent(result))
        text = R.format_case_result(3, 12, outcome)
        assert "GROUNDING:\nFAIL" in text and "- promises a refund" in text and "INTERNAL DRAFT" in text and "DRAFT REPLY:\n<none>" in text
        assert "grounding failed" in outcome.flags and not outcome.failed

    def test_generator_no_reply_is_flagged_and_summarised(self):
        reason = "the generator returned no reply: model declined: returned reply=null (no explanation given)"
        result = escalate_result(classification=Classification(intent="onboard_wifi_issue", confidence=0.9), decision=Decision(action=ESCALATE, confidence=0.5, reasons=["policy check passed: ok", reason]))
        outcome = run(FakeAgent(result))
        assert any(f.startswith("generator returned no reply (model declined") for f in outcome.flags) and not outcome.failed
        summary = R.summarize_results([outcome], S.Selection(cases=[outcome.case], unavailable=[], requested=1))
        assert summary.generator_no_reply == ["case_q"]
        assert "Cases where the generator returned no reply:\n1  (case_q)" in R.format_summary(summary)

    def test_extra_cases_are_labelled(self):
        assert "SCENARIO:\nEXTRA\n" in R.format_case_result(13, 14, run(FakeAgent(auto_result()), smoke_case(scenario=S.EXTRA)))

    def test_observation_flags_are_diagnostics_not_verdicts(self):
        case = smoke_case(escalation_signal="dm_for_account_lookup", candidate_intent="seat_reservation_issue", scenario="ambiguous_boundary_case")
        flags = run(FakeAgent(auto_result(evidence=[evidence(1, rtype="a", intent="x"), evidence(2, rtype="b", intent="y"), evidence(3, rtype="c", intent="z")])), case).flags
        for f in ("AUTO_HANDLE on a historically escalation-like case", "AUTO_HANDLE in a boundary / multi-intent / low-information scenario",
                  "retrieved cases have mixed resolution types", "retrieved cases mostly have a different candidate intent"):
            assert f in flags
        assert any(f.startswith("predicted intent differs") for f in flags)
        assert "no evidence retrieved" in run(FakeAgent(escalate_result(evidence=[]))).flags
        assert "low-information query" in run(FakeAgent(escalate_result()), smoke_case(message="@VirginTrains refund?")).flags

    def test_no_accuracy_or_score_is_reported(self):
        outcomes = [run(FakeAgent(auto_result())), run(FakeAgent(escalate_result()))]
        summary = R.format_summary(R.summarize_results(outcomes, S.Selection(cases=[o.case for o in outcomes], unavailable=[], requested=2))).lower()
        for banned in ("accuracy", "f1", "precision", "recall", "correct", "safe automation"):
            assert banned not in summary

    def test_malformed_agent_result_is_detected(self):
        bad = SimpleNamespace(model_dump=lambda: {"classification": {"intent": "x", "confidence": 3.0}, "decision": {"action": "MAYBE", "confidence": 0.5}})
        outcome = run(FakeAgent(bad))
        assert outcome.failed and outcome.failure == R.AGENT_LOGIC and "malformed" in outcome.error
        assert "STATUS: FAILED" in R.format_case_result(1, 1, outcome)

    def test_invariant_violations(self):
        def errors(result, corpus=CORPUS, splits=SPLITS):
            return R.check_result_invariants(result, INTENTS, corpus, splits)

        assert errors(auto_result()) == []
        assert any("not a configured candidate intent" in e for e in errors(auto_result(classification=Classification(intent="free_upgrade", confidence=0.5))))
        assert any("AUTO_HANDLE without a reply" in e for e in errors(auto_result(reply=None)))
        assert any("grounding failed but the final action" in e for e in errors(auto_result(grounding=GroundingResult(grounded=False, confidence=0.9, unsupported_claims=["x"]))))
        assert any("ESCALATE without any reason" in e for e in errors(escalate_result(decision=Decision(action=ESCALATE, confidence=1.0, reasons=[]))))
        assert any("not in the retrieval corpus" in e for e in errors(auto_result(evidence=[evidence(9)])))
        assert any("forbidden split" in e for e in errors(auto_result(), splits={**SPLITS, "train_1": GOLDEN}))
        assert any("no source tweet ids" in e for e in errors(auto_result(evidence=[evidence(1, tweets=())])))
        assert any("customer-facing reply" in e for e in errors(escalate_result(reply="hi")))

    def test_failed_invariant_marks_the_case_failed_but_the_run_continues(self):
        outcome = run(FakeAgent(auto_result(reply=None)))
        assert outcome.failed and outcome.result is not None and "STATUS: FAILED" in R.format_case_result(1, 1, outcome)
        outcomes = list(R.run_cases(FakeAgent(auto_result(reply=None)), [smoke_case(), smoke_case(case_id="case_r")], allowed_intents=INTENTS, corpus_case_ids=CORPUS, split_by_case=SPLITS))
        assert len(outcomes) == 2

    def test_failure_categories(self):
        assert run(FakeAgent(exc=ModelRuntimeError("timeout"))).failure == R.LIVE_MODEL
        assert run(FakeAgent(exc=RuntimeError("bug"))).failure == R.AGENT_LOGIC
        with pytest.raises(R.SetupFailure):
            run(FakeAgent(exc=ModelConfigError("bad key")))
        degraded = escalate_result(decision=Decision(action=ESCALATE, confidence=1.0, reasons=["classification failed: timed out"]))
        assert run(FakeAgent(degraded)).failure == R.LIVE_MODEL  # the agent hides runtime failures as ESCALATE; the smoke test must not

    def test_api_key_is_scrubbed_from_errors(self):
        outcome = R.run_case(FakeAgent(exc=ModelRuntimeError("failed with key SECRET123")), smoke_case(), allowed_intents=INTENTS, corpus_case_ids=CORPUS, split_by_case=SPLITS, secrets=["SECRET123"])
        assert "SECRET123" not in outcome.error and "[REDACTED]" in outcome.error

    def test_empty_message_never_reaches_the_agent(self):
        agent = FakeAgent(auto_result())
        assert run(agent, smoke_case(message="  ")).failure == R.AGENT_LOGIC and agent.calls == []


class TestSummary:
    def test_counts_and_status(self):
        outcomes = [run(FakeAgent(auto_result())), run(FakeAgent(escalate_result())), run(FakeAgent(exc=ModelRuntimeError("x")))]
        sel = S.Selection(cases=[o.case for o in outcomes], unavailable=["praise"], requested=12)
        s = R.summarize_results(outcomes, sel)
        assert (s.executed, s.auto_handle, s.escalate, s.grounding_pass, s.end_to_end_ok) == (3, 1, 1, 1, 2)
        assert s.live_status == "PARTIAL" and s.status == "PARTIAL" and s.scenarios_unavailable == ["praise"]
        text = R.format_summary(s)
        for needle in ("SMOKE TEST SUMMARY", "Requested cases:\n12", "Scenarios unavailable:\npraise", "Overall smoke-test status:\nPARTIAL", "NEXT ACTIONS"):
            assert needle in text

    def test_pass_and_fail(self):
        ok = [run(FakeAgent(auto_result()))]
        assert R.summarize_results(ok, S.Selection(cases=[ok[0].case], unavailable=[], requested=1)).status == "PASS"
        bad = [run(FakeAgent(exc=ModelRuntimeError("x")))]
        s = R.summarize_results(bad, S.Selection(cases=[bad[0].case], unavailable=[], requested=1))
        assert s.status == "FAIL" and s.live_status == "FAILED"
        assert R.summarize_results([], S.Selection(cases=[], unavailable=[], requested=1)).status == "FAIL"


# ---------------------------------------------------------------- no side effects, CLI
class TestNoSideEffects:
    def test_selection_run_and_formatting_modify_and_create_no_files(self, tmp_path, monkeypatch):
        (tmp_path / "taxonomy.yaml").write_text("status: CANDIDATE_NOT_GROUND_TRUTH", encoding="utf-8")
        (tmp_path / "golden.csv").write_text("case_id\n1\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        watched = [tmp_path / "taxonomy.yaml", tmp_path / "golden.csv", tmp_path / "missing.parquet"]
        before, listing = R.fingerprint_files(watched), sorted(os.listdir(tmp_path))
        sel = select()
        outcomes = list(R.run_cases(FakeAgent(auto_result()), sel.cases, allowed_intents=INTENTS, corpus_case_ids=CORPUS, split_by_case=SPLITS))
        for i, o in enumerate(outcomes, 1):
            R.format_case_result(i, len(outcomes), o)
        R.format_summary(R.summarize_results(outcomes, sel))
        assert R.changed_files(before, R.fingerprint_files(watched)) == [] and sorted(os.listdir(tmp_path)) == listing

    def test_changed_files_detects_modification(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("1", encoding="utf-8")
        before = R.fingerprint_files([f])
        f.write_text("2", encoding="utf-8")
        assert R.changed_files(before, R.fingerprint_files([f])) == [str(f)]


@pytest.mark.parametrize("provider,var", [("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY")])
def test_script_without_api_key_is_a_setup_failure_and_not_a_pass(tmp_path, provider, var):
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY")}
    cmd = [sys.executable, str(REPO / "scripts" / "smoke_test_agent.py"), "--provider", provider, "--limit", "3", "--env-file", str(tmp_path / "absent.env")]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 2 and "SETUP FAILURE" in proc.stderr and var in proc.stderr
    assert "PASS" not in proc.stdout and "SMOKE TEST SUMMARY" not in proc.stdout
