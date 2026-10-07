from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ingestion.resolution_memory import (
    ESCALATION_SOURCE,
    INTENT_SOURCE,
    MEMORY_COLUMNS,
    NO_REPLY,
    STRONG,
    WEAK,
    build_memory_records,
    clip,
    customer_problem_from_turns,
    escalation_signal,
    historical_response_from_turns,
    select_response_turns,
)

TS = pd.Timestamp("2017-10-24 10:00", tz="UTC")


def turn(tid: int, role: str, text: str) -> dict:
    return {"tweet_id": tid, "role": role, "text": text}


def case(case_id="case_10", turns=None, rtype="information_provided", outcome="agent_answered_unconfirmed", evidence=None, signals=(), dm=False, resolved=False,
         cont=False, starts=True, context=()):
    turns = turns if turns is not None else [
        turn(10, "customer", "@VirginTrains my wifi will not connect on the 17:43 https://t.co/abc"),
        turn(11, "brand_agent", "@123 Sorry to hear that, please try https://example.com/wifi and let us know ^MW"),
        turn(12, "customer", "thanks, that worked a treat for me today"),
    ]
    n_brand = sum(t["role"] == "brand_agent" for t in turns)
    return {
        "case_id": case_id, "conversation_id": "conv_1", "full_turns": turns, "resolution_type": rtype, "resolution_outcome": outcome,
        "resolution_evidence": evidence if evidence is not None else [{"tweet_id": 11, "signal": rtype, "snippet": "please try the link"}],
        "resolution_signals": list(signals), "dm_redirect": dm, "resolved": resolved, "source_tweet_ids": [t["tweet_id"] for t in turns],
        "context_tweet_ids": list(context), "agent_turn_count": n_brand, "is_continuation": cont, "starts_with_customer": starts, "first_timestamp": TS,
    }


def build(*cases, intent="onboard_wifi_issue", split="train_retrieval") -> pd.DataFrame:
    df = pd.DataFrame(list(cases))
    return build_memory_records(df, {0: intent, 1: None}, {c: split for c in df["case_id"]}, {c: 0 for c in df["case_id"]})


# ---- schema ----------------------------------------------------------------------------------------------------
def test_memory_schema_has_every_required_field_in_a_stable_order() -> None:
    m = build(case())
    assert list(m.columns) == MEMORY_COLUMNS
    for required in ("case_id", "intent", "customer_problem", "historical_response", "resolution_summary", "resolution_type", "resolved", "dm_redirect", "escalation_signal", "source_tweet_ids"):
        assert required in m.columns
    row = m.iloc[0]
    assert row["intent_source"] == INTENT_SOURCE and row["escalation_signal_source"] == ESCALATION_SOURCE
    assert m["resolved"].dtype == bool and m["dm_redirect"].dtype == bool and m["in_primary_corpus"].dtype == bool


def test_a_memory_record_is_an_episode_not_a_tweet_dump() -> None:
    row = build(case()).iloc[0]
    assert row["customer_problem"] == "my wifi will not connect on the 17:43"
    assert row["historical_response"] == "Sorry to hear that, please try https://example.com/wifi and let us know"
    assert "thanks" not in row["customer_problem"], "later customer follow-ups are not part of the stated problem"
    assert "@" not in row["customer_problem"] and "t.co" not in row["customer_problem"] and "^MW" not in row["historical_response"]
    assert row["resolution_summary"].startswith("Agent provided information;") and "please try the link" in row["resolution_summary"]


def test_multi_tweet_openers_are_joined_until_the_first_brand_reply() -> None:
    turns = [turn(1, "customer", "my train is late"), turn(2, "customer", "and nobody has told us why"), turn(3, "brand_agent", "Apologies for the delay, the signal failure is being fixed"), turn(4, "customer", "ok")]
    assert customer_problem_from_turns(turns) == "my train is late and nobody has told us why"


def test_original_wording_is_preserved_including_links_and_html_entities_are_decoded() -> None:
    turns = [turn(1, "customer", "q"), turn(2, "brand_agent", "We&#39;re sorry &amp; here is the form: https://virgintrains.co.uk/form ^AB")]
    out = historical_response_from_turns(select_response_turns(turns, [{"tweet_id": 2, "signal": "self_service", "snippet": "x"}], "self_service"))
    assert out == "We're sorry & here is the form: https://virgintrains.co.uk/form"


def test_long_text_is_clipped_on_a_word_boundary() -> None:
    out = clip("alpha beta gamma delta epsilon zeta", 20)
    assert out.endswith("…") and len(out) <= 20 and " " not in out[-3:] and "gamm" not in out.replace("gamma", "")


def test_response_selection_prefers_the_evidence_turn_over_the_first_reply() -> None:
    turns = [turn(1, "customer", "refund?"), turn(2, "brand_agent", "Hi there, thanks for getting in touch with us today"), turn(3, "brand_agent", "You can claim a refund via the Delay Repay form online"), turn(4, "customer", "ta")]
    chosen = select_response_turns(turns, [{"tweet_id": 3, "signal": "refund", "snippet": "claim a refund"}], "refund")
    assert [t["tweet_id"] for t in chosen] == [3]


# ---- provenance ------------------------------------------------------------------------------------------------
def test_provenance_is_preserved_and_response_ids_are_real_brand_tweets() -> None:
    row = build(case(context=[7, 8])).iloc[0]
    assert list(row["source_tweet_ids"]) == [10, 11, 12] and list(row["context_tweet_ids"]) == [7, 8]
    assert list(row["response_tweet_ids"]) == [11]
    assert set(row["response_tweet_ids"]) <= set(row["source_tweet_ids"])
    assert row["case_id"] == "case_10" and row["conversation_id"] == "conv_1" and row["split"] == "train_retrieval"
    assert all(isinstance(x, int) for x in row["source_tweet_ids"])


# ---- weak evidence stays out of the primary corpus --------------------------------------------------------------
def test_cases_without_a_brand_reply_are_never_strong_evidence() -> None:
    turns = [turn(1, "customer", "why is my train so late today, nobody is telling us anything")]
    row = build(case(turns=turns, rtype="unresolved", outcome="no_response", evidence=[])).iloc[0]
    assert row["evidence_quality"] == NO_REPLY and not row["in_primary_corpus"] and row["exclusion_reason"] == "brand_never_replied"
    assert row["historical_response"] == "" and row["n_brand_turns"] == 0
    assert "No resolving reply" in row["resolution_summary"]


@pytest.mark.parametrize("rtype", ["other", "clarification_requested", "unresolved"])
def test_weak_resolution_types_are_marked_and_excluded(rtype) -> None:
    row = build(case(rtype=rtype)).iloc[0]
    assert row["evidence_quality"] == WEAK and not row["in_primary_corpus"] and row["exclusion_reason"] == f"weak_resolution_type_{rtype}"


@pytest.mark.parametrize(
    "kwargs, intent, reason",
    [({"cont": True}, "onboard_wifi_issue", "continuation_without_opening_problem"), ({"starts": False}, "onboard_wifi_issue", "continuation_without_opening_problem"),
     ({}, None, "no_candidate_intent"), ({}, "unclear_or_media_only", "fallback_intent_problem_unclear")],
)
def test_other_unusable_cases_are_excluded_with_a_reason(kwargs, intent, reason) -> None:
    row = build(case(**kwargs), intent=intent).iloc[0]
    assert not row["in_primary_corpus"] and row["exclusion_reason"] == reason


def test_uninformative_problems_and_empty_responses_are_excluded() -> None:
    thanks = case(turns=[turn(1, "customer", "thanks!"), turn(2, "brand_agent", "You are welcome, have a lovely journey today")])
    assert build(thanks).iloc[0]["exclusion_reason"] == "uninformative_customer_problem"
    short = case(turns=[turn(1, "customer", "when is the next train to london please"), turn(2, "brand_agent", "Hi")], evidence=[])
    assert build(short).iloc[0]["exclusion_reason"] == "response_too_short"


def test_a_good_case_is_strong_and_primary() -> None:
    row = build(case()).iloc[0]
    assert row["evidence_quality"] == STRONG and row["in_primary_corpus"] and row["exclusion_reason"] == ""


def test_escalation_signal_reflects_historical_agent_behaviour() -> None:
    assert escalation_signal([]) == "none"
    assert escalation_signal(["information_provided"]) == "none"
    assert escalation_signal(["redirected_to_dm"]) == "dm_for_account_lookup"
    assert escalation_signal(["redirected_to_other_operator", "redirected_to_dm"]) == "handoff_to_other_operator"
    assert escalation_signal(["escalated", "redirected_to_dm"]) == "customer_relations_or_formal_route"


def test_summary_mentions_the_dm_redirect_once() -> None:
    s = build(case(rtype="redirected_to_dm", outcome="redirected_to_dm", dm=True, evidence=[{"tweet_id": 11, "signal": "redirected_to_dm", "snippet": "DM us"}])).iloc[0]["resolution_summary"]
    assert s.count("DM") == 3 and "also asked" not in s


# ---- real artifacts --------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MEMORY = PROCESSED / "virgintrains_resolution_memory.parquet"
ASSIGN = PROCESSED / "splits" / "virgintrains_split_assignments.csv"
CASES = PROCESSED / "virgintrains_cases.parquet"

real = pytest.mark.skipif(not (MEMORY.exists() and ASSIGN.exists() and CASES.exists()), reason="resolution memory not built")


@real
def test_real_memory_contains_only_train_cases_and_no_golden_reserve_or_dev() -> None:
    m = pd.read_parquet(MEMORY)
    a = pd.read_csv(ASSIGN)
    train = set(a.loc[a["split"] == "train_retrieval", "case_id"])
    assert set(m["split"]) == {"train_retrieval"}
    assert set(m["case_id"]) == train, "memory is exactly the train_retrieval case set"
    for forbidden in ("golden_eval", "golden_pool_reserve", "dev_calibration"):
        assert not set(m["case_id"]) & set(a.loc[a["split"] == forbidden, "case_id"]), forbidden


@real
def test_real_memory_schema_primary_corpus_and_provenance() -> None:
    m = pd.read_parquet(MEMORY)
    assert list(m.columns) == MEMORY_COLUMNS and m["case_id"].is_unique
    primary = m[m["in_primary_corpus"]]
    assert len(primary) > 5000 and (primary["evidence_quality"] == STRONG).all()
    assert (primary["historical_response"].str.len() > 0).all() and (primary["customer_problem"].str.len() > 0).all()
    assert primary["intent"].notna().all() and not primary["intent"].eq("unclear_or_media_only").any()
    assert not m.loc[m["evidence_quality"] == NO_REPLY, "in_primary_corpus"].any()
    assert (m.loc[m["evidence_quality"] == NO_REPLY, "historical_response"] == "").all()
    assert set(m["intent_source"]) == {INTENT_SOURCE}


@real
def test_real_memory_provenance_matches_the_source_cases_and_rebuilding_is_reproducible() -> None:
    m = pd.read_parquet(MEMORY).sample(300, random_state=0).set_index("case_id")
    cases = pd.read_parquet(CASES)
    cases = cases[cases["case_id"].isin(m.index)].reset_index(drop=True)
    assert len(cases) == len(m)
    for c in cases.itertuples():
        assert list(m.loc[c.case_id, "source_tweet_ids"]) == [int(x) for x in c.source_tweet_ids]
        assert set(m.loc[c.case_id, "response_tweet_ids"]) <= set(int(x) for x in c.source_tweet_ids)
    clusters = pd.read_parquet(PROCESSED / "virgintrains_case_clusters.parquet", columns=["case_id", "cluster_id"])
    intent_of = m["intent"].to_dict()
    cluster_of = dict(zip(clusters["case_id"], clusters["cluster_id"]))
    # rebuild with the intents the file recorded for each cluster, and compare every non-intent field
    cluster_to_intent = {int(cluster_of[cid]): intent_of[cid] for cid in m.index}
    rebuilt = build_memory_records(cases, cluster_to_intent, {c: "train_retrieval" for c in cases["case_id"]}, cluster_of).set_index("case_id")
    pd.testing.assert_frame_equal(rebuilt.sort_index(), m.sort_index()[rebuilt.columns], check_dtype=False)
