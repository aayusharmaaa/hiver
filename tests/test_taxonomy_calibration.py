from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.sampling import add_coverage_columns
from evaluation.taxonomy_calibration import (
    DEFAULT_QUOTAS,
    FALLBACK,
    HUMAN_COLUMNS,
    CalibrationConfig,
    _allocate,
    assert_calibration_safe,
    calibration_frame,
    exclusion_reasons,
    sample_calibration,
)


def make_pool(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    intents = list(DEFAULT_QUOTAS)
    rows = []
    for i in range(n):
        intent = intents[i % len(intents)]
        rows.append(
            {
                "case_id": f"r_{i}", "group_id": f"g_{i}", "customer_id": f"u_{i}", "conversation_id": f"c_{i}",
                "source_tweet_ids": [1000 + 2 * i, 1001 + 2 * i], "context_tweet_ids": [],
                "opening_message": f"unique opener {chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}{chr(97 + (i // 676) % 26)} text",
                "is_continuation": False, "starts_with_customer": True,
                "candidate_intent": intent, "cluster_id": intents.index(intent),
                "margin": float(rng.random()), "resolution_type": ["information_provided", "self_service", "unresolved"][i % 3],
                "turn_count": int(rng.integers(1, 12)), "dm_redirect": bool(i % 7 == 0),
                "first_timestamp": pd.Timestamp("2017-10-12", tz="UTC") + pd.Timedelta(hours=i),
            }
        )
    return add_coverage_columns(pd.DataFrame(rows))


def golden_from(pool: pd.DataFrame, **overrides) -> pd.DataFrame:
    base = {"group_id": "gg", "customer_id": "gu", "conversation_id": "gc", "source_tweet_ids": [5], "context_tweet_ids": [], "opening_message": "completely different golden text"}
    base.update(overrides)
    return pd.DataFrame([base])


def test_reserve_cases_that_neighbour_golden_are_excluded() -> None:
    pool = make_pool(20)
    cases = {
        "shares_group_with_golden": {"group_id": pool.loc[0, "group_id"]},
        "shares_customer_with_golden": {"customer_id": pool.loc[1, "customer_id"]},
        "shares_conversation_with_golden": {"conversation_id": pool.loc[2, "conversation_id"]},
        "shares_tweet_with_golden": {"source_tweet_ids": [pool.loc[3, "source_tweet_ids"][0]]},
        "opening_text_duplicates_golden": {"opening_message": "@VirginTrains " + pool.loc[4, "opening_message"].upper() + " https://t.co/abc"},
    }
    for reason, override in cases.items():
        idx = list(cases).index(reason)
        reasons = exclusion_reasons(pool, golden_from(pool, **override))
        assert reasons.iloc[idx] == reason, reason
        assert (reasons == "").sum() == len(pool) - 1


def test_context_tweet_overlap_with_golden_is_excluded() -> None:
    pool = make_pool(5)
    pool.at[2, "context_tweet_ids"] = [777]
    reasons = exclusion_reasons(pool, golden_from(pool, context_tweet_ids=[777]))
    assert reasons.iloc[2] == "shares_tweet_with_golden"


def test_continuations_and_non_customer_openers_are_excluded() -> None:
    pool = make_pool(6)
    pool.loc[0, "is_continuation"] = True
    pool.loc[1, "starts_with_customer"] = False
    reasons = exclusion_reasons(pool, golden_from(pool))
    assert reasons.iloc[0] == "continuation_episode_no_opening"
    assert reasons.iloc[1] == "does_not_start_with_customer"
    assert (reasons.iloc[2:] == "").all()


def test_sample_meets_target_and_overrepresents_hard_strata() -> None:
    pool = make_pool(1200)
    sample = sample_calibration(pool, CalibrationConfig(target_size=200))
    assert len(sample) == 200 and sample["case_id"].is_unique
    counts = sample["sampling_stratum"].value_counts()
    assert counts["service_status_delay_enquiry"] > counts["onboard_wifi_issue"]
    population_share = pool["candidate_intent"].value_counts(normalize=True)
    assert counts["service_status_delay_enquiry"] / 200 > population_share["service_status_delay_enquiry"], "hard stratum must be over-represented"
    assert set(sample["selection_reason"]) <= {"low_margin_boundary", "diverse_coverage", "all_eligible", "fallback_coverage"}


def test_low_margin_cases_are_preferred_inside_a_stratum() -> None:
    pool = make_pool(1200)
    sample = sample_calibration(pool, CalibrationConfig(target_size=200))
    hard = "journey_disruption_complaint"
    chosen = sample[(sample["candidate_intent"] == hard) & (sample["selection_reason"] == "low_margin_boundary")]
    eligible = pool[pool["candidate_intent"] == hard]
    assert len(chosen) > 0
    assert chosen["margin"].max() <= eligible["margin"].quantile(0.4) + 1e-12


def test_sampling_is_deterministic_and_seed_sensitive() -> None:
    pool = make_pool(800)
    a = sample_calibration(pool, CalibrationConfig(seed=1))
    b = sample_calibration(pool, CalibrationConfig(seed=1))
    c = sample_calibration(pool, CalibrationConfig(seed=2))
    assert a["case_id"].tolist() == b["case_id"].tolist()
    assert a["case_id"].tolist() != c["case_id"].tolist()


def test_stratum_weight_restores_stratum_size() -> None:
    pool = make_pool(1200)
    sample = sample_calibration(pool, CalibrationConfig())
    for intent, g in sample.groupby("sampling_stratum"):
        assert g["stratum_weight"].sum() == pytest.approx(len(pool[pool["candidate_intent"] == intent]))


def test_shortfall_is_redistributed_in_priority_order() -> None:
    alloc = _allocate({"a": 10, "b": 10, "c": 10}, {"a": 4, "b": 50, "c": 50}, 30)
    assert alloc["a"] == 4 and sum(alloc.values()) == 30 and alloc["b"] >= 10 and alloc["c"] >= 10


def test_small_fallback_stratum_is_taken_whole() -> None:
    pool = make_pool(900)
    keep = pool[pool["candidate_intent"] != FALLBACK]
    few = pool[pool["candidate_intent"] == FALLBACK].head(6)
    sample = sample_calibration(pd.concat([keep, few]), CalibrationConfig())
    fb = sample[sample["candidate_intent"] == FALLBACK]
    assert len(fb) == 6 and set(fb["selection_reason"]) == {"all_eligible"}


def test_human_columns_are_blank_and_required_fields_present() -> None:
    pool = make_pool(600)
    sample = sample_calibration(pool, CalibrationConfig(target_size=60))
    sample = sample.assign(
        full_turns=[[{"role": "customer", "tweet_id": 1, "text": "hi"}] for _ in range(len(sample))], cluster_name="c", second_cluster_id=1,
        runner_up_cluster_name="r", runner_up_intent="x", resolved=False, resolution_outcome="no_response", resolution_summary="s",
        split="golden_pool_reserve",
    )
    frame = calibration_frame(sample)
    for col in ("case_id", "first_customer_message", "conversation", "candidate_intent", "candidate_cluster_id", "auto_resolution_type", "auto_resolved", "sampling_stratum", "selection_reason", "stratum_weight", "source_tweet_ids", "sampling_seed"):
        assert col in frame
    for col in HUMAN_COLUMNS:
        assert (frame[col] == "").all(), f"{col} must not be pre-filled"


def _assignments(sample_ids: list[str], **splits) -> pd.DataFrame:
    rows = [{"case_id": c, "split": "golden_pool_reserve", "group_id": f"g{c}", "customer_id": f"u{c}", "conversation_id": f"v{c}"} for c in sample_ids]
    rows += [{"case_id": "G1", "split": "golden_eval", "group_id": "gG", "customer_id": "uG", "conversation_id": "vG"}]
    df = pd.DataFrame(rows)
    for case, split in splits.items():
        df.loc[df["case_id"] == case, "split"] = split
    return df


def test_safety_check_passes_on_clean_sample_and_blocks_train_dev_golden() -> None:
    sample = pd.DataFrame({"case_id": ["a", "b", "c"]})
    assert all(v == 0 for v in assert_calibration_safe(sample, _assignments(["a", "b", "c"])).values())
    for bad_split in ("train_retrieval", "dev_calibration", "golden_eval"):
        with pytest.raises(AssertionError):
            assert_calibration_safe(sample, _assignments(["a", "b", "c"], a=bad_split))


def test_safety_check_blocks_customer_shared_with_golden() -> None:
    sample = pd.DataFrame({"case_id": ["a", "b"]})
    assignments = _assignments(["a", "b"])
    assignments.loc[assignments["case_id"] == "a", "customer_id"] = "uG"
    with pytest.raises(AssertionError):
        assert_calibration_safe(sample, assignments)
