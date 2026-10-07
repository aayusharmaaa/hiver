from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.sampling import diversity_sample, length_bucket, stratum_weights, time_period
from evaluation.splits import (
    DEV,
    EXCLUDED,
    GOLDEN,
    TRAIN,
    LeakageError,
    SplitConfig,
    case_groups,
    finalize_splits,
    opener_key,
    provisional_split,
    verify_no_leakage,
)
from taxonomy.taxonomy_builder import match_labels


def _letters(i: int) -> str:
    """Digits are collapsed by `opener_key`, so make openers distinct with letters."""
    out = ""
    while True:
        out = chr(97 + i % 26) + out
        i //= 26
        if i == 0:
            return out


def synthetic_cases(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # Mostly one customer per conversation, with a few repeat customers; denser sharing chains
        # everything into a single group (which is correct behaviour but makes a useless fixture).
        customer = f"c{i // 2}" if rng.random() > 0.05 else f"c{rng.integers(0, n // 2)}"
        rows.append(
            {
                "case_id": f"case_{i}",
                "customer_id": customer,
                "conversation_id": f"conv_{i // 2}",
                "opening_message": f"@Brand question {_letters(i)} about my train today",
                "has_brand_reply": True,
                "is_continuation": False,
                "starts_with_customer": True,
                "resolution_type": ["information_provided", "self_service", "unresolved", "compensation"][i % 4],
                "turn_count": int(rng.integers(1, 12)),
                "dm_redirect": bool(i % 17 == 0),
                "resolved": bool(i % 5 == 0),
                "first_timestamp": pd.Timestamp("2017-10-10", tz="UTC") + pd.Timedelta(hours=int(i)),
                "source_tweet_ids": [10 * i, 10 * i + 1],
                "context_tweet_ids": [],
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def assigned() -> pd.DataFrame:
    cases = synthetic_cases()
    cfg = SplitConfig(golden_size=60)
    prov = provisional_split(cases, cfg)
    clusters = pd.Series(np.arange(len(cases)) % 6, index=cases["case_id"])
    return finalize_splits(cases, prov, clusters, cfg)


def test_groups_join_cases_sharing_a_customer_or_conversation() -> None:
    cases = pd.DataFrame(
        {
            "case_id": ["a", "b", "c", "d"],
            "customer_id": ["u1", "u1", "u2", "u3"],
            "conversation_id": ["x", "y", "y", "z"],
        }
    )
    groups = case_groups(cases)
    assert len({groups[0], groups[1], groups[2]}) == 1, "a-b share customer u1, b-c share conversation y"
    assert groups[3] != groups[0]


def test_split_is_deterministic_and_seed_dependent() -> None:
    cases = synthetic_cases(400)
    a = provisional_split(cases, SplitConfig(seed=1))
    b = provisional_split(cases, SplitConfig(seed=1))
    c = provisional_split(cases, SplitConfig(seed=2))
    assert a["provisional_split"].equals(b["provisional_split"])
    assert not a["provisional_split"].equals(c["provisional_split"])


def test_no_group_spans_two_provisional_splits() -> None:
    prov = provisional_split(synthetic_cases(800), SplitConfig())
    assert (prov.groupby("group_id")["provisional_split"].nunique() == 1).all()
    assert (prov.groupby("customer_id")["provisional_split"].nunique() == 1).all()


def test_golden_is_disjoint_from_train_and_dev(assigned: pd.DataFrame) -> None:
    golden = assigned[assigned["split"] == GOLDEN]
    assert 0 < len(golden) <= 60
    checks = verify_no_leakage(assigned)
    assert all(v == 0 for v in checks.values())
    other = assigned[assigned["split"].isin([TRAIN, DEV])]
    assert set(golden["customer_id"]).isdisjoint(other["customer_id"])
    assert set(golden["conversation_id"]).isdisjoint(other["conversation_id"])


def test_golden_pool_leftovers_are_neither_train_nor_dev(assigned: pd.DataFrame) -> None:
    reserve = assigned[assigned["split"] == "golden_pool_reserve"]
    assert len(reserve) > 0
    assert not set(reserve["case_id"]) & set(assigned.loc[assigned["split"].isin([TRAIN, DEV]), "case_id"])


def test_duplicate_opening_of_a_golden_case_is_removed_from_train() -> None:
    cases = synthetic_cases(1500)
    cfg = SplitConfig(golden_size=60)
    prov = provisional_split(cases, cfg)
    pool_ids = prov.loc[prov["provisional_split"] == "golden_pool", "case_id"]
    train_id = prov.loc[prov["provisional_split"] == TRAIN, "case_id"].iloc[0]
    # Make a train case reuse every pool case's opening text: whichever pool case becomes golden, it collides.
    shared = "@Brand where is the 10:15 to Euston"
    cases.loc[cases["case_id"].isin(pool_ids), "opening_message"] = shared
    cases.loc[cases["case_id"] == train_id, "opening_message"] = shared
    clusters = pd.Series(0, index=cases["case_id"])
    out = finalize_splits(cases, prov, clusters, cfg)
    row = out.set_index("case_id").loc[train_id]
    assert row["split"] == EXCLUDED and row["split_reason"] == "opening_text_duplicates_golden"
    verify_no_leakage(out)


def test_shared_context_tweet_with_golden_is_excluded_from_train() -> None:
    cases = synthetic_cases(1500)
    cfg = SplitConfig(golden_size=60)
    prov = provisional_split(cases, cfg)
    pool_ids = set(prov.loc[prov["provisional_split"] == "golden_pool", "case_id"])
    train_id = prov.loc[prov["provisional_split"] == TRAIN, "case_id"].iloc[0]
    cases["context_tweet_ids"] = [[777] if c in pool_ids or c == train_id else [] for c in cases["case_id"]]
    out = finalize_splits(cases, prov, pd.Series(0, index=cases["case_id"]), cfg)
    assert out.set_index("case_id").loc[train_id, "split"] == EXCLUDED
    verify_no_leakage(out)


def test_verify_no_leakage_raises_on_shared_customer(assigned: pd.DataFrame) -> None:
    bad = assigned.copy()
    golden_customer = bad.loc[bad["split"] == GOLDEN, "customer_id"].iloc[0]
    victim = bad.index[bad["split"] == TRAIN][0]
    bad.loc[victim, "customer_id"] = golden_customer
    with pytest.raises(LeakageError):
        verify_no_leakage(bad)


def test_ineligible_cases_are_never_golden() -> None:
    cases = synthetic_cases(1500)
    cases.loc[::3, "has_brand_reply"] = False
    cases.loc[1::3, "is_continuation"] = True
    cfg = SplitConfig(golden_size=60)
    out = finalize_splits(cases, provisional_split(cases, cfg), pd.Series(0, index=cases["case_id"]), cfg)
    golden = out[out["split"] == GOLDEN]
    assert golden["has_brand_reply"].all() and not golden["is_continuation"].any()


def test_opener_key_normalizes_mentions_urls_and_numbers() -> None:
    a = opener_key("@Brand where is the 10:15 to Euston? https://t.co/aaa")
    b = opener_key("@Other Where is the 11:20 to euston?? https://t.co/bbb")
    assert a == b and a


def test_diversity_sample_is_deterministic_and_covers_rare_levels() -> None:
    frame = pd.DataFrame({"kind": ["common"] * 950 + ["rare"] * 50, "flag": [True, False] * 500})
    a = diversity_sample(frame, ["kind", "flag"], 100, seed=3)
    b = diversity_sample(frame, ["kind", "flag"], 100, seed=3)
    assert list(a) == list(b) and len(a) == 100 and len(set(a)) == 100
    assert (frame.loc[a, "kind"] == "rare").mean() > 0.05, "rare level should be over-represented vs its 5% share"


def test_stratum_weights_restore_population_shares() -> None:
    pop = pd.DataFrame({"c": ["a"] * 90 + ["b"] * 10})
    sample = pd.DataFrame({"c": ["a"] * 5 + ["b"] * 5})
    w = stratum_weights(pop, sample, "c")
    assert w.iloc[0] == pytest.approx(1.8) and w.iloc[-1] == pytest.approx(0.2)


@pytest.mark.parametrize("n, label", [(1, "short_1-2"), (2, "short_1-2"), (3, "medium_3-4"), (8, "long_5-8"), (9, "very_long_9+")])
def test_length_buckets(n: int, label: str) -> None:
    assert length_bucket(n) == label


def test_time_period_pools_early_and_handles_missing() -> None:
    assert time_period(pd.Timestamp("2016-01-01", tz="UTC")) == "before_2017-10-09"
    assert time_period(pd.Timestamp("2017-10-12", tz="UTC")) == "week_2017-10-09"
    assert time_period(None) == "unknown_time"


def test_cluster_labels_are_matched_by_terms_not_by_numeric_id() -> None:
    specs = [
        {"name": "wifi", "anchor_terms": ["wifi", "internet", "connect"]},
        {"name": "refund", "anchor_terms": ["refund", "delay", "repay"]},
    ]
    # Same clusters, arbitrary renumbering: labels must follow the terms.
    matched = match_labels({7: ["delay", "repay", "claim"], 3: ["wifi", "connect", "train"]}, specs)
    assert matched[7]["name"] == "refund" and matched[3]["name"] == "wifi"
    unmatched = match_labels({1: ["banana", "apple"]}, specs)
    assert unmatched[1]["final_intent"].endswith("NEEDS_REVIEW"), "unmatched clusters must be flagged, never silently named"
