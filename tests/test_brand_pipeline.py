"""Brand-level pipeline: conversations -> episodes -> support cases (synthetic tweets only)."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import raw_frame
from ingestion.brand_pipeline import build_conversation_table, process_tweets
from ingestion.episodes import SegmentationConfig
from ingestion.loader import clean_tweets
from ingestion.roles import BRAND_AGENT, CUSTOMER, OTHER_AGENT, assign_roles, detect_hidden_agents

BRAND = "Brand"


def tsd(day: int, hour: int = 10, minute: int = 0, second: int = 0) -> str:
    return f"Tue Oct {day:02d} {hour:02d}:{minute:02d}:{second:02d} +0000 2017"


def run(make_tweets, rows, cfg: SegmentationConfig | None = None):
    return process_tweets(make_tweets(rows), BRAND, cfg)


def cases_by_first_tweet(result) -> dict[int, dict]:
    return {c["source_tweet_ids"][0]: c for c in result.cases}


def test_case_has_all_required_fields_in_chronological_order(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            (10, "c1", "True", tsd(2, 10, 0), "@Brand my wifi is down", "11", None),
            (11, BRAND, "False", tsd(2, 10, 5), "@c1 sorry, please DM us your booking ref", "12", 10),
            (12, "c1", "True", tsd(2, 10, 9), "@Brand here", "13", 11),
            (13, BRAND, "False", tsd(2, 10, 12), "@c1 thanks, fixed", None, 12),
        ],
    )
    assert len(result.cases) == 1
    case = result.cases[0]
    for field in [
        "case_id", "brand", "conversation_id", "customer_messages", "agent_messages", "full_turns", "turn_count",
        "first_timestamp", "last_timestamp", "resolved", "resolution_type", "resolution_summary", "dm_redirect",
        "source_tweet_ids",
    ]:
        assert field in case, field
    assert case["brand"] == BRAND
    assert case["source_tweet_ids"] == [10, 11, 12, 13]
    assert [t["role"] for t in case["full_turns"]] == [CUSTOMER, BRAND_AGENT, CUSTOMER, BRAND_AGENT]
    assert case["customer_messages"] == ["@Brand my wifi is down", "@Brand here"]
    assert len(case["agent_messages"]) == 2
    assert case["turn_count"] == 4
    assert case["first_timestamp"] < case["last_timestamp"]
    assert case["dm_redirect"] is True


def test_turns_follow_timestamps_not_file_or_id_order(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            (1, BRAND, "False", tsd(2, 10, 30), "@c1 last reply from us", None, 3),
            (3, "c1", "True", tsd(2, 10, 20), "@Brand follow up question", "1", 4),
            (4, BRAND, "False", tsd(2, 10, 10), "@c1 first reply from us", "3", 7),
            (7, "c1", "True", tsd(2, 10, 0), "@Brand where is my train", "4", None),
        ],
    )
    assert [c["source_tweet_ids"] for c in result.cases] == [[7, 4, 3, 1]]


def test_same_second_parent_precedes_child_even_with_lower_child_id(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            (50, "c1", "True", tsd(2, 10, 0, 7), "@Brand hello there", "20", None),
            (20, BRAND, "False", tsd(2, 10, 0, 7), "@c1 hi", None, 50),
        ],
    )
    assert result.cases[0]["source_tweet_ids"] == [50, 20]


def test_two_customers_in_one_thread_become_two_cases(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            (1, "c1", "True", tsd(2, 10, 0), "@Brand cancelled again??", "2", None),
            (2, BRAND, "False", tsd(2, 10, 3), "@c1 sorry to hear that, what service?", "3", 1),
            (3, "c2", "True", tsd(2, 10, 5), "@c1 @Brand same for me today", "4", 2),
            (4, BRAND, "False", tsd(2, 10, 8), "@c2 apologies, any details?", None, 3),
        ],
    )
    owners = {c["customer_id"]: c["source_tweet_ids"] for c in result.cases}
    assert owners == {"c1": [1, 2], "c2": [3, 4]}
    assert len({c["conversation_id"] for c in result.cases}) == 1
    assert all(c["thread_customer_count"] == 2 for c in result.cases)


def test_unrelated_threads_from_same_customer_are_separate_cases(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            (1, "c1", "True", tsd(2, 9, 0), "@Brand wifi not working on the 09:00", "2", None),
            (2, BRAND, "False", tsd(2, 9, 4), "@c1 try reconnecting please", None, 1),
            (30, "c1", "True", tsd(2, 12, 0), "@Brand totally different: lost my scarf", "31", None),
            (31, BRAND, "False", tsd(2, 12, 6), "@c1 please contact lost property", None, 30),
        ],
    )
    assert len(result.cases) == 2
    first, second = sorted(result.cases, key=lambda c: c["first_timestamp"])
    assert first["conversation_id"] != second["conversation_id"]
    assert first["customer_id"] == second["customer_id"] == "c1"
    assert set(first["source_tweet_ids"]).isdisjoint(second["source_tweet_ids"])
    assert second["case_id"] in first["related_case_ids"]


def test_long_silence_starts_a_new_linked_continuation_case(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            (1, "c1", "True", tsd(2, 10, 0), "@Brand where is my refund", "2", None),
            (2, BRAND, "False", tsd(2, 10, 5), "@c1 please claim via the form", "3", 1),
            (3, "c1", "True", tsd(6, 10, 0), "@Brand still no news", "4", 2),
            (4, BRAND, "False", tsd(6, 10, 4), "@c1 we are looking into it", None, 3),
        ],
        SegmentationConfig(gap_hours=24.0),
    )
    assert len(result.cases) == 2
    first, second = sorted(result.cases, key=lambda c: c["first_timestamp"])
    assert first["source_tweet_ids"] == [1, 2] and second["source_tweet_ids"] == [3, 4]
    assert second["is_continuation"] and second["continues_case_id"] == first["case_id"]
    assert not first["is_continuation"]
    assert first["conversation_id"] == second["conversation_id"]
    assert second["context_tweet_ids"] == [], "the earlier episode is linked, not duplicated as context"


ANNOUNCEMENT = (1, BRAND, "False", tsd(2, 9, 0), "Planned engineering works affect services today", "2,4", None)


def test_brand_announcement_with_several_customers_is_context_not_a_turn(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            ANNOUNCEMENT,
            (2, "c1", "True", tsd(2, 9, 30), "@Brand does this affect the 10:15?", "3", 1),
            (3, BRAND, "False", tsd(2, 9, 33), "@c1 yes, it will run 20 minutes late", None, 2),
            (4, "c2", "True", tsd(2, 9, 40), "@Brand and the 11:00?", "5", 1),
            (5, BRAND, "False", tsd(2, 9, 44), "@c2 that one is on time", None, 4),
        ],
    )
    by_customer = {c["customer_id"]: c for c in result.cases}
    assert by_customer["c1"]["source_tweet_ids"] == [2, 3]
    assert by_customer["c2"]["source_tweet_ids"] == [4, 5]
    assert all(c["context_tweet_ids"] == [1] and c["starts_with_customer"] for c in result.cases)
    announcement = result.tweets.set_index("tweet_id").loc[1]
    assert pd.isna(announcement["case_id"]) and isinstance(announcement["exclusion_reason"], str)


def test_brand_opening_in_a_single_customer_thread_is_attributed_to_that_customer(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            (1, BRAND, "False", tsd(2, 9, 0), "Planned engineering works affect services today", "2", None),
            (2, "c1", "True", tsd(2, 9, 30), "@Brand does this affect the 10:15?", "3", 1),
            (3, BRAND, "False", tsd(2, 9, 33), "@c1 yes, it will run 20 minutes late", None, 2),
        ],
    )
    (case,) = result.cases
    assert case["source_tweet_ids"] == [1, 2, 3]
    assert case["starts_with_customer"] is False, "flagged so downstream code can exclude such cases from golden"
    assert case["full_turns"][0]["owner_source"] == "sole_customer"


def test_short_gap_does_not_split_the_case(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            (1, "c1", "True", tsd(2, 8, 0), "@Brand where is my refund", "2", None),
            (2, BRAND, "False", tsd(2, 8, 5), "@c1 please claim via the form", "3", 1),
            (3, "c1", "True", tsd(2, 20, 0), "@Brand still nothing", "4", 2),
            (4, BRAND, "False", tsd(2, 20, 4), "@c1 checking", None, 3),
        ],
        SegmentationConfig(gap_hours=24.0),
    )
    assert len(result.cases) == 1


def test_missing_relationship_ids_are_bridged_by_the_other_side(make_tweets) -> None:
    # Tweet 2 lacks in_response_to; tweet 1 lists it in response_tweet_id. Tweet 6 lacks response ids
    # but names its parent 5 via in_response_to.
    result = run(
        make_tweets,
        [
            (1, "c1", "True", tsd(2, 10, 0), "@Brand first", "2", None),
            (2, BRAND, "False", tsd(2, 10, 2), "@c1 answer one", None, None),
            (5, "c2", "True", tsd(2, 11, 0), "@Brand second", None, None),
            (6, BRAND, "False", tsd(2, 11, 2), "@c2 answer two", None, 5),
        ],
    )
    grouped = {tuple(c["source_tweet_ids"]) for c in result.cases}
    assert grouped == {(1, 2), (5, 6)}


def test_parent_missing_from_the_file_is_flagged_not_dropped(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            (2, BRAND, "False", tsd(2, 10, 2), "@c1 here is the answer to your question", None, 999),
            (3, "c1", "True", tsd(2, 10, 5), "@Brand thanks", None, 2),
        ],
    )
    assert result.cases, "tweets with a dangling parent must still produce a case"
    turns = result.cases[0]["full_turns"]
    assert any(t["in_response_to_tweet_id"] == 999 for t in turns)
    assert result.cases[0]["has_missing_parent"] is True


def test_duplicate_tweets_appear_once(make_tweets) -> None:
    rows = [
        (1, "c1", "True", tsd(2, 10, 0), "@Brand hello", "2", None),
        (1, "c1", "True", tsd(2, 10, 0), "@Brand hello", "2", None),
        (2, BRAND, "False", tsd(2, 10, 3), "@c1 hi", None, 1),
    ]
    cleaned, report = clean_tweets(raw_frame(rows))
    result = process_tweets(cleaned, BRAND)
    assert result.cases[0]["source_tweet_ids"] == [1, 2]
    assert result.tweets["tweet_id"].is_unique
    assert report is not None


def test_other_brand_conversations_are_excluded(make_tweets) -> None:
    result = run(
        make_tweets,
        [
            (1, "c1", "True", tsd(2, 10, 0), "@Other hello", "2", None),
            (2, "Other", "False", tsd(2, 10, 3), "@c1 hi", None, 1),
            (3, "c2", "True", tsd(2, 10, 0), "@Brand hello", "4", None),
            (4, BRAND, "False", tsd(2, 10, 3), "@c2 hi", None, 3),
        ],
    )
    assert [c["source_tweet_ids"] for c in result.cases] == [[3, 4]]
    assert set(result.tweets["tweet_id"]) == {3, 4}


def test_provenance_is_preserved_on_every_turn(make_tweets) -> None:
    rows = [
        (10, "c1", "True", tsd(2, 10, 0), "@Brand hi", "11,12", None),
        (11, BRAND, "False", tsd(2, 10, 5), "@c1 hello", None, 10),
        (12, "Other", "False", tsd(2, 10, 6), "@c1 hello from us too", None, 10),
    ]
    result = run(make_tweets, rows)
    case = result.cases[0]
    by_id = {t["tweet_id"]: t for t in case["full_turns"]}
    assert set(by_id) == set(case["source_tweet_ids"]) == {10, 11, 12}
    assert by_id[10]["created_at_raw"] == tsd(2, 10, 0)
    assert by_id[10]["response_tweet_id_raw"] == "11,12"
    assert by_id[11]["in_response_to_tweet_id"] == 10
    assert by_id[10]["text"] == "@Brand hi"
    assert by_id[10]["inbound"] is True and by_id[11]["inbound"] is False
    assert sorted(t["source_row"] for t in case["full_turns"]) == [0, 1, 2]
    assert by_id[12]["role"] == OTHER_AGENT

    conv_table = build_conversation_table(result.tweets, BRAND)
    assert conv_table.loc[0, "tweet_ids"] == [10, 11, 12]
    assert [t["tweet_id"] for t in conv_table.loc[0, "turns"]] == [10, 11, 12]
    assert conv_table.loc[0, "turns"][0]["case_id"] == case["case_id"]


def test_inbound_flagged_account_with_agent_signoffs_is_treated_as_agent(make_tweets) -> None:
    rows = [(100 + i, "ghost", "True", tsd(2, 10, i), f"@c{i} sorry about that ^AB", None, None) for i in range(6)]
    rows += [(200, "c9", "True", tsd(2, 11, 0), "@Brand hi", None, None)]
    tweets = make_tweets(rows)
    hidden = detect_hidden_agents(tweets)
    assert hidden == {"ghost"}
    roles = assign_roles(tweets, BRAND, hidden)
    assert set(roles[tweets["author_id"] == "ghost"]) == {OTHER_AGENT}
    assert roles[tweets["author_id"] == "c9"].iloc[0] == CUSTOMER
    assert tweets.loc[tweets["author_id"] == "ghost", "inbound"].all(), "raw inbound flag must stay untouched"


def test_customer_with_few_signed_tweets_is_not_hidden_agent(make_tweets) -> None:
    rows = [(1, "x", "True", tsd(2, 10, 0), "thanks ^AB", None, None)] + [
        (2 + i, "x", "True", tsd(2, 10, 1 + i), "normal message", None, None) for i in range(6)
    ]
    assert detect_hidden_agents(make_tweets(rows)) == set()


@pytest.mark.parametrize("gap_hours", [1.0, 48.0])
def test_every_attributed_tweet_belongs_to_exactly_one_case(make_tweets, gap_hours: float) -> None:
    result = run(
        make_tweets,
        [
            (1, "c1", "True", tsd(2, 10, 0), "@Brand a", "2", None),
            (2, BRAND, "False", tsd(2, 10, 5), "@c1 b", "3", 1),
            (3, "c1", "True", tsd(2, 14, 0), "@Brand c", "4", 2),
            (4, BRAND, "False", tsd(2, 14, 5), "@c1 d", None, 3),
        ],
        SegmentationConfig(gap_hours=gap_hours),
    )
    ids = [t for c in result.cases for t in c["source_tweet_ids"]]
    assert sorted(ids) == [1, 2, 3, 4] and len(ids) == len(set(ids))
    assert result.tweets["case_id"].notna().all()
    assert pd.Series(ids).is_unique
