from __future__ import annotations

import pandas as pd

from conftest import ts
from ingestion.reconstruction import reconstruct_conversations


def turns_by_conversation(conv: pd.DataFrame) -> dict[str, list[int]]:
    return {cid: g["tweet_id"].tolist() for cid, g in conv.groupby("conversation_id", sort=False)}


def test_linear_thread_is_one_conversation(make_tweets) -> None:
    tweets = make_tweets(
        [
            (10, "c1", "True", ts(0), "@Brand my wifi is down", "11", None),
            (11, "Brand", "False", ts(1), "@c1 sorry! DM us", "12", 10),
            (12, "c1", "True", ts(2), "@Brand done", "13", 11),
            (13, "Brand", "False", ts(3), "@c1 thanks, fixed", None, 12),
        ]
    )
    conv, stats = reconstruct_conversations(tweets)
    assert turns_by_conversation(conv) == {"conv_10": [10, 11, 12, 13]}
    assert conv["turn_index"].tolist() == [0, 1, 2, 3]
    assert conv["parent_tweet_id"].tolist()[1:] == [10, 11, 12]
    assert pd.isna(conv["parent_tweet_id"].iloc[0])
    assert stats.conversations == 1 and stats.edges > 0


def test_branching_replies_and_separate_threads(make_tweets) -> None:
    tweets = make_tweets(
        [
            (1, "c1", "True", ts(0), "q", "2,3", None),
            (2, "BrandA", "False", ts(1), "a1", None, 1),
            (3, "BrandB", "False", ts(2), "a2", None, 1),
            (50, "c2", "True", ts(5), "other issue", "51", None),
            (51, "BrandA", "False", ts(6), "reply", None, 50),
        ]
    )
    conv, stats = reconstruct_conversations(tweets)
    assert turns_by_conversation(conv) == {"conv_1": [1, 2, 3], "conv_50": [50, 51]}
    assert stats.conversations == 2


def test_turns_are_ordered_by_timestamp_not_id_or_file_order(make_tweets) -> None:
    # File order and id order both disagree with the true chronology (as in twcs.csv).
    tweets = make_tweets(
        [
            (1, "Brand", "False", ts(30), "@c1 last reply", None, 3),
            (3, "c1", "True", ts(20), "@Brand follow-up", "1", 4),
            (4, "Brand", "False", ts(10), "@c1 first reply", "3", 8),
            (8, "c1", "True", ts(0), "@Brand opening", "4", None),
        ]
    )
    conv, _ = reconstruct_conversations(tweets)
    assert conv["tweet_id"].tolist() == [8, 4, 3, 1]
    assert conv["conversation_id"].unique().tolist() == ["conv_8"]
    assert conv["created_at"].is_monotonic_increasing


def test_timestamp_ties_break_by_tweet_id_and_nat_sorts_last(make_tweets) -> None:
    tweets = make_tweets(
        [
            (7, "c1", "True", ts(0), "q", "6,5", None),
            (6, "Brand", "False", ts(1), "same second b", None, 7),
            (5, "Brand", "False", ts(1), "same second a", None, 7),
            (9, "c1", "True", "garbage", "no timestamp", None, 6),
        ]
    )
    conv, _ = reconstruct_conversations(tweets)
    assert conv["tweet_id"].tolist() == [7, 5, 6, 9]


def test_missing_parent_ids_are_flagged_not_dropped(make_tweets) -> None:
    tweets = make_tweets(
        [
            (20, "Brand", "False", ts(1), "@c1 reply to a tweet not in the sample", None, 999),
            (21, "c1", "True", ts(2), "@Brand thanks", None, 20),
        ]
    )
    conv, stats = reconstruct_conversations(tweets)
    assert turns_by_conversation(conv) == {"conv_20": [20, 21]}
    assert conv.set_index("tweet_id").loc[20, "missing_parent"]
    assert not conv.set_index("tweet_id").loc[21, "missing_parent"]
    assert stats.missing_parent_refs == 1


def test_dangling_response_ids_are_counted(make_tweets) -> None:
    tweets = make_tweets([(30, "c1", "True", ts(0), "q", "31,32", None), (31, "B", "False", ts(1), "a", None, 30)])
    conv, stats = reconstruct_conversations(tweets)
    assert stats.dangling_response_refs == 1
    assert turns_by_conversation(conv) == {"conv_30": [30, 31]}


def test_link_recovered_from_response_list_when_in_response_to_missing(make_tweets) -> None:
    tweets = make_tweets(
        [
            (40, "c1", "True", ts(0), "q", "41", None),
            (41, "Brand", "False", ts(1), "a", None, None),
        ]
    )
    conv, stats = reconstruct_conversations(tweets)
    assert turns_by_conversation(conv) == {"conv_40": [40, 41]}
    row = conv.set_index("tweet_id").loc[41]
    assert row["parent_tweet_id"] == 40
    assert row["parent_source"] == "response_tweet_id"
    assert stats.parents_inferred_from_response_list == 1


def test_single_message_conversations(make_tweets) -> None:
    tweets = make_tweets(
        [
            (60, "c1", "True", ts(0), "@Brand anyone there?", None, None),
            (61, "c2", "True", ts(1), "@Brand hello?", None, None),
        ]
    )
    conv, stats = reconstruct_conversations(tweets)
    assert turns_by_conversation(conv) == {"conv_60": [60], "conv_61": [61]}
    assert conv["conversation_size"].tolist() == [1, 1]
    assert stats.single_tweet_conversations == 2


def test_duplicate_tweets_do_not_duplicate_turns(make_tweets) -> None:
    tweets = make_tweets(
        [
            (70, "c1", "True", ts(0), "q", "71", None),
            (71, "Brand", "False", ts(1), "a", None, 70),
        ]
    )
    duplicated = pd.concat([tweets, tweets.iloc[[1]]], ignore_index=True)
    conv, stats = reconstruct_conversations(duplicated)
    assert conv["tweet_id"].tolist() == [70, 71]
    assert stats.tweets == 2


def test_self_reference_and_cycles_terminate(make_tweets) -> None:
    tweets = make_tweets(
        [
            (80, "c1", "True", ts(0), "self", "81", 80),
            (81, "Brand", "False", ts(1), "cycle", "80", 80),
        ]
    )
    conv, stats = reconstruct_conversations(tweets)
    assert turns_by_conversation(conv) == {"conv_80": [80, 81]}
    assert stats.self_references == 1


def test_empty_input(make_tweets) -> None:
    tweets = make_tweets([(1, "c", "True", ts(0), "x", None, None)]).iloc[0:0]
    conv, stats = reconstruct_conversations(tweets)
    assert conv.empty and stats.conversations == 0
