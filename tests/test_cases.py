from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import ts
from ingestion.cases import build_support_cases, write_support_cases
from ingestion.reconstruction import reconstruct_conversations
from ingestion.resolution import ResolutionType, classify_resolution

BRANDS = ["AppleSupport", "Delta"]


@pytest.fixture
def cases(make_tweets) -> pd.DataFrame:
    tweets = make_tweets(
        [
            # Multi-turn, customer confirms.
            (1, "c1", "True", ts(0), "@AppleSupport my iPhone battery drains fast", "2", None),
            (2, "AppleSupport", "False", ts(5), "@c1 Try restarting in Settings > General.", "3", 1),
            (3, "c1", "True", ts(9), "@AppleSupport that worked, thanks!", None, 2),
            # Single message to a brand, no reply.
            (10, "c2", "True", ts(1), "@Delta my flight is delayed again", None, None),
            # Agent redirects to DM.
            (20, "c3", "True", ts(2), "@Delta lost my bag", "21", None),
            (21, "Delta", "False", ts(3), "@c3 So sorry! Please DM us your bag tag. *TJ", None, 20),
        ]
    )
    conv, _ = reconstruct_conversations(tweets)
    return build_support_cases(conv, BRANDS).set_index("case_id")


def test_case_fields_and_provenance(cases: pd.DataFrame) -> None:
    c = cases.loc["case_1"]
    assert c["brand"] == "AppleSupport" and c["brand_source"] == "agent_reply"
    assert c["conversation_id"] == "conv_1"
    assert list(c["customer_messages"]) == [
        "@AppleSupport my iPhone battery drains fast",
        "@AppleSupport that worked, thanks!",
    ]
    assert list(c["agent_messages"]) == ["@c1 Try restarting in Settings > General."]
    assert list(c["tweet_ids"]) == [1, 2, 3]
    assert c["turn_count"] == 3 and c["is_multi_turn"] and c["is_reconstructable"]
    assert c["first_timestamp"] == pd.Timestamp("2017-10-31 10:00", tz="UTC")
    assert c["last_timestamp"] == pd.Timestamp("2017-10-31 10:09", tz="UTC")
    assert c["first_response_minutes"] == 5.0
    turn = c["full_turns"][1]
    assert turn["tweet_id"] == 2 and turn["role"] == "agent" and turn["parent_tweet_id"] == 1
    assert turn["created_at_raw"] == ts(5) and turn["source_row"] == 1
    assert c["resolved"] and c["resolution_type"] == "customer_confirmed"


def test_single_message_case(cases: pd.DataFrame) -> None:
    c = cases.loc["case_10"]
    assert c["brand"] == "Delta" and c["brand_source"] == "mention"
    assert c["turn_count"] == 1 and list(c["agent_messages"]) == []
    assert not c["is_reconstructable"] and not c["is_multi_turn"]
    assert not c["resolved"] and c["resolution_type"] == "no_response"
    assert c["first_response_minutes"] is None or pd.isna(c["first_response_minutes"])


def test_dm_redirect_case(cases: pd.DataFrame) -> None:
    c = cases.loc["case_20"]
    assert c["resolution_type"] == "redirected_to_dm" and not c["resolved"]
    assert "DM us your bag tag" in c["resolution_summary"]


@pytest.mark.parametrize(
    "roles,texts,expected",
    [
        (["customer"], ["help"], ResolutionType.NO_RESPONSE),
        (["customer", "agent", "customer"], ["x", "y", "thanks for nothing"], ResolutionType.CUSTOMER_FOLLOWUP_UNANSWERED),
        (["customer", "agent", "customer"], ["x", "y", "still not working"], ResolutionType.CUSTOMER_FOLLOWUP_UNANSWERED),
        (["customer", "agent"], ["x", "Glad we could help! Have a great day ^KC"], ResolutionType.AGENT_CLOSED),
        (["customer", "agent"], ["x", "Please call us at 1-800-555"], ResolutionType.REDIRECTED_TO_CHANNEL),
        (["customer", "agent"], ["x", "Check here: https://t.co/abc"], ResolutionType.REDIRECTED_TO_CHANNEL),
        (["customer", "agent"], ["x", "Your refund posts in 3-5 days."], ResolutionType.AGENT_ANSWERED_UNCONFIRMED),
        (
            ["customer", "agent"],
            ["x", "We'd be happy to help. What seems to be going on?"],
            ResolutionType.AGENT_AWAITING_CUSTOMER,
        ),
    ],
)
def test_resolution_rules(roles: list[str], texts: list[str], expected: ResolutionType) -> None:
    assert classify_resolution(roles, texts).resolution_type == expected


def test_multi_party_conversation_is_flagged(make_tweets) -> None:
    tweets = make_tweets(
        [
            (1, "Delta", "False", ts(0), "Flights resume at 5pm", "2,3", None),
            (2, "c1", "True", ts(1), "@Delta mine is still delayed", None, 1),
            (3, "c2", "True", ts(2), "@Delta same here", None, 1),
        ]
    )
    conv, _ = reconstruct_conversations(tweets)
    c = build_support_cases(conv, BRANDS).iloc[0]
    assert c["customer_count"] == 2 and c["is_multi_party"]
    assert not c["starts_with_customer"]


def test_write_support_cases_roundtrip(make_tweets, tmp_path: Path) -> None:
    tweets = make_tweets(
        [
            (1, "c1", "True", ts(0), "@Delta q", "2", None),
            (2, "Delta", "False", ts(1), "@c1 a", None, 1),
            (3, "c2", "True", ts(2), "@AppleSupport q", None, None),
        ]
    )
    conv, _ = reconstruct_conversations(tweets)
    out = tmp_path / "cases.parquet"
    jsonl = tmp_path / "cases.jsonl"
    stats = write_support_cases(conv, BRANDS, out, jsonl_path=jsonl, brands={"Delta"}, chunk_size=1)
    written = pd.read_parquet(out)
    assert stats.cases == 1 and written["case_id"].tolist() == ["case_1"]
    assert len(jsonl.read_text(encoding="utf-8").splitlines()) == 1
