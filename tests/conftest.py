from __future__ import annotations

from typing import Callable

import pandas as pd
import pytest

from ingestion.loader import clean_tweets

COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def ts(minute: int, hour: int = 10) -> str:
    return f"Tue Oct 31 {hour:02d}:{minute:02d}:00 +0000 2017"


def raw_frame(rows: list[tuple]) -> pd.DataFrame:
    """Build a raw, string-typed frame in the observed twcs.csv layout.

    Each row: (tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id)
    with None for missing relationship ids.
    """
    df = pd.DataFrame(
        [[None if v is None else str(v) for v in row] for row in rows],
        columns=COLUMNS,
        dtype=object,
    )
    df.insert(0, "source_row", range(len(df)))
    return df


@pytest.fixture
def make_tweets() -> Callable[[list[tuple]], pd.DataFrame]:
    def _make(rows: list[tuple]) -> pd.DataFrame:
        cleaned, _ = clean_tweets(raw_frame(rows))
        return cleaned

    return _make
