from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import COLUMNS, raw_frame, ts
from ingestion.loader import clean_tweets, parse_created_at, read_raw, summarize_dataset
from ingestion.schema import SchemaError, validate_schema


def test_missing_required_column_raises() -> None:
    raw = raw_frame([(1, "Brand", "False", ts(0), "hi", None, None)]).drop(columns=["inbound"])
    with pytest.raises(SchemaError, match="inbound"):
        validate_schema(raw)


def test_clean_preserves_raw_text_and_provenance() -> None:
    text = "@Brand   my  order #123 never came 😡 https://t.co/x"
    raw = raw_frame([(5, "c1", "True", ts(1), text, "6,7", None)])
    cleaned, report = clean_tweets(raw)
    row = cleaned.iloc[0]
    assert row["text"] == text
    assert row["created_at_raw"] == ts(1)
    assert row["response_tweet_id_raw"] == "6,7"
    assert row["response_tweet_ids"] == [6, 7]
    assert row["source_row"] == 0
    assert row["created_at"] == pd.Timestamp("2017-10-31 10:01:00", tz="UTC")
    assert report.output_rows == 1


def test_parse_created_at_handles_offsets_and_fallback() -> None:
    values = pd.Series(
        ["Tue Oct 31 22:10:47 +0000 2017", "Tue Oct 31 22:10:47 +0530 2017", "2017-10-31 22:10:47+00:00", None, "junk"]
    )
    parsed = parse_created_at(values)
    assert parsed.iloc[0] == pd.Timestamp("2017-10-31 22:10:47", tz="UTC")
    assert parsed.iloc[1] == pd.Timestamp("2017-10-31 16:40:47", tz="UTC")
    assert parsed.iloc[2:].isna().all()


def test_invalid_rows_are_dropped_and_counted() -> None:
    raw = raw_frame(
        [
            (1, "c1", "True", ts(1), "ok", None, None),
            ("abc", "c1", "True", ts(2), "bad id", None, None),
            (3, "c1", "maybe", ts(3), "bad inbound", None, None),
            (4, "c1", "True", "not a date", "bad date kept", None, None),
        ]
    )
    cleaned, report = clean_tweets(raw)
    assert cleaned["tweet_id"].tolist() == [1, 4]
    assert report.dropped_invalid_tweet_id == 1
    assert report.dropped_invalid_inbound == 1
    assert report.unparseable_created_at == 1


def test_duplicate_tweets_keep_first_and_count_conflicts() -> None:
    raw = raw_frame(
        [
            (1, "c1", "True", ts(1), "original", None, None),
            (1, "c1", "True", ts(1), "original", None, None),
            (1, "c1", "True", ts(1), "edited copy", None, None),
            (2, "Brand", "False", ts(2), "reply", None, 1),
        ]
    )
    cleaned, report = clean_tweets(raw)
    assert cleaned["tweet_id"].tolist() == [1, 2]
    assert cleaned.loc[cleaned.tweet_id == 1, "text"].item() == "original"
    assert report.dropped_exact_duplicates == 1
    assert report.dropped_conflicting_duplicates == 1


def test_summary_counts(tmp_path: Path) -> None:
    raw = raw_frame(
        [
            (1, "c1", "True", ts(1), "help", "2", None),
            (2, "BrandA", "False", ts(2), "sure", None, 1),
            (3, "c2", "True", ts(3), "hey", None, None),
            (4, "BrandB", "False", ts(4), "hi", None, 3),
        ]
    )
    path = tmp_path / "tiny.csv"
    raw[COLUMNS].to_csv(path, index=False)
    reread = read_raw(path)
    cleaned, _ = clean_tweets(reread)
    summary = summarize_dataset(reread, cleaned)
    assert summary.row_count == 4
    assert summary.unique_brands == 2
    assert summary.inbound_count == 2 and summary.outbound_count == 2
    assert summary.missing_values["in_response_to_tweet_id"] == 2
    assert summary.missing_values["response_tweet_id"] == 3


def test_parquet_input_matches_csv(tmp_path: Path) -> None:
    raw = raw_frame([(1, "c1", "True", ts(1), "help", "2", None), (2, "B", "False", ts(2), "ok", None, 1)])
    csv_path, pq_path = tmp_path / "t.csv", tmp_path / "t.parquet"
    raw[COLUMNS].to_csv(csv_path, index=False)
    raw[COLUMNS].to_parquet(pq_path, index=False)
    a, _ = clean_tweets(read_raw(csv_path))
    b, _ = clean_tweets(read_raw(pq_path))
    pd.testing.assert_frame_equal(a, b)
