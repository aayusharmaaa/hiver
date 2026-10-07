"""Schema of the Customer Support on Twitter dataset (Kaggle: thoughtvector/customer-support-on-twitter).

The column list below was taken from the observed `twcs.csv` header, not from documentation:

    tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

Observed properties (full file, 2,811,774 rows):
  * only `response_tweet_id` and `in_response_to_tweet_id` contain nulls;
  * `response_tweet_id` may hold a comma-separated list of tweet ids;
  * `inbound` is the string "True"/"False" (True = customer, False = company);
  * `created_at` looks like "Tue Oct 31 22:10:47 +0000 2017".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

TWEET_ID = "tweet_id"
AUTHOR_ID = "author_id"
INBOUND = "inbound"
CREATED_AT = "created_at"
TEXT = "text"
RESPONSE_TWEET_ID = "response_tweet_id"
IN_RESPONSE_TO_TWEET_ID = "in_response_to_tweet_id"

REQUIRED_COLUMNS: tuple[str, ...] = (
    TWEET_ID,
    AUTHOR_ID,
    INBOUND,
    CREATED_AT,
    TEXT,
    RESPONSE_TWEET_ID,
    IN_RESPONSE_TO_TWEET_ID,
)
NULLABLE_COLUMNS: frozenset[str] = frozenset({RESPONSE_TWEET_ID, IN_RESPONSE_TO_TWEET_ID})
CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"
CREATED_AT_PATTERN = r"[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} [+-]\d{4} \d{4}"
INBOUND_VALUES: dict[str, bool] = {"true": True, "false": False}
# Columns added by the loader itself; not reported as unexpected.
INTERNAL_COLUMNS: frozenset[str] = frozenset({"source_row"})


class SchemaError(ValueError):
    """Raised when the input does not have the columns the pipeline relies on."""


@dataclass
class SchemaReport:
    """Non-fatal findings from schema validation."""

    extra_columns: list[str] = field(default_factory=list)
    unexpected_nulls: dict[str, int] = field(default_factory=dict)
    invalid_inbound_values: int = 0
    non_numeric_tweet_ids: int = 0

    @property
    def is_clean(self) -> bool:
        return not (self.unexpected_nulls or self.invalid_inbound_values or self.non_numeric_tweet_ids)


def validate_schema(df: pd.DataFrame) -> SchemaReport:
    """Validate a raw (string-typed) frame.

    Missing required columns are fatal. Value-level problems are reported and logged, and
    are handled later by the cleaning step (rows are dropped with a count, never silently).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Input is missing required columns {missing}. Found columns: {list(df.columns)}"
        )

    report = SchemaReport(
        extra_columns=[c for c in df.columns if c not in REQUIRED_COLUMNS and c not in INTERNAL_COLUMNS]
    )
    if report.extra_columns:
        logger.warning("Ignoring unexpected extra columns: %s", report.extra_columns)

    for col in REQUIRED_COLUMNS:
        if col in NULLABLE_COLUMNS:
            continue
        n_null = int(df[col].isna().sum())
        if n_null:
            report.unexpected_nulls[col] = n_null
            logger.warning("Column %r has %d unexpected nulls", col, n_null)

    inbound_norm = df[INBOUND].astype(str).str.strip().str.lower()
    report.invalid_inbound_values = int((~inbound_norm.isin(list(INBOUND_VALUES))).sum())
    if report.invalid_inbound_values:
        logger.warning("%d rows have an `inbound` value other than True/False", report.invalid_inbound_values)

    ids = pd.to_numeric(df[TWEET_ID], errors="coerce")
    report.non_numeric_tweet_ids = int((ids.isna() | (ids % 1 != 0)).sum())
    if report.non_numeric_tweet_ids:
        logger.warning("%d rows have a non-numeric tweet_id", report.non_numeric_tweet_ids)

    return report
