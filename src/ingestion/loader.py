"""Load, validate, and clean the raw tweet table.

Provenance rules:
  * the raw `text` column is never modified;
  * raw `created_at` and `response_tweet_id` strings are kept alongside parsed versions;
  * `source_row` records the 0-based row position in the input file;
  * every dropped row is counted in `CleaningReport`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ingestion import schema as S

logger = logging.getLogger(__name__)

SOURCE_ROW = "source_row"
CREATED_AT_RAW = "created_at_raw"
RESPONSE_TWEET_ID_RAW = "response_tweet_id_raw"
RESPONSE_TWEET_IDS = "response_tweet_ids"


@dataclass
class CleaningReport:
    input_rows: int = 0
    dropped_invalid_tweet_id: int = 0
    dropped_invalid_inbound: int = 0
    dropped_exact_duplicates: int = 0
    dropped_conflicting_duplicates: int = 0
    unparseable_created_at: int = 0
    unparseable_in_response_to: int = 0
    unparseable_response_ids: int = 0
    output_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class DatasetSummary:
    row_count: int
    missing_values: dict[str, int]
    unique_authors: int
    unique_brands: int
    inbound_count: int
    outbound_count: int
    brands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def read_raw(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV (optionally gzipped) or Parquet file with every column as string.

    Reading as strings keeps ids exact (no float coercion of nullable id columns) and leaves
    type decisions to `clean_tweets`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffixes = "".join(path.suffixes).lower()
    logger.info("Reading %s%s", path, f" (first {nrows:,} rows)" if nrows else "")
    if suffixes.endswith(".parquet"):
        df = pd.read_parquet(path)
        if nrows is not None:
            df = df.head(nrows)
        df = df.astype("string").astype(object).where(df.notna(), None)
    elif suffixes.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            nrows=nrows,
        )
    else:
        raise ValueError(f"Unsupported input format {path.suffix!r}; expected .csv, .csv.gz or .parquet")

    df.insert(0, SOURCE_ROW, range(len(df)))
    logger.info("Read %d rows x %d columns", len(df), df.shape[1] - 1)
    return df


_MONTHS = {m: f"{i:02d}" for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1
)}


def parse_created_at(values: pd.Series) -> pd.Series:
    """Parse Twitter timestamps ("Tue Oct 31 22:10:47 +0000 2017") to UTC.

    Values matching the observed layout are converted via string slicing (~10x faster than
    strptime with %z on 2.8M rows); anything else falls back to strptime. Unparseable
    values become NaT.
    """
    s = values.astype(object).where(values.notna(), "")
    fast = s.astype(str).str.fullmatch(S.CREATED_AT_PATTERN) & s.astype(str).str.slice(4, 7).isin(list(_MONTHS))
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")

    if fast.any():
        arr = s[fast].tolist()
        naive = pd.to_datetime(
            [f"{x[26:30]}-{_MONTHS[x[4:7]]}-{x[8:10]} {x[11:19]}" for x in arr],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        offsets = pd.Series([x[20:25] for x in arr])
        minutes = offsets.map(
            {o: (1 if o[0] == "+" else -1) * (int(o[1:3]) * 60 + int(o[3:5])) for o in offsets.unique()}
        ).to_numpy()
        utc = naive - pd.to_timedelta(minutes, unit="m")
        out.loc[fast] = utc.tz_localize("UTC")

    slow = ~fast & values.notna()
    if slow.any():
        out.loc[slow] = pd.to_datetime(values[slow], format=S.CREATED_AT_FORMAT, utc=True, errors="coerce")
    return out


def _parse_response_ids(raw: object) -> list[int] | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or raw is pd.NA:
        return []
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            return None
    return out


def deduplicate_tweets(df: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    """Drop duplicate `tweet_id`s, keeping the first occurrence in file order.

    Exact duplicates (identical content) and conflicting duplicates (same id, different
    content) are counted separately; conflicting ones are logged as warnings.
    """
    dup_mask = df.duplicated(S.TWEET_ID, keep="first")
    if not dup_mask.any():
        return df

    content_cols = [S.TWEET_ID, S.AUTHOR_ID, S.INBOUND, S.TEXT, S.CREATED_AT]
    content_cols = [c for c in content_cols if c in df.columns]
    exact_dup_mask = df.astype({c: str for c in content_cols}).duplicated(content_cols, keep="first")
    n_exact = int(exact_dup_mask.sum())
    n_conflicting = int(dup_mask.sum()) - n_exact
    if report is not None:
        report.dropped_exact_duplicates += n_exact
        report.dropped_conflicting_duplicates += n_conflicting
    if n_conflicting:
        logger.warning(
            "%d tweet_ids appear more than once with different content; keeping first occurrence",
            n_conflicting,
        )
    logger.info("Dropped %d duplicate tweet rows (%d exact)", int(dup_mask.sum()), n_exact)
    return df.loc[~dup_mask].copy()


def clean_tweets(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate and type-convert a raw frame produced by `read_raw`.

    Output columns:
        source_row, tweet_id (int64), author_id (str), inbound (bool),
        created_at (datetime64[ns, UTC]), created_at_raw (str), text (raw str),
        in_response_to_tweet_id (Int64), response_tweet_id_raw (str|None),
        response_tweet_ids (list[int])
    """
    S.validate_schema(raw)
    report = CleaningReport(input_rows=len(raw))
    df = raw.copy()
    if SOURCE_ROW not in df.columns:
        df.insert(0, SOURCE_ROW, range(len(df)))

    tweet_id = pd.to_numeric(df[S.TWEET_ID].astype("string").str.strip(), errors="coerce")
    bad_id = tweet_id.isna() | (tweet_id % 1 != 0)
    report.dropped_invalid_tweet_id = int(bad_id.sum())
    if report.dropped_invalid_tweet_id:
        logger.warning("Dropping %d rows with invalid tweet_id", report.dropped_invalid_tweet_id)
    df = df.loc[~bad_id].copy()
    df[S.TWEET_ID] = tweet_id.loc[~bad_id].astype("int64")

    inbound = df[S.INBOUND].astype("string").str.strip().str.lower().map(S.INBOUND_VALUES)
    bad_inbound = inbound.isna()
    report.dropped_invalid_inbound = int(bad_inbound.sum())
    if report.dropped_invalid_inbound:
        logger.warning("Dropping %d rows with invalid inbound flag", report.dropped_invalid_inbound)
    df = df.loc[~bad_inbound].copy()
    df[S.INBOUND] = inbound.loc[~bad_inbound].astype(bool)

    df = deduplicate_tweets(df, report)

    df[CREATED_AT_RAW] = df[S.CREATED_AT].astype(object)
    df[S.CREATED_AT] = parse_created_at(df[S.CREATED_AT])
    report.unparseable_created_at = int(df[S.CREATED_AT].isna().sum())
    if report.unparseable_created_at:
        logger.warning(
            "%d rows have unparseable created_at; they are kept and ordered by tweet_id",
            report.unparseable_created_at,
        )

    irt_raw = df[S.IN_RESPONSE_TO_TWEET_ID]
    irt = pd.to_numeric(irt_raw, errors="coerce")
    report.unparseable_in_response_to = int((irt_raw.notna() & irt.isna()).sum())
    df[S.IN_RESPONSE_TO_TWEET_ID] = irt.round().astype("Int64")

    df[RESPONSE_TWEET_ID_RAW] = df[S.RESPONSE_TWEET_ID].astype(object).where(df[S.RESPONSE_TWEET_ID].notna(), None)
    parsed = [_parse_response_ids(v) for v in df[RESPONSE_TWEET_ID_RAW].tolist()]
    report.unparseable_response_ids = sum(p is None for p in parsed)
    df[RESPONSE_TWEET_IDS] = [p if p is not None else [] for p in parsed]

    df[S.AUTHOR_ID] = df[S.AUTHOR_ID].astype(str)
    df[S.TEXT] = df[S.TEXT].astype(object).where(df[S.TEXT].notna(), "")

    columns = [
        SOURCE_ROW,
        S.TWEET_ID,
        S.AUTHOR_ID,
        S.INBOUND,
        S.CREATED_AT,
        CREATED_AT_RAW,
        S.TEXT,
        S.IN_RESPONSE_TO_TWEET_ID,
        RESPONSE_TWEET_ID_RAW,
        RESPONSE_TWEET_IDS,
    ]
    df = df[columns].reset_index(drop=True)
    report.output_rows = len(df)
    logger.info("Cleaning: %s", report.to_dict())
    return df, report


def known_brands(tweets: pd.DataFrame) -> list[str]:
    """Brands are the authors of outbound (inbound == False) tweets."""
    return sorted(tweets.loc[~tweets[S.INBOUND].astype(bool), S.AUTHOR_ID].unique().tolist())


def summarize_dataset(raw: pd.DataFrame, cleaned: pd.DataFrame) -> DatasetSummary:
    """Missing values are reported on the raw frame; counts on the cleaned one."""
    missing = {c: int(raw[c].isna().sum()) for c in S.REQUIRED_COLUMNS if c in raw.columns}
    brands = known_brands(cleaned)
    n_inbound = int(cleaned[S.INBOUND].sum())
    return DatasetSummary(
        row_count=len(cleaned),
        missing_values=missing,
        unique_authors=int(cleaned[S.AUTHOR_ID].nunique()),
        unique_brands=len(brands),
        inbound_count=n_inbound,
        outbound_count=len(cleaned) - n_inbound,
        brands=brands,
    )


def log_summary(summary: DatasetSummary) -> None:
    logger.info("Rows: %s", f"{summary.row_count:,}")
    logger.info("Missing values per column: %s", summary.missing_values)
    logger.info("Unique authors: %s | unique brands: %d", f"{summary.unique_authors:,}", summary.unique_brands)
    logger.info(
        "Inbound (customer): %s | outbound (brand): %s",
        f"{summary.inbound_count:,}",
        f"{summary.outbound_count:,}",
    )


def load_tweets(path: str | Path, nrows: int | None = None) -> tuple[pd.DataFrame, DatasetSummary, CleaningReport]:
    """Convenience: read + validate + clean + summarize."""
    raw = read_raw(path, nrows=nrows)
    cleaned, report = clean_tweets(raw)
    summary = summarize_dataset(raw, cleaned)
    log_summary(summary)
    return cleaned, summary, report
