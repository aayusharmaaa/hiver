"""Turn reconstructed conversations into normalized support cases.

One case per conversation. Every turn keeps its raw text, tweet id, author id, raw
timestamp string and original file row, so any derived field can be traced back to the
source CSV.

`is_multi_party` marks conversations with more than one customer or brand (e.g. many users
replying under one brand tweet during an outage). They are kept, but are not one-customer
support cases and should usually be excluded or split downstream.

Brand attribution:
  * `agent_reply`: the outbound author with the most turns (ties -> earliest to reply);
  * `mention`: no brand replied, but a customer turn @-mentions a known brand handle;
  * `unknown`: neither.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ingestion import schema as S
from ingestion.loader import CREATED_AT_RAW, RESPONSE_TWEET_ID_RAW, SOURCE_ROW
from ingestion.reconstruction import CONVERSATION_ID, MISSING_PARENT, PARENT_TWEET_ID, ROOT_TWEET_ID
from ingestion.resolution import classify_resolution

logger = logging.getLogger(__name__)

ROLE_CUSTOMER = "customer"
ROLE_AGENT = "agent"
BRAND_FROM_REPLY = "agent_reply"
BRAND_FROM_MENTION = "mention"
BRAND_UNKNOWN = "unknown"

_MENTION = re.compile(r"@(\w+)")

TURN_TYPE = pa.struct(
    [
        ("turn_index", pa.int32()),
        ("tweet_id", pa.int64()),
        ("author_id", pa.string()),
        ("role", pa.string()),
        ("inbound", pa.bool_()),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("created_at_raw", pa.string()),
        ("text", pa.string()),
        ("in_response_to_tweet_id", pa.int64()),
        ("parent_tweet_id", pa.int64()),
        ("response_tweet_id_raw", pa.string()),
        ("source_row", pa.int64()),
    ]
)

CASE_SCHEMA = pa.schema(
    [
        ("case_id", pa.string()),
        ("conversation_id", pa.string()),
        ("brand", pa.string()),
        ("brand_source", pa.string()),
        ("brands_involved", pa.list_(pa.string())),
        ("customer_ids", pa.list_(pa.string())),
        ("customer_messages", pa.list_(pa.string())),
        ("agent_messages", pa.list_(pa.string())),
        ("first_customer_message", pa.string()),
        ("full_turns", pa.list_(TURN_TYPE)),
        ("tweet_ids", pa.list_(pa.int64())),
        ("turn_count", pa.int32()),
        ("customer_turn_count", pa.int32()),
        ("agent_turn_count", pa.int32()),
        ("speaker_switches", pa.int32()),
        ("starts_with_customer", pa.bool_()),
        ("customer_count", pa.int32()),
        ("is_multi_party", pa.bool_()),
        ("is_reconstructable", pa.bool_()),
        ("is_multi_turn", pa.bool_()),
        ("has_missing_parent", pa.bool_()),
        ("first_timestamp", pa.timestamp("us", tz="UTC")),
        ("last_timestamp", pa.timestamp("us", tz="UTC")),
        ("first_response_minutes", pa.float64()),
        ("resolved", pa.bool_()),
        ("resolution_type", pa.string()),
        ("resolution_summary", pa.string()),
    ]
)


@dataclass
class CaseBuildStats:
    cases: int = 0
    reconstructable: int = 0
    multi_turn: int = 0
    brand_sources: dict[str, int] = field(default_factory=dict)
    resolution_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _attribute_brand(
    roles: list[str], authors: list[str], texts: list[str], known_brands_lower: dict[str, str]
) -> tuple[str | None, str, list[str]]:
    agent_authors = [a for r, a in zip(roles, authors) if r == ROLE_AGENT]
    if agent_authors:
        counts = Counter(agent_authors)
        top = max(counts.values())
        brand = next(a for a in agent_authors if counts[a] == top)
        return brand, BRAND_FROM_REPLY, list(dict.fromkeys(agent_authors))
    for text in texts:
        for handle in _MENTION.findall(text):
            brand = known_brands_lower.get(handle.lower())
            if brand:
                return brand, BRAND_FROM_MENTION, []
    return None, BRAND_UNKNOWN, []


def _conversation_slices(conv_ids: np.ndarray) -> Iterator[tuple[int, int]]:
    if len(conv_ids) == 0:
        return
    change = np.flatnonzero(conv_ids[1:] != conv_ids[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(conv_ids)]])
    yield from zip(starts.tolist(), ends.tolist())


def iter_case_records(
    conversations: pd.DataFrame, known_brands: list[str], multi_turn_min_turns: int = 3
) -> Iterator[dict]:
    """Yield one case dict per conversation.

    `conversations` must be the output of `reconstruct_conversations` (sorted by
    conversation, then turn).
    """
    brands_lower = {b.lower(): b for b in known_brands}
    cols = {
        "conv": conversations[CONVERSATION_ID].to_numpy(),
        "root": conversations[ROOT_TWEET_ID].to_numpy(),
        "tweet_id": conversations[S.TWEET_ID].to_numpy(),
        "author": conversations[S.AUTHOR_ID].to_numpy(),
        "inbound": conversations[S.INBOUND].to_numpy(),
        "created_at": np.array(
            [None if pd.isna(t) else t.to_pydatetime() for t in conversations[S.CREATED_AT].astype(object)],
            dtype=object,
        ),
        "created_at_raw": conversations[CREATED_AT_RAW].to_numpy(),
        "text": conversations[S.TEXT].to_numpy(),
        "irt": conversations[S.IN_RESPONSE_TO_TWEET_ID].astype("float64").to_numpy(),
        "parent": conversations[PARENT_TWEET_ID].astype("float64").to_numpy(),
        "resp_raw": conversations[RESPONSE_TWEET_ID_RAW].to_numpy(),
        "source_row": conversations[SOURCE_ROW].to_numpy(),
        "missing_parent": conversations[MISSING_PARENT].to_numpy(),
    }

    for start, end in _conversation_slices(cols["conv"]):
        sl = slice(start, end)
        inbound = cols["inbound"][sl].tolist()
        roles = [ROLE_CUSTOMER if ib else ROLE_AGENT for ib in inbound]
        authors = cols["author"][sl].tolist()
        texts = cols["text"][sl].tolist()
        tweet_ids = cols["tweet_id"][sl].tolist()
        created = cols["created_at"][sl].tolist()

        brand, brand_source, brands_involved = _attribute_brand(roles, authors, texts, brands_lower)
        resolution = classify_resolution(roles, texts)

        turns = []
        for i in range(end - start):
            irt = cols["irt"][start + i]
            parent = cols["parent"][start + i]
            turns.append(
                {
                    "turn_index": i,
                    "tweet_id": int(tweet_ids[i]),
                    "author_id": authors[i],
                    "role": roles[i],
                    "inbound": bool(inbound[i]),
                    "created_at": created[i],
                    "created_at_raw": cols["created_at_raw"][start + i],
                    "text": texts[i],
                    "in_response_to_tweet_id": None if np.isnan(irt) else int(irt),
                    "parent_tweet_id": None if np.isnan(parent) else int(parent),
                    "response_tweet_id_raw": cols["resp_raw"][start + i],
                    "source_row": int(cols["source_row"][start + i]),
                }
            )

        customer_msgs = [t for t, r in zip(texts, roles) if r == ROLE_CUSTOMER]
        agent_msgs = [t for t, r in zip(texts, roles) if r == ROLE_AGENT]
        n_c, n_a = len(customer_msgs), len(agent_msgs)
        reconstructable = n_c > 0 and n_a > 0
        valid_ts = [t for t in created if t is not None]

        first_response_minutes = None
        first_c = next((i for i, r in enumerate(roles) if r == ROLE_CUSTOMER), None)
        if first_c is not None:
            first_a = next((i for i in range(first_c + 1, len(roles)) if roles[i] == ROLE_AGENT), None)
            if first_a is not None and created[first_c] is not None and created[first_a] is not None:
                first_response_minutes = (created[first_a] - created[first_c]).total_seconds() / 60.0

        root = int(cols["root"][start])
        customer_ids = list(dict.fromkeys(a for a, r in zip(authors, roles) if r == ROLE_CUSTOMER))
        yield {
            "case_id": f"case_{root}",
            "conversation_id": cols["conv"][start],
            "brand": brand,
            "brand_source": brand_source,
            "brands_involved": brands_involved,
            "customer_ids": customer_ids,
            "customer_messages": customer_msgs,
            "agent_messages": agent_msgs,
            "first_customer_message": customer_msgs[0] if customer_msgs else None,
            "full_turns": turns,
            "tweet_ids": [int(t) for t in tweet_ids],
            "turn_count": len(roles),
            "customer_turn_count": n_c,
            "agent_turn_count": n_a,
            "speaker_switches": sum(roles[i] != roles[i - 1] for i in range(1, len(roles))),
            "starts_with_customer": roles[0] == ROLE_CUSTOMER,
            "customer_count": len(customer_ids),
            "is_multi_party": len(customer_ids) > 1 or len(brands_involved) > 1,
            "is_reconstructable": reconstructable,
            "is_multi_turn": reconstructable and len(roles) >= multi_turn_min_turns,
            "has_missing_parent": bool(cols["missing_parent"][sl].any()),
            "first_timestamp": min(valid_ts) if valid_ts else None,
            "last_timestamp": max(valid_ts) if valid_ts else None,
            "first_response_minutes": first_response_minutes,
            "resolved": resolution.resolved,
            "resolution_type": resolution.resolution_type.value,
            "resolution_summary": resolution.resolution_summary,
        }


def build_support_cases(conversations: pd.DataFrame, known_brands: list[str]) -> pd.DataFrame:
    """In-memory variant, convenient for tests and small subsets."""
    records = list(iter_case_records(conversations, known_brands))
    return pa.Table.from_pylist(records, schema=CASE_SCHEMA).to_pandas() if records else CASE_SCHEMA.empty_table().to_pandas()


def write_support_cases(
    conversations: pd.DataFrame,
    known_brands: list[str],
    parquet_path: str | Path,
    jsonl_path: str | Path | None = None,
    brands: set[str] | None = None,
    chunk_size: int = 50_000,
) -> CaseBuildStats:
    """Stream cases to Parquet (and optionally JSONL) in chunks to bound memory.

    If `brands` is given, only cases attributed to those brands are written.
    """
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    stats = CaseBuildStats()
    brand_sources: Counter[str] = Counter()
    resolution_types: Counter[str] = Counter()

    writer = pq.ParquetWriter(parquet_path, CASE_SCHEMA, compression="zstd")
    jsonl = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None
    buffer: list[dict] = []

    def flush() -> None:
        if not buffer:
            return
        writer.write_table(pa.Table.from_pylist(buffer, schema=CASE_SCHEMA))
        if jsonl:
            for rec in buffer:
                jsonl.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        buffer.clear()

    try:
        for rec in iter_case_records(conversations, known_brands):
            if brands is not None and rec["brand"] not in brands:
                continue
            buffer.append(rec)
            stats.cases += 1
            stats.reconstructable += rec["is_reconstructable"]
            stats.multi_turn += rec["is_multi_turn"]
            brand_sources[rec["brand_source"]] += 1
            resolution_types[rec["resolution_type"]] += 1
            if len(buffer) >= chunk_size:
                flush()
                logger.info("Wrote %s cases so far", f"{stats.cases:,}")
        flush()
    finally:
        writer.close()
        if jsonl:
            jsonl.close()

    stats.brand_sources = dict(brand_sources)
    stats.resolution_types = dict(resolution_types.most_common())
    logger.info("Case build: %s", stats.to_dict())
    return stats
