"""Build normalized support cases (one per customer episode) with full provenance."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ingestion import schema as S
from ingestion.episodes import CASE_ID, DEPTH, OWNER_SOURCE
from ingestion.loader import CREATED_AT_RAW, RESPONSE_TWEET_ID_RAW, SOURCE_ROW
from ingestion.reconstruction import CONVERSATION_ID, MISSING_PARENT, PARENT_SOURCE, PARENT_TWEET_ID
from ingestion.resolution import classify_resolution
from ingestion.resolution_signals import derive_case_resolution
from ingestion.roles import BRAND_AGENT, CUSTOMER, OTHER_AGENT, ROLE

logger = logging.getLogger(__name__)

TURN_STRUCT = pa.struct(
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
        ("parent_source", pa.string()),
        ("response_tweet_id_raw", pa.string()),
        ("owner_source", pa.string()),
        ("source_row", pa.int64()),
    ]
)
EVIDENCE_STRUCT = pa.struct([("tweet_id", pa.int64()), ("signal", pa.string()), ("snippet", pa.string())])

CASE_SCHEMA = pa.schema(
    [
        ("case_id", pa.string()),
        ("brand", pa.string()),
        ("conversation_id", pa.string()),
        ("customer_id", pa.string()),
        ("customer_messages", pa.list_(pa.string())),
        ("agent_messages", pa.list_(pa.string())),
        ("other_agent_messages", pa.list_(pa.string())),
        ("opening_message", pa.string()),
        ("full_turns", pa.list_(TURN_STRUCT)),
        ("turn_count", pa.int32()),
        ("customer_turn_count", pa.int32()),
        ("agent_turn_count", pa.int32()),
        ("other_agent_turn_count", pa.int32()),
        ("first_timestamp", pa.timestamp("us", tz="UTC")),
        ("last_timestamp", pa.timestamp("us", tz="UTC")),
        ("duration_minutes", pa.float64()),
        ("first_response_minutes", pa.float64()),
        ("max_gap_hours", pa.float64()),
        ("resolved", pa.bool_()),
        ("resolution_type", pa.string()),
        ("resolution_outcome", pa.string()),
        ("resolution_signals", pa.list_(pa.string())),
        ("resolution_evidence", pa.list_(EVIDENCE_STRUCT)),
        ("resolution_summary", pa.string()),
        ("dm_redirect", pa.bool_()),
        ("source_tweet_ids", pa.list_(pa.int64())),
        ("context_tweet_ids", pa.list_(pa.int64())),
        ("is_continuation", pa.bool_()),
        ("continues_case_id", pa.string()),
        ("thread_customer_count", pa.int32()),
        ("thread_episode_count", pa.int32()),
        ("has_brand_reply", pa.bool_()),
        ("starts_with_customer", pa.bool_()),
        ("has_missing_parent", pa.bool_()),
        ("related_case_ids", pa.list_(pa.string())),
    ]
)


def _none_if_na(v: object):
    return None if v is None or (not isinstance(v, (list, str)) and pd.isna(v)) else v


def build_cases(
    tweets: pd.DataFrame,
    episodes: pd.DataFrame,
    brand: str,
    other_operator_ids: frozenset[str] = frozenset(),
    related_window_hours: float = 24.0,
) -> list[dict]:
    """`tweets`: output of `segment_conversations`; `episodes`: its episode table."""
    ep_meta = episodes.set_index("case_id").to_dict("index")
    attributed = tweets.loc[tweets[CASE_ID].notna()].copy()
    attributed["_missing_time"] = attributed[S.CREATED_AT].isna()
    attributed = attributed.sort_values(
        [CASE_ID, "_missing_time", S.CREATED_AT, DEPTH, S.TWEET_ID], kind="mergesort"
    )
    created_py = attributed[S.CREATED_AT].astype(object).where(attributed[S.CREATED_AT].notna(), None)
    attributed["_created_py"] = [None if v is None else v.to_pydatetime() for v in created_py]

    records: list[dict] = []
    for case_id, g in attributed.groupby(CASE_ID, sort=False):
        meta = ep_meta[case_id]
        roles = g[ROLE].tolist()
        texts = g[S.TEXT].tolist()
        tids = [int(t) for t in g[S.TWEET_ID].tolist()]
        times = g["_created_py"].tolist()
        irt = g[S.IN_RESPONSE_TO_TWEET_ID].tolist()
        parent = g[PARENT_TWEET_ID].tolist()

        turns = [
            {
                "turn_index": i,
                "tweet_id": tids[i],
                "author_id": str(g[S.AUTHOR_ID].iloc[i]),
                "role": roles[i],
                "inbound": bool(g[S.INBOUND].iloc[i]),
                "created_at": times[i],
                "created_at_raw": _none_if_na(g[CREATED_AT_RAW].iloc[i]),
                "text": texts[i],
                "in_response_to_tweet_id": None if pd.isna(irt[i]) else int(irt[i]),
                "parent_tweet_id": None if pd.isna(parent[i]) else int(parent[i]),
                "parent_source": _none_if_na(g[PARENT_SOURCE].iloc[i]),
                "response_tweet_id_raw": _none_if_na(g[RESPONSE_TWEET_ID_RAW].iloc[i]),
                "owner_source": _none_if_na(g[OWNER_SOURCE].iloc[i]),
                "source_row": int(g[SOURCE_ROW].iloc[i]),
            }
            for i in range(len(g))
        ]
        customer_msgs = [t for t, r in zip(texts, roles) if r == CUSTOMER]
        agent_msgs = [t for t, r in zip(texts, roles) if r == BRAND_AGENT]
        other_msgs = [t for t, r in zip(texts, roles) if r == OTHER_AGENT]
        valid = [t for t in times if t is not None]

        first_response = None
        first_c = next((i for i, r in enumerate(roles) if r == CUSTOMER), None)
        if first_c is not None and times[first_c] is not None:
            nxt = next((i for i in range(first_c + 1, len(roles)) if roles[i] == BRAND_AGENT and times[i] is not None), None)
            if nxt is not None:
                first_response = (times[nxt] - times[first_c]).total_seconds() / 60.0
        gaps = [(b - a).total_seconds() / 3600 for a, b in zip(valid, valid[1:])]

        outcome = classify_resolution(
            [CUSTOMER if r == CUSTOMER else "agent" for r in roles], texts
        ).resolution_type.value
        res = derive_case_resolution(
            [{"tweet_id": tids[i], "role": roles[i], "text": texts[i]} for i in range(len(g))],
            outcome,
            other_operator_ids,
        )

        records.append(
            {
                "case_id": case_id,
                "brand": brand,
                "conversation_id": meta[CONVERSATION_ID],
                "customer_id": meta["customer_id"],
                "customer_messages": customer_msgs,
                "agent_messages": agent_msgs,
                "other_agent_messages": other_msgs,
                "opening_message": customer_msgs[0] if customer_msgs else None,
                "full_turns": turns,
                "turn_count": len(g),
                "customer_turn_count": len(customer_msgs),
                "agent_turn_count": len(agent_msgs),
                "other_agent_turn_count": len(other_msgs),
                "first_timestamp": min(valid) if valid else None,
                "last_timestamp": max(valid) if valid else None,
                "duration_minutes": (max(valid) - min(valid)).total_seconds() / 60 if valid else None,
                "first_response_minutes": first_response,
                "max_gap_hours": max(gaps) if gaps else None,
                "resolved": res.resolved,
                "resolution_type": res.resolution_type,
                "resolution_outcome": res.outcome,
                "resolution_signals": list(res.signals),
                "resolution_evidence": list(res.evidence),
                "resolution_summary": res.summary,
                "dm_redirect": res.dm_redirect,
                "source_tweet_ids": tids,
                "context_tweet_ids": [int(x) for x in meta["context_tweet_ids"]],
                "is_continuation": bool(meta["is_continuation"]),
                "continues_case_id": _none_if_na(meta["continues_case_id"]),
                "thread_customer_count": int(meta["thread_customer_count"]),
                "thread_episode_count": int(meta["thread_episode_count"]),
                "has_brand_reply": len(agent_msgs) > 0,
                "starts_with_customer": roles[0] == CUSTOMER,
                "has_missing_parent": bool(g[MISSING_PARENT].any()),
                "related_case_ids": [],
            }
        )
    attach_related_cases(records, related_window_hours)
    logger.info("Built %d cases", len(records))
    return records


def attach_related_cases(records: list[dict], window_hours: float) -> None:
    """Link (never merge) cases of the same customer that start/end within `window_hours`."""
    by_customer: dict[str, list[dict]] = {}
    for rec in records:
        if rec["first_timestamp"] is not None:
            by_customer.setdefault(rec["customer_id"], []).append(rec)
    window = pd.Timedelta(hours=window_hours).to_pytimedelta()
    for recs in by_customer.values():
        if len(recs) < 2:
            continue
        for a in recs:
            a["related_case_ids"] = sorted(
                b["case_id"]
                for b in recs
                if b is not a
                and b["first_timestamp"] <= a["last_timestamp"] + window
                and a["first_timestamp"] <= b["last_timestamp"] + window
            )


def write_cases(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=CASE_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    logger.info("Wrote %s (%d cases)", path, len(records))


def write_jsonl(records: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
