"""End-to-end brand extraction: cleaned tweets -> conversations -> episodes -> cases."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ingestion import schema as S
from ingestion.brand_filter import BrandFilterStats, select_brand_conversations
from ingestion.episode_cases import build_cases, write_cases
from ingestion.episodes import CASE_ID, EXCLUSION_REASON, OWNER_ID, SegmentationConfig, SegmentationStats, link_orphans, segment_conversations
from ingestion.loader import CREATED_AT_RAW, RESPONSE_TWEET_ID_RAW, RESPONSE_TWEET_IDS, SOURCE_ROW
from ingestion.reconstruction import (
    CONVERSATION_ID,
    CONVERSATION_SIZE,
    MISSING_PARENT,
    PARENT_SOURCE,
    PARENT_TWEET_ID,
    ROOT_TWEET_ID,
    TURN_INDEX,
    reconstruct_conversations,
)
from ingestion.roles import BRAND_AGENT, CUSTOMER, OTHER_AGENT, ROLE, assign_roles, detect_hidden_agents

logger = logging.getLogger(__name__)

TWEET_COLUMNS = [
    SOURCE_ROW, S.TWEET_ID, S.AUTHOR_ID, S.INBOUND, ROLE, S.CREATED_AT, CREATED_AT_RAW, S.TEXT,
    S.IN_RESPONSE_TO_TWEET_ID, RESPONSE_TWEET_ID_RAW, PARENT_TWEET_ID, PARENT_SOURCE, MISSING_PARENT,
    CONVERSATION_ID, ROOT_TWEET_ID, TURN_INDEX, CONVERSATION_SIZE, CASE_ID, OWNER_ID, "owner_source",
    EXCLUSION_REASON, "episode_seq", "reply_depth",
]


@dataclass
class BrandBuildResult:
    brand: str
    filter_stats: BrandFilterStats
    segmentation: SegmentationStats
    hidden_agents: list[str]
    cases: list[dict] = field(default_factory=list)
    tweets: pd.DataFrame | None = None
    conversations: pd.DataFrame | None = None

    def summary(self) -> dict:
        return {
            "brand": self.brand,
            "filter": self.filter_stats.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "hidden_agents": self.hidden_agents,
            "cases": len(self.cases),
        }


def build_conversation_table(tweets: pd.DataFrame, brand: str) -> pd.DataFrame:
    """One row per reconstructed conversation (thread), turns nested in chronological order."""
    rows = []
    ordered = tweets.sort_values([CONVERSATION_ID, TURN_INDEX], kind="mergesort")
    for conv_id, g in ordered.groupby(CONVERSATION_ID, sort=False):
        times = g[S.CREATED_AT].dropna()
        case_ids = [c for c in dict.fromkeys(g[CASE_ID].tolist()) if isinstance(c, str)]
        gaps = times.diff().dt.total_seconds().div(3600).dropna()
        turns = [
            {
                "turn_index": int(r[TURN_INDEX]),
                "tweet_id": int(r[S.TWEET_ID]),
                "author_id": str(r[S.AUTHOR_ID]),
                "role": r[ROLE],
                "inbound": bool(r[S.INBOUND]),
                "created_at": None if pd.isna(r[S.CREATED_AT]) else r[S.CREATED_AT].to_pydatetime(),
                "created_at_raw": r[CREATED_AT_RAW],
                "text": r[S.TEXT],
                "parent_tweet_id": None if pd.isna(r[PARENT_TWEET_ID]) else int(r[PARENT_TWEET_ID]),
                "in_response_to_tweet_id": None if pd.isna(r[S.IN_RESPONSE_TO_TWEET_ID]) else int(r[S.IN_RESPONSE_TO_TWEET_ID]),
                "response_tweet_id_raw": r[RESPONSE_TWEET_ID_RAW] if isinstance(r[RESPONSE_TWEET_ID_RAW], str) else None,
                "case_id": r[CASE_ID] if isinstance(r[CASE_ID], str) else None,
                "owner_id": r[OWNER_ID] if isinstance(r[OWNER_ID], str) else None,
                "exclusion_reason": r[EXCLUSION_REASON] if isinstance(r[EXCLUSION_REASON], str) else None,
                "source_row": int(r[SOURCE_ROW]),
            }
            for _, r in g.iterrows()
        ]
        rows.append(
            {
                "conversation_id": conv_id,
                "brand": brand,
                "root_tweet_id": int(g[ROOT_TWEET_ID].iloc[0]),
                "tweet_count": len(g),
                "customer_ids": list(dict.fromkeys(g.loc[g[ROLE] == CUSTOMER, S.AUTHOR_ID].astype(str))),
                "brand_agent_tweet_count": int((g[ROLE] == BRAND_AGENT).sum()),
                "other_agent_ids": list(dict.fromkeys(g.loc[g[ROLE] == OTHER_AGENT, S.AUTHOR_ID].astype(str))),
                "case_ids": case_ids,
                "case_count": len(case_ids),
                "excluded_tweet_count": int(g[CASE_ID].isna().sum()),
                "first_timestamp": None if times.empty else times.min().to_pydatetime(),
                "last_timestamp": None if times.empty else times.max().to_pydatetime(),
                "max_gap_hours": float(gaps.max()) if len(gaps) else None,
                "has_missing_parent": bool(g[MISSING_PARENT].any()),
                "tweet_ids": [int(t) for t in g[S.TWEET_ID]],
                "turns": turns,
            }
        )
    return pd.DataFrame(rows)


def process_tweets(
    cleaned: pd.DataFrame | None,
    brand: str,
    cfg: SegmentationConfig | None = None,
    reconstructed: pd.DataFrame | None = None,
) -> BrandBuildResult:
    """Run filter -> roles -> orphan linking -> segmentation -> cases.

    Pass `cleaned` (output of `clean_tweets`) or an already `reconstructed` frame.
    """
    cfg = cfg or SegmentationConfig()
    if reconstructed is None and cleaned is None:
        raise ValueError("Provide `cleaned` or `reconstructed` tweets")
    conversations = reconstructed if reconstructed is not None else reconstruct_conversations(cleaned)[0]
    subset, filter_stats = select_brand_conversations(conversations, brand)
    hidden = detect_hidden_agents(subset)
    subset = subset.copy()
    subset[ROLE] = assign_roles(subset, brand, hidden)
    subset, linked = link_orphans(subset, cfg)
    tweets, episodes, seg_stats = segment_conversations(subset, cfg)
    seg_stats.orphans_linked = linked

    other_ops = frozenset(hidden) | frozenset(
        tweets.loc[(tweets[ROLE] == OTHER_AGENT), S.AUTHOR_ID].astype(str)
    ) | frozenset({"VirginAtlantic", "nationalrailenq"} if brand == "VirginTrains" else set())
    cases = build_cases(tweets, episodes, brand, other_ops)
    tweets = tweets[TWEET_COLUMNS].sort_values([CONVERSATION_ID, TURN_INDEX], kind="mergesort").reset_index(drop=True)
    return BrandBuildResult(brand, filter_stats, seg_stats, sorted(hidden), cases, tweets, None)


def write_outputs(result: BrandBuildResult, out_dir: str | Path, prefix: str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "conversations": out / f"{prefix}_conversations.parquet",
        "tweets": out / f"{prefix}_tweets.parquet",
        "cases": out / f"{prefix}_cases.parquet",
    }
    conv_table = build_conversation_table(result.tweets, result.brand)
    conv_table.to_parquet(paths["conversations"], index=False, compression="zstd")
    result.tweets.to_parquet(paths["tweets"], index=False, compression="zstd")
    write_cases(result.cases, paths["cases"])
    for name, p in paths.items():
        logger.info("Wrote %s -> %s", name, p)
    return paths
