"""Split reconstructed threads into support episodes (one customer, one issue window).

A reply-graph component ("conversation") can contain several unrelated episodes:
  * many customers replying under one brand tweet (outage announcements);
  * one customer returning to a thread days or weeks later;
  * (rarely) tweets with no usable reply ids.

Boundary rules, applied in order:
  1. Thread: tweets already linked by reply ids form a conversation (never merged by author).
  2. Orphans: a tweet that has no usable link is attached to a nearby tweet only if the
     brand/customer @-mention proves the pairing and the time gap is small
     (`link_orphans`). Nothing is merged on author id alone.
  3. Owner: each customer tweet belongs to its author; each agent tweet belongs to the
     customer found by (a) walking up the reply chain, (b) the @<customer_id> it mentions,
     (c) the thread's only customer. Agent tweets that match none (brand-initiated posts) are
     excluded from cases and kept as optional context.
  4. Time: within one owner, a gap greater than `gap_hours` between consecutive tweets starts
     a new episode, flagged `is_continuation` and linked to the previous one.

Within an episode, turns are ordered by (created_at, reply depth, tweet_id): the depth term
keeps a parent before its child when both share a second.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from ingestion import schema as S
from ingestion.reconstruction import (
    CONVERSATION_ID,
    CONVERSATION_SIZE,
    PARENT_SOURCE,
    PARENT_TWEET_ID,
    ROOT_TWEET_ID,
    TURN_INDEX,
)
from ingestion.roles import CUSTOMER, ROLE

logger = logging.getLogger(__name__)

CASE_ID = "case_id"
OWNER_ID = "owner_id"
OWNER_SOURCE = "owner_source"
EXCLUSION_REASON = "exclusion_reason"
EPISODE_SEQ = "episode_seq"
DEPTH = "reply_depth"

PARENT_FROM_CONTEXT = "contextual_mention"
MENTION_ID = re.compile(r"@(\d+)")
EXCLUDED_UNATTRIBUTED = "unattributed_agent_tweet"


@dataclass(frozen=True)
class SegmentationConfig:
    gap_hours: float = 24.0
    orphan_max_gap_minutes: float = 120.0


@dataclass
class SegmentationStats:
    conversations: int = 0
    orphans_linked: int = 0
    episodes: int = 0
    continuation_episodes: int = 0
    conversations_split: int = 0
    multi_customer_conversations: int = 0
    unattributed_tweets: int = 0
    owner_sources: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def link_orphans(df: pd.DataFrame, cfg: SegmentationConfig) -> tuple[pd.DataFrame, int]:
    """Attach single-tweet conversations to a neighbour proven by an @<customer_id> mention.

    Agent orphan mentioning customer X attaches to X's latest earlier tweet within the gap.
    Customer orphan by X attaches to the latest earlier agent tweet that mentions X.
    Real public data has almost no orphans (every tweet carries a reply link); this covers
    exports where reply ids were dropped.
    """
    df = df.copy()
    orphan_mask = df[CONVERSATION_SIZE] == 1
    if not orphan_mask.any():
        return df, 0

    max_gap = pd.Timedelta(minutes=cfg.orphan_max_gap_minutes)
    order = df.sort_values([S.CREATED_AT, S.TWEET_ID], na_position="last")
    by_author: dict[str, list[int]] = {}
    by_mention: dict[str, list[int]] = {}
    for pos, (author, role, text) in zip(order.index, zip(order[S.AUTHOR_ID], order[ROLE], order[S.TEXT])):
        if role == CUSTOMER:
            by_author.setdefault(str(author), []).append(pos)
        else:
            for m in set(MENTION_ID.findall(str(text))):
                by_mention.setdefault(m, []).append(pos)

    def latest_before(candidates: list[int], row_time: pd.Timestamp, self_pos: int) -> int | None:
        best = None
        for p in candidates:
            if p == self_pos:
                continue
            t = df.at[p, S.CREATED_AT]
            if pd.isna(t) or pd.isna(row_time) or t > row_time or row_time - t > max_gap:
                continue
            if best is None or t >= df.at[best, S.CREATED_AT]:
                best = p
        return best

    linked = 0
    orphan_positions = list(order.index[order[CONVERSATION_SIZE].eq(1).to_numpy()])
    for pos in orphan_positions:
        role, author, text, t = df.at[pos, ROLE], str(df.at[pos, S.AUTHOR_ID]), str(df.at[pos, S.TEXT]), df.at[pos, S.CREATED_AT]
        target = None
        if role == CUSTOMER:
            target = latest_before(by_mention.get(author, []), t, pos)
        else:
            for m in MENTION_ID.findall(text):
                target = latest_before(by_author.get(m, []), t, pos)
                if target is not None:
                    break
        if target is None or df.at[target, CONVERSATION_ID] == df.at[pos, CONVERSATION_ID]:
            continue
        df.at[pos, CONVERSATION_ID] = df.at[target, CONVERSATION_ID]
        df.at[pos, ROOT_TWEET_ID] = df.at[target, ROOT_TWEET_ID]
        df.at[pos, PARENT_TWEET_ID] = df.at[target, S.TWEET_ID]
        df.at[pos, PARENT_SOURCE] = PARENT_FROM_CONTEXT
        linked += 1

    if linked:
        df = df.sort_values([CONVERSATION_ID, S.CREATED_AT, S.TWEET_ID], na_position="last", kind="mergesort")
        df[TURN_INDEX] = df.groupby(CONVERSATION_ID, sort=False).cumcount().astype("int64")
        df[CONVERSATION_SIZE] = df.groupby(CONVERSATION_ID, sort=False)[S.TWEET_ID].transform("size").astype("int64")
        df = df.reset_index(drop=True)
    logger.info("Linked %d orphan tweets via contextual mentions", linked)
    return df, linked


_MIN_TS = pd.Timestamp("1900-01-01", tz="UTC")


def _sort_key(times: list, depths: list[int], tids: list[int], i: int) -> tuple:
    """Order by time (NaT last), then reply depth (parent before child), then tweet id."""
    t = times[i]
    missing = pd.isna(t)
    return (missing, _MIN_TS if missing else t, depths[i], tids[i])


def _segment_conversation(g: pd.DataFrame, cfg: SegmentationConfig) -> dict:
    tids = g[S.TWEET_ID].tolist()
    roles = g[ROLE].tolist()
    authors = g[S.AUTHOR_ID].astype(str).tolist()
    texts = g[S.TEXT].astype(str).tolist()
    times = g[S.CREATED_AT].tolist()
    parents = [None if pd.isna(p) else int(p) for p in g[PARENT_TWEET_ID].tolist()]
    n = len(tids)
    pos = {t: i for i, t in enumerate(tids)}
    customers = list(dict.fromkeys(a for a, r in zip(authors, roles) if r == CUSTOMER))
    customer_set = set(customers)

    depths = [0] * n
    for i in range(n):
        seen: set[int] = set()
        j, d = i, 0
        while parents[j] is not None and parents[j] in pos and parents[j] not in seen:
            seen.add(parents[j])
            j = pos[parents[j]]
            d += 1
        depths[i] = d

    owner: list[str | None] = [None] * n
    source: list[str | None] = [None] * n
    for i in range(n):
        if roles[i] == CUSTOMER:
            owner[i], source[i] = authors[i], "self"
            continue
        seen = set()
        j = i
        while parents[j] is not None and parents[j] in pos and parents[j] not in seen:
            seen.add(parents[j])
            j = pos[parents[j]]
            if roles[j] == CUSTOMER:
                owner[i], source[i] = authors[j], "parent_chain"
                break
        if owner[i] is None:
            for m in MENTION_ID.findall(texts[i]):
                if m in customer_set:
                    owner[i], source[i] = m, "mention"
                    break
        if owner[i] is None and len(customers) == 1:
            owner[i], source[i] = customers[0], "sole_customer"

    segments: list[dict] = []
    for cust in customers:
        members = sorted((i for i in range(n) if owner[i] == cust), key=lambda i: _sort_key(times, depths, tids, i))
        current: list[int] = []
        for i in members:
            if current:
                prev_t, t = times[current[-1]], times[i]
                if not pd.isna(prev_t) and not pd.isna(t) and (t - prev_t) > pd.Timedelta(hours=cfg.gap_hours):
                    segments.append({"owner": cust, "members": current})
                    current = []
            current.append(i)
        if current:
            segments.append({"owner": cust, "members": current})

    # An agent-only tail (reply after a long silence) belongs to the previous episode.
    merged: list[dict] = []
    for seg in segments:
        has_customer = any(roles[i] == CUSTOMER for i in seg["members"])
        prev = next((m for m in reversed(merged) if m["owner"] == seg["owner"]), None)
        if not has_customer and prev is not None:
            prev["members"].extend(seg["members"])
            prev["members"].sort(key=lambda i: _sort_key(times, depths, tids, i))
        else:
            merged.append(seg)

    merged.sort(key=lambda s: _sort_key(times, depths, tids, s["members"][0]))
    last_by_owner: dict[str, str] = {}
    episodes = []
    for seq, seg in enumerate(merged):
        case_id = f"case_{tids[seg['members'][0]]}"
        cont_of = last_by_owner.get(seg["owner"])
        last_by_owner[seg["owner"]] = case_id
        first = seg["members"][0]
        first_cust = next((i for i in seg["members"] if roles[i] == CUSTOMER), first)
        context: list[int] = []
        j, seen = first_cust, set()
        while parents[j] is not None and parents[j] in pos and parents[j] not in seen:
            seen.add(parents[j])
            j = pos[parents[j]]
            if owner[j] is None:
                context.append(tids[j])
            else:
                break
        episodes.append(
            {
                "case_id": case_id,
                "owner": seg["owner"],
                "seq": seq,
                "members": [tids[i] for i in seg["members"]],
                "continues_case_id": cont_of,
                "context_tweet_ids": list(reversed(context)),
            }
        )
    return {
        "episodes": episodes,
        "owner": dict(zip(tids, owner)),
        "source": dict(zip(tids, source)),
        "depth": dict(zip(tids, depths)),
        "n_customers": len(customers),
    }


def segment_conversations(df: pd.DataFrame, cfg: SegmentationConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, SegmentationStats]:
    """Return (tweets with episode columns, episode table, stats).

    Added tweet columns: case_id (None if excluded), owner_id, owner_source, exclusion_reason,
    episode_seq, reply_depth. The episode table has one row per case.
    """
    cfg = cfg or SegmentationConfig()
    stats = SegmentationStats()
    df = df.reset_index(drop=True).copy()
    case_of: dict[int, str] = {}
    owner_of: dict[int, str | None] = {}
    source_of: dict[int, str | None] = {}
    depth_of: dict[int, int] = {}
    seq_of: dict[int, int] = {}
    episodes: list[dict] = []

    for _, g in df.groupby(CONVERSATION_ID, sort=False):
        res = _segment_conversation(g, cfg)
        stats.conversations += 1
        stats.multi_customer_conversations += res["n_customers"] > 1
        eps = res["episodes"]
        if len(eps) > res["n_customers"]:
            stats.conversations_split += 1
        conv_id = g[CONVERSATION_ID].iloc[0]
        for ep in eps:
            for t in ep["members"]:
                case_of[t] = ep["case_id"]
                seq_of[t] = ep["seq"]
            episodes.append(
                {
                    "case_id": ep["case_id"],
                    CONVERSATION_ID: conv_id,
                    "customer_id": ep["owner"],
                    EPISODE_SEQ: ep["seq"],
                    "is_continuation": ep["continues_case_id"] is not None,
                    "continues_case_id": ep["continues_case_id"],
                    "context_tweet_ids": ep["context_tweet_ids"],
                    "thread_customer_count": res["n_customers"],
                    "thread_episode_count": len(eps),
                }
            )
        owner_of.update(res["owner"])
        source_of.update(res["source"])
        depth_of.update(res["depth"])

    tid = df[S.TWEET_ID]
    df[CASE_ID] = tid.map(case_of)
    df[OWNER_ID] = tid.map(owner_of)
    df[OWNER_SOURCE] = tid.map(source_of)
    df[EPISODE_SEQ] = tid.map(seq_of).astype("Int64")
    df[DEPTH] = tid.map(depth_of).astype("int64")
    df[EXCLUSION_REASON] = None
    df.loc[df[CASE_ID].isna(), EXCLUSION_REASON] = EXCLUDED_UNATTRIBUTED

    episode_table = pd.DataFrame(episodes)
    stats.episodes = len(episode_table)
    stats.continuation_episodes = int(episode_table["is_continuation"].sum()) if len(episode_table) else 0
    stats.unattributed_tweets = int(df[CASE_ID].isna().sum())
    stats.owner_sources = df[OWNER_SOURCE].value_counts(dropna=True).to_dict()
    logger.info("Segmentation: %s", stats.to_dict())
    return df, episode_table, stats
