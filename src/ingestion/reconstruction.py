"""Rebuild multi-turn conversations from tweet reply links.

A conversation is a connected component of the reply graph, where an undirected edge joins
two tweets if either
  * child.in_response_to_tweet_id == parent.tweet_id, or
  * child.tweet_id is listed in parent.response_tweet_id,
and both tweets exist in the input. Using both directions recovers links when only one side
was recorded. References to tweets missing from the input are counted and flagged instead
of being treated as errors (the public dataset is a sample, so dangling ids are common).

Within a conversation, turns are ordered by (created_at, tweet_id); tweets with an
unparseable timestamp sort last. The earliest tweet is the root and gives the
conversation its id: ``conv_<root_tweet_id>``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ingestion import schema as S
from ingestion.loader import RESPONSE_TWEET_IDS, deduplicate_tweets

logger = logging.getLogger(__name__)

CONVERSATION_ID = "conversation_id"
ROOT_TWEET_ID = "root_tweet_id"
PARENT_TWEET_ID = "parent_tweet_id"
PARENT_SOURCE = "parent_source"
MISSING_PARENT = "missing_parent"
TURN_INDEX = "turn_index"
CONVERSATION_SIZE = "conversation_size"

PARENT_FROM_IN_RESPONSE_TO = "in_response_to_tweet_id"
PARENT_FROM_RESPONSE_LIST = "response_tweet_id"


@dataclass
class ReconstructionStats:
    tweets: int = 0
    conversations: int = 0
    edges: int = 0
    single_tweet_conversations: int = 0
    missing_parent_refs: int = 0
    dangling_response_refs: int = 0
    self_references: int = 0
    parents_inferred_from_response_list: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _connected_components(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Union-find with path compression; returns a component label (min member index) per node."""
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(src.tolist(), dst.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    return np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)


def reconstruct_conversations(tweets: pd.DataFrame) -> tuple[pd.DataFrame, ReconstructionStats]:
    """Assign every tweet to a conversation.

    Expects the output of `ingestion.loader.clean_tweets`. Returns a copy sorted by
    (conversation_id, turn_index) with these added columns:
        conversation_id, root_tweet_id, parent_tweet_id (Int64), parent_source,
        missing_parent (bool), turn_index (int), conversation_size (int)
    """
    stats = ReconstructionStats()
    df = tweets
    if df[S.TWEET_ID].duplicated().any():
        logger.warning("Input contains duplicate tweet_ids; deduplicating before reconstruction")
        df = deduplicate_tweets(df)
    df = df.reset_index(drop=True).copy()
    n = len(df)
    stats.tweets = n
    if n == 0:
        for col in (CONVERSATION_ID, ROOT_TWEET_ID, PARENT_TWEET_ID, PARENT_SOURCE, MISSING_PARENT, TURN_INDEX, CONVERSATION_SIZE):
            df[col] = pd.Series(dtype=object)
        return df, stats

    id_index = pd.Index(df[S.TWEET_ID].to_numpy())
    positions = np.arange(n, dtype=np.int64)

    # Edges from in_response_to_tweet_id (child -> parent).
    irt = df[S.IN_RESPONSE_TO_TWEET_ID]
    has_irt = irt.notna().to_numpy()
    irt_vals = irt.fillna(-1).astype("int64").to_numpy()
    irt_pos = id_index.get_indexer(irt_vals)
    self_ref = has_irt & (irt_vals == df[S.TWEET_ID].to_numpy())
    stats.self_references = int(self_ref.sum())
    irt_ok = has_irt & (irt_pos >= 0) & ~self_ref
    missing_parent = has_irt & (irt_pos < 0)
    stats.missing_parent_refs = int(missing_parent.sum())

    # Edges from response_tweet_id lists (parent -> children).
    resp = df[RESPONSE_TWEET_IDS]
    lengths = resp.map(len).to_numpy()
    resp_parent_pos = np.repeat(positions, lengths)
    resp_child_ids = np.fromiter(
        (cid for ids in resp.tolist() for cid in ids), dtype=np.int64, count=int(lengths.sum())
    )
    resp_child_pos = id_index.get_indexer(resp_child_ids) if len(resp_child_ids) else np.array([], dtype=np.int64)
    resp_ok = (resp_child_pos >= 0) & (resp_child_pos != resp_parent_pos)
    stats.dangling_response_refs = int((resp_child_pos < 0).sum())

    src = np.concatenate([positions[irt_ok], resp_parent_pos[resp_ok]])
    dst = np.concatenate([irt_pos[irt_ok], resp_child_pos[resp_ok]])
    stats.edges = len(src)
    components = _connected_components(n, src, dst)

    # Parent: explicit in_response_to when resolvable, else the (lowest-id) tweet whose
    # response list names this tweet.
    parent_pos = np.full(n, -1, dtype=np.int64)
    parent_pos[irt_ok] = irt_pos[irt_ok]
    parent_source = np.full(n, None, dtype=object)
    parent_source[irt_ok] = PARENT_FROM_IN_RESPONSE_TO
    if resp_ok.any():
        inferred = (
            pd.DataFrame({"child": resp_child_pos[resp_ok], "parent": resp_parent_pos[resp_ok]})
            .sort_values(["child", "parent"])
            .drop_duplicates("child")
        )
        child = inferred["child"].to_numpy()
        needs = parent_pos[child] < 0
        parent_pos[child[needs]] = inferred["parent"].to_numpy()[needs]
        parent_source[child[needs]] = PARENT_FROM_RESPONSE_LIST
        stats.parents_inferred_from_response_list = int(needs.sum())

    tweet_ids = df[S.TWEET_ID].to_numpy()
    parent_ids = pd.array(np.where(parent_pos >= 0, tweet_ids[np.maximum(parent_pos, 0)], 0), dtype="Int64")
    parent_ids[parent_pos < 0] = pd.NA
    df[PARENT_TWEET_ID] = parent_ids
    df[PARENT_SOURCE] = parent_source
    df[MISSING_PARENT] = missing_parent
    df["_component"] = components

    df = df.sort_values(["_component", S.CREATED_AT, S.TWEET_ID], na_position="last", kind="mergesort")
    df[TURN_INDEX] = df.groupby("_component", sort=False).cumcount().astype("int64")
    root_ids = df.groupby("_component", sort=False)[S.TWEET_ID].transform("first")
    df[ROOT_TWEET_ID] = root_ids.astype("int64")
    df[CONVERSATION_ID] = "conv_" + root_ids.astype(str)
    df[CONVERSATION_SIZE] = df.groupby("_component", sort=False)[S.TWEET_ID].transform("size").astype("int64")
    df = df.drop(columns="_component")
    df = df.sort_values([ROOT_TWEET_ID, TURN_INDEX], kind="mergesort").reset_index(drop=True)

    sizes = df.loc[df[TURN_INDEX] == 0, CONVERSATION_SIZE]
    stats.conversations = int(len(sizes))
    stats.single_tweet_conversations = int((sizes == 1).sum())
    logger.info("Reconstruction: %s", stats.to_dict())
    return df, stats
