"""Leakage-safe train / dev / golden splits for support cases.

Design:
  * The unit of assignment is a *group*: cases that share a customer or a conversation thread
    are tied together (union-find), so no customer or thread straddles two splits.
  * Groups are hashed (sha256 of seed + smallest case_id) into train / dev / golden_pool.
    This happens BEFORE any modelling: intent discovery only sees train+dev.
  * The golden set is a stratified selection from eligible golden_pool cases. The rest of the
    pool is `golden_pool_reserve` (never used for retrieval or prompts).
  * After golden is fixed, train/dev cases whose opening text duplicates a golden opener, or
    that share context tweets with a golden case, are moved to `excluded_leakage_guard`.

Usage rules (also written into the manifest):
  * retrieval corpus / prompt examples: `train_retrieval` only;
  * thresholds, calibration, prompt selection: `dev_calibration` only;
  * `golden_eval`: final reporting only, never tuning.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from evaluation.sampling import add_coverage_columns, diversity_sample, stratum_weights

logger = logging.getLogger(__name__)

TRAIN = "train_retrieval"
DEV = "dev_calibration"
GOLDEN = "golden_eval"
POOL = "golden_pool"
RESERVE = "golden_pool_reserve"
EXCLUDED = "excluded_leakage_guard"

_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_DIGITS = re.compile(r"\d+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    golden_pool_fraction: float = 0.10
    dev_fraction: float = 0.15
    golden_size: int = 250
    min_opening_tokens: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


def opener_key(text: str | None) -> str:
    """Normalized opening text used to detect duplicated openers across splits."""
    if not text:
        return ""
    t = _URL.sub(" ", text.lower())
    t = _MENTION.sub(" ", t)
    t = _DIGITS.sub("0", t)
    return " ".join(_NON_ALNUM.sub(" ", t).split())


def case_groups(cases: pd.DataFrame) -> pd.Series:
    """Group id per case: connected components over shared customer or conversation."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for case_id, cust, conv in zip(cases["case_id"], cases["customer_id"], cases["conversation_id"]):
        union(case_id, f"cust:{cust}")
        union(case_id, f"conv:{conv}")
    roots = [find(c) for c in cases["case_id"]]
    members = pd.Series(cases["case_id"].tolist(), index=roots).groupby(level=0).min()
    return pd.Series([members[r] for r in roots], index=cases.index, name="group_id")


def group_bucket(group_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def provisional_split(cases: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Cluster-free split into train / dev / golden_pool. Safe to compute before modelling."""
    out = cases[["case_id", "customer_id", "conversation_id"]].copy()
    out["group_id"] = case_groups(cases)
    u = out["group_id"].map(lambda g: group_bucket(g, cfg.seed))
    out["provisional_split"] = np.select(
        [u < cfg.golden_pool_fraction, u < cfg.golden_pool_fraction + cfg.dev_fraction],
        [POOL, DEV],
        default=TRAIN,
    )
    return out


def golden_eligible(cases: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    tokens = cases["opening_message"].fillna("").map(lambda t: len(opener_key(t).split()))
    return (
        cases["has_brand_reply"]
        & ~cases["is_continuation"]
        & (tokens >= cfg.min_opening_tokens)
        & cases["starts_with_customer"]
    )


def finalize_splits(
    cases: pd.DataFrame, provisional: pd.DataFrame, clusters: pd.Series, cfg: SplitConfig
) -> pd.DataFrame:
    """Return per-case final split with a reason. `clusters` maps case_id -> cluster id."""
    df = add_coverage_columns(cases).merge(provisional[["case_id", "group_id", "provisional_split"]], on="case_id")
    df["cluster_id"] = df["case_id"].map(clusters).fillna(-2).astype(int)
    df["split"] = df["provisional_split"].replace({POOL: RESERVE})
    df["split_reason"] = "hash_bucket"

    pool = df[(df["provisional_split"] == POOL) & golden_eligible(df, cfg)]
    picked = diversity_sample(pool, ["cluster_id", "resolution_type", "length_bucket", "dm_redirect", "resolved"], cfg.golden_size, cfg.seed)
    df.loc[picked, "split"] = GOLDEN
    df.loc[picked, "split_reason"] = "stratified_pick_from_golden_pool"
    df.loc[(df["split"] == RESERVE), "split_reason"] = "golden_pool_not_selected"

    golden = df[df["split"] == GOLDEN]
    g_keys = {k for k in golden["opening_message"].map(opener_key) if k}
    g_context = {t for ids in golden["context_tweet_ids"] for t in ids}
    open_dup = df["opening_message"].map(opener_key).isin(g_keys) & df["split"].isin([TRAIN, DEV])
    ctx_shared = df["context_tweet_ids"].map(lambda ids: bool(g_context.intersection(ids))) & df["split"].isin([TRAIN, DEV])
    df.loc[open_dup, ["split", "split_reason"]] = [EXCLUDED, "opening_text_duplicates_golden"]
    df.loc[ctx_shared & ~open_dup, ["split", "split_reason"]] = [EXCLUDED, "shares_context_tweet_with_golden"]
    logger.info("Split sizes: %s", df["split"].value_counts().to_dict())

    if len(golden):
        pop = df[(df["provisional_split"] == POOL) & golden_eligible(df, cfg)]
        weights = stratum_weights(pop, df[df["split"] == GOLDEN], "cluster_id")
        df["golden_stratum_weight"] = weights
    else:
        df["golden_stratum_weight"] = np.nan
    return df


class LeakageError(AssertionError):
    pass


def verify_no_leakage(assigned: pd.DataFrame) -> dict[str, int]:
    """Raise if golden overlaps retrieval/dev data; returns the checks performed (all zero)."""
    golden = assigned[assigned["split"] == GOLDEN]
    others = assigned[assigned["split"].isin([TRAIN, DEV])]
    checks = {
        "case_ids": len(set(golden["case_id"]) & set(others["case_id"])),
        "customers": len(set(golden["customer_id"]) & set(others["customer_id"])),
        "conversations": len(set(golden["conversation_id"]) & set(others["conversation_id"])),
        "groups": len(set(golden["group_id"]) & set(others["group_id"])),
        "source_tweets": len(
            {t for ids in golden["source_tweet_ids"] for t in ids} & {t for ids in others["source_tweet_ids"] for t in ids}
        ),
        "context_tweets": len(
            {t for ids in golden["context_tweet_ids"] for t in ids} & {t for ids in others["context_tweet_ids"] for t in ids}
        ),
        "opening_texts": len(
            {k for k in golden["opening_message"].map(opener_key) if k}
            & {k for k in others["opening_message"].map(opener_key) if k}
        ),
    }
    dev_train = {
        "train_dev_customers": len(
            set(assigned.loc[assigned["split"] == TRAIN, "customer_id"]) & set(assigned.loc[assigned["split"] == DEV, "customer_id"])
        )
    }
    checks.update(dev_train)
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        raise LeakageError(f"Split leakage detected: {bad}")
    return checks
