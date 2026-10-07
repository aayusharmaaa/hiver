"""Rank brands by how suitable their data is for building a grounded support agent.

All metrics are computed per brand from support cases and the brand's own outbound tweets.
Text-based metrics use a fixed-size random sample per brand so large brands do not get an
advantage just from having more text (e.g. more chances for templates to repeat).

Score = weighted sum of percentile ranks among eligible brands, so metrics on different
scales are comparable and a single outlier does not dominate.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ingestion.resolution import ResolutionType
from ingestion.text import latin_share, template_key
from taxonomy.keyword_intents import OTHER, classify_intent, normalized_entropy

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "conversations": 0.10,
    "reconstructable_conversations": 0.15,
    "reconstructable_rate": 0.05,
    "multi_turn_density": 0.20,
    "template_repeat_rate": 0.15,
    "intent_diversity": 0.15,
    "resolved_rate": 0.10,
    "non_dm_rate": 0.10,
}

CASE_COLUMNS = [
    "case_id",
    "brand",
    "turn_count",
    "is_reconstructable",
    "is_multi_party",
    "is_multi_turn",
    "has_missing_parent",
    "resolved",
    "resolution_type",
    "first_customer_message",
    "first_response_minutes",
]


@dataclass
class RankingConfig:
    sample_size: int = 5_000
    template_tokens: int = 8
    template_min_repeats: int = 5
    min_conversations: int = 1_000
    min_latin_share: float = 0.85
    seed: int = 42
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


def _template_stats(texts: list[str], cfg: RankingConfig) -> tuple[float, list[dict]]:
    keys = [template_key(t, cfg.template_tokens) for t in texts]
    keys = [k for k in keys if k]
    if not keys:
        return 0.0, []
    counts = Counter(keys)
    repeated = sum(c for c in counts.values() if c >= cfg.template_min_repeats)
    top = [{"template": k, "count": c} for k, c in counts.most_common(5)]
    return repeated / len(keys), top


def compute_brand_metrics(
    cases: pd.DataFrame, agent_tweets: pd.DataFrame, cfg: RankingConfig | None = None
) -> pd.DataFrame:
    """`cases`: support cases (see CASE_COLUMNS). `agent_tweets`: columns author_id, text
    for outbound tweets. Returns one row per brand."""
    cfg = cfg or RankingConfig()
    cases = cases.loc[cases["brand"].notna()]
    agent_by_brand = {b: g for b, g in agent_tweets.groupby("author_id", sort=False)}
    rows: list[dict] = []

    for brand, g in cases.groupby("brand", sort=True):
        # "Reconstructable" = one customer, one brand, at least one turn from each.
        recon = g.loc[g["is_reconstructable"] & ~g["is_multi_party"]]
        n_conv, n_recon = len(g), len(recon)
        res_counts = recon["resolution_type"].value_counts()

        recon_sample = _sample(recon, cfg.sample_size, cfg.seed)
        first_msgs = recon_sample["first_customer_message"].fillna("").tolist()
        intents = [classify_intent(t) for t in first_msgs]
        intent_counts = Counter(intents)
        latin = float(np.mean([latin_share(t) >= 0.9 for t in first_msgs])) if first_msgs else 0.0

        brand_agent = agent_by_brand.get(brand)
        agent_texts = (
            _sample(brand_agent, cfg.sample_size, cfg.seed)["text"].fillna("").tolist()
            if brand_agent is not None
            else []
        )
        repeat_rate, top_templates = _template_stats(agent_texts, cfg)

        def rate(numer: float, denom: int) -> float:
            return float(numer) / denom if denom else 0.0

        rows.append(
            {
                "brand": brand,
                "conversations": n_conv,
                "reconstructable_conversations": n_recon,
                "reconstructable_rate": rate(n_recon, n_conv),
                "multi_turn_conversations": int(recon["is_multi_turn"].sum()),
                "multi_turn_density": rate(recon["is_multi_turn"].sum(), n_recon),
                "avg_turns": float(recon["turn_count"].mean()) if n_recon else 0.0,
                "resolved_rate": rate(recon["resolved"].sum(), n_recon),
                "dm_redirect_rate": rate(res_counts.get(ResolutionType.REDIRECTED_TO_DM.value, 0), n_recon),
                "non_dm_rate": 1.0 - rate(res_counts.get(ResolutionType.REDIRECTED_TO_DM.value, 0), n_recon),
                "top_resolution_type": res_counts.index[0] if len(res_counts) else None,
                "top_resolution_share": rate(res_counts.iloc[0], n_recon) if len(res_counts) else 0.0,
                "template_repeat_rate": repeat_rate,
                "intent_diversity": normalized_entropy(intents),
                "intent_coverage": rate(sum(c for k, c in intent_counts.items() if k != OTHER), len(intents)),
                "latin_script_share": latin,
                "multi_party_rate": rate(g["is_multi_party"].sum(), n_conv),
                "missing_parent_rate": rate(g["has_missing_parent"].sum(), n_conv),
                "median_first_response_minutes": float(recon["first_response_minutes"].median()) if n_recon else None,
                "agent_tweets_total": 0 if brand_agent is None else len(brand_agent),
                "resolution_mix": {k: round(v / n_recon, 4) for k, v in res_counts.items()} if n_recon else {},
                "top_intents": [{"intent": k, "share": round(c / len(intents), 4)} for k, c in intent_counts.most_common(6)],
                "top_agent_templates": top_templates,
            }
        )
    metrics = pd.DataFrame(rows)
    logger.info("Computed metrics for %d brands", len(metrics))
    return metrics


def score_brands(metrics: pd.DataFrame, cfg: RankingConfig | None = None) -> pd.DataFrame:
    """Add `eligible`, `ineligible_reason`, per-metric percentile columns, `score`, `rank`."""
    cfg = cfg or RankingConfig()
    df = metrics.copy()
    reasons = []
    for _, r in df.iterrows():
        why = []
        if r["conversations"] < cfg.min_conversations:
            why.append(f"conversations<{cfg.min_conversations}")
        if r["latin_script_share"] < cfg.min_latin_share:
            why.append(f"latin_script_share<{cfg.min_latin_share}")
        reasons.append(";".join(why) or None)
    df["ineligible_reason"] = reasons
    df["eligible"] = df["ineligible_reason"].isna()

    eligible = df["eligible"]
    df["score"] = np.nan
    if eligible.any():
        score = pd.Series(0.0, index=df.index[eligible])
        for metric, weight in cfg.weights.items():
            pct = df.loc[eligible, metric].astype(float).rank(pct=True, method="average")
            df.loc[eligible, f"pct_{metric}"] = pct
            score += weight * pct
        df.loc[eligible, "score"] = score / sum(cfg.weights.values())

    df = df.sort_values(["eligible", "score", "conversations"], ascending=[False, False, False])
    df["rank"] = np.where(df["eligible"], np.arange(1, len(df) + 1), np.nan)
    return df.reset_index(drop=True)
