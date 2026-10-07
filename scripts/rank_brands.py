"""Rank brands for agent suitability from the outputs of build_cases.py.

Usage:
    python scripts/rank_brands.py
    python scripts/rank_brands.py --top 15 --min-conversations 2000 --sample-size 5000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from common.logging_utils import configure_logging
from evaluation.brand_ranking import CASE_COLUMNS, DEFAULT_WEIGHTS, RankingConfig, compute_brand_metrics, score_brands

logger = logging.getLogger("rank_brands")

CSV_COLUMNS = [
    "rank",
    "brand",
    "score",
    "eligible",
    "ineligible_reason",
    "conversations",
    "reconstructable_conversations",
    "reconstructable_rate",
    "multi_turn_conversations",
    "multi_turn_density",
    "avg_turns",
    "resolved_rate",
    "dm_redirect_rate",
    "top_resolution_type",
    "top_resolution_share",
    "template_repeat_rate",
    "intent_diversity",
    "intent_coverage",
    "latin_script_share",
    "multi_party_rate",
    "missing_parent_rate",
    "median_first_response_minutes",
    "agent_tweets_total",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--cases", type=Path, default=None, help="Defaults to <processed-dir>/cases.parquet")
    parser.add_argument("--tweets", type=Path, default=None, help="Defaults to <processed-dir>/tweets_clean.parquet")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min-conversations", type=int, default=RankingConfig.min_conversations)
    parser.add_argument("--min-latin-share", type=float, default=RankingConfig.min_latin_share)
    parser.add_argument("--sample-size", type=int, default=RankingConfig.sample_size)
    parser.add_argument("--weights", type=str, default=None, help='JSON overrides, e.g. \'{"multi_turn_density": 0.3}\'')
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    cases_path = args.cases or args.processed_dir / "cases.parquet"
    tweets_path = args.tweets or args.processed_dir / "tweets_clean.parquet"
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        weights.update(json.loads(args.weights))
    cfg = RankingConfig(
        sample_size=args.sample_size,
        min_conversations=args.min_conversations,
        min_latin_share=args.min_latin_share,
        weights=weights,
    )

    logger.info("Loading %s", cases_path)
    cases = pd.read_parquet(cases_path, columns=CASE_COLUMNS)
    logger.info("Loading outbound tweets from %s", tweets_path)
    tweets = pd.read_parquet(tweets_path, columns=["author_id", "inbound", "text"])
    agent_tweets = tweets.loc[~tweets["inbound"], ["author_id", "text"]]
    del tweets

    ranked = score_brands(compute_brand_metrics(cases, agent_tweets, cfg), cfg)

    out = args.processed_dir
    csv_path, json_path = out / "brand_ranking.csv", out / "brand_ranking.json"
    ranked[CSV_COLUMNS].round(4).to_csv(csv_path, index=False)
    payload = {
        "config": {
            "weights": cfg.weights,
            "sample_size": cfg.sample_size,
            "min_conversations": cfg.min_conversations,
            "min_latin_share": cfg.min_latin_share,
            "template_tokens": cfg.template_tokens,
            "template_min_repeats": cfg.template_min_repeats,
        },
        "brands": json.loads(ranked.round(4).to_json(orient="records")),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s and %s", csv_path, json_path)

    show = ranked.loc[ranked["eligible"]].head(args.top)
    cols = [
        "rank", "brand", "score", "conversations", "reconstructable_conversations", "multi_turn_density",
        "avg_turns", "resolved_rate", "dm_redirect_rate", "template_repeat_rate", "intent_diversity",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"\n=== Top {args.top} brands ===")
        print(show[cols].round(3).to_string(index=False))
        excluded = ranked.loc[~ranked["eligible"] & (ranked["conversations"] >= cfg.min_conversations)]
        if len(excluded):
            print("\nExcluded despite volume:")
            print(excluded[["brand", "conversations", "ineligible_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
