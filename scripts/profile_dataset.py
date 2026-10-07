"""Profile the raw Twitter support dataset.

Usage:
    python scripts/profile_dataset.py --input data/raw/twcs.csv
    python scripts/profile_dataset.py --input data/raw/twcs.csv --nrows 200000 --top 15
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from common.logging_utils import configure_logging
from ingestion import schema as S
from ingestion.loader import RESPONSE_TWEET_IDS, clean_tweets, log_summary, read_raw, summarize_dataset
from ingestion.reconstruction import CONVERSATION_SIZE, TURN_INDEX, reconstruct_conversations

logger = logging.getLogger("profile_dataset")


def build_profile(raw: pd.DataFrame, top: int) -> dict:
    tweets, cleaning = clean_tweets(raw)
    summary = summarize_dataset(raw, tweets)
    log_summary(summary)

    n = len(tweets)
    has_irt = tweets[S.IN_RESPONSE_TO_TWEET_ID].notna()
    has_resp = tweets[RESPONSE_TWEET_IDS].map(len) > 0
    conversations, recon = reconstruct_conversations(tweets)

    roots = conversations.loc[conversations[TURN_INDEX] == 0]
    sizes = roots[CONVERSATION_SIZE]
    per_conv = conversations.groupby("conversation_id", sort=False)[S.INBOUND].agg(["sum", "size"])
    both_roles = ((per_conv["sum"] > 0) & (per_conv["sum"] < per_conv["size"])).sum()

    outbound = tweets.loc[~tweets[S.INBOUND]]
    tweets_per_brand = outbound[S.AUTHOR_ID].value_counts()

    return {
        "input_rows": cleaning.input_rows,
        "total_tweets": n,
        "cleaning": cleaning.to_dict(),
        "missing_values": summary.missing_values,
        "unique_authors": summary.unique_authors,
        "unique_brands": summary.unique_brands,
        "inbound_tweets": summary.inbound_count,
        "outbound_tweets": summary.outbound_count,
        "inbound_outbound_ratio": round(summary.inbound_count / max(summary.outbound_count, 1), 4),
        "date_range": [str(tweets[S.CREATED_AT].min()), str(tweets[S.CREATED_AT].max())],
        "relationships": {
            "pct_with_in_response_to": round(100 * has_irt.mean(), 2),
            "pct_with_response_tweet_id": round(100 * has_resp.mean(), 2),
            "pct_with_any_relationship": round(100 * (has_irt | has_resp).mean(), 2),
            "pct_with_multiple_responses": round(100 * (tweets[RESPONSE_TWEET_IDS].map(len) > 1).mean(), 2),
            "missing_parent_refs": recon.missing_parent_refs,
            "dangling_response_refs": recon.dangling_response_refs,
        },
        "conversations": {
            "count": recon.conversations,
            "single_tweet": recon.single_tweet_conversations,
            "pct_single_tweet": round(100 * recon.single_tweet_conversations / max(recon.conversations, 1), 2),
            "with_customer_and_brand": int(both_roles),
            "size_mean": round(float(sizes.mean()), 3),
            "size_median": float(sizes.median()),
            "size_p90": float(sizes.quantile(0.9)),
            "size_p99": float(sizes.quantile(0.99)),
            "size_max": int(sizes.max()),
            "size_distribution": {
                str(k): int(v) for k, v in sizes.clip(upper=10).value_counts().sort_index().items()
            },
        },
        "tweets_per_brand_top": tweets_per_brand.head(top).to_dict(),
        "tweets_per_brand_all": tweets_per_brand.to_dict(),
    }


def print_profile(p: dict, top: int) -> None:
    rel, conv = p["relationships"], p["conversations"]
    print("\n=== Dataset profile ===")
    print(f"Total tweets:            {p['total_tweets']:,}  (input rows {p['input_rows']:,})")
    print(f"Date range:              {p['date_range'][0]} -> {p['date_range'][1]}")
    print(f"Unique authors / brands: {p['unique_authors']:,} / {p['unique_brands']}")
    print(f"Inbound / outbound:      {p['inbound_tweets']:,} / {p['outbound_tweets']:,}  (ratio {p['inbound_outbound_ratio']})")
    print(f"Missing values:          {p['missing_values']}")
    print("\n--- Relationships ---")
    print(f"% with in_response_to:   {rel['pct_with_in_response_to']}%")
    print(f"% with response ids:     {rel['pct_with_response_tweet_id']}%")
    print(f"% with any relationship: {rel['pct_with_any_relationship']}%")
    print(f"Refs to missing tweets:  parent={rel['missing_parent_refs']:,}  response={rel['dangling_response_refs']:,}")
    print("\n--- Conversations ---")
    print(f"Conversations:           {conv['count']:,}")
    print(f"Single-tweet:            {conv['single_tweet']:,} ({conv['pct_single_tweet']}%)")
    print(f"Customer + brand turns:  {conv['with_customer_and_brand']:,}")
    print(f"Size mean/median/p90/max {conv['size_mean']} / {conv['size_median']} / {conv['size_p90']} / {conv['size_max']}")
    print(f"Size distribution (10 = 10+): {conv['size_distribution']}")
    print(f"\n--- Top {top} brands by outbound tweets ---")
    for brand, count in p["tweets_per_brand_top"].items():
        print(f"  {brand:<20} {count:>9,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=_bootstrap.DEFAULT_RAW, help="CSV or Parquet file")
    parser.add_argument("--nrows", type=int, default=None, help="Read only the first N rows (breaks some threads)")
    parser.add_argument("--top", type=int, default=20, help="How many brands to print")
    parser.add_argument("--output", type=Path, default=_bootstrap.DEFAULT_PROCESSED / "profile.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    raw = read_raw(args.input, nrows=args.nrows)
    profile = build_profile(raw, args.top)
    print_profile(profile, args.top)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
