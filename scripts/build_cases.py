"""Reconstruct conversations and write normalized support cases.

Outputs (in --output-dir):
    tweets_clean.parquet   every cleaned tweet + conversation_id / parent / turn_index
    cases.parquet          one row per conversation (nested full_turns with provenance)
    cases.jsonl            optional, with --jsonl
    build_report.json      cleaning / reconstruction / case statistics

Usage:
    python scripts/build_cases.py --input data/raw/twcs.csv
    python scripts/build_cases.py --input data/raw/twcs.csv --brands AppleSupport --jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from common.logging_utils import configure_logging
from ingestion.cases import write_support_cases
from ingestion.loader import RESPONSE_TWEET_IDS, clean_tweets, log_summary, read_raw, summarize_dataset
from ingestion.reconstruction import reconstruct_conversations

logger = logging.getLogger("build_cases")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=_bootstrap.DEFAULT_RAW, help="CSV or Parquet file")
    parser.add_argument("--output-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--nrows", type=int, default=None, help="Read only the first N rows (breaks some threads)")
    parser.add_argument("--brands", nargs="*", default=None, help="Only write cases attributed to these brands")
    parser.add_argument("--jsonl", action="store_true", help="Also write cases.jsonl")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    if args.nrows:
        logger.warning("--nrows truncates the file; replies outside the first rows will be missing")

    t0 = time.perf_counter()
    raw = read_raw(args.input, nrows=args.nrows)
    tweets, cleaning = clean_tweets(raw)
    summary = summarize_dataset(raw, tweets)
    log_summary(summary)
    del raw

    conversations, recon = reconstruct_conversations(tweets)
    del tweets

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tweets_path = out / "tweets_clean.parquet"
    conversations.drop(columns=[RESPONSE_TWEET_IDS]).to_parquet(tweets_path, index=False, compression="zstd")
    logger.info("Wrote %s (%d rows)", tweets_path, len(conversations))

    suffix = "" if not args.brands else "_" + "_".join(args.brands)
    cases_path = out / f"cases{suffix}.parquet"
    jsonl_path = out / f"cases{suffix}.jsonl" if args.jsonl else None
    stats = write_support_cases(
        conversations,
        summary.brands,
        cases_path,
        jsonl_path=jsonl_path,
        brands=set(args.brands) if args.brands else None,
        chunk_size=args.chunk_size,
    )
    logger.info("Wrote %s%s", cases_path, f" and {jsonl_path}" if jsonl_path else "")

    report = {
        "input": str(args.input),
        "nrows": args.nrows,
        "brands_filter": args.brands,
        "cleaning": cleaning.to_dict(),
        "reconstruction": recon.to_dict(),
        "cases": stats.to_dict(),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / f"build_report{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Done in %.1fs", report["elapsed_seconds"])


if __name__ == "__main__":
    main()
