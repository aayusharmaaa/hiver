"""Extract one brand's conversations and build episode-level support cases.

Outputs (in --output-dir, default data/processed):
    virgintrains_conversations.parquet   one row per reconstructed thread, turns nested
    virgintrains_tweets.parquet          every tweet of those threads + role / case assignment
    virgintrains_cases.parquet           one row per support episode
    virgintrains_build_report.json

Usage:
    python scripts/build_virgintrains.py --input data/raw/twcs.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from common.logging_utils import configure_logging
from ingestion.brand_pipeline import process_tweets, write_outputs
from ingestion.episodes import SegmentationConfig
from ingestion.loader import clean_tweets, log_summary, read_raw, summarize_dataset
from ingestion.reconstruction import reconstruct_conversations

logger = logging.getLogger("build_virgintrains")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=_bootstrap.DEFAULT_RAW)
    parser.add_argument("--brand", default="VirginTrains")
    parser.add_argument("--output-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--gap-hours", type=float, default=SegmentationConfig.gap_hours,
                        help="Silence after which the same customer's next tweet starts a new episode")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    t0 = time.perf_counter()
    raw = read_raw(args.input, nrows=args.nrows)
    tweets, cleaning = clean_tweets(raw)
    summary = summarize_dataset(raw, tweets)
    log_summary(summary)
    if args.brand not in summary.brands:
        raise SystemExit(f"Brand {args.brand!r} has no outbound tweets in {args.input}")
    del raw

    conversations, recon = reconstruct_conversations(tweets)
    del tweets
    result = process_tweets(None, args.brand, SegmentationConfig(gap_hours=args.gap_hours), reconstructed=conversations)
    prefix = args.brand.lower()
    paths = write_outputs(result, args.output_dir, prefix)

    report = {
        "input": str(args.input),
        "cleaning": cleaning.to_dict(),
        "reconstruction_full_dataset": recon.to_dict(),
        **result.summary(),
        "gap_hours": args.gap_hours,
        "outputs": {k: str(v) for k, v in paths.items()},
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    (args.output_dir / f"{prefix}_build_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Done in %.1fs: %d cases", report["elapsed_seconds"], report["cases"])


if __name__ == "__main__":
    main()
