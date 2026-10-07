"""Discover candidate intent clusters from customer opening messages.

Fits on train+dev customers only; golden-pool cases are assigned to the nearest centroid
afterwards, so the golden set never influences the discovered taxonomy.

Outputs (data/processed):
    virgintrains_case_clusters.parquet   case_id -> cluster assignment + margin
    virgintrains_cluster_preview.json    raw diagnostics (terms, examples, k sweep) for manual naming

Usage:
    python scripts/discover_intents.py            # chooses k in 8..15 automatically
    python scripts/discover_intents.py --k 12
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from common.logging_utils import configure_logging
from evaluation.splits import POOL, SplitConfig, provisional_split
from taxonomy import discovery as D

logger = logging.getLogger("discover_intents")
CONTINUATION = -3


def choose_k(sweep: list[D.KSweepRow]) -> int:
    """Highest silhouette among k whose clustering is reproducible across seeds (ARI >= 0.6)."""
    stable = [r for r in sweep if r.seed_agreement_ari >= 0.6 and r.min_cluster_share >= 0.02] or sweep
    return max(stable, key=lambda r: r.silhouette).k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--k-min", type=int, default=8)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    cases = pd.read_parquet(
        args.processed_dir / "virgintrains_cases.parquet",
        columns=["case_id", "customer_id", "conversation_id", "opening_message", "is_continuation", "starts_with_customer", "has_brand_reply"],
    )
    prov = provisional_split(cases, SplitConfig(seed=args.seed))
    cases = cases.merge(prov[["case_id", "provisional_split"]], on="case_id")
    cases["norm"] = cases["opening_message"].map(D.normalize_opening)
    cases["informative"] = D.informative_mask(cases["norm"])
    usable = cases[~cases["is_continuation"] & cases["starts_with_customer"] & cases["informative"]]
    fit = usable[usable["provisional_split"] != POOL].reset_index(drop=True)
    pool = usable[usable["provisional_split"] == POOL].reset_index(drop=True)
    logger.info("Usable openers: %d (fit %d, golden-pool %d); non-informative or continuation: %d",
                len(usable), len(fit), len(pool), len(cases) - len(usable))

    emb_fit, model_used = D.embed_texts(fit["norm"].tolist(), cache_dir=args.processed_dir / "cache")
    logger.info("Embedding model: %s, shape %s", model_used, emb_fit.shape)

    sweep = D.sweep_k(emb_fit, range(args.k_min, args.k_max + 1), seed=args.seed)
    k = args.k or choose_k(sweep)
    logger.info("Using k=%d", k)
    km = D.fit_kmeans(emb_fit, k, args.seed)
    det_fit = D.assignment_details(km, emb_fit)
    fit = pd.concat([fit, det_fit], axis=1)

    emb_pool, _ = D.embed_texts(pool["norm"].tolist(), cache_dir=args.processed_dir / "cache")
    pool = pd.concat([pool, D.assignment_details(km, emb_pool)], axis=1)

    terms = D.cluster_top_terms(fit["norm"].tolist(), fit["cluster_id"].to_numpy())
    agreement = D.tfidf_agreement(fit["norm"].tolist(), fit["cluster_id"].to_numpy(), k, args.seed)
    logger.info("TF-IDF/LSA vs embedding cluster agreement: %s", agreement)

    assigned = pd.concat([fit.assign(in_fit=True), pool.assign(in_fit=False)], ignore_index=True)
    assigned = assigned[["case_id", "cluster_id", "second_cluster_id", "distance", "margin", "in_fit", "norm"]]
    rest = cases[~cases["case_id"].isin(assigned["case_id"])].copy()
    rest["cluster_id"] = np.where(rest["is_continuation"] | ~rest["starts_with_customer"], CONTINUATION, D.INSUFFICIENT)
    rest["second_cluster_id"] = rest["cluster_id"]
    rest["distance"] = np.nan
    rest["margin"] = np.nan
    rest["in_fit"] = False
    out = pd.concat([assigned, rest[["case_id", "cluster_id", "second_cluster_id", "distance", "margin", "in_fit", "norm"]]], ignore_index=True)
    out.to_parquet(args.processed_dir / "virgintrains_case_clusters.parquet", index=False)

    texts = cases.set_index("case_id")["opening_message"]
    clusters = {}
    for c in range(k):
        members = fit[fit["cluster_id"] == c]
        nearest = members.nsmallest(args.examples, "distance")
        ambiguous = members.nsmallest(args.examples, "margin")
        clusters[c] = {
            "size": len(members),
            "share": round(len(members) / len(fit), 4),
            "top_terms": terms[c],
            "nearest_examples": [{"case_id": r.case_id, "text": texts[r.case_id]} for r in nearest.itertuples()],
            "ambiguous_examples": [
                {"case_id": r.case_id, "text": texts[r.case_id], "second_cluster": int(r.second_cluster_id), "margin": round(float(r.margin), 4)}
                for r in ambiguous.itertuples()
            ],
        }
    preview = {
        "embedding_model": model_used,
        "k": k,
        "n_fit": len(fit),
        "n_golden_pool_assigned": len(pool),
        "n_insufficient_or_continuation": int(len(cases) - len(usable)),
        "k_sweep": [r.__dict__ for r in sweep],
        "tfidf_agreement": agreement,
        "clusters": clusters,
    }
    (args.processed_dir / "virgintrains_cluster_preview.json").write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote cluster assignments and preview")


if __name__ == "__main__":
    main()
