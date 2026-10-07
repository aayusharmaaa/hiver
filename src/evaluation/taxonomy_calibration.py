"""Taxonomy calibration sample drawn from the golden-pool *reserve* (never from golden, train or dev).

Purpose: a human labels ~200 cases so the candidate taxonomy can be compared with human judgement and then
frozen. The sample deliberately over-represents confusable regions, so its raw proportions are NOT prevalence;
`stratum_weight` re-weights within each candidate intent.

Safety rules enforced here (and re-checked by `assert_calibration_safe`):
  * only `golden_pool_reserve` cases are eligible;
  * a reserve case is dropped if it shares a group, customer, conversation, source/context tweet, or opening
    text with any golden case (the reserve came from the same hash bucket as the golden pool, so such overlaps exist);
  * continuation episodes and cases that do not start with a customer message are dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.inspection import conversation_text
from evaluation.sampling import diversity_sample
from evaluation.splits import GOLDEN, RESERVE, opener_key
from ingestion.resolution_signals import RESOLUTION_TYPES

logger = logging.getLogger(__name__)

FALLBACK = "unclear_or_media_only"
HUMAN_COLUMNS = ["human_intent", "human_resolution_type", "human_resolved", "human_escalation_signal", "human_notes"]
HUMAN_RESOLUTION_TYPES = tuple(RESOLUTION_TYPES) + ("unclear",)
HUMAN_RESOLVED_VALUES = ("yes", "no", "unclear")
ESCALATION_SIGNALS = (
    "none",
    "formal_complaint_or_customer_relations",
    "safety_or_vulnerability",
    "accessibility_or_assistance",
    "legal_media_or_ombudsman_threat",
    "repeated_unresolved_contact",
    "needs_account_or_booking_lookup",
    "other",
)

# Deliberately skewed toward the confusable regions (sums to 200). Intents not listed get `DEFAULT_QUOTA`.
DEFAULT_QUOTAS: dict[str, int] = {
    "service_status_delay_enquiry": 30,
    "journey_disruption_complaint": 28,
    "chitchat_non_support": 24,
    "customer_service_complaint": 24,
    "delay_repay_refund_claim": 22,
    "ticket_booking_query": 20,
    "seat_reservation_issue": 20,
    FALLBACK: 14,
    "praise_positive_feedback": 8,
    "first_class_catering_issue": 5,
    "onboard_wifi_issue": 5,
}
# Order in which spare capacity is handed out if some stratum cannot fill its quota.
TOP_UP_PRIORITY = list(DEFAULT_QUOTAS)


@dataclass(frozen=True)
class CalibrationConfig:
    seed: int = 42
    target_size: int = 200
    low_margin_fraction: float = 0.5
    low_margin_quantile: float = 0.4


def exclusion_reasons(pool: pd.DataFrame, golden: pd.DataFrame) -> pd.Series:
    """Reason a reserve case may not be used for calibration ('' = eligible)."""
    g_groups, g_cust, g_conv = set(golden["group_id"]), set(golden["customer_id"]), set(golden["conversation_id"])
    g_tweets = {int(t) for ids in golden["source_tweet_ids"] for t in ids} | {int(t) for ids in golden["context_tweet_ids"] for t in ids}
    g_open = {k for k in golden["opening_message"].map(opener_key) if k}

    reasons = pd.Series("", index=pool.index, dtype=object)

    def mark(mask: pd.Series, reason: str) -> None:
        reasons.loc[mask & (reasons == "")] = reason

    mark(pool["group_id"].isin(g_groups), "shares_group_with_golden")
    mark(pool["customer_id"].isin(g_cust), "shares_customer_with_golden")
    mark(pool["conversation_id"].isin(g_conv), "shares_conversation_with_golden")
    mark(
        pool.apply(lambda r: bool(g_tweets.intersection(int(t) for t in list(r["source_tweet_ids"]) + list(r["context_tweet_ids"]))), axis=1),
        "shares_tweet_with_golden",
    )
    mark(pool["opening_message"].map(opener_key).isin(g_open), "opening_text_duplicates_golden")
    mark(pool["is_continuation"].astype(bool), "continuation_episode_no_opening")
    mark(~pool["starts_with_customer"].astype(bool), "does_not_start_with_customer")
    return reasons


def _allocate(quotas: dict[str, int], available: dict[str, int], target: int) -> dict[str, int]:
    """Cap each quota by availability, then hand shortfall to strata with spare capacity in priority order."""
    alloc = {k: min(q, available.get(k, 0)) for k, q in quotas.items()}
    missing = target - sum(alloc.values())
    order = [k for k in TOP_UP_PRIORITY if k in available] + [k for k in available if k not in TOP_UP_PRIORITY]
    while missing > 0:
        progressed = False
        for k in order:
            if missing > 0 and alloc.get(k, 0) < available.get(k, 0):
                alloc[k] = alloc.get(k, 0) + 1
                missing -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def sample_calibration(pool: pd.DataFrame, cfg: CalibrationConfig, quotas: dict[str, int] | None = None) -> pd.DataFrame:
    """`pool`: eligible reserve cases with columns candidate_intent, cluster_id, margin, resolution_type, length_bucket, dm_redirect."""
    quotas = quotas or DEFAULT_QUOTAS
    available = pool["candidate_intent"].value_counts().to_dict()
    alloc = _allocate(quotas, available, cfg.target_size)
    picks: list[pd.DataFrame] = []
    for intent, q in alloc.items():
        eligible = pool[pool["candidate_intent"] == intent]
        if q <= 0 or eligible.empty:
            continue
        if q >= len(eligible):
            chosen = eligible.copy()
            chosen["selection_reason"] = "all_eligible"
        elif intent == FALLBACK:
            chosen = eligible.loc[diversity_sample(eligible, ["cluster_id", "resolution_type", "length_bucket"], q, cfg.seed)].copy()
            chosen["selection_reason"] = "fallback_coverage"
        else:
            n_low = int(round(q * cfg.low_margin_fraction))
            cutoff = eligible["margin"].quantile(cfg.low_margin_quantile)
            low_pool = eligible[eligible["margin"] <= cutoff]
            low_idx = diversity_sample(low_pool, ["cluster_id", "resolution_type", "length_bucket"], min(n_low, len(low_pool)), cfg.seed)
            rest = eligible.drop(index=low_idx)
            rest_idx = diversity_sample(rest, ["cluster_id", "resolution_type", "length_bucket", "dm_redirect"], q - len(low_idx), cfg.seed)
            chosen = eligible.loc[low_idx.union(rest_idx)].copy()
            chosen["selection_reason"] = np.where(chosen.index.isin(low_idx), "low_margin_boundary", "diverse_coverage")
        chosen["sampling_stratum"] = intent
        chosen["stratum_quota"] = quotas.get(intent, 0)
        chosen["stratum_allocated"] = q
        chosen["stratum_eligible"] = len(eligible)
        chosen["stratum_selected"] = len(chosen)
        chosen["stratum_weight"] = len(eligible) / len(chosen)
        picks.append(chosen)
    sample = pd.concat(picks).sort_values(["sampling_stratum", "case_id"]).reset_index(drop=True)
    sample["sampling_seed"] = cfg.seed
    shortfall = {k: (quotas[k], alloc.get(k, 0)) for k in quotas if alloc.get(k, 0) < quotas[k]}
    if shortfall:
        logger.warning("Strata that could not fill their quota (quota, allocated): %s", shortfall)
    return sample


def calibration_frame(sample: pd.DataFrame) -> pd.DataFrame:
    """The CSV handed to the human labeller. Human columns are empty on purpose."""
    out = pd.DataFrame(
        {
            "case_id": sample["case_id"],
            "first_customer_message": sample["opening_message"].fillna(""),
            "conversation": [conversation_text(list(t)) for t in sample["full_turns"]],
            "candidate_intent": sample["candidate_intent"],
            "candidate_cluster_id": sample["cluster_id"],
            "candidate_cluster_name": sample["cluster_name"],
            "runner_up_cluster_id": sample["second_cluster_id"],
            "runner_up_cluster_name": sample["runner_up_cluster_name"],
            "runner_up_intent": sample["runner_up_intent"],
            "cluster_margin": sample["margin"].round(4),
            "auto_resolution_type": sample["resolution_type"],
            "auto_resolved": sample["resolved"],
            "auto_dm_redirect": sample["dm_redirect"],
            "auto_resolution_outcome": sample["resolution_outcome"],
            "auto_resolution_summary": sample["resolution_summary"],
            "turn_count": sample["turn_count"],
            "length_bucket": sample["length_bucket"],
            "first_timestamp": sample["first_timestamp"],
            "customer_id": sample["customer_id"],
            "conversation_id": sample["conversation_id"],
            "group_id": sample["group_id"],
            "source_tweet_ids": [json.dumps([int(x) for x in ids]) for ids in sample["source_tweet_ids"]],
            "source_split": sample["split"],
            "sampling_stratum": sample["sampling_stratum"],
            "selection_reason": sample["selection_reason"],
            "stratum_quota": sample["stratum_quota"],
            "stratum_allocated": sample["stratum_allocated"],
            "stratum_eligible": sample["stratum_eligible"],
            "stratum_selected": sample["stratum_selected"],
            "stratum_weight": sample["stratum_weight"].round(4),
            "sampling_seed": sample["sampling_seed"],
        }
    )
    for col in HUMAN_COLUMNS:
        out[col] = ""
    return out


def source_columns_sha256(frame: pd.DataFrame) -> str:
    """Fingerprint of everything in the calibration CSV except the human_* columns (provenance + sampling metadata).

    Computed over the CSV re-read as strings, row order included, so it is stable across write/read round-trips.
    """
    source = frame.drop(columns=[c for c in HUMAN_COLUMNS if c in frame]).astype(str)
    payload = json.dumps({"columns": list(source.columns), "rows": source.values.tolist()}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_calibration_safe(sample: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, int]:
    """Raise if any calibration case is golden/train/dev, or shares customer/group/conversation/tweets with golden."""
    ids = set(sample["case_id"])
    split_of = assignments.set_index("case_id")["split"]
    wrong = sorted(c for c in ids if split_of.get(c) != RESERVE)
    golden = assignments[assignments["split"] == GOLDEN]
    sample_meta = assignments[assignments["case_id"].isin(ids)]
    checks = {
        "not_in_reserve": len(wrong),
        "duplicate_case_ids": int(len(sample) - len(ids)),
        "case_ids_in_golden": len(ids & set(golden["case_id"])),
        "customers_shared_with_golden": len(set(sample_meta["customer_id"]) & set(golden["customer_id"])),
        "conversations_shared_with_golden": len(set(sample_meta["conversation_id"]) & set(golden["conversation_id"])),
        "groups_shared_with_golden": len(set(sample_meta["group_id"]) & set(golden["group_id"])),
    }
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        raise AssertionError(f"Calibration sample is not leakage-safe: {bad}")
    return checks
