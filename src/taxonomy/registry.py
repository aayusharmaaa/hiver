"""Helpers for the intent registry: evidence extraction, cluster->intent mapping, hashing, status checks.

Nothing here invents policy. "Required information" and "escalation triggers" are derived from what
VirginTrains agents were observed to ask for or do in the train split, with counts as evidence;
anything else is marked NEEDS_REVIEW.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

STATUS_CANDIDATE = "CANDIDATE_NOT_GROUND_TRUTH"
STATUS_CALIBRATED = "HUMAN_CALIBRATED"
FALLBACK_INTENT = "unclear_or_media_only"
NEEDS_REVIEW = "NEEDS_REVIEW"

# Things an agent can ask the customer for. Matched only in agent turns that also look like a request.
REQUEST_PATTERNS: dict[str, re.Pattern[str]] = {
    "booking_or_reference_number": re.compile(
        r"\b(booking|vt|ticket|order|collection|claim)\s*(ref|reference|number|no)\b|\breference (number|no)\b|\bbooking ref"
    ),
    "email_address": re.compile(r"\be-?mail\b"),
    "service_or_train_details": re.compile(
        r"\bwhich (service|train)\b|\bwhat (service|train|time)\b|\bservice are you\b|\bwhich station\b|\bwhere are you (travelling|going|heading)\b"
    ),
    "photo_or_screenshot": re.compile(r"\b(photo|picture|screenshot|image)\b"),
    "name_or_contact_details": re.compile(r"\b(your name|full name|phone number|contact number|telephone number|postcode|address)\b"),
    "seat_or_coach": re.compile(r"\b(seat|coach|carriage) (number|no)\b|\bwhich coach\b|\bwhat coach\b"),
    "description_of_problem": re.compile(
        r"\bwhat (issues|issue|problem|seems|happened)\b|\bmore (details|information)\b|\bwhat (is|are) the (issue|problem)\b"
    ),
}
_REQUEST_CUE = re.compile(r"\?|\bplease\b|\bcould you\b|\bcan you\b|\bif you (can|could)\b|\blet us know\b")


def requested_items(agent_messages: list[str]) -> set[str]:
    found: set[str] = set()
    for msg in agent_messages:
        low = " ".join(str(msg).lower().split())
        if not _REQUEST_CUE.search(low):
            continue
        for item, pattern in REQUEST_PATTERNS.items():
            if pattern.search(low):
                found.add(item)
    return found


def required_information(group: pd.DataFrame, min_share: float = 0.05, min_cases: int = 15) -> list[dict[str, Any]]:
    """Items agents asked customers for in this intent's train cases, with counts. Empty -> NEEDS_REVIEW."""
    n = len(group)
    if n == 0:
        return [{"item": NEEDS_REVIEW, "evidence": "no train cases"}]
    counts: dict[str, int] = {}
    for msgs in group["agent_messages"]:
        for item in requested_items(list(msgs)):
            counts[item] = counts.get(item, 0) + 1
    out = [
        {"item": item, "evidence": f"requested by an agent in {c:,} of {n:,} train cases ({100 * c / n:.1f}%)"}
        for item, c in sorted(counts.items(), key=lambda kv: -kv[1])
        if c >= min_cases and c / n >= min_share
    ]
    return out or [{"item": NEEDS_REVIEW, "evidence": f"no request pattern reached {min_cases} cases / {100 * min_share:.0f}% of {n:,} train cases"}]


def historic_escalation_evidence(group: pd.DataFrame, min_share: float = 0.02, min_cases: int = 10) -> list[dict[str, Any]]:
    """Observed agent handling that implies a hand-off, with counts. Describes history, does not set policy."""
    n = len(group)
    signals = {
        "escalated": "agent pointed the customer to Customer Relations or a formal complaint route",
        "redirected_to_dm": "agent asked the customer to move to DM (account or booking lookup needed)",
        "redirected_to_other_operator": "agent handed the case to another train operator",
    }
    out = []
    for sig, text in signals.items():
        c = int(sum(sig in list(s) for s in group["resolution_signals"])) if n else 0
        if n and c >= min_cases and c / n >= min_share:
            out.append(
                {
                    "trigger": f"historic: {text}",
                    "source": "historical_dataset",
                    "evidence": f"seen in {c:,} of {n:,} train cases ({100 * c / n:.1f}%)",
                    "policy": NEEDS_REVIEW,
                }
            )
    return out


def load_labels(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cluster_intent_map(preview: dict, labels: dict) -> dict[int, dict[str, str | None]]:
    """cluster_id -> {cluster_name, final_intent} for the *current* numbering (matched by anchor terms)."""
    from taxonomy.taxonomy_builder import match_labels

    terms = {int(c): v["top_terms"] for c, v in preview["clusters"].items()}
    specs = match_labels(terms, labels["clusters"])
    out: dict[int, dict[str, str | None]] = {
        cid: {"cluster_name": s["name"], "final_intent": s.get("final_intent")} for cid, s in specs.items()
    }
    for sp in labels.get("special", []):
        out[int(sp["cluster_id"])] = {"cluster_name": sp["name"], "final_intent": sp.get("final_intent")}
    out.setdefault(-1, {"cluster_name": "unclear_or_media_only", "final_intent": FALLBACK_INTENT})
    out.setdefault(-3, {"cluster_name": "continuation_no_opening", "final_intent": None})
    out.setdefault(-2, {"cluster_name": "no_cluster", "final_intent": None})
    return out


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_ids_sha256(case_ids) -> str:
    return sha256_text("\n".join(sorted(str(c) for c in case_ids)))


def registry_status(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return (data.get("taxonomy") or {}).get("status")


def intent_names(taxonomy: dict) -> list[str]:
    """Production intent names plus the fallback, from a registry/frozen taxonomy dict (the inner `taxonomy` mapping)."""
    names = [i["name"] for i in taxonomy["intents"]]
    if taxonomy.get("fallback"):
        names.append(taxonomy["fallback"]["name"])
    return names
