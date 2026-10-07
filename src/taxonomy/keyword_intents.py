"""Seed keyword lexicon for coarse customer intents (no LLM, English only).

This is *not* the agent's intent classifier. It exists only to estimate how diverse a
brand's incoming issues are, so brands can be compared before a data-driven taxonomy is
built for the chosen brand. Rules are checked in priority order; first match wins.
"""

from __future__ import annotations

import re
from collections import Counter
from math import log

from ingestion.text import normalize_for_matching

OTHER = "other"

SEED_INTENTS: tuple[tuple[str, str], ...] = (
    ("account_access", r"log ?in|sign ?in|password|locked out|hacked|verification code|2fa|two.factor|can'?t access|reset my"),
    ("cancellation", r"\bcancel(?:l?ed|l?ing|lation)?\b|unsubscribe|close my account|terminate"),
    ("refund_compensation", r"refund|money back|reimburse|compensat|chargeback|credit back"),
    ("billing_payment", r"\bbill(?:ed|ing)?\b|charged|\bcharge\b|payment|\bpaid\b|invoice|overcharg|\bfee\b|subscription|price"),
    ("order_delivery", r"\border\b|deliver|package|parcel|shipping|shipped|tracking|courier|arriv|dispatch"),
    ("travel_disruption", r"flight|delay|\btrain\b|luggage|baggage|boarding|\bgate\b|\bseat\b|connection flight|missed (?:my )?connection|platform"),
    ("connectivity_outage", r"outage|no signal|no service|internet|wi-?fi|\bnetwork\b|\bdata\b|\b[345]g\b|slow speed|\bdown\b"),
    ("technical_issue", r"not working|doesn'?t work|won'?t (?:load|open|work)|error|crash|\bbug\b|broken|glitch|freez|update|battery|keeps"),
    ("how_to_question", r"^(?:how|what|when|where|why|can i|can you|do you|is there|will)\b|\?$"),
    ("complaint", r"worst|terrible|awful|disappoint|rude|ridiculous|unacceptable|useless|disgust|shocking|joke"),
    ("praise", r"\blove\b|thank|great service|amazing|awesome|brilliant|best"),
)

_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple((name, re.compile(p)) for name, p in SEED_INTENTS)
INTENT_NAMES: tuple[str, ...] = tuple(name for name, _ in SEED_INTENTS)


def classify_intent(text: str | None) -> str:
    if not text:
        return OTHER
    norm = normalize_for_matching(text)
    for name, pattern in _COMPILED:
        if pattern.search(norm):
            return name
    return OTHER


def normalized_entropy(labels: list[str], exclude: str | None = OTHER) -> float:
    """Shannon entropy of the label distribution divided by log(#possible labels), in [0, 1]."""
    counts = Counter(l for l in labels if l != exclude)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((c / total) * log(c / total) for c in counts.values())
    return entropy / log(len(INTENT_NAMES))
