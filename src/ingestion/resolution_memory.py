"""Resolution memory: one record per historical support-resolution episode (not per tweet).

Each record keeps what the customer asked for (the problem as stated before the first brand reply), the brand
reply that carried the resolution, a compact summary of what the agent did, and provenance back to the tweets.

Everything here is rule-based and extractive. The `intent` is the *candidate* taxonomy intent (clustering of opening
messages, `CANDIDATE_NOT_GROUND_TRUTH`), and `escalation_signal` describes what historical agents *did*, not a policy.
No LLM is involved.

Cases without a useful brand response are kept in the file but marked `evidence_quality != "strong"` and
`in_primary_corpus = False`, so they cannot become positive resolution evidence.
"""

from __future__ import annotations

import html
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ingestion.resolution_signals import display_text
from taxonomy.registry import FALLBACK_INTENT

INTENT_SOURCE = "candidate_taxonomy_not_ground_truth"
ESCALATION_SOURCE = "historical_agent_behaviour"

STRONG = "strong"
WEAK = "weak"
NO_REPLY = "no_brand_reply"

# Types that, on their own, do not show the agent resolving or answering anything.
WEAK_TYPES = frozenset({"other", "clarification_requested", "unresolved"})

MAX_PROBLEM_CHARS = 500
MAX_RESPONSE_CHARS = 420
MAX_RESPONSE_TURNS = 2
MIN_PROBLEM_WORDS = 3
MIN_RESPONSE_WORDS = 4

_URL = re.compile(r"https?://\S+")
_SPACES = re.compile(r"\s+")
_THANKS_ONLY = re.compile(r"^(?:ok(?:ay)?|thanks?|thank you|cheers|ta|great|brilliant|perfect|thx)\b[\s\w!.,:)(-]{0,25}$", re.IGNORECASE)

PHRASE = {
    "refund": "Agent discussed a refund",
    "compensation": "Agent discussed compensation or Delay Repay",
    "account_action": "Agent said it changed or issued something on the booking",
    "redirected_to_dm": "Agent asked the customer to continue in DM",
    "redirected_to_other_operator": "Agent pointed the customer to another operator",
    "escalated": "Agent referred or escalated the case",
    "troubleshooting": "Agent gave troubleshooting steps",
    "self_service": "Agent pointed to a self-service route (link, form or station staff)",
    "feedback_acknowledged": "Agent acknowledged and passed on the feedback",
    "information_provided": "Agent provided information",
    "clarification_requested": "Agent asked a clarifying question",
    "other": "Agent replied without a clear resolution action",
    "unresolved": "No resolving reply in the public thread",
}
OUTCOME = {
    "customer_confirmed": "customer confirmed it helped",
    "agent_closed": "agent closed the thread",
    "agent_answered_unconfirmed": "agent answered, no customer confirmation",
    "customer_followup_unanswered": "customer followed up and was not answered",
    "redirected_to_channel": "moved to another channel",
    "redirected_to_dm": "moved to DM",
    "agent_awaiting_customer": "agent was waiting for the customer",
    "no_response": "no brand reply",
}

# Historical agent behaviour -> a small vocabulary. "none" means no hand-off was visible, not that none is needed.
ESCALATION_ORDER = (
    ("escalated", "customer_relations_or_formal_route"),
    ("redirected_to_other_operator", "handoff_to_other_operator"),
    ("redirected_to_dm", "dm_for_account_lookup"),
)

MEMORY_COLUMNS = [
    "case_id", "split", "conversation_id", "intent", "intent_source", "cluster_id",
    "customer_problem", "historical_response", "resolution_summary", "resolution_type", "resolution_outcome",
    "resolved", "dm_redirect", "escalation_signal", "escalation_signal_source",
    "evidence_quality", "in_primary_corpus", "exclusion_reason",
    "source_tweet_ids", "response_tweet_ids", "context_tweet_ids", "n_brand_turns", "first_timestamp",
]


def _words(text: str) -> int:
    return len(text.split())


def strip_urls(text: str) -> str:
    return _SPACES.sub(" ", _URL.sub(" ", text)).strip()


def clip(text: str, limit: int) -> str:
    """Clip at a word boundary with an ellipsis; never mid-word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut[limit // 2 :]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:-") + "…"


def customer_problem_from_turns(turns: Iterable[Mapping[str, Any]], max_chars: int = MAX_PROBLEM_CHARS) -> str:
    """The problem as the customer stated it: all customer turns *before the first brand reply* (urls and mentions removed).

    That is exactly the information available when a new message arrives, so queries and memory are built the same way.
    """
    parts: list[str] = []
    for turn in turns:
        if turn["role"] == "brand_agent":
            break
        if turn["role"] != "customer":
            continue
        text = strip_urls(display_text(str(turn["text"])))
        if text:
            parts.append(text)
    if not parts:  # the brand spoke first (rare): fall back to the first customer turn
        first = next((t for t in turns if t["role"] == "customer"), None)
        if first is not None:
            text = strip_urls(display_text(str(first["text"])))
            parts = [text] if text else []
    return clip(" ".join(parts), max_chars)


def select_response_turns(turns: list[Mapping[str, Any]], evidence: Iterable[Mapping[str, Any]], resolution_type: str) -> list[Mapping[str, Any]]:
    """Brand turns that carried the resolution: evidence turns for the chosen type first, else the first brand reply."""
    brand = [t for t in turns if t["role"] == "brand_agent"]
    if not brand:
        return []
    evidence = list(evidence)
    wanted = [int(e["tweet_id"]) for e in evidence if e.get("signal") == resolution_type] or [int(e["tweet_id"]) for e in evidence]
    chosen = [t for t in brand if int(t["tweet_id"]) in set(wanted)]
    if not chosen:
        chosen = brand[:1]
    useful = [t for t in chosen if _words(strip_urls(display_text(str(t["text"])))) >= MIN_RESPONSE_WORDS or _URL.search(str(t["text"]))]
    return (useful or chosen)[:MAX_RESPONSE_TURNS]


def historical_response_from_turns(chosen: list[Mapping[str, Any]], max_chars: int = MAX_RESPONSE_CHARS) -> str:
    """Original wording of the selected brand reply(ies); sign-offs and leading mentions removed, links kept."""
    texts = [display_text(str(t["text"])) for t in chosen]
    texts = [t for t in texts if t]
    if not texts:
        return ""
    per_turn = max_chars // len(texts)
    return " | ".join(clip(t, per_turn) for t in texts)


def escalation_signal(signals: Iterable[str]) -> str:
    present = set(signals)
    for signal, label in ESCALATION_ORDER:
        if signal in present:
            return label
    return "none"


def summarise(resolution_type: str, outcome: str, evidence: Iterable[Mapping[str, Any]], dm_redirect: bool) -> str:
    snippet = next((str(e["snippet"]) for e in evidence if e.get("signal") == resolution_type and e.get("snippet")), "")
    text = f"{PHRASE.get(resolution_type, resolution_type)}; {OUTCOME.get(outcome, outcome.replace('_', ' '))}."
    if dm_redirect and resolution_type != "redirected_to_dm" and "dm" not in outcome:
        text += " The agent also asked to move to DM."
    if snippet:
        text += f' Key line: "{clip(html.unescape(snippet), 170)}"'
    return text


def _classify(problem: str, response: str, n_brand: int, rtype: str, intent: str | None, is_continuation: bool, starts_with_customer: bool) -> tuple[str, str]:
    """(evidence_quality, exclusion_reason). Empty reason means the case is a strong positive example."""
    if n_brand == 0:
        return NO_REPLY, "brand_never_replied"
    if is_continuation or not starts_with_customer:
        return WEAK, "continuation_without_opening_problem"
    if intent is None:
        return WEAK, "no_candidate_intent"
    if intent == FALLBACK_INTENT:
        return WEAK, "fallback_intent_problem_unclear"
    if _words(problem) < MIN_PROBLEM_WORDS or _THANKS_ONLY.match(problem or ""):
        return WEAK, "uninformative_customer_problem"
    if rtype in WEAK_TYPES:
        return WEAK, f"weak_resolution_type_{rtype}"
    if _words(strip_urls(response)) < MIN_RESPONSE_WORDS and not _URL.search(response):
        return WEAK, "response_too_short"
    return STRONG, ""


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    return list(value)


def build_memory_records(
    cases: pd.DataFrame,
    intent_of_cluster: Mapping[int, str | None],
    split_of: Mapping[str, str],
    cluster_of: Mapping[str, int],
) -> pd.DataFrame:
    """Build memory records for `cases` (any split). Callers decide which splits may be written/retrieved."""
    rows: list[dict[str, Any]] = []
    for case in cases.itertuples(index=False):
        turns = [dict(t) for t in case.full_turns]
        evidence = [dict(e) for e in _as_list(case.resolution_evidence)]
        rtype = str(case.resolution_type)
        cluster_id = int(cluster_of[case.case_id])
        intent = intent_of_cluster.get(cluster_id)
        chosen = select_response_turns(turns, evidence, rtype)
        problem = customer_problem_from_turns(turns)
        response = historical_response_from_turns(chosen)
        quality, reason = _classify(problem, response, int(case.agent_turn_count), rtype, intent, bool(case.is_continuation), bool(case.starts_with_customer))
        rows.append(
            {
                "case_id": case.case_id,
                "split": split_of[case.case_id],
                "conversation_id": case.conversation_id,
                "intent": intent,
                "intent_source": INTENT_SOURCE,
                "cluster_id": cluster_id,
                "customer_problem": problem,
                "historical_response": response,
                "resolution_summary": summarise(rtype, str(case.resolution_outcome), evidence, bool(case.dm_redirect)),
                "resolution_type": rtype,
                "resolution_outcome": str(case.resolution_outcome),
                "resolved": bool(case.resolved),
                "dm_redirect": bool(case.dm_redirect),
                "escalation_signal": escalation_signal(_as_list(case.resolution_signals)),
                "escalation_signal_source": ESCALATION_SOURCE,
                "evidence_quality": quality,
                "in_primary_corpus": quality == STRONG,
                "exclusion_reason": reason,
                "source_tweet_ids": [int(x) for x in _as_list(case.source_tweet_ids)],
                "response_tweet_ids": [int(t["tweet_id"]) for t in chosen],
                "context_tweet_ids": [int(x) for x in _as_list(case.context_tweet_ids)],
                "n_brand_turns": int(case.agent_turn_count),
                "first_timestamp": case.first_timestamp,
            }
        )
    return pd.DataFrame(rows, columns=MEMORY_COLUMNS)
