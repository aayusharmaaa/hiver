"""Deterministic risk policy: AUTO_HANDLE or ESCALATE. No LLM is involved, and every decision lists its reasons.

Conservative by construction: the agent auto-handles only when *every* check passes; a single failed check escalates.
All thresholds come from `configs/support_agent.yaml` (`PolicySettings`).

Decision.confidence: for AUTO_HANDLE, the weakest of the supporting signals (classifier confidence, best similarity, intent agreement);
for ESCALATE, 1.0 when a hard rule fired (never-auto intent, sensitive wording, multi-intent, low-information message),
otherwise 1 - weakest signal (a borderline shortfall gives a low-confidence escalation).
"""

from __future__ import annotations

import re
from collections import Counter

from agent.config import PolicySettings
from agent.schemas import AUTO_HANDLE, ESCALATE, Classification, Decision, Evidence

_NOISE = re.compile(r"https?://\S+|@\w+|#\w+")
_WORD = re.compile(r"[A-Za-z0-9£']+")


def count_words(text: str | None) -> int:
    return len(_WORD.findall(_NOISE.sub(" ", text or "")))


def sensitive_matches(message: str, patterns: list[str]) -> list[str]:
    """The distinct words that triggered any sensitive pattern (case-insensitive)."""
    found: list[str] = []
    for pattern in patterns:
        for m in re.finditer(pattern, message, flags=re.IGNORECASE):
            word = m.group(0).lower()
            if word not in found:
                found.append(word)
    return found


def usable_evidence(evidence: list[Evidence], settings: PolicySettings) -> list[Evidence]:
    """Cases similar enough to the message AND with a meaningful historical response."""
    return [
        e
        for e in evidence
        if e.semantic_similarity is not None
        and e.semantic_similarity >= settings.min_usable_similarity
        and count_words(e.historical_response) >= settings.min_response_words
    ]


def decide(
    classification: Classification,
    evidence: list[Evidence],
    message: str,
    settings: PolicySettings,
    *,
    low_information_query: bool = False,
) -> Decision:
    hard: list[str] = []  # explicit escalation triggers
    soft: list[str] = []  # shortfalls in the strength of the evidence

    # ---- explicit triggers ---------------------------------------------------------------------------------------
    if low_information_query or count_words(message) < settings.min_message_words:
        hard.append(f"low-information message (fewer than {settings.min_message_words} content words)")
    if classification.intent in settings.escalate_intents:
        hard.append(f"intent '{classification.intent}' is never auto-handled: {settings.escalate_intents[classification.intent]}")
    if classification.multi_intent:
        hard.append("message appears to contain more than one request (multi-intent)")
    words = sensitive_matches(message, settings.sensitive_patterns)
    if words:
        hard.append(f"potentially sensitive or exceptional wording: {', '.join(words[:5])}")

    # ---- confidence ----------------------------------------------------------------------------------------------
    if classification.confidence < settings.min_classifier_confidence:
        soft.append(f"classifier confidence {classification.confidence:.2f} < {settings.min_classifier_confidence:.2f}")

    # ---- retrieval strength and evidence quality -----------------------------------------------------------------
    usable = usable_evidence(evidence, settings)
    top_sim = max((e.semantic_similarity for e in evidence if e.semantic_similarity is not None), default=0.0)
    intent_agreement = (sum(e.intent == classification.intent for e in usable) / len(usable)) if usable else 0.0
    if not evidence:
        soft.append("no historical evidence was retrieved")
    else:
        if top_sim < settings.min_top_similarity:
            soft.append(f"weak retrieval: best similarity {top_sim:.2f} < {settings.min_top_similarity:.2f}")
        if len(usable) < settings.min_usable_evidence:
            soft.append(f"insufficient evidence: {len(usable)} usable case(s) < {settings.min_usable_evidence}")
    if usable:
        if intent_agreement < settings.min_intent_agreement:
            soft.append(f"evidence disagrees with the predicted intent: {intent_agreement:.0%} of usable cases share it (< {settings.min_intent_agreement:.0%})")
        consistency = Counter(e.resolution_type for e in usable).most_common(1)[0][1] / len(usable)
        if consistency < settings.min_resolution_consistency:
            soft.append(f"historical resolutions are inconsistent: most common type covers {consistency:.0%} of usable cases (< {settings.min_resolution_consistency:.0%})")
        needs_human = sum(e.escalation_signal in settings.human_required_signals for e in usable) / len(usable)
        if needs_human >= settings.escalate_human_signal_share:
            soft.append(f"{needs_human:.0%} of similar historical cases needed account access or a formal route that this agent does not have")

    weakest = min(classification.confidence, max(0.0, min(1.0, top_sim)), intent_agreement if usable else 0.0)
    if hard or soft:
        return Decision(action=ESCALATE, confidence=1.0 if hard else round(1.0 - weakest, 3), reasons=hard + soft)

    reasons = [
        f"classifier confidence {classification.confidence:.2f} >= {settings.min_classifier_confidence:.2f}",
        f"{len(usable)} usable evidence cases (best similarity {top_sim:.2f}); {intent_agreement:.0%} share the predicted intent",
        "no explicit escalation trigger (intent, multi-intent, sensitive wording, low-information)",
    ]
    return Decision(action=AUTO_HANDLE, confidence=round(weakest, 3), reasons=reasons)
