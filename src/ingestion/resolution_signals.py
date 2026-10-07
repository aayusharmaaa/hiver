"""Observable-behaviour resolution typing for support cases (rule-based, no LLM).

Each brand-agent turn is scanned for *signals* (what the agent visibly did or said). The
case-level `resolution_type` is the highest-priority signal; `resolved` is a separate,
conservative outcome judgement taken from how the thread ended.

The labels describe what the agent said in public. They never assert that an action was
completed (e.g. "refund" means the agent offered/discussed a refund, not that money moved),
and every label is backed by an extractive evidence snippet from a specific tweet.

Priority (highest first). `redirected_to_dm` applies only when the agent's *last* turn asks to
move to DM, so earlier substantive help is not hidden by a closing "DM us if still broken".

    refund > compensation > account_action > redirected_to_dm > redirected_to_other_operator
    > escalated > troubleshooting > self_service > feedback_acknowledged
    > information_provided > clarification_requested > other

`unresolved` replaces the type when the brand never replied, or when the agent's only
signals are weak (clarification/other) and the customer's last message was a negative or
unanswered follow-up.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from ingestion.resolution import ResolutionType

SIGNAL_PRIORITY: tuple[str, ...] = (
    "refund",
    "compensation",
    "account_action",
    "redirected_to_dm",
    "redirected_to_other_operator",
    "escalated",
    "troubleshooting",
    "self_service",
    "feedback_acknowledged",
    "information_provided",
    "clarification_requested",
    "other",
)
RESOLUTION_TYPES: tuple[str, ...] = SIGNAL_PRIORITY + ("unresolved",)
WEAK_SIGNALS = frozenset({"clarification_requested", "other"})
RESOLVED_OUTCOMES = frozenset({ResolutionType.CUSTOMER_CONFIRMED.value, ResolutionType.AGENT_CLOSED.value})

_LEADING_MENTIONS = re.compile(r"^(?:\s*@\w+)+\s*")
_SIGNOFF = re.compile(r"\s*\^[A-Za-z]{2,3}\s*$")
_PART_MARKER = re.compile(r"^\s*\d+/\d+\s*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_URL = re.compile(r"https?://\S+")

_DM = re.compile(
    r"\b(?:dm|dms|d\.m)\b|direct message|private message|\bpm (?:us|me)\b|message us|"
    r"(?:send|drop|shoot) us a (?:dm|message|note)"
)
_REFUND = re.compile(r"\brefund(?:s|ed|ing)?\b|money back|reimburs")
_COMPENSATION = re.compile(
    r"delay ?repay|compensat|goodwill|\bvouchers?\b|\bclaim(?:s|ed|ing)?\b|\bcredit(?:ed)?\b"
)
_BOOKING_OBJECT = r"(?:ticket|booking|seat|reservation|journey|order|card|account|reference|railcard)s?"
_ACCOUNT_ACTION = re.compile(
    rf"\b(?:i|we)(?:'ve| have|'ll| will)? (?:now )?(?:booked|reserved|amended|changed|cancell?ed|re-?issued|"
    rf"resent|re-?sent|upgraded|arranged|(?:book|reserve|amend|change|cancel|re-?issue|resend|arrange)) "
    rf"[^.?!]{{0,40}}\b{_BOOKING_OBJECT}\b|"
    rf"\byour {_BOOKING_OBJECT} (?:has|have) been (?:booked|reserved|amended|changed|cancell?ed|re-?issued|resent|"
    rf"updated|upgraded|refunded)\b"
)
_ESCALATED = re.compile(
    r"customer relations|escalat|\bchase\b|raise[ds]? (?:this|it|that) with|forwarded|(?:will|'ll|can) forward|"
    r"forward(?:ing)? (?:this|it|that|your)|refer(?:red)? (?:this|you|it)|report(?:ed|ing)? (?:this|it|that) to|"
    r"look(?:ing)? into (?:this|it|that)|(?:team|colleagues) (?:will|are|is|can) (?:look|contact|be in touch)"
)
_FEEDBACK = re.compile(
    r"feedback|pass(?:ed|ing)? (?:this|that|it|your [a-z]+)? ?(?:on|along|back)\b|"
    r"\bnoted\b|thanks for (?:letting|telling) us|thank you for (?:letting|telling) us"
)
_TROUBLESHOOTING = re.compile(
    r"have you tried|\btry (?:restarting|clearing|logging|a different|another|using a|using the|refreshing|"
    r"connecting|reconnecting|turning|switching|again later)|clear (?:your )?(?:cache|cookies|history)|"
    r"\b(?:refresh|restart|reboot|re-?install|reconnect)(?:ing)?\b|different (?:browser|device)|new (?:browser|tab)|"
    r"\bbrowser\b|does (?:it|this) (?:still|work)|\berror (?:message|code)\b|(?:turn|switch) (?:it|your [a-z-]+) "
    r"(?:off|on)|forget the network"
)
_SELF_SERVICE = re.compile(
    r"https?://|\b(?:online form|web ?form|contact form|our website|the website|our app|the app|ticket (?:office|machines?)|"
    r"station team|members? of staff|live updates|journey planner|visit)\b"
)
_CLARIFY = re.compile(
    r"(?:which|what|where|when|who|could you|can you|are you|do you|did you|have you|is it|was it)\b[^.?!]*\?|"
    r"(?:send|provide|confirm|let us know|tell us) (?:us )?(?:your|the|a|what|which)\b"
)
_OPERATOR_HANDOFF = re.compile(
    r"pass(?:ing)?|forward|speak to|contact|operate|their|travell?ing (?:with|on)|run by|responsib|"
    r"best (?:placed|contact)|may be able|can (?:help|confirm|advise)|is a|service"
)
_MENTION = re.compile(r"@(\w+)")


def display_text(raw: str) -> str:
    """Raw tweet text without addressing mentions, part markers (1/2), or the ^XX sign-off."""
    text = html.unescape(raw or "")
    text = _SIGNOFF.sub("", text)
    text = _LEADING_MENTIONS.sub("", text)
    text = _PART_MARKER.sub("", text)
    return " ".join(text.split())


def _snippet(display: str, pattern: re.Pattern[str], limit: int = 220) -> str:
    low = display.lower()
    for sentence in _SENTENCE_SPLIT.split(display):
        if pattern.search(sentence.lower()):
            return sentence if len(sentence) <= limit else sentence[: limit - 1] + "…"
    return display if len(display) <= limit else display[: limit - 1] + "…"


def detect_turn_signals(raw_text: str, other_operator_ids: frozenset[str] = frozenset()) -> dict[str, str]:
    """Return {signal: evidence_snippet} for one brand-agent turn."""
    display = display_text(raw_text)
    low = display.lower()
    found: dict[str, str] = {}

    def add(name: str, pattern: re.Pattern[str]) -> bool:
        if pattern.search(low):
            found[name] = _snippet(display, pattern)
            return True
        return False

    add("redirected_to_dm", _DM)
    add("refund", _REFUND)
    add("compensation", _COMPENSATION)
    add("account_action", _ACCOUNT_ACTION)

    handles = {h.lower() for h in _MENTION.findall(display)}
    if handles & {h.lower() for h in other_operator_ids} and _OPERATOR_HANDOFF.search(low):
        found["redirected_to_other_operator"] = _snippet(display, _OPERATOR_HANDOFF)

    if not add("escalated", _ESCALATED):
        add("feedback_acknowledged", _FEEDBACK)
    add("troubleshooting", _TROUBLESHOOTING)
    add("self_service", _SELF_SERVICE)

    asks_question = "?" in display
    add("clarification_requested", _CLARIFY)
    informative = len(_URL.sub("", display).split()) >= 6 and not asks_question
    if informative and not (set(found) - {"clarification_requested"}):
        found["information_provided"] = display if len(display) <= 220 else display[:219] + "…"
    if not found:
        found["other"] = display if len(display) <= 220 else display[:219] + "…"
    return found


@dataclass(frozen=True)
class CaseResolution:
    resolution_type: str
    resolved: bool
    outcome: str
    dm_redirect: bool
    signals: tuple[str, ...]
    evidence: tuple[dict, ...]
    summary: str


def derive_case_resolution(
    turns: list[dict], outcome: str, other_operator_ids: frozenset[str] = frozenset()
) -> CaseResolution:
    """`turns`: chronological dicts with tweet_id, role ('customer'/'brand_agent'/'other_agent'), text.

    `outcome` is the thread-ending label from `ingestion.resolution.classify_resolution`.
    """
    agent_turns = [t for t in turns if t["role"] == "brand_agent"]
    per_turn = [(t, detect_turn_signals(t["text"], other_operator_ids)) for t in agent_turns]
    all_signals = {s for _, sig in per_turn for s in sig}
    dm_any = any("redirected_to_dm" in sig for _, sig in per_turn)
    other_agent_seen = any(t["role"] == "other_agent" for t in turns)

    if not per_turn:
        suffix = " Another operator's account replied, but the brand did not." if other_agent_seen else ""
        return CaseResolution(
            "unresolved", False, outcome, False, (), (),
            f"Brand never replied in the public thread (outcome: {outcome}).{suffix}",
        )

    last_signals = per_turn[-1][1]
    candidates = set(all_signals)
    if "redirected_to_dm" in candidates and "redirected_to_dm" not in last_signals:
        candidates.discard("redirected_to_dm")
    chosen = next(s for s in SIGNAL_PRIORITY if s in candidates)

    last_customer = next((t for t in reversed(turns) if t["role"] == "customer"), None)
    customer_spoke_last = turns[-1]["role"] == "customer"
    negative_tail = customer_spoke_last and outcome == ResolutionType.CUSTOMER_FOLLOWUP_UNANSWERED.value
    if chosen in WEAK_SIGNALS and negative_tail and last_customer is not None:
        chosen = "unresolved"

    resolved = chosen != "unresolved" and outcome in RESOLVED_OUTCOMES
    ordered_signals = tuple(s for s in SIGNAL_PRIORITY if s in all_signals)
    evidence: list[dict] = []
    for turn, sig in per_turn:
        for name in ordered_signals:
            if name in sig and name != "other":
                evidence.append({"tweet_id": int(turn["tweet_id"]), "signal": name, "snippet": sig[name]})
    evidence = [e for e in evidence if e["signal"] in ({chosen} | {"redirected_to_dm"} if dm_any else {chosen})][:4]
    if chosen == "unresolved" or not evidence:
        t, sig = per_turn[-1]
        name = next(iter(sig))
        evidence = [{"tweet_id": int(t["tweet_id"]), "signal": name, "snippet": sig[name]}]

    quoted = "; ".join(f'[tweet {e["tweet_id"]}, {e["signal"]}] "{e["snippet"]}"' for e in evidence)
    summary = (
        f"Type: {chosen}. Outcome: {outcome}. Brand agent turns: {len(per_turn)}. "
        f"Signals seen: {', '.join(ordered_signals) or 'none'}. Evidence: {quoted}"
    )
    return CaseResolution(chosen, resolved, outcome, dm_any, ordered_signals, tuple(evidence), summary)
