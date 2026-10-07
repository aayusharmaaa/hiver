"""Rule-based resolution labelling for a reconstructed conversation (no LLM).

The labels describe how a public Twitter thread *ended*; they are weak signals, not ground
truth. Many real resolutions happen in DMs we cannot see, which is why "redirected_to_dm"
is a separate outcome rather than "unresolved".

Decision order:
  1. no agent turn                                  -> no_response
  2. customer spoke last (after an agent turn):
       gratitude/confirmation without negation      -> customer_confirmed      (resolved)
       otherwise                                    -> customer_followup_unanswered
  3. agent spoke last:
       closing phrase and no question               -> agent_closed            (resolved)
       asks to move to DM                           -> redirected_to_dm
       points to another channel (url/phone/etc.)   -> redirected_to_channel
       asks the customer a question                 -> agent_awaiting_customer
       otherwise                                    -> agent_answered_unconfirmed
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ingestion.text import normalize_for_matching


class ResolutionType(str, Enum):
    NO_RESPONSE = "no_response"
    CUSTOMER_CONFIRMED = "customer_confirmed"
    CUSTOMER_FOLLOWUP_UNANSWERED = "customer_followup_unanswered"
    AGENT_CLOSED = "agent_closed"
    REDIRECTED_TO_DM = "redirected_to_dm"
    REDIRECTED_TO_CHANNEL = "redirected_to_channel"
    AGENT_AWAITING_CUSTOMER = "agent_awaiting_customer"
    AGENT_ANSWERED_UNCONFIRMED = "agent_answered_unconfirmed"


RESOLVED_TYPES = frozenset({ResolutionType.CUSTOMER_CONFIRMED, ResolutionType.AGENT_CLOSED})

_CUSTOMER_CONFIRM = re.compile(
    r"\b(thanks|thank you|thank u|thx|ty|cheers|appreciate|sorted|fixed|resolved|solved|"
    r"works now|working now|that worked|it worked|all good|got it|perfect|brilliant|awesome)\b"
)
_NEGATION = re.compile(
    r"\b(not|no|never|still|nothing|doesn't|doesnt|didn't|didnt|won't|wont|can't|cant|"
    r"isn't|isnt|useless|worst|terrible|awful|ridiculous|joke)\b|\bfor nothing\b"
)
# "happy to help" / "glad you reached out" are deliberately excluded: agents use them as openers.
_AGENT_CLOSING = re.compile(
    r"(glad (?:we|i) (?:could|were able)|glad (?:to hear|it's|it is|that's|that worked|you got|everything|"
    r"you're all set|you were able)|happy to hear|you're welcome|you are welcome|\banytime\b|"
    r"have a (?:great|good|nice|lovely|wonderful|fantastic)|\benjoy\b|no problem|my pleasure|"
    r"great to hear|good to hear|pleased to hear|we're here if|here if you need|take care|safe travels)"
)
_DM = re.compile(
    r"(\bdm\b|\bd\.m\b|direct message|private message|\bpm\b|send us a (?:dm|message|note)|"
    r"message us|via dm|in dm|over dm|\bdms\b|follow and dm|inbox)"
)
_CHANNEL = re.compile(
    r"(<url>|call us|give us a call|\bphone\b|contact (?:us|our)|\bvisit\b|support page|"
    r"\bemail\b|live chat|chat with|help ?cent(?:er|re)|1-?800|\bhotline\b)"
)


@dataclass(frozen=True)
class Resolution:
    resolved: bool
    resolution_type: ResolutionType
    resolution_summary: str


def _truncate(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def classify_resolution(roles: list[str], texts: list[str]) -> Resolution:
    """`roles` are "customer"/"agent" in turn order; `texts` are the raw turn texts."""
    n_customer = sum(r == "customer" for r in roles)
    n_agent = len(roles) - n_customer
    counts = f"{n_customer} customer / {n_agent} agent turn(s)"

    if n_agent == 0:
        rtype = ResolutionType.NO_RESPONSE
        summary = f"{counts}; brand never replied publicly."
        return Resolution(False, rtype, summary)

    last_agent_idx = max(i for i, r in enumerate(roles) if r == "agent")
    last_agent_text = texts[last_agent_idx]
    tail = f'Final agent reply: "{_truncate(last_agent_text)}"'

    if roles[-1] == "customer":
        last_customer = normalize_for_matching(texts[-1])
        if _CUSTOMER_CONFIRM.search(last_customer) and not _NEGATION.search(last_customer):
            rtype = ResolutionType.CUSTOMER_CONFIRMED
            summary = f"{counts}; customer confirmed/thanked after agent reply. {tail}"
        else:
            rtype = ResolutionType.CUSTOMER_FOLLOWUP_UNANSWERED
            summary = f"{counts}; customer's last follow-up got no public reply. {tail}"
    else:
        last = normalize_for_matching(last_agent_text)
        asks_question = "?" in last
        if _AGENT_CLOSING.search(last) and not asks_question:
            rtype = ResolutionType.AGENT_CLOSED
            summary = f"{counts}; agent closed the thread. {tail}"
        elif _DM.search(last):
            rtype = ResolutionType.REDIRECTED_TO_DM
            summary = f"{counts}; agent moved the conversation to DM. {tail}"
        elif _CHANNEL.search(last):
            rtype = ResolutionType.REDIRECTED_TO_CHANNEL
            summary = f"{counts}; agent pointed to another channel/resource. {tail}"
        elif asks_question:
            rtype = ResolutionType.AGENT_AWAITING_CUSTOMER
            summary = f"{counts}; agent asked for details, customer never replied publicly. {tail}"
        else:
            rtype = ResolutionType.AGENT_ANSWERED_UNCONFIRMED
            summary = f"{counts}; agent replied last, no customer confirmation. {tail}"

    return Resolution(rtype in RESOLVED_TYPES, rtype, summary)
