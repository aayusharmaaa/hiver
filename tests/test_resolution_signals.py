from __future__ import annotations

import pytest

from ingestion.resolution_signals import derive_case_resolution, detect_turn_signals, display_text

OTHER = frozenset({"120576"})


def turns(*items: tuple[int, str, str]) -> list[dict]:
    return [{"tweet_id": t, "role": r, "text": x} for t, r, x in items]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("@c1 Could you DM us your booking reference please?", "redirected_to_dm"),
        ("@c1 Delay Repay compensation is available to those delayed by over 30 minutes.", "compensation"),
        ("@c1 Live Updates can be found via https://t.co/abc", "self_service"),
        ("@c1 Please forget the network and reconnect to the WiFi, does that help?", "troubleshooting"),
        ("@c1 This can be escalated further with Customer Relations via the online complaints link", "escalated"),
        ("@c1 Which service are you on today?", "clarification_requested"),
    ],
)
def test_expected_signal_detected_with_evidence_from_the_text(text: str, expected: str) -> None:
    signals = detect_turn_signals(text, OTHER)
    assert expected in signals
    snippet = signals[expected].rstrip("…")
    assert snippet in display_text(text), "evidence must be quoted from the tweet, not invented"


@pytest.mark.parametrize(
    "text, forbidden",
    [
        ("@c1 We look forward to welcoming you onboard", "escalated"),
        ("@c1 We'll try to work it out for you", "troubleshooting"),
        ("@c1 We'll update you as soon as we can", "account_action"),
    ],
)
def test_known_false_positive_phrases_do_not_fire(text: str, forbidden: str) -> None:
    assert forbidden not in detect_turn_signals(text, OTHER)


def test_hand_off_to_other_operator_requires_their_handle() -> None:
    with_handle = detect_turn_signals("@c1 The @120576 team may be able to advise on this", OTHER)
    assert "redirected_to_other_operator" in with_handle
    without = detect_turn_signals("@c1 The @someoneelse team may be able to advise on this", OTHER)
    assert "redirected_to_other_operator" not in without


def test_sign_off_and_leading_mentions_are_stripped_from_display_text() -> None:
    assert display_text("@c1 @c2 Sorry for the delay ^AB") == "Sorry for the delay"


def test_no_brand_reply_is_unresolved_and_says_so() -> None:
    res = derive_case_resolution(turns((1, "customer", "@Brand where is my train")), "no_response")
    assert res.resolution_type == "unresolved" and not res.resolved and not res.dm_redirect
    assert "never replied" in res.summary


def test_other_operator_reply_without_brand_reply_is_noted_not_credited() -> None:
    res = derive_case_resolution(
        turns((1, "customer", "@Brand help"), (2, "other_agent", "@c1 we can help")), "customer_followup_unanswered"
    )
    assert res.resolution_type == "unresolved"
    assert "Another operator" in res.summary


def test_dm_request_is_flagged_even_when_other_signals_win() -> None:
    res = derive_case_resolution(
        turns(
            (1, "customer", "@Brand my seat was taken"),
            (2, "brand_agent", "@c1 Please DM us your booking reference"),
            (3, "customer", "@Brand sent"),
            (4, "brand_agent", "@c1 You can claim compensation via https://t.co/x for the lost reservation"),
        ),
        "agent_answered_unconfirmed",
    )
    assert res.dm_redirect is True
    assert res.resolution_type == "compensation"


def test_resolved_requires_a_positive_outcome_and_is_conservative() -> None:
    base = (
        (1, "customer", "@Brand where can I claim"),
        (2, "brand_agent", "@c1 Please claim via our Delay Repay form https://t.co/x"),
    )
    answered = derive_case_resolution(turns(*base), "agent_answered_unconfirmed")
    confirmed = derive_case_resolution(turns(*base, (3, "customer", "@Brand thanks")), "customer_confirmed")
    assert answered.resolution_type == confirmed.resolution_type == "compensation"
    assert not answered.resolved, "an unconfirmed answer must not count as resolved"
    assert confirmed.resolved


def test_evidence_always_cites_a_real_brand_turn() -> None:
    items = turns(
        (7, "customer", "@Brand wifi broken"),
        (8, "brand_agent", "@c1 Have you tried disconnecting from the WiFi and reconnecting?"),
    )
    res = derive_case_resolution(items, "agent_answered_unconfirmed")
    assert res.evidence and all(e["tweet_id"] == 8 for e in res.evidence)
