from __future__ import annotations

import pandas as pd

from evaluation.brand_ranking import RankingConfig, compute_brand_metrics, score_brands
from ingestion.text import template_key
from taxonomy.keyword_intents import classify_intent


def _cases(brand: str, n: int, multi: bool, text: str, resolution: str) -> list[dict]:
    return [
        {
            "case_id": f"{brand}_{i}",
            "brand": brand,
            "turn_count": 4 if multi else 2,
            "is_reconstructable": True,
            "is_multi_party": False,
            "is_multi_turn": multi,
            "has_missing_parent": False,
            "resolved": resolution == "agent_closed",
            "resolution_type": resolution,
            "first_customer_message": text,
            "first_response_minutes": 3.0,
        }
        for i in range(n)
    ]


def test_ranking_prefers_multi_turn_resolved_brand() -> None:
    cases = pd.DataFrame(
        _cases("Good", 20, True, "@Good my order never arrived", "agent_closed")
        + _cases("Meh", 20, False, "@Meh hi", "redirected_to_dm")
        + _cases("Tiny", 2, True, "@Tiny refund please", "agent_closed")
    )
    agent = pd.DataFrame({"author_id": ["Good"] * 10 + ["Meh"] * 10, "text": ["@x Glad we could help! ^AB"] * 10 + ["@y hello"] * 10})
    cfg = RankingConfig(min_conversations=10, template_min_repeats=5)
    ranked = score_brands(compute_brand_metrics(cases, agent, cfg), cfg)
    assert ranked.iloc[0]["brand"] == "Good"
    tiny = ranked.set_index("brand").loc["Tiny"]
    assert not tiny["eligible"] and "conversations" in tiny["ineligible_reason"]


def test_template_key_masks_mentions_digits_and_signatures() -> None:
    a = template_key("@user1 Please DM us your order #123 ^KC")
    b = template_key("@other Please DM us your order #456 -JS")
    assert a == b


def test_seed_intents() -> None:
    assert classify_intent("@Brand I can't log in to my account") == "account_access"
    assert classify_intent("@Delta my flight is delayed 3 hours") == "travel_disruption"
    assert classify_intent("@Brand") == "other"
