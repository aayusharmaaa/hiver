"""Decision support shown next to a case in the labeling tool.

Everything here is *evidence already in the registry*: definitions, criteria, confusable intents, what agents were seen to ask
for, and historical (train-split) examples. It never ranks, recommends, or fills in a label, and it never says a merge or a
split is warranted. Those are decisions for the human reviewer, made later from the comparison report.
"""

from __future__ import annotations

import html
from typing import Any

from evaluation.calibration_guide import ESCALATION_DEFINITIONS, RESOLUTION_DEFINITIONS

NEEDS_REVIEW = "NEEDS_REVIEW"
MAX_EXAMPLES = 4
MAX_TEXT = 240


def _clip(text: object, n: int = MAX_TEXT) -> str:
    t = " ".join(html.unescape(str(text)).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _criteria(items: list) -> list[str]:
    return [str(x) for x in items if x != NEEDS_REVIEW]


def intent_card(intent: dict[str, Any]) -> dict[str, Any]:
    confusable = [
        {
            "intent": c["intent"],
            "evidence": c.get("evidence", ""),
            "distinguishing_note": "" if c.get("distinguishing_note", NEEDS_REVIEW) == NEEDS_REVIEW else c["distinguishing_note"],
        }
        for c in intent.get("known_confusable_intents", [])
    ]
    required = [
        {"item": r["item"], "evidence": r.get("evidence", "")} for r in intent.get("required_information", []) if r.get("item") != NEEDS_REVIEW
    ]
    historical = [t for t in intent.get("escalation_triggers", []) if t.get("source") == "historical_dataset"]
    drafted = [t for t in intent.get("escalation_triggers", []) if t.get("source") not in ("historical_dataset", "none") and t.get("trigger") != NEEDS_REVIEW]
    positives = [e for e in intent.get("positive_examples", []) if isinstance(e, dict)][:MAX_EXAMPLES]
    negatives = [e for e in intent.get("negative_examples", []) if isinstance(e, dict)][:MAX_EXAMPLES]
    return {
        "name": intent["name"],
        "is_fallback": bool(intent.get("fallback")),
        "definition": " ".join(str(intent["definition"]).split()),
        "fits_when": _criteria(intent.get("inclusion_criteria", [])),
        "does_not_fit_when": _criteria(intent.get("exclusion_criteria", [])),
        "confusable_intents": confusable,
        "required_information": required,
        "historical_escalation_evidence": [{"what": t["trigger"].replace("historic: ", ""), "evidence": t.get("evidence", "")} for t in historical],
        "reviewer_drafted_escalation_triggers": [t["trigger"] for t in drafted],
        "resolution_types_in_train": intent.get("resolution_types", {}),
        "historical_examples": [{"case_id": e["case_id"], "text": _clip(e["text"])} for e in positives],
        "near_miss_examples": [
            {"case_id": e["case_id"], "text": _clip(e["text"]), "closer_to": e.get("belongs_to", "")} for e in negatives
        ],
        "n_train_cases": intent.get("n_cases"),
    }


def build_reference(taxonomy: dict[str, Any], confusable_notes: list[dict]) -> dict[str, Any]:
    """Neutral reference material for all intents (alphabetical, so it does not hint at the candidate)."""
    intents = list(taxonomy["intents"]) + ([taxonomy["fallback"]] if taxonomy.get("fallback") else [])
    return {
        "intents": sorted((intent_card(i) for i in intents), key=lambda c: (c["is_fallback"], c["name"])),
        "confusable_notes": [{"pair": n["pair"], "note": " ".join(str(n["note"]).split())} for n in confusable_notes],
        "resolution_types": dict(RESOLUTION_DEFINITIONS),
        "escalation_signals": dict(ESCALATION_DEFINITIONS),
        "disclaimer": (
            "Evidence from the train split and the candidate registry. It describes what the data shows; it is not a recommendation, "
            "and escalation handling policy is NEEDS_REVIEW."
        ),
    }
