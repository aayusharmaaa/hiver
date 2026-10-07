"""Assemble the candidate cluster report and draft taxonomy from clusters + curated labels.

Statistics use train+dev cases (clusters were fit on them). Every *example text* comes from the
train split only, so the output can later be used for retrieval/prompting without touching
dev or golden data.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from ingestion.resolution_signals import display_text
from ingestion.text import template_key
from taxonomy.registry import (
    FALLBACK_INTENT,
    NEEDS_REVIEW,
    STATUS_CANDIDATE,
    historic_escalation_evidence,
    required_information,
)

logger = logging.getLogger(__name__)

STAT_SPLITS = ("train_retrieval", "dev_calibration")
EXAMPLE_SPLIT = "train_retrieval"
SPECIAL_NAMES = {-1: "unclear_or_media_only", -3: "continuation_no_opening"}


def match_labels(current_terms: dict[int, list[str]], specs: list[dict], depth: int = 12, min_score: float = 0.35) -> dict[int, dict]:
    """Greedy one-to-one match of clusters to curated specs by anchor-term coverage."""
    scored = []
    for cid, terms in current_terms.items():
        top = set(terms[:depth])
        for i, spec in enumerate(specs):
            anchors = set(spec["anchor_terms"])
            scored.append((len(anchors & top) / len(anchors), cid, i))
    scored.sort(reverse=True)
    out: dict[int, dict] = {}
    used: set[int] = set()
    for score, cid, i in scored:
        if cid in out or i in used or score < min_score:
            continue
        out[cid] = {**specs[i], "match_score": round(score, 2)}
        used.add(i)
    for cid in current_terms:
        if cid not in out:
            logger.warning("Cluster %s has no curated label; marking NEEDS_REVIEW", cid)
            out[cid] = {
                "name": f"cluster_{cid}_NEEDS_REVIEW", "final_intent": f"cluster_{cid}_NEEDS_REVIEW",
                "description": "NEEDS_REVIEW: no curated label matched this cluster.",
                "escalation_triggers": ["NEEDS_REVIEW"], "auto_handle_hint": "unclear", "match_score": 0.0,
            }
    for cid, spec in out.items():
        review = spec.get("cluster_id_at_review")
        if review is not None and int(review) != int(cid):
            logger.info("Cluster numbering changed since review: label %r reviewed as %s, now %s", spec["name"], review, cid)
    return out


def _clip(text: str, n: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _first_reply_patterns(group: pd.DataFrame, top: int = 5) -> list[dict]:
    keyed: dict[str, list[str]] = {}
    for msgs in group["agent_messages"]:
        if len(msgs):
            key = template_key(msgs[0], 6).replace("we're", "we are").replace("we ve", "we have")
            if key:
                keyed.setdefault(key, []).append(msgs[0])
    ranked = sorted(keyed.items(), key=lambda kv: -len(kv[1]))[:top]
    return [
        {"template": key, "count": len(v), "example": _clip(display_text(Counter(v).most_common(1)[0][0]), 160)}
        for key, v in ranked
        if len(v) >= 3
    ]


def _mix(group: pd.DataFrame, top: int = 6) -> dict[str, float]:
    return {k: round(float(v), 3) for k, v in group["resolution_type"].value_counts(normalize=True).head(top).items()}


def build_cluster_table(cases: pd.DataFrame, clusters: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """One frame with case text, behaviour, split and cluster diagnostics."""
    keep = ["case_id", "opening_message", "resolved", "dm_redirect", "resolution_type", "resolution_signals", "agent_messages", "turn_count"]
    df = clusters.merge(cases[keep], on="case_id").merge(assignments[["case_id", "split"]], on="case_id")
    return df


def build_cluster_report(table: pd.DataFrame, specs: dict[int, dict], examples: int = 5) -> list[dict]:
    names = {cid: s["name"] for cid, s in specs.items()} | SPECIAL_NAMES
    pop = table[table["split"].isin(STAT_SPLITS) & (table["cluster_id"] >= 0) & table["in_fit"]]
    total = len(pop)
    report = []
    for cid in sorted(specs):
        members = pop[pop["cluster_id"] == cid]
        train = members[members["split"] == EXAMPLE_SPLIT]
        threshold = members["margin"].quantile(0.25) if len(members) else 0
        low_margin = members[members["margin"] <= threshold]
        neighbors = (
            low_margin["second_cluster_id"].value_counts(normalize=True).head(3)
        )
        spec = specs[cid]
        report.append(
            {
                "cluster_id": int(cid),
                "proposed_intent_name": spec["name"],
                "recommended_final_intent": spec.get("final_intent"),
                "description": spec["description"],
                "label_match_score": spec.get("match_score"),
                "n_cases": len(members),
                "share_of_clustered": round(len(members) / total, 4) if total else 0.0,
                "pct_resolved": round(float(members["resolved"].mean()), 4) if len(members) else None,
                "pct_dm_redirect": round(float(members["dm_redirect"].mean()), 4) if len(members) else None,
                "resolution_type_mix": _mix(members),
                "representative_examples": [
                    {"case_id": r.case_id, "text": _clip(r.opening_message)} for r in train.nsmallest(examples, "distance").itertuples()
                ],
                "ambiguous_examples": [
                    {
                        "case_id": r.case_id,
                        "text": _clip(r.opening_message),
                        "also_close_to": f"{int(r.second_cluster_id)}:{names.get(int(r.second_cluster_id), '?')}",
                        "margin": round(float(r.margin), 4),
                    }
                    for r in train.nsmallest(examples, "margin").itertuples()
                ],
                "confusable_neighbors": [
                    {"cluster_id": int(k), "name": names.get(int(k), "?"), "share_of_low_margin_cases": round(float(v), 3)}
                    for k, v in neighbors.items()
                ],
                "representative_resolution_patterns": _first_reply_patterns(train),
                "auto_handle_hint": spec.get("auto_handle_hint", "unclear"),
                "notes": spec.get("notes", ""),
            }
        )
    return report


def build_taxonomy(
    table: pd.DataFrame,
    specs: dict[int, dict],
    report: list[dict],
    special: list[dict],
    examples: int = 6,
    final_descriptions: dict[str, str] | None = None,
    criteria: dict[str, dict] | None = None,
    confusable_notes: list[dict] | None = None,
) -> dict:
    """Registry-format taxonomy: merge clusters sharing `final_intent`; everything uncertain is NEEDS_REVIEW.

    Production intents go under `intents`; the fallback is a separate `fallback` entry.
    """
    final_descriptions = final_descriptions or {}
    criteria = criteria or {}
    notes = {frozenset(n["pair"]): n["note"] for n in (confusable_notes or [])}
    by_cluster = {r["cluster_id"]: r for r in report}
    pop = table[table["split"].isin(STAT_SPLITS) & table["in_fit"]]
    names = {cid: s["name"] for cid, s in specs.items()}
    intent_of = {cid: s.get("final_intent") for cid, s in specs.items()}

    groups: dict[str, list[int]] = {}
    for cid, intent in intent_of.items():
        if intent:
            groups.setdefault(intent, []).append(cid)

    intents = []
    for intent, cids in groups.items():
        members = pop[pop["cluster_id"].isin(cids)]
        train = members[members["split"] == EXAMPLE_SPLIT]
        spec_list = [specs[c] for c in cids]
        positives = []
        per_cluster = max(1, examples // len(cids))
        for c in cids:
            for r in train[train["cluster_id"] == c].nsmallest(per_cluster, "distance").itertuples():
                positives.append({"text": _clip(r.opening_message), "case_id": r.case_id})
        neighbor_ids = [
            n["cluster_id"] for c in cids for n in by_cluster[c]["confusable_neighbors"] if intent_of.get(n["cluster_id"]) != intent
        ]
        negatives = []
        for nb in dict.fromkeys(neighbor_ids):
            nb_train = table[(table["split"] == EXAMPLE_SPLIT) & (table["cluster_id"] == nb) & table["in_fit"]]
            pick = nb_train.nsmallest(2, "distance")
            for r in pick.itertuples():
                negatives.append({"text": _clip(r.opening_message), "case_id": r.case_id, "belongs_to": intent_of.get(nb) or names.get(nb, str(nb)),
                                  "why_confusable": "NEEDS_REVIEW: lies near this intent in embedding space"})
            if len(negatives) >= 4:
                break
        patterns: list[dict] = []
        for c in cids:
            patterns += by_cluster[c]["representative_resolution_patterns"]
        patterns = sorted(patterns, key=lambda p: -p["count"])[:5]
        reviewer_triggers = list(dict.fromkeys(t for s in spec_list for t in s.get("escalation_triggers", [])))
        hints = {s.get("auto_handle_hint", "unclear") for s in spec_list}
        mix = members["resolution_type"].value_counts(normalize=True).head(6)
        crit = criteria.get(intent, {})
        intents.append(
            {
                "name": intent,
                "status": "candidate_NEEDS_REVIEW",
                "definition": " ".join(final_descriptions.get(intent, " / ".join(s["description"] for s in spec_list)).split()),
                "inclusion_criteria": crit.get("inclusion") or [NEEDS_REVIEW],
                "exclusion_criteria": crit.get("exclusion") or [NEEDS_REVIEW],
                "source_clusters": [{"cluster_id": c, "name": names[c]} for c in cids],
                "n_cases": len(members),
                "pct_resolved": round(float(members["resolved"].mean()), 3),
                "pct_dm_redirect": round(float(members["dm_redirect"].mean()), 3),
                "positive_examples": positives,
                "negative_examples": negatives or ["NEEDS_REVIEW: no cross-intent neighbours found"],
                "common_resolution_patterns": [
                    f'"{p["example"]}" (n={p["count"]}, template: {p["template"]})' for p in patterns
                ] or ["NEEDS_REVIEW: no recurring first-reply template (>=3) found"],
                "resolution_types": {k: round(float(v), 3) for k, v in mix.items()},
                "required_information": required_information(train),
                "known_confusable_intents": _known_confusable(intent, cids, by_cluster, intent_of, notes),
                "escalation_triggers": historic_escalation_evidence(train)
                + [{"trigger": t.removeprefix("NEEDS_REVIEW: "), "source": "reviewer_draft_from_examples", "policy": NEEDS_REVIEW} for t in reviewer_triggers],
                "safe_to_auto_handle": NEEDS_REVIEW,
                "auto_handle_evidence": {
                    "hint": "likely_yes" if hints == {"likely_yes"} else "likely_no" if hints == {"likely_no"} else "unclear",
                    "brand_reply_pattern_count": len(patterns),
                    "reason": " ".join(s.get("notes", "").strip() for s in spec_list if s.get("notes")) or "NEEDS_REVIEW",
                },
            }
        )

    fallback = None
    for sp in special:
        if sp.get("final_intent") != FALLBACK_INTENT:
            continue
        members = table[table["split"].isin(STAT_SPLITS) & (table["cluster_id"] == sp["cluster_id"])]
        train = members[members["split"] == EXAMPLE_SPLIT]
        mix = members["resolution_type"].value_counts(normalize=True).head(6)
        crit = criteria.get(FALLBACK_INTENT, {})
        fallback = {
            "name": sp["final_intent"],
            "fallback": True,
            "status": "candidate_NEEDS_REVIEW",
            "definition": " ".join(sp["description"].split()),
            "inclusion_criteria": crit.get("inclusion") or [NEEDS_REVIEW],
            "exclusion_criteria": crit.get("exclusion") or [NEEDS_REVIEW],
            "source_clusters": [{"cluster_id": sp["cluster_id"], "name": sp["name"]}],
            "n_cases": len(members),
            "pct_resolved": round(float(members["resolved"].mean()), 3) if len(members) else None,
            "pct_dm_redirect": round(float(members["dm_redirect"].mean()), 3) if len(members) else None,
            "positive_examples": [{"text": _clip(t), "case_id": c} for t, c in zip(train["opening_message"].head(4), train["case_id"].head(4))],
            "negative_examples": ["NEEDS_REVIEW"],
            "common_resolution_patterns": [f'"{p["example"]}" (n={p["count"]}, template: {p["template"]})' for p in _first_reply_patterns(train, 3)]
            or ["NEEDS_REVIEW"],
            "resolution_types": {k: round(float(v), 3) for k, v in mix.items()},
            "required_information": required_information(train),
            "known_confusable_intents": [
                {"intent": "any", "evidence": "fallback is, by design, near every intent", "distinguishing_note": NEEDS_REVIEW}
            ],
            "escalation_triggers": historic_escalation_evidence(train)
            + [{"trigger": t.removeprefix("NEEDS_REVIEW: "), "source": "reviewer_draft_from_examples", "policy": NEEDS_REVIEW} for t in sp.get("escalation_triggers", [])],
            "safe_to_auto_handle": NEEDS_REVIEW,
            "auto_handle_evidence": {"hint": sp.get("auto_handle_hint", "unclear"), "reason": "NEEDS_REVIEW"},
        }
    intents.sort(key=lambda i: -i["n_cases"])
    return {
        "taxonomy": {
            "brand": "VirginTrains",
            "registry_schema_version": 2,
            "status": STATUS_CANDIDATE,
            "warning": (
                "Discovered by unsupervised clustering of opening messages and curated by hand once. NOT yet calibrated against "
                "human labels. Run scripts/finalize_taxonomy.py after the taxonomy calibration set has been labelled. "
                "Every field marked NEEDS_REVIEW (including safe_to_auto_handle on every intent) must be reviewed by a human "
                "before use. Clusters were fit on train+dev only; examples come from the train split only."
            ),
            "calibration": {"status": "NOT_YET_CALIBRATED", "set": "golden_pool_reserve", "labelled_cases": 0},
            "examples_source_split": EXAMPLE_SPLIT,
            "intents": intents,
            "fallback": fallback,
        }
    }


def _known_confusable(intent: str, cids: list[int], by_cluster: dict[int, dict], intent_of: dict[int, str | None], notes: dict) -> list[dict]:
    """Neighbouring intents ranked by how often they are the runner-up for this intent's lowest-margin cases."""
    weight: dict[str, float] = {}
    total = sum(by_cluster[c]["n_cases"] for c in cids) or 1
    for c in cids:
        for nb in by_cluster[c]["confusable_neighbors"]:
            other = intent_of.get(nb["cluster_id"])
            if other and other != intent:
                weight[other] = weight.get(other, 0.0) + nb["share_of_low_margin_cases"] * by_cluster[c]["n_cases"] / total
    out = [
        {
            "intent": other,
            "evidence": f"runner-up cluster for {100 * w:.0f}% of this intent's lowest-margin (closest to a boundary) cases",
            "distinguishing_note": notes.get(frozenset({intent, other}), NEEDS_REVIEW),
        }
        for other, w in sorted(weight.items(), key=lambda kv: -kv[1])[:4]
    ]
    listed = {o["intent"] for o in out}
    for pair, note in notes.items():
        if intent in pair:
            (other,) = pair - {intent}
            if other not in listed:
                out.append({"intent": other, "evidence": "reviewer-noted confusable pair", "distinguishing_note": note})
    return out
