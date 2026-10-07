"""Apply a reviewer's decisions to the candidate registry, validate it, and build the frozen taxonomy.

The tool does not decide anything. Merges, renames, demotions to fallback, new intents, and field edits all come
from `configs/virgintrains_taxonomy_decisions.yaml`, written by the human reviewer after reading the calibration report.
"""

from __future__ import annotations

import copy
import datetime as dt
import re
from itertools import zip_longest
from typing import Any

from taxonomy.registry import (
    NEEDS_REVIEW,
    STATUS_CALIBRATED,
    canonical_json,
    sha256_text,
)

MIN_INTENTS, MAX_INTENTS = 8, 12
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
REQUIRED_FIELDS = (
    "name", "definition", "inclusion_criteria", "exclusion_criteria", "positive_examples", "negative_examples",
    "common_resolution_patterns", "resolution_types", "required_information", "known_confusable_intents", "escalation_triggers",
)
EDITABLE_FIELDS = (set(REQUIRED_FIELDS) - {"name"}) | {"safe_to_auto_handle", "auto_handle_evidence"}
TAXONOMY_VERSION = "v1"


class DecisionError(ValueError):
    pass


def validate_decisions(decisions: dict, intent_names: set[str]) -> list[str]:
    errors: list[str] = []
    if not str(decisions.get("reviewed_by", "")).strip():
        errors.append("decisions.reviewed_by is empty: a named human reviewer must sign the decisions")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(decisions.get("reviewed_on", ""))):
        errors.append("decisions.reviewed_on must be YYYY-MM-DD")
    known = set(intent_names)
    for m in decisions.get("merges") or []:
        if not m.get("into") or not m.get("from"):
            errors.append(f"merge needs `into` and `from`: {m}")
            continue
        if not str(m.get("definition", "")).strip():
            errors.append(f"merge into {m['into']} needs a reviewer-written `definition`")
        for s in m["from"]:
            if s not in known:
                errors.append(f"merge source {s!r} is not an intent in the registry")
    for old in (decisions.get("renames") or {}):
        if old not in known:
            errors.append(f"rename source {old!r} is not an intent in the registry")
    for name in decisions.get("drop_to_fallback") or []:
        if name not in known:
            errors.append(f"drop_to_fallback {name!r} is not an intent in the registry")
    for name, fields in (decisions.get("edits") or {}).items():
        bad = set(fields) - EDITABLE_FIELDS
        if bad:
            errors.append(f"edits for {name!r} touch non-editable fields {sorted(bad)}")
    for n in decisions.get("new_intents") or []:
        for key in ("name", "definition", "inclusion_criteria", "exclusion_criteria"):
            if not n.get(key):
                errors.append(f"new intent {n.get('name')!r} is missing `{key}`")
    return errors


def _weighted(a: dict[str, float], wa: int, b: dict[str, float], wb: int) -> dict[str, float]:
    total = (wa + wb) or 1
    merged = {k: (a.get(k, 0.0) * wa + b.get(k, 0.0) * wb) / total for k in set(a) | set(b)}
    return {k: round(v, 3) for k, v in sorted(merged.items(), key=lambda kv: -kv[1])[:6]}


def _dedupe(items: list, key) -> list:
    seen, out = set(), []
    for it in items:
        k = key(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def merge_intents(parts: list[dict], into: str, definition: str) -> dict:
    base = copy.deepcopy(parts[0])
    merged_names = {p["name"] for p in parts}
    n_total = sum(p["n_cases"] for p in parts)

    def weighted_pct(field: str) -> float | None:
        vals = [(p[field], p["n_cases"]) for p in parts if p.get(field) is not None]
        return round(sum(v * w for v, w in vals) / (sum(w for _, w in vals) or 1), 3) if vals else None

    resolution_types: dict[str, float] = {}
    seen_cases = 0
    for p in parts:
        resolution_types = _weighted(resolution_types, seen_cases, p["resolution_types"], p["n_cases"])
        seen_cases += p["n_cases"]

    def listy(field: str) -> list:
        return [x for p in parts for x in p[field] if x != NEEDS_REVIEW]

    # Round-robin across the merged parts so no single source dominates the examples.
    per_part = [[x for x in p["positive_examples"] if isinstance(x, dict)] for p in parts]
    positives = _dedupe([x for tup in zip_longest(*per_part) for x in tup if x], lambda e: e["case_id"])[:8]
    negatives = [x for p in parts for x in p["negative_examples"] if not (isinstance(x, dict) and x.get("belongs_to") in merged_names)]

    required: dict[str, str] = {}
    for p in parts:
        for r in p["required_information"]:
            if r["item"] != NEEDS_REVIEW:
                required[r["item"]] = f"{required[r['item']]}; {r['evidence']}" if r["item"] in required else r["evidence"]
    confusable = _dedupe([c for p in parts for c in p["known_confusable_intents"] if c["intent"] not in merged_names], lambda c: c["intent"])
    base.update(
        {
            "name": into,
            "definition": " ".join(definition.split()),
            "inclusion_criteria": _dedupe(listy("inclusion_criteria"), str) or [NEEDS_REVIEW],
            "exclusion_criteria": _dedupe(listy("exclusion_criteria"), str) or [NEEDS_REVIEW],
            "source_clusters": _dedupe([c for p in parts for c in p.get("source_clusters", [])], lambda c: c["cluster_id"]),
            "n_cases": n_total,
            "pct_resolved": weighted_pct("pct_resolved"),
            "pct_dm_redirect": weighted_pct("pct_dm_redirect"),
            "positive_examples": positives or [NEEDS_REVIEW],
            "negative_examples": negatives[:6] or [NEEDS_REVIEW],
            "common_resolution_patterns": _dedupe(listy("common_resolution_patterns"), str)[:6] or [NEEDS_REVIEW],
            "resolution_types": resolution_types,
            "required_information": [{"item": k, "evidence": v} for k, v in required.items()]
            or [{"item": NEEDS_REVIEW, "evidence": "none observed"}],
            "known_confusable_intents": confusable,
            "escalation_triggers": _dedupe([t for p in parts for t in p["escalation_triggers"]], lambda t: t["trigger"]),
            "safe_to_auto_handle": NEEDS_REVIEW,
            "auto_handle_evidence": {"hint": "unclear", "reason": "merged intent; NEEDS_REVIEW"},
        }
    )
    return base


def _new_intent(spec: dict) -> dict:
    nr = [NEEDS_REVIEW]
    return {
        "name": spec["name"],
        "status": "human_proposed",
        "definition": " ".join(str(spec["definition"]).split()),
        "inclusion_criteria": spec["inclusion_criteria"],
        "exclusion_criteria": spec["exclusion_criteria"],
        "source_clusters": [],
        "n_cases": 0,
        "pct_resolved": None,
        "pct_dm_redirect": None,
        "positive_examples": spec.get("positive_examples") or nr,
        "negative_examples": spec.get("negative_examples") or nr,
        "common_resolution_patterns": spec.get("common_resolution_patterns") or nr,
        "resolution_types": spec.get("resolution_types") or {"NEEDS_REVIEW": 1.0},
        "required_information": spec.get("required_information") or [{"item": NEEDS_REVIEW, "evidence": "new intent, no historic evidence yet"}],
        "known_confusable_intents": spec.get("known_confusable_intents") or [],
        "escalation_triggers": spec.get("escalation_triggers") or [{"trigger": NEEDS_REVIEW, "source": "none", "policy": NEEDS_REVIEW}],
        "safe_to_auto_handle": NEEDS_REVIEW,
        "auto_handle_evidence": {"hint": "unclear", "reason": "new intent proposed by reviewer"},
    }


def apply_decisions(taxonomy: dict, decisions: dict) -> dict:
    """Return a new inner taxonomy dict with the reviewer's decisions applied."""
    tax = copy.deepcopy(taxonomy)
    intents = {i["name"]: i for i in tax["intents"]}
    fallback = tax["fallback"]
    errors = validate_decisions(decisions, set(intents) | {fallback["name"]})
    if errors:
        raise DecisionError("; ".join(errors))

    renamed: dict[str, str] = {}
    for m in decisions.get("merges") or []:
        parts = [intents.pop(s) for s in m["from"] if s in intents]
        if m["into"] in intents:
            parts.insert(0, intents.pop(m["into"]))
        if not parts:
            continue
        for p in parts:
            renamed[p["name"]] = m["into"]
        intents[m["into"]] = merge_intents(parts, m["into"], m["definition"])
    for old, new in (decisions.get("renames") or {}).items():
        if old in intents:
            intents[new] = {**intents.pop(old), "name": new}
            renamed[old] = new
    for name in decisions.get("drop_to_fallback") or []:
        gone = intents.pop(name, None)
        if gone:
            fallback["source_clusters"] = _dedupe(fallback["source_clusters"] + gone.get("source_clusters", []), lambda c: c["cluster_id"])
            fallback["n_cases"] += gone["n_cases"]
            renamed[name] = fallback["name"]
    for spec in decisions.get("new_intents") or []:
        intents[spec["name"]] = _new_intent(spec)
    for name, fields in (decisions.get("edits") or {}).items():
        target = renamed.get(name, name)
        holder = fallback if target == fallback["name"] else intents.get(target)
        if holder is None:
            raise DecisionError(f"edits refer to {name!r}, which no longer exists after merges/renames")
        holder.update(copy.deepcopy(fields))

    for it in intents.values():
        fixed = [{**c, "intent": renamed.get(c["intent"], c["intent"])} for c in it.get("known_confusable_intents", [])]
        it["known_confusable_intents"] = _dedupe([c for c in fixed if c["intent"] != it["name"]], lambda c: c["intent"])
        if it.get("status") != "human_proposed":
            it["status"] = "human_calibrated"
    fallback["status"] = "human_calibrated"
    tax["intents"] = sorted(intents.values(), key=lambda i: -i["n_cases"])
    tax["fallback"] = fallback
    tax["renames_applied"] = renamed
    return tax


def validate_registry(tax: dict, train_case_ids: set[str] | None = None, forbidden_case_ids: set[str] | None = None) -> list[str]:
    """Structural checks for a finalized taxonomy. Returns a list of errors (empty = valid)."""
    errors: list[str] = []
    intents = tax["intents"]
    if not MIN_INTENTS <= len(intents) <= MAX_INTENTS:
        errors.append(f"{len(intents)} production intents; expected {MIN_INTENTS}-{MAX_INTENTS} plus a fallback")
    fb = tax.get("fallback")
    if not fb:
        errors.append("no fallback intent")
    everything = intents + ([fb] if fb else [])
    names = [i["name"] for i in everything]
    if len(set(names)) != len(names):
        errors.append("duplicate intent names")
    for n in names:
        if not NAME_RE.match(n):
            errors.append(f"intent name {n!r} is not snake_case")
    for it in everything:
        for f in REQUIRED_FIELDS:
            if f not in it or it[f] in (None, "", [], {}):
                errors.append(f"{it.get('name')}: missing {f}")
        for c in it.get("known_confusable_intents", []):
            if c["intent"] not in names and c["intent"] != "any":
                errors.append(f"{it['name']}: known_confusable_intents refers to unknown intent {c['intent']!r}")
        for ex in it.get("positive_examples", []) + it.get("negative_examples", []):
            if isinstance(ex, dict) and "case_id" in ex:
                if train_case_ids is not None and ex["case_id"] not in train_case_ids:
                    errors.append(f"{it['name']}: example {ex['case_id']} is not a train_retrieval case")
                if forbidden_case_ids and ex["case_id"] in forbidden_case_ids:
                    errors.append(f"{it['name']}: example {ex['case_id']} is a golden/calibration case")
    return errors


def finalize_registry(tax: dict, decisions: dict, calibration: dict[str, Any]) -> dict:
    """Wrap the decided taxonomy as the final registry (status HUMAN_CALIBRATED, no candidate marker)."""
    inner = copy.deepcopy(tax)
    inner.pop("warning", None)
    inner["status"] = STATUS_CALIBRATED
    inner["calibration"] = {
        "status": "HUMAN_CALIBRATED",
        "statement": (
            "This taxonomy was human-calibrated: a person labelled the calibration sample drawn from the golden_pool_reserve split, "
            "the candidate intents were compared with those labels, and the reviewer recorded merges/splits/edits in the decisions file. "
            "No golden_eval, train_retrieval, or dev_calibration case was used for the calibration. "
            "Fields still marked NEEDS_REVIEW (e.g. safe_to_auto_handle) were not settled by calibration and must not be read as approved."
        ),
        "reviewed_by": decisions["reviewed_by"],
        "reviewed_on": decisions["reviewed_on"],
        **calibration,
    }
    return {"taxonomy": inner}


def content_hash(frozen: dict) -> str:
    body = copy.deepcopy(frozen)
    body.get("metadata", {}).pop("content_sha256", None)
    return sha256_text(canonical_json(body))


def build_frozen(registry: dict, metadata: dict[str, Any]) -> dict:
    inner = registry["taxonomy"]
    meta = dict(metadata)
    frozen = {
        "metadata": {
            "taxonomy_version": TAXONOMY_VERSION,
            "frozen": True,
            "date_created": meta.pop("date_created", dt.date.today().isoformat()),
            "source_split": "golden_pool_reserve",
            **meta,
        },
        "labels": [i["name"] for i in inner["intents"]] + [inner["fallback"]["name"]],
        "fallback": inner["fallback"]["name"],
        "taxonomy": inner,
    }
    frozen["metadata"]["content_sha256"] = content_hash(frozen)
    return frozen


def verify_frozen(frozen: dict) -> list[str]:
    """Integrity checks for a frozen taxonomy file. Empty list = intact."""
    errors = []
    meta = frozen.get("metadata", {})
    if not meta.get("frozen"):
        errors.append("metadata.frozen is not true")
    for key in ("taxonomy_version", "calibration_set_size", "date_created", "source_split", "content_sha256", "golden_case_ids_sha256"):
        if key not in meta:
            errors.append(f"metadata.{key} missing")
    if meta.get("content_sha256") and meta["content_sha256"] != content_hash(frozen):
        errors.append("content_sha256 does not match the file contents: the frozen taxonomy was edited")
    if frozen.get("taxonomy", {}).get("status") == "CANDIDATE_NOT_GROUND_TRUTH":
        errors.append("frozen taxonomy still carries the CANDIDATE_NOT_GROUND_TRUTH marker")
    labels = frozen.get("labels", [])
    if labels and labels[-1] != frozen.get("fallback"):
        errors.append("fallback must be the last label")
    return errors
