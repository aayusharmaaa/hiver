"""Failure analysis built only from existing artifacts: golden labels, cached LLM intent predictions, the retrieval proxy
evaluation and the end-to-end agent run (which may be partial).

No model is called. Every example is a real case id with its real opening message. Each failure mode states its scope (which
cases it was measured on) so a count from the 50-case agent run is never mixed up with one from the 100 blind cases.

The policy-oracle analysis applies the agent's hard escalation rules (never-auto-handle intents, sensitive wording, very short
messages) to the GOLD intent, i.e. it assumes a perfect classifier. It shows how much escalation is built into the policy
itself, independent of model quality.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from agent.policy import count_words, sensitive_matches

EXCERPT = 150
MAX_EXAMPLES = 3


def excerpt(text: str, n: int = EXCERPT) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def example(case_id: str, message: str, detail: str) -> dict[str, str]:
    return {"case_id": case_id, "message": excerpt(message), "detail": detail}


def mode(key: str, title: str, layer: str, scope: str, count: int | None, denominator: int | None, examples: list[dict], observed: str, why: str, fix: str) -> dict[str, Any]:
    return {"key": key, "title": title, "layer": layer, "scope": scope, "count": count, "denominator": denominator, "examples": examples[:MAX_EXAMPLES], "observed": observed, "why": why, "fix": fix}


# --------------------------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------------------------
def golden_frame(pack: pd.DataFrame, provenance: dict[str, str], candidate_of: dict[str, str], case_extra: pd.DataFrame) -> pd.DataFrame:
    """One row per golden case: gold labels, provenance (`blind` = labelled blind by a human), candidate intent, thread facts."""
    df = pack.copy()
    df["provenance"] = df["case_id"].map(provenance)
    df["blind"] = df["provenance"] == "human"
    df["candidate_intent"] = df["case_id"].map(candidate_of)
    df = df.merge(case_extra[["case_id", "other_agent_turn_count", "dm_redirect"]], on="case_id", how="left")
    df["human_notes"] = df["human_notes"].fillna("").astype(str)
    df["gold_confidence"] = df["gold_confidence"].fillna("").astype(str)
    return df.sort_values("labeling_order").reset_index(drop=True)


def load_llm_predictions(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if not rec.get("error"):
                    out[rec["case_id"]] = rec
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------------------------------------------------
# Intent
# --------------------------------------------------------------------------------------------------------------------
def intent_errors(golden: pd.DataFrame, preds: dict[str, dict]) -> tuple[dict, dict]:
    """(LLM intent errors on the blind 100, the confusable-boundary view of the same errors)."""
    blind = golden[golden["blind"] & golden["case_id"].isin(preds)]
    wrong = blind[[preds[c]["intent"] != g for c, g in zip(blind["case_id"], blind["gold_intent"])]]
    pairs = Counter(tuple(sorted((g, preds[c]["intent"]))) for c, g in zip(wrong["case_id"], wrong["gold_intent"]))
    confident = wrong[[preds[c]["confidence"] >= 0.9 for c in wrong["case_id"]]]
    errors = mode(
        "intent_errors", "Intent classification errors", "model + taxonomy", "LLM classifier (gpt-oss-120b) on the 100 blind-human golden cases",
        len(wrong), len(blind),
        [example(r.case_id, r.first_customer_message, f"gold `{r.gold_intent}`, predicted `{preds[r.case_id]['intent']}` (confidence {preds[r.case_id]['confidence']:.2f})") for r in confident.itertuples()],
        f"{len(wrong)} of {len(blind)} blind cases misclassified; {len(confident)} of those errors had self-reported confidence >= 0.90, "
        "so the confidence threshold (0.75) does not catch them.",
        "Self-reported LLM confidence is uncalibrated; the classifier sees only the opening tweet while the labeler saw the whole conversation.",
        "Calibrate confidence on dev (reliability curve) or replace it with retrieval-agreement signals; give the classifier the thread context the labeler had.",
    )
    top = pairs.most_common(4)
    pair_rows = [w for w in wrong.itertuples() if top and tuple(sorted((w.gold_intent, preds[w.case_id]["intent"]))) == top[0][0]]
    boundary = mode(
        "intent_boundary", "Intent boundary ambiguity", "taxonomy", "LLM errors on the blind 100, grouped into unordered intent pairs",
        sum(n for _, n in top), len(wrong),
        [example(r.case_id, r.first_customer_message, f"gold `{r.gold_intent}`, predicted `{preds[r.case_id]['intent']}`; label confidence {r.gold_confidence or 'n/a'}") for r in pair_rows],
        "Most confused pairs: " + "; ".join(f"{a} / {b} ({n})" for (a, b), n in top) + ". "
        f"Independently, the candidate taxonomy disagrees with the final gold intent on {int((golden['candidate_intent'] != golden['gold_intent']).sum())} of {len(golden)} golden cases.",
        "Several intents describe the same tweet from different angles (a delayed train is both a status question and a complaint); the taxonomy came from soft clusters (silhouette about 0.05).",
        "Define intents by the action the brand must take (live info / compensation / acknowledge) and add decision rules for the top pairs; re-label only after freezing the new definitions.",
    )
    return errors, boundary


def low_information(golden: pd.DataFrame, preds: dict[str, dict], min_words: int) -> dict:
    unclear = golden[golden["gold_intent"] == "unclear_or_media_only"]
    found = sum(preds.get(c, {}).get("intent") == "unclear_or_media_only" for c in unclear["case_id"])
    short = golden[[count_words(m) < min_words for m in golden["first_customer_message"]]]
    rows = list(unclear.itertuples())
    return mode(
        "low_information", "Low-information and media-only messages", "data + model", "all 250 golden cases",
        len(unclear), len(golden),
        [example(r.case_id, r.first_customer_message, f"gold `unclear_or_media_only`; LLM predicted `{preds.get(r.case_id, {}).get('intent', 'n/a')}`") for r in rows],
        f"{len(unclear)} cases are gold `unclear_or_media_only`; the LLM labelled {found} of them as unclear. "
        f"Opening messages with fewer than {min_words} content words (which the policy escalates): {len(short)}.",
        "The real request is in an image or a link the system never sees, so the model guesses an intent from the surrounding words.",
        "Treat bare links/images as an explicit low-information trigger and ask a clarifying question instead of classifying.",
    )


def other_operator(golden: pd.DataFrame) -> dict:
    noted_mask = golden["human_notes"].str.contains(r"another operator|other operator", case=False)
    other = golden[(golden["other_agent_turn_count"].fillna(0) > 0) | noted_mask]
    noted = golden[noted_mask]
    return mode(
        "other_operator", "Cases involving another train operator", "data", "all 250 golden cases (thread data + label notes)",
        len(other), len(golden),
        [example(r.case_id, r.first_customer_message, f"note: {excerpt(r.human_notes, 90)}") for r in noted.itertuples()],
        f"{len(other)} golden cases either have another operator's agent in the thread or a label note saying the service was another operator's "
        f"({int(other['blind'].sum())} of them in the blind 100).",
        "Customers tag VirginTrains about trains it does not run; the right answer is a redirect, which no intent captures.",
        "Add an operator-detection rule (station/route and operator names) and a 'redirect to operator' resolution path.",
    )


# --------------------------------------------------------------------------------------------------------------------
# Policy and agent
# --------------------------------------------------------------------------------------------------------------------
def oracle_hard_reasons(message: str, gold_intent: str, settings: Any) -> list[str]:
    reasons = []
    if gold_intent in settings.escalate_intents:
        reasons.append(f"intent rule ({gold_intent})")
    words = sensitive_matches(message, settings.sensitive_patterns)
    if words:
        reasons.append("sensitive wording (" + ", ".join(words[:3]) + ")")
    if count_words(message) < settings.min_message_words:
        reasons.append("low-information message")
    return reasons


def policy_over_escalation(golden: pd.DataFrame, settings: Any) -> dict:
    """With a perfect classifier, how many gold 'should not escalate' blind cases would a hard rule still escalate?"""
    blind = golden[golden["blind"]]
    no = blind[blind["gold_should_escalate"] == "no"]
    hits = [(r, oracle_hard_reasons(r.first_customer_message, r.gold_intent, settings)) for r in no.itertuples()]
    hits = [(r, h) for r, h in hits if h]
    kinds = Counter(h.split(" (")[0] for _, h_list in hits for h in h_list)
    sensitive = [(r, h) for r, h in hits if any(x.startswith("sensitive") for x in h)]
    rule = [(r, h) for r, h in hits if not any(x.startswith("sensitive") for x in h)]
    shown = sensitive[:2] + rule[:1]
    return mode(
        "policy_over_escalation", "Over-escalation built into the policy", "policy", "hard rules applied to the GOLD intent of the blind 100 (perfect-classifier oracle)",
        len(hits), len(no),
        [example(r.case_id, r.first_customer_message, f"gold intent `{r.gold_intent}`, gold should_escalate = no; fired: {'; '.join(h)}") for r, h in shown],
        f"Of {len(no)} blind cases a human said need not escalate, {len(hits)} would still be escalated by a hard rule even with perfect classification "
        f"(intent rule {kinds.get('intent rule', 0)}, sensitive wording {kinds.get('sensitive wording', 0)}, low-information {kinds.get('low-information message', 0)}; a case can fire several).",
        "Whole intents are blocked (status, disruption, Delay Repay, complaints, unclear), and the sensitive-word list matches words like 'wheelchair' or 'refund' even in praise or simple questions.",
        "Gate on the action needed rather than the intent (a timetable question is not a live-status question), and scope sensitive words to complaints; tune on dev, not golden.",
    )


def hard_rule_gaps(golden: pd.DataFrame, settings: Any, records: list[dict]) -> dict:
    blind = golden[golden["blind"]]
    yes = blind[blind["gold_should_escalate"] == "yes"]
    gaps = [r for r in yes.itertuples() if not oracle_hard_reasons(r.first_customer_message, r.gold_intent, settings)]
    unsafe = [r for r in records if r["gold_should_escalate"] == "yes" and r["final_action"] == "AUTO_HANDLE"]
    return mode(
        "unsafe_auto_handle", "Potential unsafe auto-handles", "policy", "blind 100 (oracle) + the end-to-end agent run",
        len(gaps), len(yes),
        [example(r.case_id, r.first_customer_message, f"gold intent `{r.gold_intent}`, gold should_escalate = yes; no hard rule fires; note: {excerpt(r.human_notes, 80) or '-'}") for r in gaps],
        f"{len(gaps)} of {len(yes)} blind cases that should escalate are not caught by any hard rule; only the soft checks (confidence, evidence, grounding) stand between them and an auto-reply. "
        f"Observed in the agent run: {len(unsafe)} of {len(records)} scored cases auto-handled although gold says escalate"
        + (f" ({', '.join(r['case_id'] for r in unsafe)})." if unsafe else "."),
        "Escalation-worthy cases that look like routine intents (e.g. a booking question that actually needs an account lookup) slip past intent-level rules.",
        "Measure auto-handle precision on the full 50-case slice, then add the specific triggers found (account/booking-reference requests) as rules.",
    )


def agent_observed(records: list[dict], expected: int) -> list[dict]:
    """Failure modes visible only in the end-to-end run. Counts are over the cases actually run (may be partial)."""
    n = len(records)
    scope = f"end-to-end agent run: {n} of {expected} cases" + (" (PARTIAL)" if n < expected else "")
    over = [r for r in records if r["gold_should_escalate"] == "no" and r["final_action"] == "ESCALATE"]
    attempted = [r for r in records if r["generation"] != "not_attempted"]
    gen_fail = [r for r in attempted if r["generation"] in ("declined", "error")]
    grounded = [r for r in records if r["grounding"] != "not_run"]
    ground_fail = [r for r in grounded if r["grounding"] in ("fail", "unavailable")]
    weak = [r for r in records if any(x.startswith(("weak retrieval", "insufficient evidence", "evidence disagrees")) for x in r["reasons"])]

    def ex(rows: list[dict], detail) -> list[dict]:
        return [example(r["case_id"], r["message"], detail(r)) for r in rows]

    return [
        mode("observed_over_escalation", "Over-escalation (observed)", "policy", scope, len(over), sum(r["gold_should_escalate"] == "no" for r in records),
             ex(over, lambda r: f"gold `{r['gold_intent']}`, predicted `{r['pred_intent']}`; " + "; ".join(r["reasons"])[:160]),
             f"{len(over)} cases a human said need not escalate were escalated.", "See the policy oracle above: most are blanket intent rules or sensitive words.",
             "Same as the policy oracle fix."),
        mode("generation_failures", "Reply-generation failures", "model", scope, len(gen_fail), len(attempted),
             ex(gen_fail, lambda r: f"generation = {r['generation']}"),
             f"{len(gen_fail)} of {len(attempted)} attempted generations declined or errored.", "Generator refuses when evidence does not cover the question, or the model output is unusable.",
             "Inspect declined drafts once the full run exists."),
        mode("grounding_failures", "Grounding failures", "model + retrieval", scope, len(ground_fail), len(grounded),
             ex(ground_fail, lambda r: f"unsupported: {'; '.join(r.get('unsupported_claims') or [])[:120]}"),
             f"{len(ground_fail)} of {len(grounded)} grounding checks failed or could not run.", "Drafts add details (times, links, promises) the evidence does not contain.",
             "Keep fail-closed; report blocked drafts alongside sent ones."),
        mode("insufficient_evidence", "Insufficient or inconsistent historical evidence", "retrieval + data", scope, len(weak), n,
             ex(weak, lambda r: "; ".join(x for x in r["reasons"] if x.startswith(("weak", "insufficient", "evidence")))[:160]),
             f"{len(weak)} of {n} cases had weak, too few or intent-inconsistent evidence.", "Rare or unusual complaints have no close historical match in the train split.",
             "Expected behaviour (abstain); track how often it is the only reason for escalation."),
    ]


# --------------------------------------------------------------------------------------------------------------------
# Retrieval (proxy evaluation on dev, not golden)
# --------------------------------------------------------------------------------------------------------------------
def retrieval_misses(retrieval: dict[str, Any]) -> dict:
    f = retrieval.get("failures", {})
    kinds = f.get("kinds", {})
    n_test = retrieval.get("meta", {}).get("n_test")
    bad = retrieval.get("examples", {}).get("bad", [])
    rows = []
    for b in bad:
        top = (b.get("results") or [{}])[0]
        rows.append(example(b["case_id"], b["query"], f"proxy relevance = same intent `{b['intent']}` + resolution `{b['resolution_type']}`; "
                            f"top-1 was `{top.get('intent')}` / `{top.get('resolution_type')}`; first relevant at rank {b.get('first_relevant_rank')}"))
    return mode(
        "retrieval_misses", "Retrieval finds the topic but not the handling", "retrieval", f"proxy retrieval evaluation on {n_test} dev queries (labels are heuristic, not human)",
        f.get("n_queries_without_hit_in_top5"), n_test, rows,
        f"{f.get('n_queries_without_hit_in_top5')} queries had no relevant case in the top 5; {kinds.get('same_intent_but_other_resolution_type', 0)} of them retrieved the "
        "same intent but a different resolution type.",
        "The opening message rarely contains what decided the historical handling (ticket type, eligibility, whether a DM happened).",
        "Retrieve on the full thread when available and show resolution-type diversity in the evidence instead of top-5 by similarity alone.",
    )


# --------------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------------
def analyse(golden: pd.DataFrame, preds: dict[str, dict], retrieval: dict, records: list[dict], settings: Any, *, expected_agent_cases: int) -> list[dict]:
    errors, boundary = intent_errors(golden, preds)
    return [
        errors,
        boundary,
        retrieval_misses(retrieval),
        policy_over_escalation(golden, settings),
        hard_rule_gaps(golden, settings, records),
        *agent_observed(records, expected_agent_cases),
        other_operator(golden),
        low_information(golden, preds, settings.min_message_words),
    ]


def render_markdown(modes: list[dict], meta: dict[str, Any]) -> str:
    lines = [
        "# Failure analysis",
        "",
        "Generated by `python scripts/analyze_failures.py` from existing artifacts only (no model calls). Every example is a real "
        "case id and opening message. Scopes differ per section; read the scope line before comparing counts.",
        "",
        f"- Golden: {meta['golden_cases']} cases ({meta['blind_cases']} labelled blind by a human; the rest human-reviewed assistant drafts).",
        f"- LLM intent predictions: {meta['llm_predictions']} cached (`{meta['llm_model']}`).",
        f"- End-to-end agent run: {meta['agent_cases']} of {meta['agent_expected']} cases" + (" — **partial; agent-run counts are not final**." if meta["agent_cases"] < meta["agent_expected"] else "."),
        "",
    ]
    for m in modes:
        count = "–" if m["count"] is None else (f"{m['count']} / {m['denominator']}" if m["denominator"] else str(m["count"]))
        lines += [f"## {m['title']}", "", f"- **Layer:** {m['layer']}", f"- **Scope:** {m['scope']}", f"- **Count:** {count}", f"- **Observed:** {m['observed']}",
                  f"- **Likely cause:** {m['why']}", f"- **Hypothesis / fix:** {m['fix']}", ""]
        if m["examples"]:
            lines += ["| case | opening message | what happened |", "|---|---|---|"]
            lines += [f"| {e['case_id']} | {e['message'].replace('|', '/')} | {e['detail'].replace('|', '/')} |" for e in m["examples"]]
        else:
            lines.append("_No examples in the data available so far._")
        lines.append("")
    return "\n".join(lines)
