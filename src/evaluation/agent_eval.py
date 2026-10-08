"""End-to-end agent evaluation on a stratified slice of the golden set.

The real `SupportAgent.handle` runs on each case's opening message (classification -> retrieval -> policy -> generation ->
grounding); nothing in the agent is changed. The gold `gold_should_escalate` label is the reference for routing.

* Selection: about 50 reviewed golden cases, stratified by gold intent and gold escalation, at least `min_per_intent` per
  intent. Blind-human labels are preferred within a stratum; `NEW:` intents are skipped because no system can predict them.
* Caching: every model call goes through `CachingModel`, keyed by model, prompt, system prompt and generation settings. A
  usable or unusable *output* is cached (it is the model's behaviour). A *runtime* failure (rate limit, timeout) is not:
  that case is left unscored, the run stops, and a rerun continues from the cache without repeating any call.
* Metrics (all on cases that finished without a runtime failure), with E = final ESCALATE, A = final AUTO_HANDLE,
  Y = gold says escalate, N = gold says no:
    escalation coverage      |E| / n            share of cases routed to a human
    safe automation rate     |A and N| / n      share of all cases correctly answered automatically
    false auto-handle rate   |A and Y| / |A|    share of sent replies that should have gone to a human
    escalation precision     |E and Y| / |E|
    escalation recall        |E and Y| / |Y|
  plus stage counts (policy AUTO_HANDLE, generation outcome, grounding outcome) and reply statistics.
* Reply quality: every produced draft (sent or blocked by grounding) is written to a judge-ready JSONL with empty judge fields.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable

import pandas as pd

from agent.schemas import AUTO_HANDLE, ESCALATE, AgentResult
from evaluation.golden_eval import GoldenLabelStore
from evaluation.intent_eval import ALL_REVIEWED, BLIND_HUMAN, golden_eval_frame
from evaluation.smoke_report import check_result_invariants
from models.base import ModelConfigError, ModelOutputError, ModelRuntimeError

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE = 50
MIN_PER_INTENT = 3
SAMPLE_SEED = 42

GEN_NOT_ATTEMPTED, GEN_PRODUCED, GEN_DECLINED, GEN_ERROR = "not_attempted", "produced", "declined", "error"
GROUND_NOT_RUN, GROUND_PASS, GROUND_FAIL, GROUND_UNAVAILABLE = "not_run", "pass", "fail", "unavailable"

JUDGE_RUBRIC = {
    "grounded": "Every fact, link, price, time and procedure in the reply is supported by the evidence (yes/no).",
    "answers_question": "The reply addresses what this customer actually asked (yes/partly/no).",
    "no_invented_commitments": "No promised action, compensation or claim that something was already done (yes/no).",
    "tone": "Polite, plain and on-brand for a UK train operator (good/acceptable/poor).",
    "overall": "Would a support lead send this unchanged? (1 = never ... 5 = yes)",
}
_URL = re.compile(r"https?://\S+")


class AgentEvalError(RuntimeError):
    """The evaluation inputs are unusable (e.g. too few eligible golden cases)."""


# --------------------------------------------------------------------------------------------------------------------
# Golden slice
# --------------------------------------------------------------------------------------------------------------------
def golden_agent_frame(store: GoldenLabelStore) -> pd.DataFrame:
    """`golden_eval_frame` plus the routing and resolution labels."""
    base = golden_eval_frame(store)
    labels = store.verified_frame()[["case_id", "gold_should_escalate", "gold_resolution_type"]]
    return base.merge(labels, on="case_id", how="left", validate="one_to_one")


def _rank(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()


def _allocate(available: dict[Any, int], total: int, minimum: int) -> dict[Any, int]:
    """Proportional quotas with a per-key floor, capped by availability, summing to `total` when possible."""
    pool = sum(available.values())
    quota = {k: min(v, max(minimum, round(total * v / pool))) for k, v in available.items()}
    while sum(quota.values()) > total:
        k = max((k for k in quota if quota[k] > min(minimum, available[k])), key=lambda k: (quota[k], str(k)), default=None)
        if k is None:
            break
        quota[k] -= 1
    while sum(quota.values()) < total:
        k = max((k for k in quota if quota[k] < available[k]), key=lambda k: (available[k] - quota[k], str(k)), default=None)
        if k is None:
            break
        quota[k] += 1
    return quota


def select_stratified(golden: pd.DataFrame, taxonomy_intents: Iterable[str], *, n: int = DEFAULT_SAMPLE, min_per_intent: int = MIN_PER_INTENT, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """Deterministic stratified slice of reviewed golden cases (strata: gold intent x gold escalation)."""
    allowed = set(taxonomy_intents)
    pool = golden[golden[ALL_REVIEWED] & golden["gold_intent"].isin(allowed) & golden["gold_should_escalate"].isin(["yes", "no"])].copy()
    if len(pool) < n:
        raise AgentEvalError(f"only {len(pool)} eligible reviewed golden cases; cannot select {n}")
    pool["_rank"] = [(not blind, _rank(seed, cid)) for blind, cid in zip(pool[BLIND_HUMAN], pool["case_id"])]
    picked: list[pd.DataFrame] = []
    for intent, k in _allocate(pool["gold_intent"].value_counts().to_dict(), n, min_per_intent).items():
        rows = pool[pool["gold_intent"] == intent]
        by_route = rows["gold_should_escalate"].value_counts().to_dict()
        floor = 1 if k >= len(by_route) else 0
        for route, m in _allocate(by_route, k, floor).items():
            stratum = rows[rows["gold_should_escalate"] == route].sort_values("_rank")
            picked.append(stratum.head(m))
    out = pd.concat(picked).drop(columns="_rank").sort_values("labeling_order").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------------------------------------
# Cached model calls
# --------------------------------------------------------------------------------------------------------------------
class CachingModel:
    """`LanguageModel` wrapper that replays cached outputs and records new ones in a JSONL file."""

    def __init__(self, inner: Any, cache_path: Path):
        self.inner, self.name, self.cache_path = inner, inner.name, Path(cache_path)
        self.cache = self._load()
        self.hits = self.misses = 0
        self.runtime_errors: list[str] = []

    def _load(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    out[rec["key"]] = rec
        return out

    def key(self, kind: str, prompt: str, system: str | None, temperature: float, max_output_tokens: int, schema: Any) -> str:
        payload = json.dumps([self.name, kind, system, prompt, float(temperature), int(max_output_tokens), schema], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _call(self, kind: str, fn: Callable[[], Any], key: str) -> Any:
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            if hit.get("error"):
                raise ModelOutputError(hit["error"])
            return hit["output"]
        self.misses += 1
        try:
            output, error = fn(), None
        except ModelOutputError as exc:
            output, error = None, str(exc)[:500]
        except ModelConfigError:
            raise
        except ModelRuntimeError as exc:
            self.runtime_errors.append(str(exc)[:300])
            raise
        rec = {"key": key, "model": self.name, "kind": kind, "output": output, "error": error}
        self.cache[key] = rec
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if error:
            raise ModelOutputError(error)
        return output

    def generate_json(self, prompt: str, *, system: str | None = None, temperature: float = 0.0, max_output_tokens: int = 1024, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        key = self.key("json", prompt, system, temperature, max_output_tokens, schema)
        return dict(self._call("json", lambda: self.inner.generate_json(prompt, system=system, temperature=temperature, max_output_tokens=max_output_tokens, schema=schema), key))

    def generate_text(self, prompt: str, *, system: str | None = None, temperature: float = 0.0, max_output_tokens: int = 1024) -> str:
        key = self.key("text", prompt, system, temperature, max_output_tokens, None)
        return str(self._call("text", lambda: self.inner.generate_text(prompt, system=system, temperature=temperature, max_output_tokens=max_output_tokens), key))


# --------------------------------------------------------------------------------------------------------------------
# Running and recording
# --------------------------------------------------------------------------------------------------------------------
def stage_outcomes(result: AgentResult) -> dict[str, Any]:
    """What happened at each stage, read from the AgentResult the agent returned."""
    reasons = result.decision.reasons
    gen_error = any(r.startswith("reply generation failed") for r in reasons)
    declined = any(r.startswith("the generator returned no reply") for r in reasons)
    produced = bool(result.reply or result.internal_draft_reply)
    generation = GEN_ERROR if gen_error else GEN_DECLINED if declined else GEN_PRODUCED if produced else GEN_NOT_ATTEMPTED
    g = result.grounding
    if g is None:
        grounding = GROUND_NOT_RUN
    elif any(c.startswith("could not verify") for c in g.unsupported_claims):
        grounding = GROUND_UNAVAILABLE
    else:
        grounding = GROUND_PASS if g.grounded else GROUND_FAIL
    return {
        "policy_action": AUTO_HANDLE if generation != GEN_NOT_ATTEMPTED else ESCALATE,
        "generation": generation,
        "grounding": grounding,
        "classification_failed": any(r.startswith("classification failed") for r in reasons),
    }


def case_record(case: Any, result: AgentResult, invariant_errors: list[str]) -> dict[str, Any]:
    c, d = result.classification, result.decision
    return {
        "case_id": case.case_id,
        "labeling_order": int(case.labeling_order),
        "provenance": case.provenance,
        "blind_human": bool(getattr(case, BLIND_HUMAN)),
        "message": case.text,
        "gold_intent": case.gold_intent,
        "gold_should_escalate": case.gold_should_escalate,
        "gold_resolution_type": case.gold_resolution_type,
        "pred_intent": c.intent,
        "pred_confidence": c.confidence,
        "multi_intent": c.multi_intent,
        **stage_outcomes(result),
        "final_action": d.action,
        "reasons": d.reasons,
        "reply": result.reply,
        "internal_draft_reply": result.internal_draft_reply,
        "grounding_confidence": result.grounding.confidence if result.grounding else None,
        "unsupported_claims": result.grounding.unsupported_claims if result.grounding else [],
        "evidence": [
            {"case_id": e.case_id, "similarity": e.semantic_similarity, "intent": e.intent, "resolution_type": e.resolution_type,
             "customer_problem": e.customer_problem[:300], "historical_response": e.historical_response[:400]}
            for e in result.evidence
        ],
        "invariant_errors": invariant_errors,
    }


def run_agent_cases(
    agent: Any,
    cases: pd.DataFrame,
    model: CachingModel,
    *,
    allowed_intents: Iterable[str],
    corpus_case_ids: set[str],
    split_by_case: dict[str, str],
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Run `agent.handle` on each opening message. Stops at the first runtime model failure; that case is not recorded."""
    records: list[dict[str, Any]] = []
    allowed = list(allowed_intents)
    for i, case in enumerate(cases.itertuples(index=False), 1):
        before = len(model.runtime_errors)
        result = agent.handle(message=case.text, conversation_context=None)
        if len(model.runtime_errors) > before:
            return records, f"stopped at case {i}/{len(cases)} ({case.case_id}): {model.runtime_errors[-1]}"
        rec = case_record(case, result, check_result_invariants(result, allowed, corpus_case_ids, split_by_case))
        records.append(rec)
        if progress:
            progress(i, len(cases), rec)
    return records, None


# --------------------------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------------------------
def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def reply_stats(texts: list[str]) -> dict[str, Any]:
    words = [len(t.split()) for t in texts]
    return {
        "count": len(texts),
        "mean_words": mean(words) if words else None,
        "median_words": median(words) if words else None,
        "share_under_60_words": _ratio(sum(w <= 60 for w in words), len(words)),
        "share_with_link": _ratio(sum(bool(_URL.search(t)) for t in texts), len(texts)),
        "share_asking_question": _ratio(sum("?" in t for t in texts), len(texts)),
        "share_mentioning_dm": _ratio(sum(bool(re.search(r"\b(dm|direct message)\b", t, re.I)) for t in texts), len(texts)),
    }


def agent_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    final_auto = [r for r in records if r["final_action"] == AUTO_HANDLE]
    final_esc = [r for r in records if r["final_action"] == ESCALATE]
    gold_yes = [r for r in records if r["gold_should_escalate"] == "yes"]
    esc_yes = sum(r["gold_should_escalate"] == "yes" for r in final_esc)
    auto_yes = sum(r["gold_should_escalate"] == "yes" for r in final_auto)
    auto_no = len(final_auto) - auto_yes
    policy_auto = [r for r in records if r["policy_action"] == AUTO_HANDLE]
    gen = Counter(r["generation"] for r in policy_auto)
    ground = Counter(r["grounding"] for r in records if r["grounding"] != GROUND_NOT_RUN)
    drafts = [r["reply"] or r["internal_draft_reply"] for r in records if r["reply"] or r["internal_draft_reply"]]
    return {
        "n": n,
        "gold_should_escalate": {"yes": len(gold_yes), "no": n - len(gold_yes)},
        "final": {"auto_handle": len(final_auto), "escalate": len(final_esc)},
        "routing_confusion": {
            "auto_handle_gold_no": auto_no, "auto_handle_gold_yes": auto_yes,
            "escalate_gold_yes": esc_yes, "escalate_gold_no": len(final_esc) - esc_yes,
        },
        "escalation_coverage": _ratio(len(final_esc), n),
        "safe_automation_rate": _ratio(auto_no, n),
        "false_auto_handle_rate": _ratio(auto_yes, len(final_auto)),
        "escalation_precision": _ratio(esc_yes, len(final_esc)),
        "escalation_recall": _ratio(esc_yes, len(gold_yes)),
        "intent_accuracy": _ratio(sum(r["pred_intent"] == r["gold_intent"] for r in records), n),
        "policy_auto_handle": len(policy_auto),
        "generation": {
            "attempted": len(policy_auto), "produced": gen[GEN_PRODUCED], "declined": gen[GEN_DECLINED], "error": gen[GEN_ERROR],
            "success_rate": _ratio(gen[GEN_PRODUCED], len(policy_auto)),
        },
        "grounding": {
            "run": sum(ground.values()), "pass": ground[GROUND_PASS], "fail": ground[GROUND_FAIL], "unavailable": ground[GROUND_UNAVAILABLE],
            "pass_rate": _ratio(ground[GROUND_PASS], sum(ground.values())),
        },
        "classification_failures": sum(r["classification_failed"] for r in records),
        "invariant_violations": sum(bool(r["invariant_errors"]) for r in records),
        "reply_quality": {
            "all_drafts": reply_stats(drafts),
            "sent_replies": reply_stats([r["reply"] for r in records if r["reply"]]),
            "drafts_awaiting_judge": len(drafts),
        },
        "escalation_reasons": dict(Counter(_reason_key(r) for r in final_esc).most_common()),
    }


def _reason_key(record: dict[str, Any]) -> str:
    """The first reason a case was escalated, with case-specific detail stripped."""
    if record["generation"] == GEN_ERROR:
        return "reply generation failed"
    if record["generation"] == GEN_DECLINED:
        return "generator declined to reply"
    if record["grounding"] in (GROUND_FAIL, GROUND_UNAVAILABLE):
        return f"grounding {record['grounding']}"
    first = (record["reasons"] or ["no reason"])[0]
    first = re.sub(r"'[^']*'", "'…'", first)
    return re.split(r"[:(]", first, maxsplit=1)[0].strip()


def judge_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One item per produced draft, with the evidence it was written from and blank judge fields."""
    items = []
    for r in records:
        draft = r["reply"] or r["internal_draft_reply"]
        if not draft:
            continue
        items.append({
            "case_id": r["case_id"], "provenance": r["provenance"], "customer_message": r["message"],
            "gold_intent": r["gold_intent"], "gold_should_escalate": r["gold_should_escalate"], "gold_resolution_type": r["gold_resolution_type"],
            "reply": draft, "sent_to_customer": bool(r["reply"]), "grounding": r["grounding"], "unsupported_claims": r["unsupported_claims"],
            "evidence": [{"customer_problem": e["customer_problem"], "historical_response": e["historical_response"]} for e in r["evidence"]],
            "judge": {k: None for k in JUDGE_RUBRIC} | {"notes": ""},
        })
    return items


# --------------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------------
def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None) -> str:
    return "–" if x is None else f"{x:.1f}"


def render_markdown(results: dict[str, Any]) -> str:
    meta, subsets = results["meta"], results["subsets"]
    names = list(subsets)
    lines = [
        "# End-to-end agent evaluation on a golden slice",
        "",
        "Generated by `python scripts/evaluate_agent.py`. The real `SupportAgent.handle` ran on each case's opening message; "
        "gold routing is `gold_should_escalate`.",
        "",
        f"- Model: `{meta['model']}` via {meta['provider']}. Model calls: {meta['cache_hits']} from cache, {meta['cache_misses']} new.",
        f"- Sample: {meta['selected']} reviewed golden cases, stratified by gold intent and gold escalation (seed {meta['seed']}); "
        f"{meta['blind_human_selected']} have blind human labels.",
        f"- Scored: {meta['scored']} / {meta['selected']}." + (f" **Incomplete: {meta['stopped']}**" if meta.get("stopped") else ""),
        "",
        "## Routing",
        "",
        "| metric | definition | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    rows = [
        ("cases", "", lambda m: str(m["n"])),
        ("escalation coverage", "share of cases sent to a human", lambda m: _pct(m["escalation_coverage"])),
        ("safe automation rate", "auto-handled and gold says no escalation, over all cases", lambda m: _pct(m["safe_automation_rate"])),
        ("false auto-handle rate", "auto-handled but gold says escalate, over auto-handled", lambda m: _pct(m["false_auto_handle_rate"])),
        ("escalation precision", "gold says escalate, over escalated", lambda m: _pct(m["escalation_precision"])),
        ("escalation recall", "escalated, over gold says escalate", lambda m: _pct(m["escalation_recall"])),
        ("intent accuracy", "predicted intent equals gold intent", lambda m: _pct(m["intent_accuracy"])),
    ]
    for label, definition, fn in rows:
        lines.append(f"| {label} | {definition} | " + " | ".join(fn(subsets[s]) for s in names) + " |")
    m = subsets[names[0]]
    rc = m["routing_confusion"]
    lines += [
        "",
        f"Routing confusion ({names[0]}): auto-handled {rc['auto_handle_gold_no']} correctly and {rc['auto_handle_gold_yes']} that needed a human; "
        f"escalated {rc['escalate_gold_yes']} correctly and {rc['escalate_gold_no']} that could have been automated.",
        "",
        "## Pipeline stages",
        "",
        f"- Policy AUTO_HANDLE: {m['policy_auto_handle']} of {m['n']}.",
        f"- Reply generation: {m['generation']['produced']} produced, {m['generation']['declined']} declined, {m['generation']['error']} errors "
        f"(success rate {_pct(m['generation']['success_rate'])}).",
        f"- Grounding: {m['grounding']['pass']} pass, {m['grounding']['fail']} fail, {m['grounding']['unavailable']} unavailable "
        f"(pass rate {_pct(m['grounding']['pass_rate'])}).",
        f"- Classification failures: {m['classification_failures']}. Invariant violations: {m['invariant_violations']}.",
        "",
        "Why cases were escalated:",
        "",
        "| reason | cases |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in m["escalation_reasons"].items()] or ["| none | 0 |"]
    lines += ["", "## Replies (inputs for the LLM judge)", "", "| | drafts produced | sent to customer |", "|---|---|---|"]
    q = m["reply_quality"]
    for label, key, fmt in [
        ("count", "count", str), ("mean words", "mean_words", _num), ("median words", "median_words", _num),
        ("60 words or fewer", "share_under_60_words", _pct), ("contains a link", "share_with_link", _pct),
        ("asks a question", "share_asking_question", _pct), ("mentions DM", "share_mentioning_dm", _pct),
    ]:
        lines.append(f"| {label} | {fmt(q['all_drafts'][key])} | {fmt(q['sent_replies'][key])} |")
    lines += [
        "",
        f"{q['drafts_awaiting_judge']} draft(s) are in `judge_inputs.jsonl` with empty judge fields. Rubric:",
        "",
    ]
    lines += [f"- `{k}`: {v}" for k, v in JUDGE_RUBRIC.items()]
    lines += ["", "## Per case", "", "| order | case | gold intent | gold escalate | predicted | policy | generation | grounding | final |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results["records"]:
        lines.append(
            f"| {r['labeling_order']} | {r['case_id']} | {r['gold_intent']} | {r['gold_should_escalate']} | {r['pred_intent']} | "
            f"{r['policy_action']} | {r['generation']} | {r['grounding']} | {r['final_action']} |"
        )
    return "\n".join(lines) + "\n"
