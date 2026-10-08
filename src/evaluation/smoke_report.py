"""Run the agent on smoke-test cases and format what happened. Diagnostic only: no accuracy, no scores, no verdict on correctness.

All "historical"/"candidate" values come from weak rule-derived metadata and are NOT human ground truth.
"""

from __future__ import annotations

import hashlib
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agent import policy
from agent.config import PolicySettings
from agent.schemas import AUTO_HANDLE, ESCALATE, AgentResult
from evaluation.smoke_selection import ESCALATION_LIKE_SIGNALS, EXTRA, Selection, SmokeCase
from evaluation.splits import TRAIN
from models.base import ModelConfigError, ModelError
from taxonomy.registry import FALLBACK_INTENT

SETUP, LIVE_MODEL, AGENT_LOGIC = "SETUP FAILURE", "LIVE MODEL FAILURE", "AGENT LOGIC FAILURE"
WIDTH = 100
NO_REPLY_PREFIX = "the generator returned no reply"
WEAK_NOTE = "Historical metadata is weak/candidate reference only and is NOT human ground truth."

RETRIEVAL_FLAGS = ("no evidence retrieved", "retrieved cases have mixed resolution types", "retrieved cases mostly have a different candidate intent", "query case retrieved itself")
POLICY_FLAGS = ("AUTO_HANDLE on a historically escalation-like case", "AUTO_HANDLE in a boundary / multi-intent / low-information scenario")


class SetupFailure(RuntimeError):
    """The run cannot continue at all (e.g. the API key is missing or rejected)."""


@dataclass
class CaseOutcome:
    case: SmokeCase
    result: AgentResult | None = None
    failure: str | None = None  # one of SETUP / LIVE_MODEL / AGENT_LOGIC
    error: str | None = None
    invariant_errors: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    low_information: bool = False

    @property
    def failed(self) -> bool:
        return self.failure is not None or bool(self.invariant_errors)


# --------------------------------------------------------------------------------------------------------------------
# Invariants and observation flags
# --------------------------------------------------------------------------------------------------------------------
def check_result_invariants(result: Any, allowed_intents: Iterable[str], corpus_case_ids: set[str], split_by_case: dict[str, str]) -> list[str]:
    """Structural invariants of an AgentResult. Returns violations (empty = fine); never raises on a malformed result."""
    try:
        r = AgentResult.model_validate(result.model_dump() if hasattr(result, "model_dump") else result)
    except Exception as exc:  # noqa: BLE001 - any validation failure is a violation to report
        return [f"malformed AgentResult: {str(exc).splitlines()[0][:200]}"]
    errors: list[str] = []
    if r.classification.intent not in set(allowed_intents):
        errors.append(f"predicted intent {r.classification.intent!r} is not a configured candidate intent")
    if not 0.0 <= r.classification.confidence <= 1.0:
        errors.append("classifier confidence outside [0, 1]")
    if r.decision.action not in (AUTO_HANDLE, ESCALATE):
        errors.append(f"invalid decision action {r.decision.action!r}")
    for e in r.evidence:
        if e.case_id not in corpus_case_ids:
            errors.append(f"evidence {e.case_id} is not in the retrieval corpus")
        split = split_by_case.get(e.case_id)
        if split != TRAIN:
            errors.append(f"evidence {e.case_id} belongs to forbidden split {split!r}")
        if not e.source_tweet_ids:
            errors.append(f"evidence {e.case_id} has no source tweet ids")
    if r.decision.action == AUTO_HANDLE:
        if not (r.reply and r.reply.strip()):
            errors.append("AUTO_HANDLE without a reply")
        if r.grounding is None or not r.grounding.grounded:
            errors.append("AUTO_HANDLE without a passing grounding result")
    if r.grounding is not None and not r.grounding.grounded and r.decision.action != ESCALATE:
        errors.append("grounding failed but the final action is not ESCALATE")
    if r.decision.action == ESCALATE:
        if not r.decision.reasons:
            errors.append("ESCALATE without any reason")
        if r.reply:
            errors.append("ESCALATE but a customer-facing reply is set")
    return errors


def model_failure_notes(result: AgentResult) -> list[str]:
    """Signs in a result that a live model call failed or returned unusable output (the agent degrades these to ESCALATE)."""
    notes = [r for r in result.decision.reasons if r.startswith(("classification failed", "reply generation failed"))]
    if result.grounding is not None:
        notes += [c for c in result.grounding.unsupported_claims if c.startswith("could not verify")]
    return notes


def observation_flags(case: SmokeCase, result: AgentResult, low_information: bool) -> list[str]:
    """Diagnostic flags that need human inspection. They are not verdicts."""
    flags: list[str] = []
    c, d, ev = result.classification, result.decision, result.evidence
    if case.candidate_intent and c.intent != case.candidate_intent:
        flags.append(f"predicted intent differs from the historical candidate intent ({c.intent} vs {case.candidate_intent})")
    if not ev:
        flags.append("no evidence retrieved")
    else:
        top = Counter(e.resolution_type for e in ev).most_common(1)[0][1] / len(ev)
        if top < 0.5:
            flags.append("retrieved cases have mixed resolution types")
        if sum(e.intent != c.intent for e in ev) / len(ev) > 0.5:
            flags.append("retrieved cases mostly have a different candidate intent")
        if case.case_id in {e.case_id for e in ev}:
            flags.append("query case retrieved itself")
    if low_information:
        flags.append("low-information query")
    if c.multi_intent:
        flags.append("multi-intent escalation" if d.action == ESCALATE else "multi-intent flagged")
    if c.intent == FALLBACK_INTENT:
        flags.append("classified as unclear_or_media_only")
    flags += [f"generator returned no reply ({r.split(': ', 1)[-1][:160]})" for r in d.reasons if r.startswith(NO_REPLY_PREFIX)]
    if result.grounding is not None and not result.grounding.grounded:
        flags.append("grounding failed")
    auto = d.action == AUTO_HANDLE
    if auto and (case.escalation_signal in ESCALATION_LIKE_SIGNALS or case.dm_redirect):
        flags.append("AUTO_HANDLE on a historically escalation-like case")
    if auto and (case.scenario in {"ambiguous_boundary_case", "multi_intent", "short_low_information"} or low_information or c.multi_intent):
        flags.append("AUTO_HANDLE in a boundary / multi-intent / low-information scenario")
    flags += [f"model failure: {n[:120]}" for n in model_failure_notes(result)]
    return flags


# --------------------------------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------------------------------
def scrub_secret(text: str, secrets: Iterable[str]) -> str:
    """Remove any API key from text that may be printed."""
    for s in secrets:
        if s:
            text = text.replace(s, "[REDACTED]")
    return text


def run_case(
    agent: Any,
    case: SmokeCase,
    *,
    allowed_intents: Iterable[str],
    corpus_case_ids: set[str],
    split_by_case: dict[str, str],
    policy_settings: PolicySettings | None = None,
    secrets: Iterable[str] = (),
) -> CaseOutcome:
    """Send ONE opening message through the real `agent.handle` and check the result. Never calls the pieces separately."""
    out = CaseOutcome(case=case)
    if not isinstance(case.message, str) or not case.message.strip():
        out.failure, out.error = AGENT_LOGIC, "empty customer message (refusing to call the model)"
        return out
    try:
        result = agent.handle(message=case.message, conversation_context=None)
    except ModelConfigError as exc:
        raise SetupFailure(scrub_secret(str(exc), secrets)) from exc
    except ModelError as exc:
        out.failure, out.error = LIVE_MODEL, scrub_secret(f"{type(exc).__name__}: {exc}", secrets)
        return out
    except Exception as exc:  # noqa: BLE001 - report, then continue with the next case
        out.failure, out.error = AGENT_LOGIC, scrub_secret(f"{type(exc).__name__}: {exc}", secrets)
        return out
    out.invariant_errors = check_result_invariants(result, allowed_intents, corpus_case_ids, split_by_case)
    if out.invariant_errors and str(out.invariant_errors[0]).startswith("malformed"):
        out.failure, out.error = AGENT_LOGIC, out.invariant_errors[0]
        return out
    out.result = AgentResult.model_validate(result.model_dump())
    out.low_information = policy.count_words(case.message) < (policy_settings.min_message_words if policy_settings else 3)
    out.flags = observation_flags(case, out.result, out.low_information)
    if model_failure_notes(out.result):
        out.failure, out.error = LIVE_MODEL, "; ".join(model_failure_notes(out.result))[:300]
    return out


def run_cases(agent: Any, cases: Iterable[SmokeCase], **kwargs: Any) -> Iterable[CaseOutcome]:
    """Yield outcomes one at a time so the caller can print progress. A SetupFailure propagates and stops the run."""
    for case in cases:
        yield run_case(agent, case, **kwargs)


# --------------------------------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------------------------------
def _wrap(text: str, indent: str = "") -> str:
    return textwrap.fill(" ".join(str(text).split()), width=WIDTH, initial_indent=indent, subsequent_indent=indent) or f"{indent}<none>"


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def format_case_result(index: int, total: int, outcome: CaseOutcome, *, top_evidence: int = 3) -> str:
    """One clearly separated block per case, following the smoke-test layout."""
    case, bar = outcome.case, "=" * 60
    lines = [
        bar, f"CASE {index:02d} / {total:02d}", bar,
        "CASE ID:", case.case_id, "SOURCE SPLIT:", case.split, "SCENARIO:", case.scenario,
        "WHY SELECTED:", case.why_selected,
        "CUSTOMER MESSAGE:", _wrap(f'"{case.message}"'),
        "", "HISTORICAL METADATA", "-" * 19, WEAK_NOTE,
        "Candidate intent:", str(case.candidate_intent),
        "Historical resolution type:", case.resolution_type,
        "Historical resolved:", str(case.resolved),
        "Historical DM redirect:", str(case.dm_redirect),
        f"Historical escalation signal: {case.escalation_signal}",
    ]
    if case.boundary_intent:
        lines.append(f"Nearest other candidate intent (cluster boundary): {case.boundary_intent}")
    lines += ["", "AGENT OUTPUT", "-" * 12]
    if outcome.result is None:
        lines += [f"STATUS: FAILED  [{outcome.failure}]", _wrap(outcome.error or "no result", "  "), "END CASE", bar]
        return "\n".join(lines)
    r = outcome.result
    c, d = r.classification, r.decision
    lines += [
        "PREDICTED INTENT:", c.intent, "CLASSIFIER CONFIDENCE:", f"{c.confidence:.2f}",
        "ALTERNATIVE INTENT:", str(c.alternative_intent), "MULTI-INTENT:", str(c.multi_intent).lower(),
        "LOW-INFORMATION:", str(outcome.low_information).lower(), "TOP RETRIEVED EVIDENCE:",
    ]
    if not r.evidence:
        lines.append("  <none>")
    for i, e in enumerate(r.evidence[:top_evidence], 1):
        sim = f"{e.semantic_similarity:.2f}" if e.semantic_similarity is not None else "n/a"
        lines.append(f"{i}. {e.case_id} | score {e.score:.3f} | similarity {sim} | {e.resolution_type} | {e.intent}")
        lines.append(_wrap(f"problem: {_clip(e.customer_problem, 180)}", "     "))
        lines.append(_wrap(f"resolution: {_clip(e.resolution_summary, 140)}", "     "))
    lines += ["DECISION:", d.action, "DECISION CONFIDENCE:", f"{d.confidence:.2f}", "DECISION REASONS:"]
    lines += [_wrap(f"- {x}", "") for x in d.reasons] or ["- <none>"]
    lines += ["DRAFT REPLY:", _wrap(r.reply) if r.reply else "<none>"]
    if r.internal_draft_reply:
        lines += ["INTERNAL DRAFT (failed grounding; never sent):", _wrap(r.internal_draft_reply)]
    if r.grounding is None:
        lines += ["GROUNDING:", "NOT RUN", "GROUNDING CONFIDENCE:", "n/a", "UNSUPPORTED CLAIMS:", "none"]
    else:
        g = r.grounding
        lines += ["GROUNDING:", "PASS" if g.grounded else "FAIL", "GROUNDING CONFIDENCE:", f"{g.confidence:.2f}", "UNSUPPORTED CLAIMS:"]
        lines += [_wrap(f"- {u}") for u in g.unsupported_claims] or ["none"]
    lines += ["", "OBSERVATION FLAGS (diagnostic, need human inspection):"] + ([f"- {f}" for f in outcome.flags] or ["- none"])
    if outcome.invariant_errors or outcome.failure:
        lines += ["", f"STATUS: FAILED  [{outcome.failure or AGENT_LOGIC}]"] + [f"- {e}" for e in outcome.invariant_errors] + ([f"- {outcome.error}"] if outcome.error else [])
    lines += ["END CASE", bar]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Summary:
    requested: int
    executed: int
    scenarios_covered: list[str]
    scenarios_unavailable: list[str]
    live_status: str  # SUCCESS / PARTIAL / FAILED
    end_to_end_ok: int
    auto_handle: int
    escalate: int
    grounding_pass: int
    grounding_fail: int
    generator_no_reply: list[str]
    retrieval_concerns: list[str]
    policy_concerns: list[str]
    model_failures: list[str]
    logic_failures: list[str]
    status: str  # PASS / PARTIAL / FAIL
    next_actions: list[str]


def summarize_results(outcomes: list[CaseOutcome], selection: Selection) -> Summary:
    """Counts and concerns only. No accuracy, no headline score."""
    ok = [o for o in outcomes if o.result is not None and not o.invariant_errors]
    results = [o.result for o in outcomes if o.result is not None]
    retrieval = [o.case.case_id for o in outcomes if any(f in RETRIEVAL_FLAGS for f in o.flags)]
    policy_c = [o.case.case_id for o in outcomes if any(f in POLICY_FLAGS for f in o.flags)]
    model_f = [o.case.case_id for o in outcomes if o.failure in (LIVE_MODEL, SETUP)]
    logic_f = [o.case.case_id for o in outcomes if o.failure == AGENT_LOGIC or o.invariant_errors]
    failed = [o for o in outcomes if o.failed]
    if not outcomes or len(model_f) == len(outcomes):
        live = "FAILED"
    else:
        live = "PARTIAL" if model_f else "SUCCESS"
    if not outcomes or len(failed) == len(outcomes):
        status = "FAIL"
    elif failed or selection.unavailable or len(outcomes) < selection.requested:
        status = "PARTIAL"
    else:
        status = "PASS"
    grounding_fail = [o.case.case_id for o in outcomes if o.result is not None and o.result.grounding is not None and not o.result.grounding.grounded]
    no_reply = [o.case.case_id for o in outcomes if any(f.startswith("generator returned no reply") for f in o.flags)]
    actions = []
    if failed:
        actions.append(f"Look at the failed case(s) first: {', '.join(o.case.case_id for o in failed)} (categories: {', '.join(sorted({o.failure or AGENT_LOGIC for o in failed}))}).")
    if retrieval:
        actions.append(f"Read the retrieved evidence for the {len(retrieval)} case(s) with retrieval flags: {', '.join(retrieval)}.")
    if policy_c:
        actions.append(f"Read the replies of AUTO_HANDLE cases flagged for policy review: {', '.join(policy_c)}.")
    if no_reply:
        actions.append(f"Read why the generator returned no reply (DECISION REASONS) for: {', '.join(no_reply)}.")
    if grounding_fail:
        actions.append(f"Compare the internal drafts with their evidence for grounding failures: {', '.join(grounding_fail)}.")
    if selection.unavailable:
        actions.append(f"No safe example was found for: {', '.join(selection.unavailable)}.")
    auto_ids = [o.case.case_id for o in outcomes if o.result is not None and o.result.decision.action == AUTO_HANDLE]
    actions.append("Read every AUTO_HANDLE reply by eye against its cited evidence" + (f" ({', '.join(auto_ids)})." if auto_ids else " (none were auto-handled in this run)."))
    actions.append("Treat this as a smoke test only: the cases are few, chosen by rules, and the metadata is not human ground truth.")
    return Summary(
        requested=selection.requested, executed=len(outcomes), scenarios_covered=[o.case.scenario for o in outcomes if o.case.scenario != EXTRA],
        scenarios_unavailable=list(selection.unavailable), live_status=live, end_to_end_ok=len(ok),
        auto_handle=sum(r.decision.action == AUTO_HANDLE for r in results), escalate=sum(r.decision.action == ESCALATE for r in results),
        grounding_pass=sum(r.grounding is not None and r.grounding.grounded for r in results),
        grounding_fail=len(grounding_fail), generator_no_reply=no_reply, retrieval_concerns=retrieval, policy_concerns=policy_c, model_failures=model_f, logic_failures=logic_f,
        status=status, next_actions=actions,
    )


def format_summary(s: Summary) -> str:
    bar = "=" * 60
    covered = ", ".join(s.scenarios_covered) or "none"
    lines = [
        bar, "SMOKE TEST SUMMARY", bar,
        f"Requested cases:\n{s.requested}", f"Executed cases:\n{s.executed}", f"Scenarios covered:\n{covered}",
        "Scenarios unavailable:\n" + (", ".join(s.scenarios_unavailable) or "none"),
        f"Live Gemini requests:\n{s.live_status}", f"Cases with successful end-to-end AgentResult:\n{s.end_to_end_ok}",
        f"Cases with AUTO_HANDLE:\n{s.auto_handle}", f"Cases with ESCALATE:\n{s.escalate}",
        f"Grounding passes:\n{s.grounding_pass}", f"Grounding failures:\n{s.grounding_fail}",
        f"Cases where the generator returned no reply:\n{len(s.generator_no_reply)}" + (f"  ({', '.join(s.generator_no_reply)})" if s.generator_no_reply else ""),
        f"Cases with retrieval concerns:\n{len(s.retrieval_concerns)}" + (f"  ({', '.join(s.retrieval_concerns)})" if s.retrieval_concerns else ""),
        f"Cases with policy concerns:\n{len(s.policy_concerns)}" + (f"  ({', '.join(s.policy_concerns)})" if s.policy_concerns else ""),
        f"Cases with model/schema failures:\n{len(s.model_failures)}" + (f"  ({', '.join(s.model_failures)})" if s.model_failures else ""),
        f"Cases with agent-logic / invariant failures:\n{len(s.logic_failures)}" + (f"  ({', '.join(s.logic_failures)})" if s.logic_failures else ""),
        f"Overall smoke-test status:\n{s.status}", "", "NEXT ACTIONS", "-" * 12,
    ]
    lines += [_wrap(f"{i}. {a}") for i, a in enumerate(s.next_actions, 1)]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# Untouched-files check
# --------------------------------------------------------------------------------------------------------------------
def fingerprint_files(paths: Iterable[Path]) -> dict[str, str]:
    """sha256 per existing file ('MISSING' otherwise), to prove a run changed nothing."""
    out: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"
    return out


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(k for k in before if before[k] != after.get(k))
