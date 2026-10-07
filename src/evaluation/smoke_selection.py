"""Deterministic, leakage-safe case selection for the live smoke test (no model code lives here).

Queries are opening customer messages of NON-golden cases. Selection is rule-based over the message text (plus weak candidate
metadata for the scenarios that need it), with ties broken by a seeded hash, never by row order or a random draw.

Everything labelled "historical" or "candidate" here is weak metadata (rule-derived resolution labels, candidate-taxonomy
intents). It is used to choose and later to *inspect* cases; it is never sent to the model and is NOT human ground truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from evaluation.splits import DEV, EXCLUDED, GOLDEN, RESERVE, TRAIN, opener_key
from ingestion.resolution_memory import build_memory_records
from taxonomy.registry import FALLBACK_INTENT, cluster_intent_map, load_labels

ALLOWED_SPLITS = frozenset({TRAIN, DEV})
# Queries are drawn from dev_calibration only: train cases are in the retrieval corpus, so a train query would retrieve itself.
SELECTION_SPLITS = frozenset({DEV})
EXTRA = "EXTRA"
ESCALATION_LIKE_SIGNALS = frozenset({"customer_relations_or_formal_route", "dm_for_account_lookup", "handoff_to_other_operator"})

_MENTION, _URL, _HASHTAG = re.compile(r"@\w+"), re.compile(r"https?://\S+"), re.compile(r"#\w+")
_WORD = re.compile(r"[a-z0-9£']+")
_TIME = re.compile(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b")


class SmokeSafetyError(RuntimeError):
    """A selected case is not safe to send to the model (golden, reserve, excluded, unknown split, or golden provenance overlap)."""


# --------------------------------------------------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SafetyContext:
    """Split metadata for every case plus the provenance of the golden set (customers, conversations, groups, tweets, openers)."""

    split_by_case: dict[str, str]
    golden_customers: frozenset[str] = frozenset()
    golden_conversations: frozenset[str] = frozenset()
    golden_groups: frozenset[str] = frozenset()
    golden_tweets: frozenset[int] = frozenset()
    golden_openers: frozenset[str] = frozenset()


def _ids(values: Iterable[Any]) -> set[int]:
    return {int(v) for ids in values for v in ids}


def build_safety_context(assigned: pd.DataFrame) -> SafetyContext:
    """From a frame with case_id, split, customer_id, conversation_id, group_id, source_tweet_ids, context_tweet_ids, opening_message."""
    golden = assigned[assigned["split"] == GOLDEN]
    return SafetyContext(
        split_by_case=dict(zip(assigned["case_id"], assigned["split"])),
        golden_customers=frozenset(golden["customer_id"].astype(str)),
        golden_conversations=frozenset(golden["conversation_id"].astype(str)),
        golden_groups=frozenset(golden["group_id"].astype(str)),
        golden_tweets=frozenset(_ids(golden["source_tweet_ids"]) | _ids(golden["context_tweet_ids"])),
        golden_openers=frozenset(k for k in golden["opening_message"].map(opener_key) if k),
    )


def validate_case_safety(case: "SmokeCase", safety: SafetyContext) -> None:
    """Raise SmokeSafetyError unless the case is train/dev AND shares no customer, conversation, group, tweet or opener with golden."""
    split = safety.split_by_case.get(case.case_id)
    if split is None or (isinstance(split, float) and split != split):
        raise SmokeSafetyError(f"{case.case_id}: no split metadata; refusing to use it")
    if split in (GOLDEN, RESERVE, EXCLUDED) or split not in ALLOWED_SPLITS:
        raise SmokeSafetyError(f"{case.case_id}: split is {split!r}; only {sorted(ALLOWED_SPLITS)} may be sent to the model")
    if case.split != split:
        raise SmokeSafetyError(f"{case.case_id}: the case says split {case.split!r} but the split assignments say {split!r}")
    if not isinstance(case.message, str) or not case.message.strip():
        raise SmokeSafetyError(f"{case.case_id}: empty customer message")
    overlaps = []
    if case.customer_id in safety.golden_customers:
        overlaps.append("customer")
    if case.conversation_id in safety.golden_conversations:
        overlaps.append("conversation")
    if case.group_id in safety.golden_groups:
        overlaps.append("group")
    if safety.golden_tweets & set(case.tweet_ids):
        overlaps.append("tweets")
    if opener_key(case.message) in safety.golden_openers:
        overlaps.append("opening text")
    if overlaps:
        raise SmokeSafetyError(f"{case.case_id}: shares golden provenance ({', '.join(overlaps)})")


# --------------------------------------------------------------------------------------------------------------------
# Cases and scenarios
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SmokeCase:
    """A selected opening message plus WEAK historical/candidate metadata used only for post-hoc inspection."""

    case_id: str
    split: str
    scenario: str
    message: str  # the only text sent to the agent
    why_selected: str
    candidate_intent: str | None
    resolution_type: str
    resolved: bool
    dm_redirect: bool
    escalation_signal: str
    customer_id: str = ""
    conversation_id: str = ""
    group_id: str = ""
    tweet_ids: tuple[int, ...] = ()
    boundary_intent: str | None = None  # second-nearest candidate cluster's intent (ambiguous scenario only)
    extra: dict[str, Any] = field(default_factory=dict)


def clean_message(text: str) -> str:
    """Lower-cased text with @mentions, URLs and hashtags removed (used only for selection rules)."""
    return " ".join(_HASHTAG.sub(" ", _URL.sub(" ", _MENTION.sub(" ", str(text or "")))).lower().split())


def n_words(text: str) -> int:
    return len(_WORD.findall(text))


def tiebreak(seed: int, case_id: str) -> float:
    """Seeded, order-independent value in [0, 1) used to break score ties deterministically."""
    return int(hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()[:12], 16) / float(16**12)


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


ANY_EXCLUDE = _rx(r"\b(refund\w*|compensat\w*|delay repay|complain\w*|rude|disgrace\w*|unacceptable)\b")

# Scenario scorers return None when the row is not a valid candidate, otherwise (score, short reason). Higher is better.
Scorer = Callable[[pd.Series, str, "PoolStats"], "tuple[float, str] | None"]


@dataclass(frozen=True)
class PoolStats:
    margin_cut: float  # 25th percentile of cluster margins in the pool: "close to a cluster boundary"


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    scorer: Scorer


def _keyword_scorer(intent: str, require: str, *, also: str | None = None, exclude: str | None = None, min_words: int = 5, max_words: int = 35, question: bool = False, avoid_all: bool = False) -> Scorer:
    """A scorer for single-intent scenarios: right candidate intent, required keywords, optional exclusions and length limits."""
    req, alt, exc = _rx(require), _rx(also) if also else None, _rx(exclude) if exclude else None

    def score(row: pd.Series, text: str, stats: PoolStats) -> tuple[float, str] | None:
        if row["intent"] != intent or not (min_words <= n_words(text) <= max_words):
            return None
        if question and "?" not in text:
            return None
        if exc and exc.search(text):
            return None
        if avoid_all and ANY_EXCLUDE.search(text):
            return None
        hits = {m.group(0) for m in req.finditer(text)}
        if not hits or (alt and not alt.search(text)):
            return None
        bonus = (1.0 if "?" in text else 0.0) + (1.0 if _TIME.search(text) else 0.0)
        return len(hits) + bonus, f"intent={intent}; keywords: {', '.join(sorted(hits)[:4])}"

    return score


def _score_service_status(row: pd.Series, text: str, stats: PoolStats) -> tuple[float, str] | None:
    inner = _keyword_scorer(
        "service_status_delay_enquiry", r"\b(running|delayed|cancel\w*|on time|late|leaving|depart\w*)\b",
        exclude=r"\b(refund\w*|compensat\w*|delay repay|complain\w*|disgrace\w*|unacceptable|ticket\w*|seat\w*)\b", min_words=5, max_words=25, question=True,
    )(row, text, stats)
    if inner is None:
        return None
    return inner[0] + (1.0 if re.search(r"\b(is|are|will|has)\b", text) else 0.0), inner[1] + "; explicit service-status question"


def _score_short(row: pd.Series, text: str, stats: PoolStats) -> tuple[float, str] | None:
    words = _WORD.findall(text)
    if not 1 <= len(words) <= 3 or not any(len(w) >= 3 and w.isalpha() for w in words):
        return None
    if re.fullmatch(r"(hi|hello|hey|thanks|thank you|morning|good morning|lol|ok|okay)\W*", text):
        return None
    request = bool(_rx(r"\b(refund\w*|update|help|delay\w*|compensat\w*|wifi|seat\w*|ticket\w*|cancel\w*|status|when|why|how)\b").search(text))
    return (2.0 if request else 0.0) + (1.0 if "?" in text else 0.0) + 0.1 * len(words), f"{len(words)} word(s)" + ("; contains a request word" if request else "")


_GROUPS = {
    "delay": _rx(r"\b(delay\w*|late|cancel\w*|stuck)\b"),
    "refund": _rx(r"\b(refund\w*|compensat\w*|delay repay|claim\w*)\b"),
    "seat": _rx(r"\b(seat\w*|reserv\w*)\b"),
    "complaint": _rx(r"\b(complain\w*|rude|unacceptable|disgrace\w*|appalling|shocking|awful|terrible)\b"),
    "booking": _rx(r"\b(ticket\w*|book\w*|fare\w*)\b"),
}
_MULTI_PAIRS = (("delay", "refund"), ("seat", "complaint"), ("booking", "refund"))


def _score_multi(row: pd.Series, text: str, stats: PoolStats) -> tuple[float, str] | None:
    if n_words(text) < 12 or n_words(text) > 60:
        return None
    present = {g for g, rx in _GROUPS.items() if rx.search(text)}
    pairs = [f"{a}+{b}" for a, b in _MULTI_PAIRS if a in present and b in present]
    if not pairs:
        return None
    return 3.0 * len(pairs) + (1.0 if "?" in text else 0.0), f"two support goals in one message: {', '.join(pairs)}"


_BOUNDARY_PAIRS = (
    frozenset({"service_status_delay_enquiry", "journey_disruption_complaint"}),
    frozenset({"ticket_booking_query", "delay_repay_refund_claim"}),
    frozenset({"seat_reservation_issue", "journey_disruption_complaint"}),
)


def _score_ambiguous(row: pd.Series, text: str, stats: PoolStats) -> tuple[float, str] | None:
    a, b, margin = row["intent"], row["boundary_intent"], row["margin"]
    if n_words(text) < 5 or pd.isna(margin) or not isinstance(a, str) or not isinstance(b, str) or a == b or FALLBACK_INTENT in (a, b) or margin > stats.margin_cut:
        return None
    named = frozenset({a, b}) in _BOUNDARY_PAIRS
    return (2.0 if named else 0.0) + (1.0 - margin / max(stats.margin_cut, 1e-9)), f"cluster margin {margin:.3f}; nearest intents {a} vs {b}" + ("; named boundary pair" if named else "")


_ESCALATION_TEXT = _rx(r"\b(complain\w*|compensat\w*|refund\w*|ombudsman|formal\w*|reference|still (waiting|no)|again|weeks|months|ignored|unacceptable|claim\w*)\b")


def _score_escalation(row: pd.Series, text: str, stats: PoolStats) -> tuple[float, str] | None:
    signal = row["escalation_signal"]
    hits = {m.group(0) for m in _ESCALATION_TEXT.finditer(text)}
    if n_words(text) < 8 or signal not in ESCALATION_LIKE_SIGNALS or not hits:
        return None
    return (2.0 if signal == "customer_relations_or_formal_route" else 1.0) + len(hits), f"historical signal {signal}; wording: {', '.join(sorted(hits)[:4])}"


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("service_status", "explicit train/service timing or status question", _score_service_status),
    Scenario("ticket_booking", "ticket purchase / booking / change question", _keyword_scorer("ticket_booking_query", r"\b(tickets?|book\w*|railcard|fares?|buy|purchase|advance|upgrade)\b", exclude=r"\b(refund\w*|compensat\w*|delay\w*|complain\w*|cancel\w*)\b", min_words=6, max_words=30, question=True)),
    Scenario("seat_reservation", "missing / unavailable / unhonoured reservation", _keyword_scorer("seat_reservation_issue", r"\b(seats?|reserv\w*)\b", also=r"\b(no|not|taken|double|wrong|someone|standing|without|missing|unavailable|didn't|isn't|wasn't|won't|can't|cannot)\b", exclude=r"\b(complain\w*|compensat\w*|refund\w*)\b", min_words=6)),
    Scenario("delay_repay_or_refund", "explicit compensation / refund / Delay Repay question", _keyword_scorer("delay_repay_refund_claim", r"\b(delay repay|refund\w*|compensat\w*|claim\w*)\b", exclude=r"\b(complain\w*|rude|disgrace\w*)\b", min_words=6)),
    Scenario("onboard_wifi", "specific on-board Wi-Fi problem", _keyword_scorer("onboard_wifi_issue", r"\b(wi-?fi|internet)\b", also=r"\b(not|n't|won't|can't|no|doesn't|keeps|slow|drop\w*|down|broken|working|connect\w*)\b", exclude=r"\b(complain\w*)\b")),
    Scenario("first_class_catering", "first-class food / drink / lounge issue", _keyword_scorer("first_class_catering_issue", r"\b(first class|1st class|catering|lounge|food|drinks?|breakfast|meal|coffee|tea)\b", also=r"\b(first class|1st class|catering|lounge)\b", min_words=5)),
    Scenario("customer_service_complaint", "staff / customer-service complaint, not a generic delay complaint", _keyword_scorer("customer_service_complaint", r"\b(staff|rude|guard|manager|conductor|customer service|attitude|unhelpful|ignored|treated)\b", exclude=r"\b(delay\w*|cancel\w*|refund\w*|compensat\w*)\b", min_words=8, max_words=40)),
    Scenario("praise", "explicit positive feedback or thanks", _keyword_scorer("praise_positive_feedback", r"\b(thank\w*|great|brilliant|excellent|fantastic|well done|lovely|helpful|amazing|good|best)\b", exclude=r"\?|\b(but|however|not|delay\w*|late|cancel\w*|refund\w*|complain\w*)\b|n't", max_words=25)),
    Scenario("short_low_information", "genuinely short opener (1-3 words)", _score_short),
    Scenario("multi_intent", "one message with two distinct support goals", _score_multi),
    Scenario("ambiguous_boundary_case", "message near a known candidate-cluster boundary", _score_ambiguous),
    Scenario("likely_escalation", "formal-complaint / compensation wording whose historical case went to a formal route, DM lookup or other operator", _score_escalation),
)
SCENARIO_NAMES = tuple(s.name for s in SCENARIOS)
# When --limit is smaller than the number of scenarios, keep the most diverse ones (edge cases first, then one of each intent family).
PRIORITY = ("service_status", "short_low_information", "multi_intent", "likely_escalation", "onboard_wifi", "delay_repay_or_refund", "ticket_booking",
            "seat_reservation", "first_class_catering", "customer_service_complaint", "praise", "ambiguous_boundary_case")
assert set(PRIORITY) == set(SCENARIO_NAMES)


# --------------------------------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------------------------------
def pool_stats(pool: pd.DataFrame) -> PoolStats:
    margins = pool["margin"].dropna()
    return PoolStats(margin_cut=float(margins.quantile(0.25)) if len(margins) else 0.0)


def eligible_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Rows that may be queries: from SELECTION_SPLITS, customer-opened, brand-answered, not a continuation. Raises on any unsafe split."""
    bad = pool[~pool["split"].isin(ALLOWED_SPLITS)]
    if len(bad):
        raise SmokeSafetyError(f"the candidate pool contains {len(bad)} case(s) from disallowed splits {sorted(set(bad['split']))}, e.g. {bad['case_id'].iloc[0]}")
    mask = pool["split"].isin(SELECTION_SPLITS) & pool["starts_with_customer"] & pool["has_brand_reply"] & ~pool["is_continuation"]
    return pool[mask & pool["message"].fillna("").str.strip().ne("")]


def select_case_for_scenario(scenario: Scenario, pool: pd.DataFrame, seed: int, used: set[str], stats: PoolStats) -> SmokeCase | None:
    """Best unused candidate for one scenario (highest score; ties broken by seeded hash), or None when nothing qualifies."""
    best_key: tuple[float, float] | None = None
    best: tuple[pd.Series, str] | None = None
    for row in pool.itertuples(index=False):
        if row.case_id in used:
            continue
        series = pd.Series(row._asdict())
        scored = scenario.scorer(series, clean_message(row.message), stats)
        if scored is None:
            continue
        key = (scored[0], -tiebreak(seed, row.case_id))
        if best_key is None or key > best_key:
            best_key, best = key, (series, scored[1])
    return None if best is None else _to_case(best[0], scenario.name, best[1])


def _to_case(row: pd.Series, scenario: str, why: str) -> SmokeCase:
    return SmokeCase(
        case_id=row["case_id"], split=row["split"], scenario=scenario, message=row["message"], why_selected=why,
        candidate_intent=row["intent"] if isinstance(row["intent"], str) else None,
        resolution_type=str(row["resolution_type"]), resolved=bool(row["resolved"]), dm_redirect=bool(row["dm_redirect"]),
        escalation_signal=str(row["escalation_signal"]), customer_id=str(row["customer_id"]), conversation_id=str(row["conversation_id"]),
        group_id=str(row["group_id"]), tweet_ids=tuple(int(t) for t in row["tweet_ids"]),
        boundary_intent=row["boundary_intent"] if isinstance(row["boundary_intent"], str) else None,
        extra={"evidence_quality": str(row.get("evidence_quality", "")), "cluster_margin": None if pd.isna(row["margin"]) else round(float(row["margin"]), 4)},
    )


@dataclass(frozen=True)
class Selection:
    cases: list[SmokeCase]
    unavailable: list[str]  # scenarios that were attempted but have no safe example
    requested: int


def select_smoke_cases(pool: pd.DataFrame, safety: SafetyContext, *, limit: int = 12, seed: int = 42) -> Selection:
    """Pick up to `limit` cases: scenarios in PRIORITY order first (skipping unavailable ones), then deterministic EXTRA cases.

    Every selected case is validated with `validate_case_safety`, which raises rather than skipping.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    candidates = eligible_pool(pool)
    stats = pool_stats(candidates)
    by_name = {s.name: s for s in SCENARIOS}
    chosen: dict[str, SmokeCase] = {}
    unavailable: list[str] = []
    used: set[str] = set()
    for name in PRIORITY:
        if len(chosen) >= limit:
            break
        case = select_case_for_scenario(by_name[name], candidates, seed, used, stats)
        if case is None:
            unavailable.append(name)
            continue
        chosen[name] = case
        used.add(case.case_id)
    cases = [chosen[n] for n in SCENARIO_NAMES if n in chosen]  # canonical order for display
    if limit > len(SCENARIOS):
        rest = candidates[~candidates["case_id"].isin(used) & (candidates["message"].map(lambda m: n_words(clean_message(m))) >= 4)]
        order = sorted(rest["case_id"], key=lambda c: tiebreak(seed, c))[: limit - len(SCENARIOS)]
        for cid in order:
            cases.append(_to_case(rest[rest["case_id"] == cid].iloc[0], EXTRA, "deterministic extra case (seeded hash order)"))
    for case in cases:
        validate_case_safety(case, safety)
    return Selection(cases=cases, unavailable=unavailable, requested=limit)


# --------------------------------------------------------------------------------------------------------------------
# Loading real data (the only part that reads files)
# --------------------------------------------------------------------------------------------------------------------
POOL_COLUMNS = (
    "case_id", "split", "message", "intent", "boundary_intent", "margin", "resolution_type", "resolved", "dm_redirect", "escalation_signal", "evidence_quality",
    "customer_id", "conversation_id", "group_id", "tweet_ids", "starts_with_customer", "has_brand_reply", "is_continuation",
)


def load_safety_frame(processed_dir: str | Path) -> pd.DataFrame:
    """All cases joined with their split assignment, in the layout `evaluation.splits.verify_no_leakage` expects."""
    p = Path(processed_dir)
    assignments = pd.read_csv(p / "splits" / "virgintrains_split_assignments.csv", dtype={"customer_id": str})
    cases = pd.read_parquet(p / "virgintrains_cases.parquet", columns=["case_id", "opening_message", "source_tweet_ids", "context_tweet_ids"])
    return assignments.merge(cases, on="case_id", how="left")


def load_pool(processed_dir: str | Path, cluster_labels_path: str | Path) -> pd.DataFrame:
    """Dev-calibration cases with weak candidate/historical metadata, built with the same code path as the retrieval evaluation."""
    p = Path(processed_dir)
    assignments = pd.read_csv(p / "splits" / "virgintrains_split_assignments.csv")
    split_of = dict(zip(assignments["case_id"], assignments["split"]))
    dev = pd.read_parquet(p / "splits" / "virgintrains_dev_calibration.parquet")
    preview = json.loads((p / "virgintrains_cluster_preview.json").read_text(encoding="utf-8"))
    cmap = cluster_intent_map(preview, load_labels(cluster_labels_path))
    intent_of = {cid: spec.get("final_intent") for cid, spec in cmap.items()}
    records = build_memory_records(dev, intent_of, split_of=split_of, cluster_of=dict(zip(dev["case_id"], dev["cluster_id"])))
    clusters = pd.read_parquet(p / "virgintrains_case_clusters.parquet", columns=["case_id", "second_cluster_id", "margin"]).drop_duplicates("case_id")
    df = dev[["case_id", "split", "opening_message", "customer_id", "conversation_id", "group_id", "source_tweet_ids", "starts_with_customer", "has_brand_reply", "is_continuation"]]
    df = df.merge(records[["case_id", "intent", "resolution_type", "resolved", "dm_redirect", "escalation_signal", "evidence_quality"]], on="case_id")
    df = df.merge(clusters, on="case_id", how="left")
    df["boundary_intent"] = df["second_cluster_id"].map(lambda c: None if pd.isna(c) else intent_of.get(int(c)))
    df = df.rename(columns={"opening_message": "message", "source_tweet_ids": "tweet_ids"})
    df["customer_id"] = df["customer_id"].astype(str)
    return df[list(POOL_COLUMNS)].reset_index(drop=True)
