"""Compare candidate intents with human labels on the taxonomy calibration set.

The point is a human-meaningful taxonomy, not a clustering score. Everything here is descriptive and
deterministic: confusion counts, purity, systematic disagreements, evidence for merges and splits, and
fallback quality. The calibration sample over-represents confusable regions, so rates *conditional on the
candidate intent* are meaningful but overall accuracy is not prevalence-faithful (a weighted figure is also given).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from evaluation.taxonomy_calibration import (
    ESCALATION_SIGNALS,
    HUMAN_COLUMNS,
    HUMAN_RESOLUTION_TYPES,
    HUMAN_RESOLVED_VALUES,
)

NEW_PREFIX = "new:"
FALLBACK_ALIASES = {"fallback", "unclear", "unclear_or_media_only"}
_ALSO = re.compile(r"also:\s*([a-z_]+)")


@dataclass
class Thresholds:
    min_support: int = 8          # candidate intents with fewer labelled cases get a low_support flag
    min_pair: int = 3             # minimum count for a (candidate, human) disagreement to be called systematic
    merge_rate: float = 0.30      # mutual confusion rate that suggests a merge
    split_purity: float = 0.60    # purity below this suggests heterogeneity
    split_share: float = 0.25     # each of the two main human intents must hold at least this share
    complete_fraction: float = 0.95  # below this the report is marked PRELIMINARY


@dataclass
class LabelLoad:
    frame: pd.DataFrame
    problems: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_complete: int = 0


def _norm(s: object) -> str:
    return " ".join(str(s).strip().lower().split()) if isinstance(s, str) else ""


def load_labelled(frame: pd.DataFrame, allowed_intents: list[str], fallback: str) -> LabelLoad:
    """Normalise and validate human labels. Returns only fully-labelled rows in `.frame`."""
    problems: list[str] = []
    df = frame.copy()
    for col in HUMAN_COLUMNS:
        if col not in df:
            problems.append(f"missing column {col}")
            df[col] = ""
    if df["case_id"].duplicated().any():
        problems.append("duplicate case_id rows: " + ", ".join(sorted(df.loc[df["case_id"].duplicated(), "case_id"].unique())[:5]))
    for col in HUMAN_COLUMNS:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else "") if col == "human_notes" else df[col].map(_norm)
    df["human_intent"] = df["human_intent"].map(lambda v: fallback if v in FALLBACK_ALIASES else v)

    allowed = set(allowed_intents)
    allowed_res, allowed_resolved, allowed_esc = set(HUMAN_RESOLUTION_TYPES), set(HUMAN_RESOLVED_VALUES), set(ESCALATION_SIGNALS)
    required = ["human_intent", "human_resolution_type", "human_resolved", "human_escalation_signal"]
    has_any = df[required].ne("").any(axis=1)
    complete = df[required].ne("").all(axis=1)
    for r in df[has_any & ~complete].itertuples():
        problems.append(f"{r.case_id}: partially labelled (missing {[c for c in required if getattr(r, c) == '']})")
    for r in df[complete].itertuples():
        if r.human_intent not in allowed and not r.human_intent.startswith(NEW_PREFIX):
            problems.append(f"{r.case_id}: unknown human_intent {r.human_intent!r}")
        if r.human_resolution_type not in allowed_res:
            problems.append(f"{r.case_id}: unknown human_resolution_type {r.human_resolution_type!r}")
        if r.human_resolved not in allowed_resolved:
            problems.append(f"{r.case_id}: unknown human_resolved {r.human_resolved!r}")
        if r.human_escalation_signal not in allowed_esc:
            problems.append(f"{r.case_id}: unknown human_escalation_signal {r.human_escalation_signal!r}")
    return LabelLoad(df[complete].reset_index(drop=True), problems, len(df), int(complete.sum()))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    pe = float(sum(pa.get(k, 0) * pb.get(k, 0) for k in set(pa.index) | set(pb.index)))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def _clip(text: object, n: int = 220) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def compare(df: pd.DataFrame, candidate_intents: list[str], fallback: str, th: Thresholds | None = None) -> dict:
    """`df`: fully-labelled rows with candidate_intent + human_* columns (and auto_* resolution columns)."""
    th = th or Thresholds()
    n = len(df)
    out: dict = {"n_labelled": n}
    if n == 0:
        return out
    cand, hum = df["candidate_intent"], df["human_intent"]
    agree = cand == hum
    weights = df["stratum_weight"].astype(float) if "stratum_weight" in df else pd.Series(1.0, index=df.index)
    out["overall"] = {
        "agreement": float(agree.mean()),
        "agreement_weighted_to_reserve_eligible": float((weights * agree).sum() / weights.sum()),
        "cohens_kappa": cohen_kappa(cand, hum),
        "note": "The sample over-represents confusable intents; the weighted figure re-balances within each candidate intent only.",
    }

    matrix = pd.crosstab(cand, hum)
    for name in candidate_intents:
        if name not in matrix.index:
            matrix.loc[name] = 0
    matrix = matrix.sort_index()
    pairs = (
        df.groupby(["candidate_intent", "human_intent"]).size().rename("n").reset_index().sort_values("n", ascending=False).reset_index(drop=True)
    )
    out["confusion_matrix"] = matrix
    out["confusion_pairs"] = pairs

    purity = []
    for name, g in df.groupby("candidate_intent"):
        vc = g["human_intent"].value_counts()
        k_agree = int((g["human_intent"] == name).sum())
        lo, hi = wilson(k_agree, len(g))
        modal, modal_n = vc.index[0], int(vc.iloc[0])
        purity.append(
            {
                "candidate_intent": name,
                "n": len(g),
                "agree_with_human": k_agree,
                "precision": k_agree / len(g),
                "precision_ci95": [round(lo, 3), round(hi, 3)],
                "purity": modal_n / len(g),
                "modal_human_intent": modal,
                "human_mix": {k: int(v) for k, v in vc.head(5).items()},
                "low_support": len(g) < th.min_support,
            }
        )
    out["purity"] = sorted(purity, key=lambda r: r["purity"])

    recall = []
    for name, g in df.groupby("human_intent"):
        k = int((g["candidate_intent"] == name).sum())
        recall.append(
            {
                "human_intent": name,
                "n": len(g),
                "recovered_by_candidate": k,
                "recall": k / len(g),
                "candidate_mix": {a: int(b) for a, b in g["candidate_intent"].value_counts().head(5).items()},
                "low_support": len(g) < th.min_support,
            }
        )
    out["recall"] = sorted(recall, key=lambda r: r["recall"])

    examples = []
    for r in pairs[(pairs["candidate_intent"] != pairs["human_intent"]) & (pairs["n"] >= th.min_pair)].itertuples():
        sub = df[(df["candidate_intent"] == r.candidate_intent) & (df["human_intent"] == r.human_intent)]
        examples.append(
            {
                "candidate_intent": r.candidate_intent,
                "human_intent": r.human_intent,
                "n": int(r.n),
                "examples": [
                    {"case_id": e.case_id, "message": _clip(e.first_customer_message), "human_notes": _clip(e.human_notes, 160)}
                    for e in sub.head(3).itertuples()
                ],
            }
        )
    out["systematic_disagreements"] = examples

    n_by_cand = cand.value_counts().to_dict()
    notes_also: dict[tuple[str, str], int] = {}
    for r in df.itertuples():
        for other in _ALSO.findall(r.human_notes.lower()):
            notes_also[(r.human_intent, other)] = notes_also.get((r.human_intent, other), 0) + 1
    merges = []
    names = sorted(set(cand) | set(candidate_intents))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if fallback in (a, b):
                continue
            ab = int(((cand == a) & (hum == b)).sum())
            ba = int(((cand == b) & (hum == a)).sum())
            denom = n_by_cand.get(a, 0) + n_by_cand.get(b, 0)
            rate = (ab + ba) / denom if denom else 0.0
            both_ways = min(ab, ba) >= 1
            if rate >= th.merge_rate and (ab + ba) >= th.min_pair and both_ways:
                hr = {x: df.loc[hum == x, "human_resolution_type"].value_counts(normalize=True) for x in (a, b)}
                shared = float(sum(min(hr[a].get(k, 0), hr[b].get(k, 0)) for k in set(hr[a].index) | set(hr[b].index)))
                merges.append(
                    {
                        "intents": [a, b],
                        "a_labelled_as_b": ab,
                        "b_labelled_as_a": ba,
                        "mutual_confusion_rate": round(rate, 3),
                        "human_resolution_type_overlap": round(shared, 3),
                        "human_also_mentions": notes_also.get((a, b), 0) + notes_also.get((b, a), 0),
                        "reading": "Humans use the two labels interchangeably, and agents handled them with similar resolution types."
                        if shared >= 0.6
                        else "Humans confuse the two labels, but agents handled them differently: keep separate unless the reviewer sees a shared customer goal.",
                    }
                )
    out["merge_candidates"] = sorted(merges, key=lambda m: -m["mutual_confusion_rate"])

    splits = []
    for p in purity:
        if p["low_support"] or p["candidate_intent"] == fallback:
            continue
        mix = df.loc[cand == p["candidate_intent"], "human_intent"].value_counts(normalize=True)
        if p["purity"] < th.split_purity and len(mix) >= 2 and mix.iloc[1] >= th.split_share:
            splits.append(
                {
                    "candidate_intent": p["candidate_intent"],
                    "n": p["n"],
                    "purity": round(p["purity"], 3),
                    "human_mix_shares": {k: round(float(v), 3) for k, v in mix.head(4).items()},
                    "reading": "Humans put this intent's cases into several different intents; split it or move its boundary.",
                }
            )
    out["split_candidates"] = sorted(splits, key=lambda s: s["purity"])

    new = df[df["human_intent"].str.startswith(NEW_PREFIX)]
    out["proposed_new_intents"] = [
        {
            "name": k[len(NEW_PREFIX):],
            "n": len(g),
            "from_candidates": {a: int(b) for a, b in g["candidate_intent"].value_counts().items()},
            "examples": [{"case_id": e.case_id, "message": _clip(e.first_customer_message), "human_notes": _clip(e.human_notes, 160)} for e in g.head(3).itertuples()],
        }
        for k, g in new.groupby("human_intent")
    ]

    cand_fb, hum_fb = cand == fallback, hum == fallback
    out["fallback_quality"] = {
        "candidate_fallback_n": int(cand_fb.sum()),
        "human_fallback_n": int(hum_fb.sum()),
        "precision": float((hum_fb & cand_fb).sum() / cand_fb.sum()) if cand_fb.sum() else None,
        "recall": float((hum_fb & cand_fb).sum() / hum_fb.sum()) if hum_fb.sum() else None,
        "candidate_fallback_cases_humans_assigned_real_intent": {
            k: int(v) for k, v in hum[cand_fb & ~hum_fb].value_counts().items()
        },
        "human_fallback_cases_candidate_missed": {k: int(v) for k, v in cand[hum_fb & ~cand_fb].value_counts().items()},
        "reading": "Fallback precision is the share of fallback cases that humans also could not assign; low precision means the fallback is "
        "hiding assignable requests. Fallback recall is the share of human-unclear cases the system sent to fallback.",
    }

    if "auto_resolution_type" in df:
        res_agree = df["auto_resolution_type"] == df["human_resolution_type"]
        out["resolution_type"] = {
            "agreement": float(res_agree.mean()),
            "per_auto_type": {
                t: {"n": len(g), "agreement": float((g["auto_resolution_type"] == g["human_resolution_type"]).mean()), "human_mix": {k: int(v) for k, v in g["human_resolution_type"].value_counts().head(4).items()}}
                for t, g in df.groupby("auto_resolution_type")
            },
            "confusion": pd.crosstab(df["auto_resolution_type"], df["human_resolution_type"]),
        }
        known = df[df["human_resolved"] != "unclear"]
        auto_resolved = known["auto_resolved"].astype(str).str.lower().map({"true": "yes", "false": "no"})
        out["resolved"] = {
            "n_comparable": len(known),
            "agreement": float((auto_resolved == known["human_resolved"]).mean()) if len(known) else None,
            "auto_yes_human_no": int(((auto_resolved == "yes") & (known["human_resolved"] == "no")).sum()),
            "auto_no_human_yes": int(((auto_resolved == "no") & (known["human_resolved"] == "yes")).sum()),
            "human_unclear": int((df["human_resolved"] == "unclear").sum()),
        }
    out["escalation_signals_by_human_intent"] = {
        k: {a: int(b) for a, b in g["human_escalation_signal"].value_counts().items()} for k, g in df.groupby("human_intent")
    }
    return out
