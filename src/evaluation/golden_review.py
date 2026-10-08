"""Taxonomy review after the golden set is fully labelled: candidate (cluster-derived) intent vs human gold intent.

This reviews the TAXONOMY, not the agent: there are no accuracy figures, and nothing here changes the taxonomy. Signals
(merge / split / rename / add / unused) are simple count heuristics meant for a human decision.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pandas as pd

from evaluation.golden_eval import TAXONOMY_HEADING, case_status
from taxonomy.registry import cluster_intent_map

MERGE_MIN = 3
SPLIT_MIN_GOLD_INTENTS = 3
SPLIT_MAX_TOP_SHARE = 0.6
RENAME_MIN_SHARE = 0.5
NEW_MIN_CASES = 3


class ReviewNotReadyError(RuntimeError):
    pass


def candidate_intents(golden: pd.DataFrame, preview: dict, labels: dict) -> dict[str, str]:
    """case_id -> the candidate taxonomy's intent for the case's cluster (used only for this post-labeling report)."""
    cmap = cluster_intent_map(preview, labels)
    return {cid: str((cmap.get(int(cl)) or {}).get("final_intent") or "(no candidate intent)") for cid, cl in zip(golden["case_id"], golden["cluster_id"])}


def review_frame(pack: pd.DataFrame, candidate_of: dict[str, str]) -> pd.DataFrame:
    statuses = pack.apply(case_status, axis=1)
    if (statuses != "labelled").any():
        raise ReviewNotReadyError(f"{int((statuses == 'labelled').sum())} / {len(pack)} cases labelled; the review runs only when every case is labelled")
    out = pack.copy()
    out["candidate_intent"] = out["case_id"].map(candidate_of)
    if out["candidate_intent"].isna().any():
        raise ReviewNotReadyError("some golden cases have no candidate intent mapping")
    return out


def _md_table(header: list[str], rows: list[list]) -> list[str]:
    def cell(v) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]


def _short(text: str, n: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def merge_split_rename_signals(df: pd.DataFrame) -> list[str]:
    signals: list[str] = []
    cand_counts = Counter(df["candidate_intent"])
    gold_counts = Counter(df["gold_intent"])
    pair = Counter(zip(df["candidate_intent"], df["gold_intent"]))

    for a, b in combinations(sorted(set(cand_counts) | set(gold_counts)), 2):
        if a.startswith("NEW:") or b.startswith("NEW:"):
            continue
        ab, ba = pair[(a, b)], pair[(b, a)]
        if ab and ba and ab + ba >= MERGE_MIN:
            signals.append(f"**merge?** `{a}` and `{b}` are confused in both directions ({ab} candidate {a} → gold {b}, {ba} the other way).")
    for intent, n in sorted(cand_counts.items()):
        gold = Counter(df.loc[df["candidate_intent"] == intent, "gold_intent"])
        top, top_n = gold.most_common(1)[0]
        if len(gold) >= SPLIT_MIN_GOLD_INTENTS and top_n / n <= SPLIT_MAX_TOP_SHARE:
            spread = ", ".join(f"{g} {c}" for g, c in gold.most_common(4))
            signals.append(f"**split / redefine?** candidate `{intent}` ({n} cases) spreads over {len(gold)} gold intents: {spread}.")
        if top != intent and top_n / n >= RENAME_MIN_SHARE:
            signals.append(f"**rename / redefine?** {top_n} of {n} cases of candidate `{intent}` were labelled `{top}`.")
    unused = sorted(i for i in cand_counts if gold_counts[i] == 0 and not i.startswith("("))
    if unused:
        signals.append("**unused as gold:** " + ", ".join(f"`{i}`" for i in unused) + " (never chosen by the labeler in this sample).")
    for new, n in sorted(((g, c) for g, c in gold_counts.items() if g.startswith("NEW:")), key=lambda x: -x[1]):
        if n >= NEW_MIN_CASES:
            signals.append(f"**add?** `{new}` was proposed for {n} cases.")
    return signals or ["No signal reached the thresholds."]


def build_review_markdown(df: pd.DataFrame, *, registry_status: str | None, max_examples: int = 20) -> str:
    n = len(df)
    cand_counts = Counter(df["candidate_intent"])
    gold_counts = Counter(df["gold_intent"])
    differs = df["candidate_intent"] != df["gold_intent"]

    lines = [
        "# Golden set: taxonomy review",
        "",
        f"All {n} golden cases have final human-reviewed labels: 100 were labelled blind by a human, the rest began as assistant",
        "drafts that a human reviewed (see the README's provenance note). This compares the candidate taxonomy's cluster-derived",
        "intent with the final `gold_intent`, to decide whether the taxonomy should change before it is frozen. It is **not** an evaluation of the agent and",
        "reports no accuracy figures. Nothing here was applied automatically.",
        "",
        f"- Taxonomy reference: candidate registry, status `{registry_status}`. ({TAXONOMY_HEADING})",
        "- Golden cases were sampled with stratification, so these counts are not natural prevalence (see `golden_stratum_weight`).",
        "- The candidate intent was never shown to the labeler; it is joined here only after labeling finished.",
        "- If the taxonomy changes, the gold labels stay as the human entered them; any re-mapping must be explicit and reviewed.",
        "",
        "## Intent counts",
        "",
    ]
    names = sorted(set(cand_counts) | set(gold_counts), key=lambda x: (x.startswith("NEW:"), x))
    lines += _md_table(["intent", "candidate", "gold"], [[f"`{i}`", cand_counts.get(i, 0), gold_counts.get(i, 0)] for i in names])

    lines += ["", "## Candidate vs gold disagreement", "", f"Gold differs from the candidate intent in {int(differs.sum())} of {n} cases.", ""]
    rows = []
    for intent in sorted(cand_counts):
        sub = df[df["candidate_intent"] == intent]
        other = Counter(sub.loc[sub["gold_intent"] != intent, "gold_intent"])
        rows.append([f"`{intent}`", len(sub), int((sub["gold_intent"] == intent).sum()), int(len(sub) - (sub["gold_intent"] == intent).sum()), ", ".join(f"{g} ({c})" for g, c in other.most_common(3)) or "-"])
    lines += _md_table(["candidate intent", "cases", "gold same", "gold different", "most common gold instead"], rows)

    lines += ["", "## NEW intents proposed by the labeler", ""]
    new = df[df["gold_intent"].str.startswith("NEW:")]
    if new.empty:
        lines.append("None.")
    else:
        for name, sub in sorted(new.groupby("gold_intent"), key=lambda kv: -len(kv[1])):
            lines.append(f"### `{name}` ({len(sub)} case{'s' if len(sub) != 1 else ''})")
            lines.append("")
            lines += [f"- {r.case_id} (candidate `{r.candidate_intent}`): {_short(r.human_notes, 200)}  \n  > {_short(r.first_customer_message)}" for r in sub.head(max_examples).itertuples()]
            lines.append("")

    lines += ["", "## Major confusion pairs", "", "Unordered pairs of different intents, counted over cases where candidate and gold disagree.", ""]
    pairs = Counter(tuple(sorted((a, b))) for a, b in zip(df.loc[differs, "candidate_intent"], df.loc[differs, "gold_intent"]))
    lines += _md_table(["intent A", "intent B", "cases"], [[f"`{a}`", f"`{b}`", c] for (a, b), c in pairs.most_common(15)]) if pairs else ["None."]

    lines += ["", "## Difficult boundary examples", "", "Disagreements the labeler marked as low or medium confidence, then other disagreements.", ""]
    rank = df["gold_confidence"].map({"low": 0, "medium": 1, "": 2, "high": 3}).fillna(2)
    hard = df[differs].assign(_r=rank[differs]).sort_values(["_r", "case_id"]).head(max_examples)
    if hard.empty:
        lines.append("None.")
    else:
        lines += _md_table(
            ["case", "candidate", "gold", "confidence", "opening message", "notes"],
            [[r.case_id, r.candidate_intent, r.gold_intent, r.gold_confidence or "-", _short(r.first_customer_message), _short(r.human_notes, 120) or "-"] for r in hard.itertuples()],
        )

    lines += [
        "",
        "## Merge / split / rename signals",
        "",
        f"Heuristics for a human decision (merge: confused both ways in at least {MERGE_MIN} cases; split: at least {SPLIT_MIN_GOLD_INTENTS} gold "
        f"intents with none above {SPLIT_MAX_TOP_SHARE:.0%}; rename: at least {RENAME_MIN_SHARE:.0%} relabelled to one other intent; add: a NEW "
        f"intent with at least {NEW_MIN_CASES} cases).",
        "",
    ]
    lines += [f"- {s}" for s in merge_split_rename_signals(df)]
    return "\n".join(lines) + "\n"
