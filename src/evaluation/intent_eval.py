"""Intent-classification evaluation on the frozen VirginTrains golden set.

Systems: a majority-class baseline, TF-IDF + logistic regression, and the agent's LLM `IntentClassifier` (Groq or Gemini
backend; the prompt and taxonomy are the agent's, unchanged). Every system sees
only the opening customer message (what the agent receives in production); the human labels were made from the full
conversation.

Baselines are fitted on the weak, cluster-derived candidate intents of `train_retrieval` only; the logistic-regression C is
chosen on `dev_calibration`. Golden cases are never used for fitting or selection, and the loaders refuse any frame that
contains one.

Two reporting subsets:
  blind_human    labels written blind by a human (audit provenance `human`)
  all_reviewed   blind_human + assistant drafts a human confirmed or corrected
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from evaluation.golden_eval import HUMAN, GoldenLabelStore, label_provenance
from evaluation.splits import DEV, GOLDEN, TRAIN
from taxonomy.registry import cluster_intent_map, load_labels

logger = logging.getLogger(__name__)

BLIND_HUMAN = "blind_human"
ALL_REVIEWED = "all_reviewed"
SUBSETS = (BLIND_HUMAN, ALL_REVIEWED)
REVIEWED_PROVENANCE = frozenset({HUMAN, "assistant_draft_confirmed", "assistant_draft_corrected"})
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)

_URL = re.compile(r"https?://\S+")
_HANDLE = re.compile(r"@\w+")


class IntentEvalError(RuntimeError):
    """The evaluation inputs are unsafe or inconsistent (e.g. golden cases in training data)."""


def normalise_text(text: str) -> str:
    """Lower-case, with URLs and @handles replaced by placeholder tokens (the brand handle carries no intent signal)."""
    text = _URL.sub(" urltoken ", str(text or ""))
    text = _HANDLE.sub(" handletoken ", text)
    return " ".join(text.lower().split())


# --------------------------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------------------------
def golden_eval_frame(store: GoldenLabelStore) -> pd.DataFrame:
    """case_id, text, gold_intent, provenance and one boolean column per subset, from the verified golden pack."""
    frame = store.verified_frame()
    provenance = label_provenance(store.audit_path)
    out = pd.DataFrame(
        {
            "case_id": frame["case_id"],
            "labeling_order": frame["labeling_order"].astype(int),
            "text": frame["first_customer_message"],
            "gold_intent": frame["gold_intent"].str.strip(),
            "provenance": frame["case_id"].map(provenance).fillna(""),
        }
    )
    out[BLIND_HUMAN] = out["provenance"].eq(HUMAN) & out["gold_intent"].ne("")
    out[ALL_REVIEWED] = out["provenance"].isin(REVIEWED_PROVENANCE) & out["gold_intent"].ne("")
    return out.sort_values("labeling_order").reset_index(drop=True)


def weak_labelled_split(processed_dir: str | Path, cluster_labels_path: str | Path, split: str) -> pd.DataFrame:
    """case_id, text, label for one non-golden split; label = the candidate intent of the case's cluster (weak, not human)."""
    if split not in (TRAIN, DEV):
        raise IntentEvalError(f"baselines may only use {TRAIN} or {DEV}, not {split!r}")
    processed = Path(processed_dir)
    frame = pd.read_parquet(processed / "splits" / f"virgintrains_{split}.parquet", columns=["case_id", "cluster_id", "opening_message", "split"])
    assignments = pd.read_csv(processed / "splits" / "virgintrains_split_assignments.csv", usecols=["case_id", "split"])
    golden_ids = set(assignments.loc[assignments["split"] == GOLDEN, "case_id"])
    preview = json.loads((processed / "virgintrains_cluster_preview.json").read_text(encoding="utf-8"))
    intent_of = {cid: spec.get("final_intent") for cid, spec in cluster_intent_map(preview, load_labels(cluster_labels_path)).items()}
    out = pd.DataFrame(
        {
            "case_id": frame["case_id"],
            "split": frame["split"],
            "text": frame["opening_message"].fillna(""),
            "label": frame["cluster_id"].map(lambda c: intent_of.get(int(c))),
        }
    )
    out = out[out["label"].notna() & out["text"].str.strip().ne("")].reset_index(drop=True)
    assert_no_golden(out, golden_ids, expected_split=split)
    return out


def assert_no_golden(frame: pd.DataFrame, golden_ids: Iterable[str], *, expected_split: str) -> None:
    """Refuse a fitting frame that holds a golden case or rows from another split."""
    overlap = set(frame["case_id"]) & set(golden_ids)
    if overlap:
        raise IntentEvalError(f"{len(overlap)} golden case(s) found in {expected_split} data (e.g. {sorted(overlap)[0]}); refusing to fit")
    other = set(frame["split"]) - {expected_split}
    if other:
        raise IntentEvalError(f"{expected_split} data contains rows from other splits: {sorted(other)}")


# --------------------------------------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------------------------------------
class MajorityBaseline:
    """Always predicts the most frequent training label (ties broken alphabetically)."""

    name = "majority_class"

    def fit(self, texts: list[str], labels: list[str]) -> "MajorityBaseline":
        counts = Counter(labels)
        if not counts:
            raise IntentEvalError("cannot fit the majority baseline on no labels")
        top = max(counts.values())
        self.label_ = sorted(k for k, v in counts.items() if v == top)[0]
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [self.label_] * len(texts)


def tfidf_logreg(c: float, seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=normalise_text, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=seed)),
        ]
    )


@dataclass
class TfidfSelection:
    model: Pipeline
    c: float
    dev_macro_f1: dict[float, float]


def fit_tfidf_logreg(train: pd.DataFrame, dev: pd.DataFrame, grid: Iterable[float] = C_GRID, seed: int = 0) -> TfidfSelection:
    """Fit on train for each C, pick the best dev macro-F1 (weak labels), return the train-fitted model for that C."""
    scores: dict[float, float] = {}
    models: dict[float, Pipeline] = {}
    for c in grid:
        model = tfidf_logreg(c, seed).fit(train["text"].tolist(), train["label"].tolist())
        scores[c] = float(f1_score(dev["label"], model.predict(dev["text"].tolist()), average="macro", zero_division=0))
        models[c] = model
    best = max(scores, key=lambda c: (scores[c], -c))
    return TfidfSelection(models[best], best, scores)


# --------------------------------------------------------------------------------------------------------------------
# LLM classifier predictions (cached, resumable)
# --------------------------------------------------------------------------------------------------------------------
def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def load_prediction_cache(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[(rec["case_id"], rec["model"], rec["prompt_sha"])] = rec
    return cache


def llm_predictions(
    classifier: Any,
    cases: pd.DataFrame,
    cache_path: Path,
    *,
    model_name: str,
    fallback_intent: str,
    output_errors: tuple[type[BaseException], ...],
    stop_errors: tuple[type[BaseException], ...],
    pause_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, str], str | None]:
    """case_id -> predicted intent, reusing cached calls for the same model and prompt.

    An unparseable model output is scored as the fallback intent (what the agent does). A transport or quota error stops
    the run; cases without a prediction stay missing and the reason is returned.
    """
    cache = load_prediction_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    preds: dict[str, str] = {}
    called = 0
    for row in cases.itertuples(index=False):
        sha = prompt_sha(classifier.build_prompt(row.text, None))
        hit = cache.get((row.case_id, model_name, sha))
        if hit is None:
            if called and pause_seconds > 0:
                sleep(pause_seconds)
            called += 1
            try:
                c = classifier.classify(row.text, None)
                rec = {"intent": c.intent, "confidence": c.confidence, "error": None}
            except output_errors as exc:
                rec = {"intent": fallback_intent, "confidence": 0.0, "error": f"unparseable output: {exc}"[:300]}
            except stop_errors as exc:
                return preds, f"stopped after {len(preds)} prediction(s): {exc}"[:300]
            hit = {"case_id": row.case_id, "model": model_name, "prompt_sha": sha, **rec}
            with open(cache_path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(hit, ensure_ascii=False) + "\n")
        preds[row.case_id] = hit["intent"]
    return preds, None


# --------------------------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------------------------
def label_order(taxonomy_intents: list[str], *label_lists: Iterable[str]) -> list[str]:
    """Taxonomy order first, then any extra labels (e.g. NEW:...) alphabetically."""
    seen = set().union(*[set(x) for x in label_lists]) if label_lists else set()
    return [i for i in taxonomy_intents] + sorted(seen - set(taxonomy_intents))


def intent_metrics(gold: list[str], pred: list[str], taxonomy_intents: list[str]) -> dict[str, Any]:
    """Accuracy, macro-F1 over the intents present in gold, per-intent P/R/F1/support and the confusion matrix.

    Gold labels outside the taxonomy (NEW:...) stay in: no system can predict them, so they count as errors for everyone.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    if not gold:
        return {"n": 0}
    labels = label_order(taxonomy_intents, gold, pred)
    gold_labels = [l for l in labels if l in set(gold)]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    per = {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, l in enumerate(labels)}
    return {
        "n": len(gold),
        "accuracy": float(np.mean(np.array(gold) == np.array(pred))),
        "macro_f1": float(np.mean([per[l]["f1"] for l in gold_labels])),
        "macro_f1_labels": gold_labels,
        "per_intent": per,
        "confusion": {"labels": labels, "matrix": confusion_matrix(gold, pred, labels=labels).tolist()},
    }


def evaluate_systems(golden: pd.DataFrame, predictions: dict[str, dict[str, str]], taxonomy_intents: list[str]) -> dict[str, dict[str, Any]]:
    """subset -> system -> metrics (+ coverage). A system is scored only on the subset cases it has predictions for."""
    out: dict[str, dict[str, Any]] = {}
    for subset in SUBSETS:
        rows = golden[golden[subset]]
        out[subset] = {}
        for system, preds in predictions.items():
            have = rows[rows["case_id"].isin(preds)]
            m = intent_metrics(have["gold_intent"].tolist(), [preds[c] for c in have["case_id"]], taxonomy_intents)
            m["coverage"] = {"predicted": len(have), "subset_size": len(rows)}
            out[subset][system] = m
    return out


# --------------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------------
SUBSET_TITLES = {
    BLIND_HUMAN: "Blind human labels (primary)",
    ALL_REVIEWED: "All reviewed labels (blind human + human-confirmed assistant drafts)",
}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _short(label: str) -> str:
    return "".join(w[0] for w in label.replace("NEW:", "new_").split("_") if w).upper()


def render_markdown(results: dict[str, Any]) -> str:
    meta = results["meta"]
    lines = [
        "# Intent classification on the golden set",
        "",
        "Generated by `python scripts/evaluate_intents.py`. Input to every system: the opening customer message only. "
        "Gold: human labels made from the full conversation.",
        "",
        "## Setup",
        "",
        f"- Baselines fitted on `{TRAIN}` ({meta['train_cases']:,} cases), C chosen on `{DEV}` ({meta['dev_cases']:,} cases); "
        "their labels are weak cluster-derived candidate intents, not human labels. Golden was never used for fitting or selection.",
        f"- Majority class: `{meta['majority_label']}`.",
        f"- TF-IDF + logistic regression: C = {meta['tfidf_c']} (dev macro-F1 by C: "
        + ", ".join(f"{c}: {v:.3f}" for c, v in meta["tfidf_dev_macro_f1"].items())
        + ").",
        (
            "- LLM classifier: not run (`--skip-llm`)."
            if not meta.get("llm_model")
            else f"- LLM classifier: the agent's `IntentClassifier` on {meta['llm_provider']}, model `{meta['llm_model']}`. "
            + (f"**Incomplete: {meta['llm_stopped']}**" if meta.get("llm_stopped") else "All cases predicted.")
            + (f" {meta['llm_unparseable']} unparseable output(s) scored as the fallback intent." if meta.get("llm_unparseable") else "")
        ),
        f"- Label provenance: {meta['provenance']}.",
        "",
    ]
    for subset in SUBSETS:
        systems = results["subsets"][subset]
        lines += [f"## {SUBSET_TITLES[subset]}", "", "| system | cases scored | accuracy | macro-F1 |", "|---|---|---|---|"]
        for name, m in systems.items():
            cov = m["coverage"]
            if m["n"] == 0:
                lines.append(f"| {name} | 0 / {cov['subset_size']} | – | – |")
            else:
                lines.append(f"| {name} | {cov['predicted']} / {cov['subset_size']} | {_pct(m['accuracy'])} | {m['macro_f1']:.3f} |")
        lines.append("")
        for name, m in systems.items():
            if m["n"] == 0:
                continue
            lines += [f"### {name}: per intent", "", "| intent | support | precision | recall | F1 |", "|---|---|---|---|---|"]
            for label, v in m["per_intent"].items():
                lines.append(f"| {label} | {v['support']} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} |")
            conf = m["confusion"]
            abbrev = [_short(l) for l in conf["labels"]]
            lines += ["", f"<details><summary>{name}: confusion matrix (rows = gold, columns = predicted)</summary>", "", "| gold \\ pred | " + " | ".join(abbrev) + " |", "|---" * (len(abbrev) + 1) + "|"]
            for label, row in zip(conf["labels"], conf["matrix"]):
                lines.append(f"| {label} | " + " | ".join(str(v) if v else "·" for v in row) + " |")
            lines += ["", "Abbreviations: " + ", ".join(f"{a} = {l}" for a, l in zip(abbrev, conf["labels"])), "", "</details>", ""]
    return "\n".join(lines)
