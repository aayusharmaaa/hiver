"""Proxy retrieval evaluation for the resolution memory. NOT human relevance ground truth.

Queries are non-golden `dev_calibration` cases (what a customer wrote before the first brand reply). The corpus is the
`train_retrieval` resolution memory. A historical case counts as *relevant* when, per the rule:

    intent_and_resolution (headline): same CANDIDATE intent AND same resolution type
    intent (secondary):               same CANDIDATE intent

Both labels are machine-derived: the intent from clustering opening messages (candidate, not validated by a human) and the
resolution type from rule-based signals in the brand's reply. Consequences to keep in mind when reading the numbers:

* the "+ intent" strategy and the relevance rule share the same candidate taxonomy, so that strategy is partly
  self-confirming; a noisy-intent stress test is included to show how it degrades when the candidate intent is wrong;
* the resolution type is a heuristic label, so some "irrelevant" results may in fact be sensible;
* Recall@k is the hit-rate form: the share of queries with at least one relevant case in the top k.

Weights are chosen on a tune half of the dev queries and every reported number is on the disjoint test half. Golden
cases are never used (as queries, as corpus, or for tuning).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from retrieval import ResolutionRetriever, clean_text, minmax_rows, tokenize
from taxonomy.registry import FALLBACK_INTENT

KS = (1, 3, 5)
RULES = ("intent_and_resolution", "intent")
HEADLINE_RULE = "intent_and_resolution"
SEM_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.0001, 0.1))
INTENT_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
NOISE_RATE = 0.3
SEED = 42


# ---- queries and relevance ------------------------------------------------------------------------------------------
def select_queries(records: pd.DataFrame) -> pd.DataFrame:
    """Dev queries with a usable proxy label: strong evidence, a real (non-fallback) candidate intent."""
    q = records[(records["evidence_quality"] == "strong") & records["intent"].notna() & (records["intent"] != FALLBACK_INTENT)]
    q = q[q["customer_problem"].map(lambda t: len(tokenize(t)) >= 1)]
    return q.rename(columns={"customer_problem": "query"}).reset_index(drop=True)


def split_tune_test(case_ids: Iterable[str], tune_fraction: float = 0.5, salt: str = "retrieval-eval-v1") -> np.ndarray:
    """Stable hash split, independent of row order. True = tune."""
    def bucket(cid: str) -> float:
        return int(hashlib.sha256(f"{salt}:{cid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return np.array([bucket(c) < tune_fraction for c in case_ids])


def _codes(*arrays: Sequence[Any]) -> list[np.ndarray]:
    """Integer-code several arrays against one shared vocabulary."""
    flat = pd.Series([x for a in arrays for x in a])
    codes, _ = pd.factorize(flat)
    out, start = [], 0
    for a in arrays:
        out.append(codes[start : start + len(a)])
        start += len(a)
    return out


def relevance_mask(queries: pd.DataFrame, corpus: pd.DataFrame, rule: str) -> np.ndarray:
    if rule not in RULES:
        raise ValueError(f"unknown relevance rule {rule!r}")
    qi, di = _codes(queries["intent"].tolist(), corpus["intent"].tolist())
    mask = qi[:, None] == di[None, :]
    if rule == "intent":
        return mask
    qr, dr = _codes(queries["resolution_type"].tolist(), corpus["resolution_type"].tolist())
    return mask & (qr[:, None] == dr[None, :])


# ---- metrics --------------------------------------------------------------------------------------------------------
def best_relevant_rank(scores: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """1-based rank of the best-ranked relevant document per query; ties broken by corpus position (as in search)."""
    n_q, n_d = scores.shape
    best = np.where(rel, scores, -np.inf).argmax(axis=1)  # argmax returns the first (lowest-index) maximum
    best_score = scores[np.arange(n_q), best]
    higher = (scores > best_score[:, None]).sum(axis=1)
    tied_before = ((scores == best_score[:, None]) & (np.arange(n_d)[None, :] < best[:, None])).sum(axis=1)
    return 1 + higher + tied_before


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def summarise_ranks(ranks: np.ndarray, rel: np.ndarray, scores: np.ndarray, seed: int = SEED, n_boot: int = 1000) -> dict[str, Any]:
    out: dict[str, Any] = {"n_queries": int(len(ranks))}
    for k in KS:
        out[f"recall@{k}"] = float((ranks <= k).mean())
    rr = 1.0 / ranks
    out["mrr"] = float(rr.mean())
    top5 = top_indices(scores, 5)
    out["precision@5"] = float(np.take_along_axis(rel, top5, axis=1).mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ranks), size=(n_boot, len(ranks)))
    out["mrr_ci95"] = [float(x) for x in np.percentile(rr[idx].mean(axis=1), [2.5, 97.5])]
    out["recall@1_ci95"] = [float(x) for x in np.percentile((ranks <= 1)[idx].mean(axis=1), [2.5, 97.5])]
    out["recall@5_ci95"] = [float(x) for x in np.percentile((ranks <= 5)[idx].mean(axis=1), [2.5, 97.5])]
    return out


def random_baseline(n_queries: int, n_docs: int, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).random((n_queries, n_docs), dtype=np.float32)


def noisy_intents(intents: Sequence[str], pool: Sequence[str], rate: float, seed: int = SEED) -> list[str]:
    """Replace a `rate` share of intents with a different intent from the pool (simulates a wrong candidate intent)."""
    rng = np.random.default_rng(seed)
    pool = sorted(set(pool))
    out = []
    for intent in intents:
        if rng.random() < rate:
            out.append(str(rng.choice([p for p in pool if p != intent])))
        else:
            out.append(intent)
    return out


# ---- scoring shared across strategies --------------------------------------------------------------------------------
@dataclass
class ScoreCache:
    """Normalised lexical and semantic matrices for a query set, so weight sweeps are cheap."""

    bm25: np.ndarray
    emb: np.ndarray

    @classmethod
    def build(cls, retriever: ResolutionRetriever, queries: Sequence[str]) -> "ScoreCache":
        comp = retriever.components([clean_text(q) for q in queries], "hybrid")
        return cls(minmax_rows(comp["bm25"]), minmax_rows(comp["embedding"]))

    def take(self, mask: np.ndarray) -> "ScoreCache":
        return ScoreCache(self.bm25[mask], self.emb[mask])

    def hybrid(self, sem_weight: float) -> np.ndarray:
        return sem_weight * self.emb + (1.0 - sem_weight) * self.bm25


def intent_bonus(retriever: ResolutionRetriever, intents: Sequence[str | None], weight: float) -> np.ndarray:
    return retriever.intent_bonus(intents, weight)


def strategy_scores(cache: ScoreCache, retriever: ResolutionRetriever, intents: Sequence[str], sem_weight: float, intent_weight: float) -> dict[str, np.ndarray]:
    base = cache.hybrid(sem_weight)
    return {
        "BM25": cache.bm25,
        "Embedding": cache.emb,
        "Hybrid": base,
        "Hybrid + intent": base + intent_bonus(retriever, intents, intent_weight),
    }


def evaluate_strategies(strategies: dict[str, np.ndarray], rel: np.ndarray) -> dict[str, dict[str, Any]]:
    return {name: summarise_ranks(best_relevant_rank(s, rel), rel, s) for name, s in strategies.items()}


def tune(cache: ScoreCache, retriever: ResolutionRetriever, intents: Sequence[str], rel: np.ndarray, intent_pool: Sequence[str]) -> dict[str, Any]:
    """Pick the semantic weight on clean hybrid MRR, then the intent weight on the mean MRR over clean and noisy intents."""
    sem_rows = []
    for w in SEM_GRID:
        s = cache.hybrid(w)
        sem_rows.append({"sem_weight": w, "mrr": float((1.0 / best_relevant_rank(s, rel)).mean())})
    best_sem = max(sem_rows, key=lambda r: (r["mrr"], -abs(r["sem_weight"] - 0.5)))["sem_weight"]
    base = cache.hybrid(best_sem)
    wrong = noisy_intents(intents, intent_pool, NOISE_RATE)
    int_rows = []
    for w in INTENT_GRID:
        clean = float((1.0 / best_relevant_rank(base + intent_bonus(retriever, intents, w), rel)).mean())
        noisy = float((1.0 / best_relevant_rank(base + intent_bonus(retriever, wrong, w), rel)).mean())
        int_rows.append({"intent_weight": w, "mrr_clean_intent": clean, "mrr_noisy_intent": noisy, "objective": (clean + noisy) / 2})
    best_int = max(int_rows, key=lambda r: (r["objective"], -r["intent_weight"]))["intent_weight"]
    return {"sem_weight": best_sem, "intent_weight": best_int, "sem_sweep": sem_rows, "intent_sweep": int_rows}


def intent_stress_test(cache: ScoreCache, retriever: ResolutionRetriever, intents: Sequence[str], rel: np.ndarray, sem_weight: float, intent_weight: float, intent_pool: Sequence[str]) -> list[dict[str, Any]]:
    """Soft bonus vs hard filter when a share of candidate intents is wrong. Shows why the intent is a signal, not a filter."""
    base = cache.hybrid(sem_weight)
    rows = []
    for rate in (0.0, 0.15, NOISE_RATE, 0.5):
        given = noisy_intents(intents, intent_pool, rate) if rate else list(intents)
        soft = base + intent_bonus(retriever, given, intent_weight)
        hard = base - 10.0 * (1.0 - (intent_bonus(retriever, given, 1.0) > 0))
        for name, s in (("query only", base), ("soft intent bonus", soft), ("hard intent filter", hard)):
            ranks = best_relevant_rank(s, rel)
            rows.append({"wrong_intent_rate": rate, "mode": name, "recall@5": float((ranks <= 5).mean()), "mrr": float((1.0 / ranks).mean())})
    return rows


# ---- explanations and failure analysis ---------------------------------------------------------------------------------
def explain_match(query: pd.Series, doc: pd.Series, rule: str = HEADLINE_RULE) -> dict[str, Any]:
    same_intent = query["intent"] == doc["intent"]
    same_res = query["resolution_type"] == doc["resolution_type"]
    relevant = same_intent and (same_res or rule == "intent")
    shared = sorted(set(tokenize(query["query"])) & set(tokenize(doc["customer_problem"])))
    reasons = [f"same candidate intent ({doc['intent']})" if same_intent else f"different candidate intent ({doc['intent']} vs query {query['intent']})"]
    reasons.append(f"same resolution type ({doc['resolution_type']})" if same_res else f"different resolution type ({doc['resolution_type']} vs query's {query['resolution_type']})")
    if shared:
        reasons.append("shared terms: " + ", ".join(shared[:8]))
    return {"relevant": bool(relevant), "reasons": reasons}


def failure_kind(query: pd.Series, top: pd.DataFrame) -> str:
    if len(tokenize(query["query"])) < 4:
        return "very_short_query"
    same = top["intent"] == query["intent"]
    if not same.any():
        return "no_same_intent_in_top5"
    return "same_intent_but_other_resolution_type"


def breakdown_by(queries: pd.DataFrame, ranks: np.ndarray, key: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"key": key.to_numpy(), "hit5": ranks <= 5, "hit1": ranks <= 1, "rr": 1.0 / ranks})
    g = df.groupby("key", sort=False).agg(n=("rr", "size"), recall_at_1=("hit1", "mean"), recall_at_5=("hit5", "mean"), mrr=("rr", "mean"))
    return g.reset_index()


def length_bucket(n_tokens: int) -> str:
    return "1-3 tokens" if n_tokens <= 3 else "4-7 tokens" if n_tokens <= 7 else "8-15 tokens" if n_tokens <= 15 else "16+ tokens"


def pick_examples(queries: pd.DataFrame, ranks: np.ndarray, per_intent: int = 1, limit: int = 6, good: bool = True) -> list[int]:
    """Deterministic, intent-diverse examples. good: relevant case at rank 1-3; bad: none in the top 5."""
    ok = ranks <= 3 if good else ranks > 5
    order = sorted(np.flatnonzero(ok), key=lambda i: hashlib.sha256(queries.iloc[i]["case_id"].encode()).hexdigest())
    seen: dict[str, int] = {}
    picked = []
    for i in order:
        intent = queries.iloc[i]["intent"]
        if seen.get(intent, 0) >= per_intent:
            continue
        seen[intent] = seen.get(intent, 0) + 1
        picked.append(int(i))
        if len(picked) >= limit:
            break
    return picked


def snippet(text: str, n: int = 170) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


# ---- orchestration ---------------------------------------------------------------------------------------------------
def _example_record(i: int, queries: pd.DataFrame, scores: np.ndarray, corpus: pd.DataFrame, rank: int, rule: str, top_show: int) -> dict[str, Any]:
    q = queries.iloc[i]
    order = top_indices(scores[i : i + 1], top_show)[0]
    shown = []
    for r, d in enumerate(order, 1):
        doc = corpus.iloc[int(d)]
        why = explain_match(q, doc, rule)
        shown.append(
            {
                "rank": r, "case_id": doc["case_id"], "score": float(scores[i, d]), "intent": doc["intent"],
                "resolution_type": doc["resolution_type"], "problem": snippet(doc["customer_problem"]),
                "response": snippet(doc["historical_response"]), "relevant": why["relevant"], "reasons": why["reasons"],
            }
        )
    return {"case_id": q["case_id"], "query": snippet(q["query"], 260), "intent": q["intent"], "resolution_type": q["resolution_type"], "first_relevant_rank": int(rank), "results": shown}


def run_evaluation(
    retriever: ResolutionRetriever,
    queries: pd.DataFrame,
    *,
    alt_retriever: ResolutionRetriever | None = None,
    sem_weight: float | None = None,
    intent_weight: float | None = None,
    n_examples: int = 6,
    top_show: int = 3,
) -> dict[str, Any]:
    corpus = retriever.corpus
    overlap = set(queries["case_id"]) & set(corpus["case_id"])
    if overlap:
        raise ValueError(f"{len(overlap)} evaluation queries are also in the retrieval corpus (e.g. {sorted(overlap)[:3]})")
    rel_all = {rule: relevance_mask(queries, corpus, rule) for rule in RULES}
    keep = rel_all[HEADLINE_RULE].any(axis=1)
    queries = queries[keep].reset_index(drop=True)
    rel_all = {r: m[keep] for r, m in rel_all.items()}
    is_tune = split_tune_test(queries["case_id"])
    test = ~is_tune
    intents = queries["intent"].tolist()
    pool = sorted(corpus["intent"].dropna().unique())

    cache = ScoreCache.build(retriever, queries["query"].tolist())
    tuning = tune(cache.take(is_tune), retriever, [x for x, t in zip(intents, is_tune) if t], rel_all[HEADLINE_RULE][is_tune], pool)
    sem = tuning["sem_weight"] if sem_weight is None else float(sem_weight)
    iw = tuning["intent_weight"] if intent_weight is None else float(intent_weight)

    q_test = queries[test].reset_index(drop=True)
    i_test = q_test["intent"].tolist()
    c_test = cache.take(test)
    strategies = strategy_scores(c_test, retriever, i_test, sem, iw)
    rel_test = {r: m[test] for r, m in rel_all.items()}
    headline = {rule: evaluate_strategies(strategies, rel_test[rule]) for rule in RULES}
    rand = random_baseline(len(q_test), len(corpus))
    random_metrics = {rule: summarise_ranks(best_relevant_rank(rand, rel_test[rule]), rel_test[rule], rand) for rule in RULES}
    tune_strategies = strategy_scores(cache.take(is_tune), retriever, [x for x, t in zip(intents, is_tune) if t], sem, iw)
    tune_metrics = evaluate_strategies(tune_strategies, rel_all[HEADLINE_RULE][is_tune])
    stress = intent_stress_test(c_test, retriever, i_test, rel_test[HEADLINE_RULE], sem, iw, pool)

    ablation = None
    if alt_retriever is not None:
        if list(alt_retriever.corpus["case_id"]) != list(corpus["case_id"]):
            raise ValueError("the alternative retriever must index the same corpus in the same order")
        alt_cache = ScoreCache.build(alt_retriever, q_test["query"].tolist())
        ablation = {
            "index_fields": list(alt_retriever.index_fields),
            "metrics": evaluate_strategies(strategy_scores(alt_cache, alt_retriever, i_test, sem, iw), rel_test[HEADLINE_RULE]),
        }

    final = strategies["Hybrid + intent"]
    ranks = best_relevant_rank(final, rel_test[HEADLINE_RULE])
    n_tokens = q_test["query"].map(lambda t: len(tokenize(t)))
    top5 = top_indices(final, 5)
    kinds = [failure_kind(q_test.iloc[i], corpus.iloc[top5[i]]) for i in np.flatnonzero(ranks > 5)]
    top1_hybrid = top_indices(strategies["Hybrid"], 1)[:, 0]
    dup = float(np.mean([clean_text(q).lower() == clean_text(corpus.iloc[int(d)]["customer_problem"]).lower() for q, d in zip(q_test["query"], top1_hybrid)]))
    good = [_example_record(i, q_test, final, corpus, ranks[i], HEADLINE_RULE, top_show) for i in pick_examples(q_test, ranks, 1, n_examples, True)]
    bad = [_example_record(i, q_test, final, corpus, ranks[i], HEADLINE_RULE, top_show) for i in pick_examples(q_test, ranks, 1, n_examples, False)]

    return {
        "meta": {
            "n_corpus": int(len(corpus)), "n_queries_eligible": int(len(queries)), "n_queries_dropped_no_relevant": int((~keep).sum()),
            "n_tune": int(is_tune.sum()), "n_test": int(test.sum()), "sem_weight": sem, "intent_weight": iw,
            "weights_source": "tuned on the tune half" if sem_weight is None and intent_weight is None else "overridden on the command line",
            "headline_rule": HEADLINE_RULE, "retriever": retriever.config, "noise_rate_for_tuning": NOISE_RATE,
            "mean_relevant_per_query": {r: float(m.sum(axis=1).mean()) for r, m in rel_test.items()},
        },
        "tuning": tuning,
        "test": headline,
        "tune_half": tune_metrics,
        "random": random_metrics,
        "stress": stress,
        "ablation": ablation,
        "per_intent": breakdown_by(q_test, ranks, q_test["intent"]).sort_values("n", ascending=False).to_dict("records"),
        "by_length": breakdown_by(q_test, ranks, n_tokens.map(length_bucket)).to_dict("records"),
        "by_resolution_type": breakdown_by(q_test, ranks, q_test["resolution_type"]).sort_values("n", ascending=False).to_dict("records"),
        "failures": {"n_queries_without_hit_in_top5": int((ranks > 5).sum()), "kinds": pd.Series(kinds, dtype=object).value_counts().to_dict()},
        "near_duplicate_top1_rate": dup,
        "examples": {"good": good, "bad": bad},
    }


# ---- rendering -------------------------------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{x:.3f}"


def md_table(rows: list[dict[str, Any]], columns: list[str], headers: list[str] | None = None) -> str:
    headers = headers or columns
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        cells = [(_fmt(r[c]) if isinstance(r[c], float) else str(r[c])) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def metrics_rows(metrics: dict[str, dict[str, Any]], random_row: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    for name, m in metrics.items():
        rows.append({"strategy": name, **{f"R@{k}": m[f"recall@{k}"] for k in KS}, "MRR": m["mrr"], "P@5": m["precision@5"], "MRR 95% CI": f"{m['mrr_ci95'][0]:.3f}-{m['mrr_ci95'][1]:.3f}"})
    if random_row is not None:
        rows.append({"strategy": "Random (baseline)", **{f"R@{k}": random_row[f"recall@{k}"] for k in KS}, "MRR": random_row["mrr"], "P@5": random_row["precision@5"], "MRR 95% CI": "-"})
    return rows


METRIC_COLUMNS = ["strategy", "R@1", "R@3", "R@5", "MRR", "P@5", "MRR 95% CI"]


def best_strategy(results: dict[str, Any]) -> str:
    return max(results["test"][HEADLINE_RULE].items(), key=lambda kv: kv[1]["mrr"])[0]


def render_console(results: dict[str, Any]) -> str:
    m = results["meta"]
    out = [
        "RETRIEVAL EVALUATION (proxy relevance, not human ground truth)",
        f"corpus={m['n_corpus']} historical episodes | queries: {m['n_queries_eligible']} eligible dev cases ({m['n_tune']} tune / {m['n_test']} test) | weights: semantic={m['sem_weight']}, intent={m['intent_weight']} ({m['weights_source']})",
        "",
    ]
    for rule, label in (("intent_and_resolution", "HEADLINE relevance = same candidate intent AND same resolution type"), ("intent", "SECONDARY relevance = same candidate intent")):
        out.append(f"{label}  [test half, n={m['n_test']}, avg relevant docs per query={m['mean_relevant_per_query'][rule]:.0f}]")
        rows = metrics_rows(results["test"][rule], results["random"][rule])
        width = max(len(r["strategy"]) for r in rows)
        out.append(f"{'strategy'.ljust(width)}   R@1    R@3    R@5    MRR    P@5")
        for r in rows:
            out.append(f"{r['strategy'].ljust(width)}  {r['R@1']:.3f}  {r['R@3']:.3f}  {r['R@5']:.3f}  {r['MRR']:.3f}  {r['P@5']:.3f}")
        out.append("")
    out.append(f"Best strategy by headline MRR: {best_strategy(results)}")
    out.append("")
    for title, key in (("GOOD RETRIEVAL EXAMPLES", "good"), ("BAD RETRIEVAL EXAMPLES", "bad")):
        out.append("=" * 100)
        out.append(title)
        for ex in results["examples"][key]:
            out.append("-" * 100)
            out.append(f"QUERY [{ex['case_id']}] intent={ex['intent']} (agent resolution: {ex['resolution_type']}); first relevant result at rank {ex['first_relevant_rank']}")
            out.append(f"  {ex['query']}")
            out.append("TOP RETRIEVED CASES")
            for r in ex["results"]:
                out.append(f"  #{r['rank']} {r['case_id']} score={r['score']:.2f} intent={r['intent']} type={r['resolution_type']} {'RELEVANT' if r['relevant'] else 'not relevant'}")
                out.append(f"     problem : {r['problem']}")
                out.append(f"     response: {r['response']}")
                out.append(f"     why     : {'; '.join(r['reasons'])}")
        out.append("")
    return "\n".join(out)


def render_markdown(results: dict[str, Any]) -> str:
    m = results["meta"]
    t = results["tuning"]
    lines = [
        "# VirginTrains retrieval evaluation (proxy)",
        "",
        "> **This is a proxy retrieval evaluation, not human relevance ground truth.** The taxonomy is a *candidate* taxonomy derived from data exploration "
        "(`CANDIDATE_NOT_GROUND_TRUTH`); human taxonomy calibration is intentionally not a blocking dependency for this prototype. "
        "No golden case is used as a query, as corpus, or for tuning.",
        "",
        "## Setup",
        f"- **Corpus:** {m['n_corpus']:,} historical resolution episodes (`train_retrieval`, strong evidence only).",
        f"- **Queries:** {m['n_queries_eligible']:,} non-golden `dev_calibration` cases with a strong resolution label and a non-fallback candidate intent "
        f"({m['n_queries_dropped_no_relevant']} dropped for having no relevant case in the corpus). Query text = what the customer said before the first brand reply.",
        f"- **Split:** {m['n_tune']:,} tune / {m['n_test']:,} test (stable hash). Weights are chosen on the tune half; **all numbers below are on the test half**.",
        f"- **Relevance (headline):** same candidate intent **and** same rule-derived resolution type (avg {m['mean_relevant_per_query']['intent_and_resolution']:.0f} relevant cases per query). "
        f"**Secondary:** same candidate intent only (avg {m['mean_relevant_per_query']['intent']:.0f}).",
        "- **Recall@k** is the hit-rate form: the share of queries with at least one relevant case in the top k.",
        f"- **Weights ({m['weights_source']}):** semantic weight {m['sem_weight']}, intent bonus {m['intent_weight']} (a soft additive bonus on the 0-1 normalised score, not a filter).",
        "",
        "### Caveats that matter",
        "- The *Hybrid + intent* strategy and the relevance rule share the same candidate taxonomy, so that row is partly self-confirming. See the wrong-intent stress test.",
        "- The resolution type is a heuristic label from the brand's reply, so some results counted as irrelevant may be sensible, and vice versa.",
        "- Cross-split near-duplicates (templated tweets) can inflate scores: "
        f"{results['near_duplicate_top1_rate']:.1%} of test queries have a top-1 hybrid result with identical cleaned text.",
        "",
        "## Headline results (test half)",
        md_table(metrics_rows(results["test"][HEADLINE_RULE], results["random"][HEADLINE_RULE]), METRIC_COLUMNS),
        "",
        f"**Best by MRR:** {best_strategy(results)}.",
        "",
        "## Secondary relevance: same candidate intent only",
        md_table(metrics_rows(results["test"]["intent"], results["random"]["intent"]), METRIC_COLUMNS),
        "",
        "## Tune half (sanity check that the test half is not unusual)",
        md_table(metrics_rows(results["tune_half"]), METRIC_COLUMNS),
        "",
        "## Weight selection (tune half only)",
        "Semantic weight (clean hybrid MRR):",
        md_table(t["sem_sweep"], ["sem_weight", "mrr"]),
        "",
        f"Intent bonus (chosen on the mean of clean and {int(m['noise_rate_for_tuning'] * 100)}%-wrong intents, so the weight does not over-trust an imperfect candidate intent):",
        md_table(t["intent_sweep"], ["intent_weight", "mrr_clean_intent", "mrr_noisy_intent", "objective"]),
        "",
        "## Stress test: what if the candidate intent is wrong? (test half)",
        md_table(results["stress"], ["wrong_intent_rate", "mode", "recall@5", "mrr"]),
        "",
    ]
    if results["ablation"]:
        a = results["ablation"]
        lines += [f"## Index-text ablation: index = {' + '.join(a['index_fields'])} (same weights, test half)", md_table(metrics_rows(a["metrics"]), METRIC_COLUMNS), ""]
    lines += [
        "## Where it works and where it fails (Hybrid + intent, headline relevance)",
        "By candidate intent:",
        md_table(results["per_intent"], ["key", "n", "recall_at_1", "recall_at_5", "mrr"], ["intent", "n", "R@1", "R@5", "MRR"]),
        "",
        "By query length:",
        md_table(results["by_length"], ["key", "n", "recall_at_1", "recall_at_5", "mrr"], ["length", "n", "R@1", "R@5", "MRR"]),
        "",
        "By the historical agent's resolution type:",
        md_table(results["by_resolution_type"], ["key", "n", "recall_at_1", "recall_at_5", "mrr"], ["resolution type", "n", "R@1", "R@5", "MRR"]),
        "",
        f"Queries with no relevant case in the top 5: **{results['failures']['n_queries_without_hit_in_top5']}** of {m['n_test']}. Kinds: {results['failures']['kinds']}.",
        "",
    ]
    for title, key in (("Good retrieval examples", "good"), ("Bad retrieval examples", "bad")):
        lines += [f"## {title}", ""]
        for ex in results["examples"][key]:
            lines += [f"### QUERY `{ex['case_id']}` (candidate intent: {ex['intent']}; agent resolution: {ex['resolution_type']}; first relevant at rank {ex['first_relevant_rank']})", f"> {ex['query']}", "", "**TOP RETRIEVED CASES**", ""]
            for r in ex["results"]:
                lines += [
                    f"{r['rank']}. `{r['case_id']}` score {r['score']:.2f} - {r['intent']} / {r['resolution_type']} - **{'relevant' if r['relevant'] else 'not relevant'}**",
                    f"   - problem: {r['problem']}",
                    f"   - response: {r['response']}",
                    f"   - why: {'; '.join(r['reasons'])}",
                ]
            lines.append("")
    return "\n".join(lines) + "\n"
