from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from retrieval_fixtures import FakeEncoder, make_memory

from evaluation import retrieval as ev
from retrieval import ResolutionRetriever


def test_best_relevant_rank_matches_a_brute_force_ranking_including_ties() -> None:
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 4, size=(40, 30)).astype(np.float32)  # many ties on purpose
    rel = rng.random((40, 30)) < 0.15
    rel[:, 0] |= ~rel.any(axis=1)
    got = ev.best_relevant_rank(scores, rel)
    for i in range(40):
        order = np.lexsort((np.arange(30), -scores[i]))
        assert got[i] == int(np.flatnonzero(rel[i][order])[0]) + 1


def test_metrics_on_a_hand_computed_example() -> None:
    scores = np.array([[0.9, 0.5, 0.1, 0.0], [0.1, 0.9, 0.8, 0.0], [0.2, 0.3, 0.9, 0.8]], dtype=np.float32)
    rel = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
    ranks = ev.best_relevant_rank(scores, rel)
    assert ranks.tolist() == [1, 2, 2]
    m = ev.summarise_ranks(ranks, rel, scores, n_boot=50)
    assert m["recall@1"] == pytest.approx(1 / 3) and m["recall@3"] == 1.0 and m["recall@5"] == 1.0
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert m["precision@5"] == pytest.approx(3 / 12)
    assert m["mrr_ci95"][0] <= m["mrr"] <= m["mrr_ci95"][1]


def test_relevance_rules() -> None:
    q = pd.DataFrame({"intent": ["a", "b"], "resolution_type": ["x", "y"]})
    d = pd.DataFrame({"intent": ["a", "a", "b", "c"], "resolution_type": ["x", "y", "y", "y"]})
    assert ev.relevance_mask(q, d, "intent").tolist() == [[True, True, False, False], [False, False, True, False]]
    assert ev.relevance_mask(q, d, "intent_and_resolution").tolist() == [[True, False, False, False], [False, False, True, False]]
    with pytest.raises(ValueError):
        ev.relevance_mask(q, d, "bogus")


def test_tune_test_split_is_stable_order_independent_and_balanced() -> None:
    ids = [f"case_{i}" for i in range(2000)]
    a = ev.split_tune_test(ids)
    b = ev.split_tune_test(list(reversed(ids)))[::-1]
    assert (a == b).all() and 0.45 < a.mean() < 0.55


def test_noisy_intents_changes_roughly_the_requested_share_and_never_keeps_the_same_intent_when_replacing() -> None:
    intents = ["a", "b", "c"] * 400
    noisy = ev.noisy_intents(intents, ["a", "b", "c", "d"], 0.3)
    changed = np.mean([x != y for x, y in zip(intents, noisy)])
    assert 0.25 < changed < 0.35 and noisy == ev.noisy_intents(intents, ["a", "b", "c", "d"], 0.3)


def test_select_queries_keeps_only_strong_non_fallback_cases() -> None:
    rec = pd.DataFrame(
        {
            "case_id": list("abcde"), "evidence_quality": ["strong", "weak", "strong", "strong", "no_brand_reply"],
            "intent": ["x", "x", "unclear_or_media_only", None, "x"], "customer_problem": ["wifi down"] * 5,
        }
    )
    assert ev.select_queries(rec)["case_id"].tolist() == ["a"]


def corpus_and_queries():
    memory = make_memory(per_intent=40)
    retr = ResolutionRetriever(memory, encoder=FakeEncoder())
    intents = list(memory["intent"].unique())
    rows = []
    for i in range(60):
        intent = intents[i % 3]
        words = {"onboard_wifi_issue": "wifi internet connect", "seat_reservation_issue": "seat reservation coach", "delay_repay_refund_claim": "delay refund claim"}[intent]
        rows.append({"case_id": f"dev_{i}", "query": f"my {words} issue number {i}", "intent": intent, "resolution_type": "self_service" if i % 2 else "information_provided"})
    return retr, pd.DataFrame(rows)


def test_run_evaluation_on_synthetic_data_returns_the_four_strategies_and_beats_random() -> None:
    retr, queries = corpus_and_queries()
    res = ev.run_evaluation(retr, queries, n_examples=2)
    assert res["meta"]["n_tune"] + res["meta"]["n_test"] == len(queries)
    test = res["test"][ev.HEADLINE_RULE]
    assert list(test) == ["BM25", "Embedding", "Hybrid", "Hybrid + intent"]
    for m in test.values():
        assert all(f"recall@{k}" in m for k in (1, 3, 5)) and "mrr" in m
        assert m["mrr"] > res["random"][ev.HEADLINE_RULE]["mrr"]
    assert test["Hybrid + intent"]["mrr"] >= test["Hybrid"]["mrr"]
    assert res["meta"]["sem_weight"] in ev.SEM_GRID and res["meta"]["intent_weight"] in ev.INTENT_GRID
    assert {r["mode"] for r in res["stress"]} == {"query only", "soft intent bonus", "hard intent filter"}
    assert res["examples"]["good"] and ev.best_strategy(res) in test


def test_evaluation_is_deterministic() -> None:
    retr, queries = corpus_and_queries()
    a = ev.run_evaluation(retr, queries, n_examples=2)
    b = ev.run_evaluation(ResolutionRetriever(retr.corpus, encoder=FakeEncoder()), queries, n_examples=2)
    assert a == b


def test_weight_overrides_skip_tuning_values() -> None:
    retr, queries = corpus_and_queries()
    res = ev.run_evaluation(retr, queries, sem_weight=0.25, intent_weight=0.4, n_examples=1)
    assert res["meta"]["sem_weight"] == 0.25 and res["meta"]["intent_weight"] == 0.4 and "overridden" in res["meta"]["weights_source"]


def test_queries_that_are_in_the_corpus_are_refused() -> None:
    retr, queries = corpus_and_queries()
    leaky = queries.copy()
    leaky.loc[0, "case_id"] = "case_1"
    with pytest.raises(ValueError, match="also in the retrieval corpus"):
        ev.run_evaluation(retr, leaky)


def test_a_mismatched_alternative_index_is_refused() -> None:
    retr, queries = corpus_and_queries()
    alt = ResolutionRetriever(make_memory(per_intent=30), encoder=FakeEncoder())
    with pytest.raises(ValueError, match="same corpus"):
        ev.run_evaluation(retr, queries, alt_retriever=alt)


def test_reports_render_with_all_sections_and_the_proxy_disclaimer() -> None:
    retr, queries = corpus_and_queries()
    alt = ResolutionRetriever(retr.corpus, encoder=FakeEncoder(), index_fields=("customer_problem", "historical_response"))
    res = ev.run_evaluation(retr, queries, alt_retriever=alt, n_examples=2)
    md, console = ev.render_markdown(res), ev.render_console(res)
    for needle in ("proxy retrieval evaluation, not human relevance ground truth", "CANDIDATE_NOT_GROUND_TRUTH", "Stress test", "Index-text ablation", "Good retrieval examples", "Bad retrieval examples", "Hybrid + intent"):
        assert needle in md, needle
    for needle in ("QUERY", "TOP RETRIEVED CASES", "why", "BM25", "Embedding", "Hybrid", "R@1", "MRR"):
        assert needle in console, needle
    assert "gold" not in console.lower().replace("golden", "")


def test_stress_test_hard_filter_degrades_faster_than_the_soft_bonus() -> None:
    retr, queries = corpus_and_queries()
    res = ev.run_evaluation(retr, queries, n_examples=1)
    rows = {(r["wrong_intent_rate"], r["mode"]): r["mrr"] for r in res["stress"]}
    assert rows[(0.5, "hard intent filter")] < rows[(0.0, "hard intent filter")]
    assert rows[(0.5, "soft intent bonus")] >= rows[(0.5, "hard intent filter")]
