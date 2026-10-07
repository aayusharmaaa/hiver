from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from retrieval_fixtures import FakeEncoder, make_memory

from retrieval import (
    BM25Index,
    ResolutionRetriever,
    RetrievalError,
    RetrievalLeakageError,
    clean_text,
    embed_cached,
    minmax_rows,
    tokenize,
)


def build(memory=None, **kw) -> ResolutionRetriever:
    kw.setdefault("encoder", FakeEncoder())
    return ResolutionRetriever(make_memory() if memory is None else memory, **kw)


# ---- text handling ----------------------------------------------------------------------------------------------
def test_clean_text_removes_urls_mentions_and_entities_but_keeps_hashtag_words() -> None:
    assert clean_text("@VirginTrains the wifi &amp; power is down https://t.co/abc #Fail") == "the wifi & power is down Fail"


def test_tokenizer_keeps_negations_drops_stop_words_and_stems_plurals() -> None:
    assert tokenize("The wifi is NOT working on trains") == ["wifi", "not", "working", "train"]
    assert tokenize("don't") == ["dont"]
    assert tokenize("@Virgin https://x.co") == []


# ---- BM25 ------------------------------------------------------------------------------------------------------
def test_bm25_ranks_the_matching_document_first_and_rare_terms_outweigh_common_ones() -> None:
    idx = BM25Index(["wifi not connecting", "seat reservation missing", "refund for delay", "wifi and seat and refund"])
    s = idx.scores(["wifi connecting"])[0]
    assert int(np.argmax(s)) == 0 and s[0] > s[3] > 0 and s[1] == 0 and s[2] == 0
    assert idx.scores(["zzzz unseen"]).sum() == 0


def test_bm25_retrieval_finds_the_right_topic() -> None:
    r = build()
    res = r.search("wifi will not connect, no internet signal", top_k=5, method="bm25")
    assert len(res) == 5 and all(x.intent == "onboard_wifi_issue" for x in res[:3])
    assert r._doc_emb is None, "BM25 alone must not need the embedding model"
    assert [x.rank for x in res] == [1, 2, 3, 4, 5] and res[0].score >= res[-1].score


# ---- embeddings ------------------------------------------------------------------------------------------------
def test_embedding_retrieval_finds_the_right_topic_with_unit_norm_vectors() -> None:
    r = build()
    res = r.search("seat reservation coach standing", top_k=3, method="embedding")
    assert all(x.intent == "seat_reservation_issue" for x in res)
    assert np.allclose(np.linalg.norm(r.doc_embeddings, axis=1), 1.0, atol=1e-5)
    assert r._bm25 is None, "embedding retrieval must not build the BM25 index"


def test_missing_encoder_model_fails_loudly_instead_of_silently_degrading(monkeypatch) -> None:
    import sys
    import types

    from retrieval import SentenceTransformerEncoder

    def unavailable(*args, **kwargs):
        raise OSError("model not found (simulated)")

    # A stub module: the real package (and torch) is never imported, so the test is fast and offline.
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=unavailable))
    enc = SentenceTransformerEncoder(model_name="definitely/not-a-real-model-xyz")
    with pytest.raises(RetrievalError, match="unavailable"):
        enc.encode(["hello"])


# ---- hybrid ----------------------------------------------------------------------------------------------------
def order(res) -> list[str]:
    return [x.case_id for x in res]


def test_hybrid_weights_reduce_to_the_two_baselines_at_the_extremes() -> None:
    q = "refund and compensation claim for the delay"
    pure_bm25 = order(build(sem_weight=0.0).search(q, top_k=10, method="hybrid"))
    pure_emb = order(build(sem_weight=1.0).search(q, top_k=10, method="hybrid"))
    assert pure_bm25 == order(build().search(q, top_k=10, method="bm25"))
    assert pure_emb == order(build().search(q, top_k=10, method="embedding"))


def test_hybrid_scores_are_the_documented_weighted_sum_of_normalised_scores() -> None:
    r = build(sem_weight=0.3)
    q = ["wifi connect portal"]
    comp = r.components(q, "hybrid")
    expected = 0.3 * minmax_rows(comp["embedding"]) + 0.7 * minmax_rows(comp["bm25"])
    np.testing.assert_allclose(r.score_batch(q, method="hybrid"), expected, atol=1e-6)
    res = r.search(q[0], top_k=1)[0]
    assert res.score == pytest.approx(float(expected.max()), abs=1e-6)
    assert set(res.scores) == {"bm25", "embedding", "intent_bonus"}


@pytest.mark.parametrize("w", [-0.1, 1.1])
def test_invalid_hybrid_weight_is_rejected(w) -> None:
    with pytest.raises(RetrievalError):
        build(sem_weight=w)
    with pytest.raises(RetrievalError):
        build(intent_weight=-1)


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(RetrievalError):
        build().search("wifi", method="magic")


# ---- candidate intent as a signal -------------------------------------------------------------------------------
def test_intent_bonus_prefers_matching_intent_without_filtering_others_out() -> None:
    r = build(intent_weight=0.15)
    q = "wifi seat refund delay problem"  # deliberately ambiguous
    plain = r.search(q, top_k=24, method="hybrid")
    boosted = r.search(q, intent="seat_reservation_issue", top_k=24, method="hybrid")
    assert len(boosted) == 24, "no hard filter: every intent is still retrievable"
    mean_rank = lambda res, intent: np.mean([x.rank for x in res if x.intent == intent])  # noqa: E731
    assert mean_rank(boosted, "seat_reservation_issue") < mean_rank(plain, "seat_reservation_issue")
    assert any(x.intent != "seat_reservation_issue" for x in boosted[:10]) or len({x.intent for x in boosted}) == 3
    assert all(x.intent_match == (x.intent == "seat_reservation_issue") for x in boosted)
    assert all(x.scores["intent_bonus"] == (0.15 if x.intent_match else 0.0) for x in boosted)


def test_a_strong_cross_intent_match_beats_a_weak_same_intent_match() -> None:
    r = build(intent_weight=0.05)
    res = r.search("wifi internet connect login portal signal", intent="seat_reservation_issue", top_k=1, method="bm25")
    assert res[0].intent == "onboard_wifi_issue"


@pytest.mark.parametrize("bad", ["not_an_intent", "  ", "", "unclear_or_media_only", None])
def test_missing_unknown_or_fallback_intent_means_query_only_retrieval(bad) -> None:
    r = build()
    q = "reserved seat standing coach"
    assert order(r.search(q, intent=bad, top_k=6)) == order(r.search(q, top_k=6))


def test_non_string_intent_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        build().search("wifi", intent=5)


def test_intent_weight_zero_equals_query_only() -> None:
    r = build(intent_weight=0.0)
    assert order(r.search("seat", intent="seat_reservation_issue", top_k=8)) == order(r.search("seat", top_k=8))


# ---- top_k, empty and short queries -----------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 3, 5, 24])
def test_top_k_is_respected_and_results_are_a_prefix_of_a_longer_list(k) -> None:
    r = build()
    res = r.search("wifi connect", top_k=k)
    assert len(res) == k
    assert order(res) == order(r.search("wifi connect", top_k=24))[:k]


def test_top_k_larger_than_the_corpus_returns_everything() -> None:
    assert len(build().search("wifi", top_k=1000)) == 24


@pytest.mark.parametrize("bad", [0, -1, 2.5, "3", None, True])
def test_invalid_top_k_is_rejected(bad) -> None:
    with pytest.raises(RetrievalError):
        build().search("wifi", top_k=bad)


@pytest.mark.parametrize("q", ["", "   ", "@VirginTrains", "@VirginTrains https://t.co/xyz", "the and of"])
def test_empty_queries_return_no_evidence_instead_of_noise(q) -> None:
    assert build().search(q, top_k=5) == []


def test_very_short_queries_are_answered_but_flagged() -> None:
    res = build().search("wifi", top_k=3)
    assert len(res) == 3 and all(x.low_information_query for x in res)
    assert not any(x.low_information_query for x in build().search("wifi not connecting on the train", top_k=3))


def test_batch_search_keeps_positions_when_some_queries_are_empty() -> None:
    out = build().search_batch(["wifi connect", "", "seat reservation"], [None, None, "seat_reservation_issue"], top_k=2)
    assert [len(x) for x in out] == [2, 0, 2]
    with pytest.raises(RetrievalError):
        build().search_batch(["a b", "c d"], ["x"], top_k=2)
    with pytest.raises(TypeError):
        build().search(123)  # type: ignore[arg-type]


# ---- determinism -----------------------------------------------------------------------------------------------
def test_results_are_deterministic_across_instances_and_calls() -> None:
    q = "refund for delay compensation claim"
    a = build().search(q, intent="delay_repay_refund_claim", top_k=10)
    b = build().search(q, intent="delay_repay_refund_claim", top_k=10)
    r = build()
    c, d = r.search(q, intent="delay_repay_refund_claim", top_k=10), r.search(q, intent="delay_repay_refund_claim", top_k=10)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b] == [x.to_dict() for x in c] == [x.to_dict() for x in d]


def test_ties_break_on_corpus_order() -> None:
    m = make_memory(per_intent=2)
    dup = m.iloc[[0]].copy()
    dup["case_id"] = "case_dup"
    m = pd.concat([m, dup], ignore_index=True)
    res = build(m).search(m.iloc[0]["customer_problem"], top_k=3, method="bm25")
    ids = order(res)
    assert ids.index("case_1") < ids.index("case_dup"), "identical documents tie; the earlier corpus row wins"


# ---- cache reuse -----------------------------------------------------------------------------------------------
def test_document_embeddings_are_cached_and_reused_across_runs(tmp_path: Path) -> None:
    first = FakeEncoder()
    r1 = build(encoder=first, cache_dir=tmp_path)
    r1.search("wifi connect", top_k=3)
    files = sorted(tmp_path.glob("retrieval_embeddings_*.npy"))
    assert len(files) >= 1 and first.calls >= 1

    second = FakeEncoder()
    r2 = build(encoder=second, cache_dir=tmp_path)
    _ = r2.doc_embeddings
    assert second.calls == 0, "document embeddings must be loaded from the cache, not recomputed"
    np.testing.assert_array_equal(r1.doc_embeddings, r2.doc_embeddings)


def test_cache_is_keyed_by_model_and_corpus_text(tmp_path: Path) -> None:
    texts = ["a b c", "d e f"]
    e1, e2 = FakeEncoder("model-a"), FakeEncoder("model-b")
    embed_cached(texts, e1, tmp_path)
    embed_cached(texts, e1, tmp_path)
    assert e1.calls == 1
    embed_cached(texts, e2, tmp_path)
    embed_cached(texts + ["g h i"], e1, tmp_path)
    assert e2.calls == 1 and e1.calls == 2
    assert len(list(tmp_path.glob("retrieval_embeddings_*.npy"))) == 3
    assert not list(tmp_path.glob("*.tmp*")), "no temp files left behind"


def test_repeated_queries_are_not_re_encoded() -> None:
    enc = FakeEncoder()
    r = build(encoder=enc)
    r.search("wifi connect", top_k=3)
    calls = enc.calls
    r.search("wifi connect", top_k=3)
    r.search("wifi connect", intent="onboard_wifi_issue", top_k=5)
    assert enc.calls == calls


# ---- safety: leakage and corpus rules ---------------------------------------------------------------------------
@pytest.mark.parametrize("split", ["golden_eval", "golden_pool_reserve", "dev_calibration", "golden_pool"])
def test_memory_from_a_forbidden_split_is_refused(split) -> None:
    with pytest.raises(RetrievalLeakageError):
        build(make_memory(split=split))


def test_a_single_leaked_row_is_enough_to_refuse() -> None:
    m = make_memory()
    m.loc[3, "split"] = "golden_eval"
    with pytest.raises(RetrievalLeakageError, match="golden_eval"):
        build(m)


def test_forbidden_case_ids_are_refused_even_with_a_train_label() -> None:
    with pytest.raises(RetrievalLeakageError):
        build(forbidden_case_ids={"case_5"})


def test_from_files_forbids_every_non_train_id_in_the_split_assignments(tmp_path: Path) -> None:
    memory = make_memory()
    memory.to_parquet(tmp_path / "memory.parquet")
    pd.DataFrame({"case_id": ["case_1", "case_2", "case_9999"], "split": ["train_retrieval", "train_retrieval", "golden_eval"]}).to_csv(tmp_path / "a.csv", index=False)
    r = ResolutionRetriever.from_files(tmp_path / "memory.parquet", tmp_path / "a.csv", encoder=FakeEncoder())
    assert len(r.corpus) == 24
    pd.DataFrame({"case_id": ["case_1", "case_2"], "split": ["golden_eval", "train_retrieval"]}).to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(RetrievalLeakageError):
        ResolutionRetriever.from_files(tmp_path / "memory.parquet", tmp_path / "b.csv", encoder=FakeEncoder())


def test_weak_and_unreplied_cases_never_enter_the_primary_corpus() -> None:
    m = make_memory()
    m.loc[0, ["in_primary_corpus", "evidence_quality"]] = [False, "weak"]
    m.loc[1, ["in_primary_corpus", "evidence_quality"]] = [False, "no_brand_reply"]
    r = build(m)
    assert len(r.corpus) == 22 and not {"case_1", "case_2"} & set(r.corpus["case_id"])
    for q in ("wifi", "wifi internet connect", m.loc[0, "customer_problem"]):
        assert not {"case_1", "case_2"} & set(order(r.search(q, top_k=24)))
    assert len(build(m, primary_only=False).corpus) == 24


def test_corpus_validation() -> None:
    with pytest.raises(RetrievalError, match="missing columns"):
        build(make_memory().drop(columns=["historical_response"]))
    dup = make_memory()
    dup.loc[1, "case_id"] = dup.loc[0, "case_id"]
    with pytest.raises(RetrievalError, match="duplicate"):
        build(dup)
    empty = make_memory()
    empty["in_primary_corpus"] = False
    with pytest.raises(RetrievalError, match="empty"):
        build(empty)
    with pytest.raises(RetrievalError):
        build(index_fields=("nope",))


# ---- result contract and provenance -----------------------------------------------------------------------------
def test_results_carry_the_required_fields_and_provenance() -> None:
    m = make_memory()
    res = build(m).search("wifi connect", intent="onboard_wifi_issue", top_k=3)[0]
    row = m[m["case_id"] == res.case_id].iloc[0]
    assert res.customer_problem == row["customer_problem"] and res.historical_response == row["historical_response"]
    assert res.resolution_summary == row["resolution_summary"] and res.resolution_type == row["resolution_type"]
    assert res.provenance["source_tweet_ids"] == list(row["source_tweet_ids"])
    assert res.provenance["response_tweet_ids"] == list(row["response_tweet_ids"])
    assert res.provenance["split"] == "train_retrieval"
    assert res.provenance["intent_source"] == "candidate_taxonomy_not_ground_truth"
    json.dumps(res.to_dict())


def test_index_fields_change_what_is_searchable() -> None:
    m = make_memory()
    m.loc[0, "historical_response"] = "Please clear the zebra cache"
    only_problem = build(m).search("zebra cache", top_k=1, method="bm25")
    with_response = build(m, index_fields=("customer_problem", "historical_response")).search("zebra cache", top_k=1, method="bm25")
    assert only_problem[0].scores["bm25"] == 0.0, "the phrase is not in any customer problem"
    assert with_response[0].case_id == "case_1" and with_response[0].scores["bm25"] > 0


# ---- real artifacts (BM25 only: no model needed) ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "data" / "processed" / "virgintrains_resolution_memory.parquet"
ASSIGN = ROOT / "data" / "processed" / "splits" / "virgintrains_split_assignments.csv"


@pytest.mark.skipif(not (MEMORY.exists() and ASSIGN.exists()), reason="resolution memory not built")
def test_real_retrieval_never_returns_golden_reserve_or_dev_cases() -> None:
    r = ResolutionRetriever.from_files(MEMORY, ASSIGN)
    a = pd.read_csv(ASSIGN)
    train = set(a.loc[a["split"] == "train_retrieval", "case_id"])
    assert set(r.corpus["case_id"]) <= train
    for q in ("train cancelled how do I claim delay repay", "no wifi on the 17:43 euston to glasgow", "my reserved seat was taken", "cheers"):
        res = r.search(q, top_k=10, method="bm25")
        assert len(res) == 10 and {x.case_id for x in res} <= train
        assert all(x.provenance["split"] == "train_retrieval" and x.provenance["source_tweet_ids"] for x in res)
        assert [x.case_id for x in res] == [x.case_id for x in r.search(q, top_k=10, method="bm25")]
