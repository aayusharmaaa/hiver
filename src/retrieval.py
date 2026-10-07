"""Retrieval of historical support resolutions (no LLM, no vector database).

    memory (train_retrieval resolution episodes)
        -> BM25 index        (lexical, in-memory sparse matrix)
        -> embedding index   (sentence-transformers, cached .npy)
        -> hybrid            (weighted sum of per-query min-max normalised scores)
        -> optional soft candidate-intent bonus

The candidate intent is a *signal*, never a filter: a matching intent adds a small bonus to the normalised score, so
strong cross-intent evidence can still win. The intent comes from a candidate taxonomy that is not human validated.

Safety: the retriever refuses a corpus containing anything outside `allowed_splits` (default: train_retrieval only) or
any id in `forbidden_case_ids` (golden / reserve), and results are fully deterministic (ties broken by corpus order).
"""

from __future__ import annotations

import hashlib
import html
import logging
import math
import re
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from evaluation.splits import TRAIN
from taxonomy.registry import FALLBACK_INTENT

logger = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METHODS = ("bm25", "embedding", "hybrid")
REQUIRED_COLUMNS = (
    "case_id", "split", "intent", "customer_problem", "historical_response", "resolution_summary", "resolution_type",
    "resolved", "dm_redirect", "escalation_signal", "source_tweet_ids",
)
# Defaults chosen on the tune half of the non-golden dev queries (scripts/evaluate_retrieval.py); see the evaluation report.
DEFAULT_SEM_WEIGHT = 0.9
DEFAULT_INTENT_WEIGHT = 0.15

_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_TOKEN = re.compile(r"[a-z0-9£]+")
_SPACES = re.compile(r"\s+")
# Negations carry meaning in complaints ("wifi not working"), so they stay even though they are usually stop words.
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - {"no", "not", "nor", "never", "cannot", "nothing", "without", "none"}


class RetrievalError(ValueError):
    pass


class RetrievalLeakageError(RetrievalError):
    """The memory contains cases that must never be retrievable (golden, reserve, dev) or a split that is not allowed."""


def clean_text(text: object) -> str:
    """Entity-free text for indexing and querying: unescape, drop urls and @mentions, keep hashtag words."""
    t = html.unescape(str(text or "")).replace("’", "'")
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(r" \1 ", t)
    return _SPACES.sub(" ", t).strip()


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: object) -> list[str]:
    low = clean_text(text).lower().replace("'", "")
    return [_stem(t) for t in _TOKEN.findall(low) if t not in STOP_WORDS]


class BM25Index:
    """Okapi BM25 over a fixed corpus, stored as a sparse (docs x vocab) weight matrix. Deterministic."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        if not len(texts):
            raise RetrievalError("cannot build a BM25 index over an empty corpus")
        self.k1, self.b = k1, b
        docs = [Counter(tokenize(t)) for t in texts]
        self.vocab = {tok: i for i, tok in enumerate(sorted({t for d in docs for t in d}))}
        indptr, indices, data = [0], [], []
        for d in docs:
            for tok in sorted(d):
                indices.append(self.vocab[tok])
                data.append(float(d[tok]))
            indptr.append(len(indices))
        tf = sparse.csr_matrix((data, indices, indptr), shape=(len(docs), len(self.vocab)), dtype=np.float32)
        n = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg = max(float(doc_len.mean()), 1e-9)
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        weights = tf.tocoo()
        norm = k1 * (1.0 - b + b * doc_len[weights.row] / avg)
        values = weights.data * (k1 + 1.0) / (weights.data + norm) * idf[weights.col]
        self.weights = sparse.csr_matrix((values.astype(np.float32), (weights.row, weights.col)), shape=tf.shape)
        self.n_docs = n

    def query_vector(self, queries: Sequence[str]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for qi, q in enumerate(queries):
            for tok, count in sorted(Counter(tokenize(q)).items()):
                if tok in self.vocab:
                    rows.append(qi)
                    cols.append(self.vocab[tok])
                    vals.append(float(count))
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(queries), len(self.vocab)), dtype=np.float32)

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        """(n_queries, n_docs) raw BM25 scores; zero for queries with no vocabulary overlap."""
        return np.asarray((self.query_vector(queries) @ self.weights.T).todense(), dtype=np.float32)


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """all-MiniLM-L6-v2 by default; loaded lazily so BM25-only use never needs the model. Fails loudly if unavailable."""

    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = 128):
        self.name = model_name
        self.batch_size = batch_size
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name)
            except Exception as exc:
                raise RetrievalError(f"sentence-transformer model {self.name!r} is unavailable: {exc}") from exc
        emb = self._model.encode(list(texts), batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


def embed_cached(texts: Sequence[str], encoder: Encoder, cache_dir: str | Path | None) -> np.ndarray:
    """Unit-normalised embeddings, cached on disk by a hash of (model, texts). Repeated runs do not recompute."""
    texts = list(texts)
    key = hashlib.sha256((encoder.name + "\x00" + "\x1f".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = Path(cache_dir) / f"retrieval_embeddings_{key}.npy" if cache_dir else None
    if path is not None and path.exists():
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            logger.info("Loaded cached embeddings %s", path.name)
            return cached
    emb = np.asarray(encoder.encode(texts), dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms > 0, norms, 1.0)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, emb)
        tmp.replace(path)
    return emb


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Per-query min-max to [0, 1]; a constant row (e.g. no lexical overlap) becomes all zeros."""
    lo = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - lo
    return np.where(span > 0, (scores - lo) / np.where(span > 0, span, 1.0), 0.0).astype(np.float32)


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    case_id: str
    score: float
    customer_problem: str
    historical_response: str
    resolution_summary: str
    resolution_type: str
    intent: str | None
    intent_match: bool
    resolved: bool
    dm_redirect: bool
    escalation_signal: str
    scores: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    low_information_query: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResolutionRetriever:
    def __init__(
        self,
        memory: pd.DataFrame,
        *,
        encoder: Encoder | None = None,
        cache_dir: str | Path | None = None,
        index_fields: Sequence[str] = ("customer_problem",),
        sem_weight: float = DEFAULT_SEM_WEIGHT,
        intent_weight: float = DEFAULT_INTENT_WEIGHT,
        k1: float = 1.5,
        b: float = 0.75,
        allowed_splits: Iterable[str] = (TRAIN,),
        forbidden_case_ids: Iterable[str] = (),
        primary_only: bool = True,
    ):
        missing = [c for c in REQUIRED_COLUMNS if c not in memory.columns]
        if missing:
            raise RetrievalError(f"resolution memory is missing columns: {missing}")
        if not 0.0 <= sem_weight <= 1.0:
            raise RetrievalError("sem_weight must be in [0, 1]")
        if intent_weight < 0:
            raise RetrievalError("intent_weight must be >= 0")
        for col in index_fields:
            if col not in memory.columns:
                raise RetrievalError(f"unknown index field {col!r}")
        allowed = set(allowed_splits)
        bad_splits = sorted(set(memory["split"]) - allowed)
        if bad_splits:
            raise RetrievalLeakageError(f"memory contains splits that may not be retrieved from: {bad_splits}")
        leaked = sorted(set(memory["case_id"]) & set(forbidden_case_ids))
        if leaked:
            raise RetrievalLeakageError(f"memory contains {len(leaked)} forbidden (golden/reserve) cases, e.g. {leaked[:3]}")
        if memory["case_id"].duplicated().any():
            raise RetrievalError("duplicate case_id in the resolution memory")
        corpus = memory
        if primary_only:
            corpus = memory[memory["in_primary_corpus"]] if "in_primary_corpus" in memory.columns else memory
        if corpus.empty:
            raise RetrievalError("the retrieval corpus is empty")
        self.corpus = corpus.reset_index(drop=True)
        self.encoder = encoder
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.index_fields = tuple(index_fields)
        self.sem_weight, self.intent_weight = float(sem_weight), float(intent_weight)
        self.k1, self.b = k1, b
        self.index_texts = [clean_text(" ".join(str(r) for r in row)) for row in self.corpus[list(self.index_fields)].itertuples(index=False)]
        self.intents = set(self.corpus["intent"].dropna())
        self._intent_arr = self.corpus["intent"].to_numpy(dtype=object)
        self._bm25: BM25Index | None = None
        self._doc_emb: np.ndarray | None = None
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._warned: set[str] = set()

    @classmethod
    def from_files(cls, memory_path: str | Path, assignments_path: str | Path | None = None, **kwargs: Any) -> "ResolutionRetriever":
        """Load the memory parquet; when the split assignments are given, every golden/reserve/dev case id is forbidden."""
        memory = pd.read_parquet(memory_path)
        forbidden: set[str] = set(kwargs.pop("forbidden_case_ids", ()))
        if assignments_path is not None:
            a = pd.read_csv(assignments_path)
            forbidden |= set(a.loc[a["split"] != TRAIN, "case_id"])
        return cls(memory, forbidden_case_ids=forbidden, **kwargs)

    # ---- indexes -------------------------------------------------------------------------------------------------
    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.index_texts, self.k1, self.b)
        return self._bm25

    @property
    def doc_embeddings(self) -> np.ndarray:
        if self._doc_emb is None:
            self._doc_emb = embed_cached(self.index_texts, self._require_encoder(), self.cache_dir)
        return self._doc_emb

    def _require_encoder(self) -> Encoder:
        if self.encoder is None:
            self.encoder = SentenceTransformerEncoder()
        return self.encoder

    def _embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        missing = [q for q in dict.fromkeys(queries) if q not in self._query_cache]
        if missing:
            emb = embed_cached(missing, self._require_encoder(), self.cache_dir if len(missing) > 8 else None)
            for q, v in zip(missing, emb):
                self._query_cache[q] = v
            while len(self._query_cache) > 4096:
                self._query_cache.popitem(last=False)
        return np.stack([self._query_cache[q] for q in queries])

    # ---- scoring -------------------------------------------------------------------------------------------------
    def normalise_intent(self, intent: object) -> str | None:
        """A usable intent or None. Unknown / fallback intents carry no signal (and never raise): the taxonomy is a candidate."""
        if intent is None:
            return None
        if not isinstance(intent, str):
            raise TypeError(f"intent must be a string or None, got {type(intent).__name__}")
        name = intent.strip()
        if not name or name == FALLBACK_INTENT:
            return None
        if name not in self.intents:
            if name not in self._warned:
                logger.warning("Intent %r is not in the memory; ignoring it and searching on the query alone", name)
                self._warned.add(name)
            return None
        return name

    def components(self, queries: Sequence[str], method: str) -> dict[str, np.ndarray]:
        if method not in METHODS:
            raise RetrievalError(f"method must be one of {METHODS}, got {method!r}")
        out: dict[str, np.ndarray] = {}
        if method in ("bm25", "hybrid"):
            out["bm25"] = self.bm25.scores(queries)
        if method in ("embedding", "hybrid"):
            out["embedding"] = self._embed_queries(queries) @ self.doc_embeddings.T
        return out

    def combine(self, comp: dict[str, np.ndarray], method: str, sem_weight: float | None = None) -> np.ndarray:
        if method == "bm25":
            return minmax_rows(comp["bm25"])
        if method == "embedding":
            return minmax_rows(comp["embedding"])
        w = self.sem_weight if sem_weight is None else sem_weight
        return w * minmax_rows(comp["embedding"]) + (1.0 - w) * minmax_rows(comp["bm25"])

    def intent_bonus(self, intents: Sequence[str | None], intent_weight: float | None = None) -> np.ndarray:
        w = self.intent_weight if intent_weight is None else intent_weight
        bonus = np.zeros((len(intents), len(self.corpus)), dtype=np.float32)
        for i, intent in enumerate(intents):
            name = self.normalise_intent(intent)
            if name is not None:
                bonus[i] = w * (self._intent_arr == name)
        return bonus

    def score_batch(self, queries: Sequence[str], intents: Sequence[str | None] | None = None, method: str = "hybrid", *, sem_weight: float | None = None, intent_weight: float | None = None) -> np.ndarray:
        """Final (n_queries, n_docs) scores on a common 0..1(+intent bonus) scale."""
        cleaned = [clean_text(q) for q in queries]
        scores = self.combine(self.components(cleaned, method), method, sem_weight)
        if intents is not None:
            if len(intents) != len(queries):
                raise RetrievalError("intents must have one entry per query")
            scores = scores + self.intent_bonus(intents, intent_weight)
        return scores

    @staticmethod
    def rank(scores_row: np.ndarray) -> np.ndarray:
        """Document order by descending score; ties break on corpus position so output is deterministic."""
        return np.lexsort((np.arange(scores_row.shape[0]), -scores_row))

    # ---- public search -------------------------------------------------------------------------------------------
    def search(self, query: str, intent: str | None = None, top_k: int = 5, method: str = "hybrid") -> list[RetrievalResult]:
        return self.search_batch([query], [intent], top_k=top_k, method=method)[0]

    def search_batch(self, queries: Sequence[str], intents: Sequence[str | None] | None = None, top_k: int = 5, method: str = "hybrid") -> list[list[RetrievalResult]]:
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise RetrievalError(f"top_k must be a positive integer, got {top_k!r}")
        queries = list(queries)
        if any(not isinstance(q, str) for q in queries):
            raise TypeError("queries must be strings")
        intents = list(intents) if intents is not None else [None] * len(queries)
        if len(intents) != len(queries):
            raise RetrievalError("intents must have one entry per query")
        usable = [bool(tokenize(q)) for q in queries]
        results: list[list[RetrievalResult]] = [[] for _ in queries]
        todo = [i for i, ok in enumerate(usable) if ok]
        if len(todo) < len(queries):
            logger.warning("%d empty queries (nothing left after removing urls/mentions/stop words) returned no results", len(queries) - len(todo))
        if not todo:
            return results
        sub_q = [queries[i] for i in todo]
        sub_i = [intents[i] for i in todo]
        cleaned = [clean_text(q) for q in sub_q]
        comp = self.components(cleaned, method)
        base = self.combine(comp, method)
        total = base + self.intent_bonus(sub_i)
        for row, qi in enumerate(todo):
            order = self.rank(total[row])[: min(int(top_k), len(self.corpus))]
            low_info = len(tokenize(queries[qi])) < 2
            name = self.normalise_intent(intents[qi])
            results[qi] = [self._result(r + 1, int(d), row, total, comp, name, low_info) for r, d in enumerate(order)]
        return results

    def _result(self, rank: int, doc: int, row: int, total: np.ndarray, comp: dict[str, np.ndarray], intent: str | None, low_info: bool) -> RetrievalResult:
        rec = self.corpus.iloc[doc]
        match = intent is not None and rec["intent"] == intent
        parts = {k: float(v[row, doc]) for k, v in comp.items()}
        parts["intent_bonus"] = float(self.intent_weight if match else 0.0)
        return RetrievalResult(
            rank=rank,
            case_id=str(rec["case_id"]),
            score=float(total[row, doc]),
            customer_problem=str(rec["customer_problem"]),
            historical_response=str(rec["historical_response"]),
            resolution_summary=str(rec["resolution_summary"]),
            resolution_type=str(rec["resolution_type"]),
            intent=None if pd.isna(rec["intent"]) else str(rec["intent"]),
            intent_match=bool(match),
            resolved=bool(rec["resolved"]),
            dm_redirect=bool(rec["dm_redirect"]),
            escalation_signal=str(rec["escalation_signal"]),
            scores=parts,
            provenance={
                "source_tweet_ids": [int(x) for x in rec["source_tweet_ids"]],
                "response_tweet_ids": [int(x) for x in rec["response_tweet_ids"]] if "response_tweet_ids" in rec else [],
                "conversation_id": str(rec["conversation_id"]) if "conversation_id" in rec else "",
                "split": str(rec["split"]),
                "intent_source": str(rec["intent_source"]) if "intent_source" in rec else "",
            },
            low_information_query=low_info,
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "n_corpus": int(len(self.corpus)),
            "index_fields": list(self.index_fields),
            "sem_weight": self.sem_weight,
            "intent_weight": self.intent_weight,
            "bm25": {"k1": self.k1, "b": self.b},
            "embedding_model": getattr(self.encoder, "name", EMBED_MODEL),
        }
