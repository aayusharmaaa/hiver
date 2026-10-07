"""Data-driven intent discovery from customer opening messages (no LLM generation).

Pipeline: normalize -> TF-IDF exploration -> sentence embeddings -> KMeans over k in a range
-> per-cluster diagnostics. A TF-IDF/LSA clustering is run alongside purely as an
independent agreement check on the embedding clusters.

Banking77 is intentionally not used: the taxonomy must come from this brand's own data.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

from taxonomy.entity_masking import mask_entities

logger = logging.getLogger(__name__)

_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_DIGITS = re.compile(r"\d+")
_NON_TEXT = re.compile(r"[^a-z0-9'<> ]+")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INSUFFICIENT = -1


def normalize_opening(text: str | None, mask: bool = True) -> str:
    """Lowercase, drop mentions/urls, keep hashtag words, mask entities and digits.

    The stored/displayed text is never changed; this key is only used for modelling.
    """
    if not text:
        return ""
    t = html.unescape(text).replace("’", "'")
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(r" \1 ", t).lower()
    if mask:
        t = mask_entities(t)
    t = _DIGITS.sub("0", t)
    return " ".join(_NON_TEXT.sub(" ", t).split())


def informative_mask(norm: pd.Series, min_tokens: int = 3) -> pd.Series:
    """True when the opener has at least `min_tokens` words besides placeholders and digits."""
    return norm.map(lambda s: sum(1 for w in s.split() if not (w.startswith("<") or w.isdigit() or w == "0")) >= min_tokens)


def tfidf_matrix(norm_texts: list[str], min_df: int = 5):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.4, stop_words="english", sublinear_tf=True)
    return vec, vec.fit_transform(norm_texts)


def top_global_terms(vec: TfidfVectorizer, matrix, n: int = 40) -> list[tuple[str, float]]:
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(-scores)[:n]
    return [(terms[i], float(scores[i])) for i in order]


def embed_texts(texts: list[str], cache_dir: str | Path | None = None, model_name: str = EMBED_MODEL) -> tuple[np.ndarray, str]:
    """Unit-normalized sentence embeddings, cached by content hash. Falls back to TF-IDF+LSA."""
    key = hashlib.sha256(("\n".join(texts) + model_name).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"embeddings_{key}.npy" if cache_dir else None
    if cache is not None and cache.exists():
        logger.info("Loaded cached embeddings %s", cache)
        return np.load(cache), model_name
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        emb = model.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        used = model_name
    except Exception as exc:  # network/model unavailable
        logger.warning("Sentence-transformer unavailable (%s); falling back to TF-IDF + LSA embeddings", exc)
        _, mat = tfidf_matrix(texts, min_df=2)
        emb = normalize(TruncatedSVD(n_components=128, random_state=0).fit_transform(mat))
        used = "tfidf_lsa_128"
    emb = np.asarray(emb, dtype=np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
    return emb, used


@dataclass
class KSweepRow:
    k: int
    silhouette: float
    inertia: float
    seed_agreement_ari: float
    min_cluster_share: float


def sweep_k(emb: np.ndarray, ks: range, seed: int = 42, silhouette_sample: int = 5000) -> list[KSweepRow]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emb), size=min(silhouette_sample, len(emb)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(emb)
        km2 = KMeans(n_clusters=k, n_init=5, random_state=seed + 1).fit(emb)
        sil = float(silhouette_score(emb[idx], km.labels_[idx], metric="cosine"))
        shares = np.bincount(km.labels_, minlength=k) / len(emb)
        rows.append(KSweepRow(k, sil, float(km.inertia_), float(adjusted_rand_score(km.labels_, km2.labels_)), float(shares.min())))
        logger.info("k=%d silhouette=%.3f seed_ARI=%.3f min_share=%.3f", k, sil, rows[-1].seed_agreement_ari, rows[-1].min_cluster_share)
    return rows


def fit_kmeans(emb: np.ndarray, k: int, seed: int = 42) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb)


def assignment_details(km: KMeans, emb: np.ndarray) -> pd.DataFrame:
    """Nearest/second-nearest cluster and the distance margin between them (small = ambiguous)."""
    dist = km.transform(emb)
    order = np.argsort(dist, axis=1)
    rows = np.arange(len(emb))
    first, second = order[:, 0], order[:, 1]
    return pd.DataFrame(
        {
            "cluster_id": first,
            "second_cluster_id": second,
            "distance": dist[rows, first],
            "second_distance": dist[rows, second],
            "margin": dist[rows, second] - dist[rows, first],
        }
    )


def cluster_top_terms(norm_texts: list[str], labels: np.ndarray, n_terms: int = 12) -> dict[int, list[str]]:
    """Terms with the highest mean TF-IDF inside a cluster relative to the corpus."""
    vec, mat = tfidf_matrix(norm_texts, min_df=3)
    terms = np.array(vec.get_feature_names_out())
    overall = np.asarray(mat.mean(axis=0)).ravel()
    out = {}
    for c in sorted(set(labels.tolist())):
        inside = np.asarray(mat[labels == c].mean(axis=0)).ravel()
        out[int(c)] = terms[np.argsort(-(inside - 0.5 * overall))[:n_terms]].tolist()
    return out


def tfidf_agreement(norm_texts: list[str], embed_labels: np.ndarray, k: int, seed: int = 42) -> dict[str, float]:
    """Independent check: cluster TF-IDF/LSA vectors and compare partitions with the embedding clusters."""
    _, mat = tfidf_matrix(norm_texts, min_df=3)
    lsa = normalize(TruncatedSVD(n_components=100, random_state=seed).fit_transform(mat))
    other = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(lsa)
    return {
        "nmi": float(normalized_mutual_info_score(embed_labels, other)),
        "ari": float(adjusted_rand_score(embed_labels, other)),
    }
