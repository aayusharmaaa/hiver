"""Synthetic resolution memory and a deterministic fake encoder, so retrieval tests need no model download."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

from retrieval import tokenize

INTENT_VOCAB = {
    "onboard_wifi_issue": ["wifi", "internet", "connect", "login", "portal", "signal"],
    "seat_reservation_issue": ["seat", "reservation", "coach", "reserved", "booked", "standing"],
    "delay_repay_refund_claim": ["delay", "refund", "compensation", "claim", "repay", "money"],
}
FALLBACK = "unclear_or_media_only"


class FakeEncoder:
    """Hashed bag-of-words embedding: lexical overlap -> cosine similarity. Counts encode() calls for cache tests."""

    def __init__(self, name: str = "fake-encoder", dim: int = 64):
        self.name, self.dim, self.calls, self.texts_seen = name, dim, 0, 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        self.texts_seen += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) % 2 == 0 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms > 0, norms, 1.0)


def make_memory(per_intent: int = 8, split: str = "train_retrieval") -> pd.DataFrame:
    rows = []
    n = 0
    for intent, words in INTENT_VOCAB.items():
        for i in range(per_intent):
            n += 1
            w = [words[(i + j) % len(words)] for j in range(3)]
            rows.append(
                {
                    "case_id": f"case_{n}",
                    "split": split,
                    "conversation_id": f"conv_{n}",
                    "intent": intent,
                    "intent_source": "candidate_taxonomy_not_ground_truth",
                    "cluster_id": 0,
                    "customer_problem": f"my {w[0]} and {w[1]} problem with {w[2]} on the train number {n}",
                    "historical_response": f"Sorry about the {w[0]}, please see https://example.com/{n}",
                    "resolution_summary": f"Agent provided information about {w[0]}.",
                    "resolution_type": "self_service" if i % 2 else "information_provided",
                    "resolution_outcome": "agent_answered_unconfirmed",
                    "resolved": bool(i % 2),
                    "dm_redirect": False,
                    "escalation_signal": "none",
                    "escalation_signal_source": "historical_agent_behaviour",
                    "evidence_quality": "strong",
                    "in_primary_corpus": True,
                    "exclusion_reason": "",
                    "source_tweet_ids": [n * 10, n * 10 + 1],
                    "response_tweet_ids": [n * 10 + 1],
                    "context_tweet_ids": [],
                    "n_brand_turns": 1,
                    "first_timestamp": pd.Timestamp("2017-10-24", tz="UTC"),
                }
            )
    return pd.DataFrame(rows)
