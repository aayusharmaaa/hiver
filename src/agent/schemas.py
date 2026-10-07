"""Typed contracts between the agent's stages. `Classification` uses the *candidate* intent taxonomy (not human-validated)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AUTO_HANDLE = "AUTO_HANDLE"
ESCALATE = "ESCALATE"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Classification(_Model):
    """Which candidate intent the customer message most likely belongs to. `confidence` is the model's self-report, not a calibrated probability."""

    intent: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    alternative_intent: str | None = None
    multi_intent: bool = False
    rationale: str | None = None


class Evidence(_Model):
    """One retrieved historical case, with the provenance needed to trace it back to the source tweets."""

    case_id: str
    score: float  # final retrieval score (hybrid + intent bonus); min-max normalised per query, so NOT comparable across queries
    customer_problem: str
    historical_response: str
    resolution_summary: str
    resolution_type: str
    source_tweet_ids: list[str] = Field(default_factory=list)
    # Extra provenance / signals used by the risk policy (all optional)
    semantic_similarity: float | None = None  # raw cosine similarity of query and historical problem: comparable across queries
    lexical_score: float | None = None  # raw BM25
    intent: str | None = None  # candidate intent of the historical case
    escalation_signal: str = "none"  # what the historical agent did (hand-off, DM, ...); descriptive, not a policy
    dm_redirect: bool = False
    response_tweet_ids: list[str] = Field(default_factory=list)


class Decision(_Model):
    action: Literal["AUTO_HANDLE", "ESCALATE"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class GroundingResult(_Model):
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)


class AgentResult(_Model):
    classification: Classification
    evidence: list[Evidence] = Field(default_factory=list)
    decision: Decision
    reply: str | None = None  # only ever set when the decision is AUTO_HANDLE and the reply passed grounding
    grounding: GroundingResult | None = None
    internal_draft_reply: str | None = None  # a reply that failed grounding; for a human reviewer only, never to be sent
