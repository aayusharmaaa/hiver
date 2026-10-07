"""Typed loader for `configs/support_agent.yaml` (the single place for model settings and policy thresholds)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "support_agent.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSettings(_Section):
    name: str = "gemini-2.5-flash"
    timeout_seconds: float = Field(30.0, gt=0)
    max_retries: int = Field(2, ge=0)
    max_output_tokens: int = Field(2048, ge=64)
    response_schema: bool = False
    classifier_temperature: float = Field(0.0, ge=0, le=2)
    generator_temperature: float = Field(0.2, ge=0, le=2)
    verifier_temperature: float = Field(0.0, ge=0, le=2)


class RetrievalSettings(_Section):
    top_k: int = Field(5, ge=1)


class ClassifierSettings(_Section):
    examples_per_intent: int = Field(2, ge=0)


class PolicySettings(_Section):
    min_classifier_confidence: float = Field(0.75, ge=0, le=1)
    min_top_similarity: float = Field(0.60, ge=-1, le=1)
    min_usable_similarity: float = Field(0.55, ge=-1, le=1)
    min_usable_evidence: int = Field(3, ge=1)
    min_response_words: int = Field(4, ge=0)
    min_intent_agreement: float = Field(0.6, ge=0, le=1)
    min_resolution_consistency: float = Field(0.4, ge=0, le=1)
    human_required_signals: list[str] = Field(default_factory=lambda: ["dm_for_account_lookup", "customer_relations_or_formal_route"])
    escalate_human_signal_share: float = Field(0.4, gt=0, le=1)
    min_message_words: int = Field(3, ge=0)
    escalate_intents: dict[str, str] = Field(default_factory=dict)
    sensitive_patterns: list[str] = Field(default_factory=list)

    @field_validator("sensitive_patterns")
    @classmethod
    def _valid_regexes(cls, patterns: list[str]) -> list[str]:
        for p in patterns:
            try:
                re.compile(p)
            except re.error as exc:
                raise ValueError(f"invalid sensitive pattern {p!r}: {exc}") from None
        return patterns


class GroundingSettings(_Section):
    min_confidence: float = Field(0.7, ge=0, le=1)


class AgentConfig(_Section):
    model: ModelSettings = Field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    grounding: GroundingSettings = Field(default_factory=GroundingSettings)


def load_config(path: str | Path | None = None) -> AgentConfig:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"support-agent config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AgentConfig.model_validate(raw)
