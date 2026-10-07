"""End-to-end support agent: classify -> retrieve -> risk policy -> (generate -> verify grounding) or escalate.

    message -> IntentClassifier -> ResolutionRetriever.search(query=message, intent=...) -> policy.decide
            -> AUTO_HANDLE: ReplyGenerator -> GroundingVerifier -> reply   (any failure -> ESCALATE, no reply)
            -> ESCALATE:    no reply

Model *runtime* failures (timeouts, unusable output) degrade to ESCALATE; *configuration* failures (e.g. no API key) are raised.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import yaml

from agent import policy
from agent.classifier import IntentClassifier
from agent.config import AgentConfig, load_config
from agent.generator import ReplyGenerator
from agent.grounding import GroundingVerifier
from agent.schemas import AUTO_HANDLE, ESCALATE, AgentResult, Classification, Decision, Evidence, GroundingResult
from models.base import LanguageModel, ModelOutputError, ModelRuntimeError
from taxonomy.registry import FALLBACK_INTENT

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_TAXONOMY = REPO_ROOT / "configs" / "virgintrains_intents.yaml"

_RECOVERABLE = (ModelRuntimeError, ModelOutputError)  # includes ClassifierError


class Retriever(Protocol):
    def search(self, query: str, intent: str | None = None, top_k: int = 5, method: str = "hybrid") -> list[Any]: ...


def evidence_from_result(result: Any) -> Evidence:
    """Map a `retrieval.RetrievalResult` onto `Evidence`, keeping all provenance (tweet ids, resolution type, scores)."""
    provenance = result.provenance or {}
    return Evidence(
        case_id=str(result.case_id),
        score=float(result.score),
        customer_problem=result.customer_problem,
        historical_response=result.historical_response,
        resolution_summary=result.resolution_summary,
        resolution_type=result.resolution_type,
        source_tweet_ids=[str(t) for t in provenance.get("source_tweet_ids", [])],
        response_tweet_ids=[str(t) for t in provenance.get("response_tweet_ids", [])],
        semantic_similarity=result.scores.get("embedding"),
        lexical_score=result.scores.get("bm25"),
        intent=result.intent,
        escalation_signal=result.escalation_signal,
        dm_redirect=bool(result.dm_redirect),
    )


class SupportAgent:
    def __init__(self, classifier: IntentClassifier, retriever: Retriever, generator: ReplyGenerator, verifier: GroundingVerifier, config: AgentConfig | None = None):
        self.classifier, self.retriever, self.generator, self.verifier = classifier, retriever, generator, verifier
        self.config = config or AgentConfig()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        processed_dir: str | Path | None = None,
        taxonomy_path: str | Path | None = None,
        model: LanguageModel | None = None,
    ) -> "SupportAgent":
        """Wire the real components. The model is created first so a missing API key fails before the large indexes are loaded."""
        from models.gemini import GeminiModel  # lazy: keeps `import agent` free of network libraries
        from retrieval import ResolutionRetriever, SentenceTransformerEncoder

        config = load_config(config_path)
        if model is None:
            m = config.model
            model = GeminiModel(model_name=m.name, timeout_seconds=m.timeout_seconds, max_retries=m.max_retries, response_schema=m.response_schema)
        processed = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED
        memory = processed / "virgintrains_resolution_memory.parquet"
        assignments = processed / "splits" / "virgintrains_split_assignments.csv"
        if not memory.exists():
            raise FileNotFoundError(f"{memory} not found. Run: python scripts/build_resolution_memory.py")
        taxonomy = yaml.safe_load(Path(taxonomy_path or DEFAULT_TAXONOMY).read_text(encoding="utf-8"))["taxonomy"]
        retriever = ResolutionRetriever.from_files(memory, assignments if assignments.exists() else None, encoder=SentenceTransformerEncoder(), cache_dir=processed / "cache")
        return cls.from_parts(model, retriever, taxonomy, config)

    @classmethod
    def from_parts(cls, model: LanguageModel, retriever: Retriever, taxonomy: dict[str, Any], config: AgentConfig | None = None) -> "SupportAgent":
        config = config or AgentConfig()
        m = config.model
        return cls(
            IntentClassifier(model, taxonomy, examples_per_intent=config.classifier.examples_per_intent, temperature=m.classifier_temperature, max_output_tokens=m.max_output_tokens, use_schema=m.response_schema),
            retriever,
            ReplyGenerator(model, temperature=m.generator_temperature, max_output_tokens=m.max_output_tokens, use_schema=m.response_schema),
            GroundingVerifier(model, min_confidence=config.grounding.min_confidence, temperature=m.verifier_temperature, max_output_tokens=m.max_output_tokens, use_schema=m.response_schema),
            config,
        )

    def handle(self, message: str, conversation_context: str | None = None) -> AgentResult:
        if not isinstance(message, str):
            raise TypeError(f"message must be a string, got {type(message).__name__}")
        if not message.strip():
            unclear = Classification(intent=FALLBACK_INTENT, confidence=0.0, rationale="empty message")
            return AgentResult(classification=unclear, decision=Decision(action=ESCALATE, confidence=1.0, reasons=["empty message"]))

        # 1. classify (candidate taxonomy; failures degrade to "unclear" and escalate)
        extra: list[str] = []
        try:
            classification = self.classifier.classify(message, conversation_context)
        except _RECOVERABLE as exc:
            logger.warning("Classification failed: %s", exc)
            classification = Classification(intent=FALLBACK_INTENT, confidence=0.0, rationale=f"classification unavailable: {exc}"[:300])
            extra.append(f"classification failed: {exc}"[:300])

        # 2. retrieve (existing retriever, no extra retrieval logic here)
        results = self.retriever.search(query=message, intent=classification.intent, top_k=self.config.retrieval.top_k)
        evidence = [evidence_from_result(r) for r in results]

        # 3. risk policy (deterministic)
        decision = policy.decide(classification, evidence, message, self.config.policy, low_information_query=any(getattr(r, "low_information_query", False) for r in results))
        if extra:
            decision = decision.model_copy(update={"reasons": extra + decision.reasons})
        if decision.action != AUTO_HANDLE:
            return AgentResult(classification=classification, evidence=evidence, decision=decision)

        # 4. generate, then 5. verify grounding; any failure -> escalate with no reply
        try:
            draft = self.generator.generate(message, conversation_context, classification, evidence, decision)
        except _RECOVERABLE as exc:
            logger.warning("Generation failed: %s", exc)
            return self._escalate(classification, evidence, decision, f"reply generation failed: {exc}"[:300])
        if not draft:
            return self._escalate(classification, evidence, decision, "the generator could not produce a reply supported by the evidence")

        try:
            grounding = self.verifier.verify(draft, message, evidence)
        except _RECOVERABLE as exc:
            logger.warning("Grounding verification unavailable: %s", exc)
            grounding = GroundingResult(grounded=False, confidence=0.0, unsupported_claims=[f"could not verify: {exc}"[:300]])
        if not grounding.grounded:
            claims = "; ".join(grounding.unsupported_claims[:3]) or "no detail"
            return self._escalate(classification, evidence, decision, f"grounding verification failed: {claims}", grounding=grounding, draft=draft)
        return AgentResult(classification=classification, evidence=evidence, decision=decision, reply=draft, grounding=grounding)

    @staticmethod
    def _escalate(classification: Classification, evidence: list[Evidence], decision: Decision, reason: str, *, grounding: GroundingResult | None = None, draft: str | None = None) -> AgentResult:
        escalated = Decision(action=ESCALATE, confidence=max(0.5, grounding.confidence) if grounding else 0.5, reasons=[f"policy check passed: {r}" for r in decision.reasons] + [reason])
        return AgentResult(classification=classification, evidence=evidence, decision=escalated, reply=None, grounding=grounding, internal_draft_reply=draft)
