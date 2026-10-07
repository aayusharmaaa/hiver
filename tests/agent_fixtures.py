"""Test doubles for the support agent: a scripted language model and a fake retriever. No network, no API key."""

from __future__ import annotations

from typing import Any

from agent import classifier as classifier_mod
from agent import generator as generator_mod
from agent import grounding as grounding_mod
from agent.config import AgentConfig, PolicySettings
from models.base import ModelError
from retrieval import RetrievalResult

TAXONOMY = {
    "status": "CANDIDATE_NOT_GROUND_TRUTH",
    "intents": [
        {"name": "onboard_wifi_issue", "definition": "Wifi on the train does not work.", "inclusion_criteria": ["wifi, internet on board"], "exclusion_criteria": [], "positive_examples": [{"text": "wifi not working on my train", "case_id": "c1"}]},
        {"name": "ticket_booking_query", "definition": "Questions about tickets and booking.", "inclusion_criteria": ["ticket types"], "exclusion_criteria": [], "positive_examples": [{"text": "can I use my railcard on this ticket?", "case_id": "c2"}]},
        {"name": "service_status_delay_enquiry", "definition": "Is a train running?", "inclusion_criteria": [], "exclusion_criteria": [], "positive_examples": []},
    ],
    "fallback": {"name": "unclear_or_media_only", "definition": "Unclear or media only.", "inclusion_criteria": [], "exclusion_criteria": [], "positive_examples": []},
}

GOOD_CLASSIFICATION = {"intent": "onboard_wifi_issue", "confidence": 0.92, "alternative_intent": None, "multi_intent": False, "rationale": "wifi problem"}
GOOD_VERIFICATION = {"grounded": True, "confidence": 0.95, "unsupported_claims": []}
GOOD_REPLY = {"reply": "Sorry the wifi is not working. Please try reconnecting and see https://example.com/wifi for help."}
MESSAGE = "The wifi on my train keeps dropping and I cannot log in"


def settings(**overrides: Any) -> PolicySettings:
    base = {"escalate_intents": {"service_status_delay_enquiry": "needs live data"}, "sensitive_patterns": [r"\b(ombudsman|solicitor|injur\w*)\b"]}
    base.update(overrides)
    return PolicySettings(**base)


def agent_config(**policy_overrides: Any) -> AgentConfig:
    return AgentConfig(policy=settings(**policy_overrides))


def make_result(i: int = 1, *, similarity: float = 0.8, intent: str | None = "onboard_wifi_issue", response: str = "Sorry about the wifi, please see https://example.com/wifi for help", rtype: str = "self_service", signal: str = "none", score: float = 1.1, tweets=None) -> RetrievalResult:
    return RetrievalResult(
        rank=i,
        case_id=f"case_{i}",
        score=score,
        customer_problem=f"wifi problem number {i} on the train",
        historical_response=response,
        resolution_summary="Agent shared the wifi help page.",
        resolution_type=rtype,
        intent=intent,
        intent_match=True,
        resolved=False,
        dm_redirect=False,
        escalation_signal=signal,
        scores={"bm25": 3.0, "embedding": similarity, "intent_bonus": 0.15},
        provenance={"source_tweet_ids": tweets if tweets is not None else [i * 10, i * 10 + 1], "response_tweet_ids": [i * 10 + 1], "conversation_id": f"conv_{i}", "split": "train_retrieval", "intent_source": "candidate_taxonomy_not_ground_truth"},
    )


def good_results(n: int = 5, **kwargs: Any) -> list[RetrievalResult]:
    return [make_result(i, **kwargs) for i in range(1, n + 1)]


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None):
        self.results = good_results() if results is None else results
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, intent: str | None = None, top_k: int = 5, method: str = "hybrid") -> list[RetrievalResult]:
        self.calls.append({"query": query, "intent": intent, "top_k": top_k, "method": method})
        return self.results[:top_k]


class ScriptedModel:
    """Routes by system prompt to the classifier / generator / verifier script. A script is a dict, an Exception, or a list of them (consumed in order)."""

    name = "scripted-model"

    def __init__(self, classifier: Any = GOOD_CLASSIFICATION, generator: Any = GOOD_REPLY, verifier: Any = GOOD_VERIFICATION):
        self.scripts = {"classifier": classifier, "generator": generator, "verifier": verifier}
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _role(system: str | None) -> str:
        return {classifier_mod.SYSTEM_PROMPT: "classifier", generator_mod.SYSTEM_PROMPT: "generator", grounding_mod.SYSTEM_PROMPT: "verifier"}[system]

    def count(self, role: str) -> int:
        return sum(c["role"] == role for c in self.calls)

    def generate_text(self, prompt: str, *, system: str | None = None, temperature: float = 0.0, max_output_tokens: int = 1024) -> str:
        raise AssertionError("the agent only uses JSON generation")

    def generate_json(self, prompt: str, *, system: str | None = None, temperature: float = 0.0, max_output_tokens: int = 1024, schema: dict | None = None) -> dict[str, Any]:
        role = self._role(system)
        self.calls.append({"role": role, "prompt": prompt, "temperature": temperature})
        script = self.scripts[role]
        if isinstance(script, list):
            script = script.pop(0)
        if isinstance(script, (ModelError, Exception)):
            raise script
        return dict(script)
