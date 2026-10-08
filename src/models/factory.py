"""Pick the LLM backend by name. Both clients read their key from the environment (or the git-ignored .env)."""

from __future__ import annotations

from typing import Any

PROVIDERS = ("groq", "gemini")
KEY_ENV = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}


def provider_key(provider: str) -> str | None:
    if provider == "groq":
        from models.groq import groq_key_from_env

        return groq_key_from_env()
    from models.gemini import api_key_from_env

    return api_key_from_env()


def build_model(provider: str, settings: Any) -> Any:
    """`settings` is the agent config's `model` section (Gemini name, timeouts, retries, schema flag)."""
    if provider == "groq":
        from models.groq import GroqModel

        return GroqModel(max_retries=settings.max_retries)
    if provider == "gemini":
        from models.gemini import GeminiModel

        return GeminiModel(model_name=settings.name, timeout_seconds=settings.timeout_seconds, max_retries=settings.max_retries, response_schema=settings.response_schema)
    raise ValueError(f"unknown provider {provider!r}; choose one of {PROVIDERS}")
