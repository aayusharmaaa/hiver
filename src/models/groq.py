"""Groq client: an OpenAI-compatible `chat/completions` backend implementing the same `LanguageModel` interface as Gemini.

* API key: `GROQ_API_KEY` from the environment. Never hard-coded, never logged, never put in an error message.
* Model name: `GROQ_MODEL` env var > `model_name` argument > `DEFAULT_MODEL`.
* Rate limits: a 429 whose `retry-after` is short (the per-minute token/request window) is waited out; a long one (a daily
  cap) raises `ModelRuntimeError` straight away so callers can stop and resume later.
* gpt-oss models reason before answering; `reasoning_effort` keeps that short. The reasoning is returned separately and
  never mixed into the answer text.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Callable

import requests

from models.base import ModelConfigError, ModelOutputError, ModelRuntimeError, parse_json_object

# Groq error messages name the account's organization id; error text ends up in committed reports.
_ORG_ID = re.compile(r"\borg_[A-Za-z0-9]+")

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
API_KEY_ENV_VAR = "GROQ_API_KEY"
MODEL_ENV_VAR = "GROQ_MODEL"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


def groq_key_from_env(environ: dict[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    return (env.get(API_KEY_ENV_VAR) or "").strip() or None


class GroqModel:
    """`LanguageModel` implementation backed by the Groq chat-completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        max_rate_limit_wait_seconds: float = 90.0,
        max_rate_limit_waits: int = 20,
        reasoning_effort: str | None = "low",
        session: Any | None = None,
        environ: dict[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        key = (api_key or "").strip() or groq_key_from_env(environ)
        if not key:
            raise ModelConfigError(f"No Groq API key found. Set {API_KEY_ENV_VAR} in the environment or the git-ignored .env file.")
        env = os.environ if environ is None else environ
        self.name = (env.get(MODEL_ENV_VAR) or "").strip() or model_name or DEFAULT_MODEL
        if timeout_seconds <= 0 or max_retries < 0 or max_rate_limit_waits < 0:
            raise ModelConfigError("timeout_seconds must be positive and retry counts must be >= 0")
        self._api_key = key
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.max_rate_limit_wait_seconds = float(max_rate_limit_wait_seconds)
        self.max_rate_limit_waits = int(max_rate_limit_waits)
        self.reasoning_effort = reasoning_effort
        self._session = session if session is not None else requests.Session()
        self._sleep = sleep

    def __repr__(self) -> str:  # never expose the key
        return f"GroqModel(name={self.name!r})"

    # ---- public interface (see models.base.LanguageModel) ----------------------------------------------------------
    def generate_text(self, prompt: str, *, system: str | None = None, temperature: float = 0.0, max_output_tokens: int = 1024) -> str:
        return self._generate(prompt, system, temperature, max_output_tokens, json_mode=False)

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return parse_json_object(self._generate(prompt, system, temperature, max_output_tokens, json_mode=True))

    # ---- internals -------------------------------------------------------------------------------------------------
    def build_request(self, prompt: str, system: str | None, temperature: float, max_output_tokens: int, json_mode: bool) -> dict[str, Any]:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {"model": self.name, "messages": messages, "temperature": float(temperature), "max_completion_tokens": int(max_output_tokens)}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort and "gpt-oss" in self.name:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _generate(self, prompt: str, system: str | None, temperature: float, max_output_tokens: int, *, json_mode: bool) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ModelOutputError("cannot call the model with an empty prompt")
        payload = self._post(self.build_request(prompt, system, temperature, max_output_tokens, json_mode))
        return self._extract_text(payload)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        last, attempt, waits = "unknown error", 0, 0
        while True:
            try:
                response = self._session.post(ENDPOINT, headers=headers, json=body, timeout=self.timeout_seconds)
            except requests.Timeout:
                last = f"timed out after {self.timeout_seconds:g}s"
            except requests.RequestException as exc:
                last = f"network error ({type(exc).__name__})"
            else:
                status = response.status_code
                if status == 200:
                    try:
                        return response.json()
                    except ValueError:
                        raise ModelOutputError("Groq returned a non-JSON HTTP body") from None
                last = f"HTTP {status}: {self._error_message(response)}"
                if status == 429:
                    delay = self._retry_after(response)
                    if delay is None or delay > self.max_rate_limit_wait_seconds or waits >= self.max_rate_limit_waits:
                        raise ModelRuntimeError(f"Groq rate limit reached ({last}; retry after {delay if delay is not None else '?'}s)")
                    waits += 1
                    logger.info("Groq rate limit; waiting %.1fs (%d/%d)", delay, waits, self.max_rate_limit_waits)
                    self._sleep(delay)
                    continue
                if status in (401, 403):
                    raise ModelConfigError(f"Groq rejected the API key or its permissions: {last}")
                if status not in RETRYABLE_STATUS:
                    hint = " (check the model name / GROQ_MODEL)" if status == 404 else ""
                    raise ModelRuntimeError(f"Groq request failed: {last}{hint}")
            if attempt >= self.max_retries:
                raise ModelRuntimeError(f"Groq request failed after {attempt + 1} attempt(s): {last}")
            delay = self.retry_backoff_seconds * (2**attempt)
            attempt += 1
            logger.warning("Groq call failed (%s); retrying in %.1fs (%d/%d)", last, delay, attempt, self.max_retries)
            if delay > 0:
                self._sleep(delay)

    @staticmethod
    def _retry_after(response: Any) -> float | None:
        try:
            return float((getattr(response, "headers", None) or {}).get("retry-after"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _error_message(response: Any) -> str:
        try:
            message = str(response.json().get("error", {}).get("message", ""))
            return _ORG_ID.sub("org_[redacted]", message)[:300] or "no error message"
        except Exception:  # noqa: BLE001 - best-effort diagnostics only
            return "no error message"

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ModelOutputError("Groq returned no choices")
        choice = choices[0]
        text = str((choice.get("message") or {}).get("content") or "").strip()
        if choice.get("finish_reason") == "length":
            raise ModelOutputError("Groq output was cut off (finish_reason=length); the response may be incomplete")
        if not text:
            raise ModelOutputError(f"Groq returned no text (finish_reason={choice.get('finish_reason')})")
        return text
