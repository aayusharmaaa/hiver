"""Gemini client: the only place in the project that talks to the Gemini API.

It calls the public `generateContent` REST endpoint with `requests`, so it does not depend on a particular Gemini SDK version
(the SDK has changed names and the old `google-generativeai` package is deprecated).

* API key: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) from the environment. Never hard-coded, never logged, never put in an error message.
* Model name: `GEMINI_MODEL` env var > `model_name` argument / config file > `DEFAULT_MODEL`.
* The HTTP session is injectable, so every code path is unit-testable without network access or a key.
* Structured output: JSON mode (`responseMimeType: application/json`) is always requested for `generate_json`; the JSON shape is
  described in the prompt and the reply is validated by the caller. Passing `response_schema` also sends a `responseSchema`
  (off by default).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from models.base import ModelConfigError, ModelOutputError, ModelRuntimeError, parse_json_object

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
MODEL_ENV_VAR = "GEMINI_MODEL"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def api_key_from_env(environ: dict[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    for name in API_KEY_ENV_VARS:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return None


class GeminiModel:
    """`LanguageModel` implementation backed by the Gemini REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        response_schema: bool = False,
        session: Any | None = None,
        environ: dict[str, str] | None = None,
    ):
        key = (api_key or "").strip() or api_key_from_env(environ)
        if not key:
            raise ModelConfigError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment variable (get a key at https://aistudio.google.com/apikey); "
                "the key is read from the environment and is never stored in the repository."
            )
        env = os.environ if environ is None else environ
        self.name = (env.get(MODEL_ENV_VAR) or "").strip() or model_name or DEFAULT_MODEL
        if timeout_seconds <= 0:
            raise ModelConfigError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ModelConfigError("max_retries must be >= 0")
        self._api_key = key
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.send_response_schema = bool(response_schema)
        self._session = session if session is not None else requests.Session()

    def __repr__(self) -> str:  # never expose the key
        return f"GeminiModel(name={self.name!r})"

    # ---- public interface (see models.base.LanguageModel) ----------------------------------------------------------
    def generate_text(self, prompt: str, *, system: str | None = None, temperature: float = 0.0, max_output_tokens: int = 1024) -> str:
        return self._generate(prompt, system, temperature, max_output_tokens, json_mode=False, schema=None)

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = self._generate(prompt, system, temperature, max_output_tokens, json_mode=True, schema=schema)
        return parse_json_object(text)

    # ---- internals -------------------------------------------------------------------------------------------------
    def build_request(self, prompt: str, system: str | None, temperature: float, max_output_tokens: int, json_mode: bool, schema: dict[str, Any] | None) -> dict[str, Any]:
        config: dict[str, Any] = {"temperature": float(temperature), "maxOutputTokens": int(max_output_tokens)}
        if json_mode:
            config["responseMimeType"] = "application/json"
            if schema is not None and self.send_response_schema:
                config["responseSchema"] = schema
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _generate(self, prompt: str, system: str | None, temperature: float, max_output_tokens: int, *, json_mode: bool, schema: dict[str, Any] | None) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ModelOutputError("cannot call the model with an empty prompt")
        body = self.build_request(prompt, system, temperature, max_output_tokens, json_mode, schema)
        payload = self._post(body)
        return self._extract_text(payload)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = ENDPOINT.format(model=self.name)
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        last = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(url, headers=headers, json=body, timeout=self.timeout_seconds)
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
                        raise ModelOutputError("Gemini returned a non-JSON HTTP body") from None
                last = f"HTTP {status}: {self._error_message(response)}"
                if status in (400, 404):
                    hint = " (check the model name / GEMINI_MODEL)" if status == 404 else ""
                    raise ModelRuntimeError(f"Gemini rejected the request: {last}{hint}")
                if status in (401, 403):
                    raise ModelConfigError(f"Gemini rejected the API key or its permissions: {last}")
                if status not in RETRYABLE_STATUS:
                    raise ModelRuntimeError(f"Gemini request failed: {last}")
            if attempt < self.max_retries:
                delay = self.retry_backoff_seconds * (2**attempt)
                logger.warning("Gemini call failed (%s); retrying in %.1fs (%d/%d)", last, delay, attempt + 1, self.max_retries)
                if delay > 0:
                    time.sleep(delay)
        raise ModelRuntimeError(f"Gemini request failed after {self.max_retries + 1} attempt(s): {last}")

    @staticmethod
    def _error_message(response: Any) -> str:
        try:
            return str(response.json().get("error", {}).get("message", ""))[:300] or "no error message"
        except Exception:  # noqa: BLE001 - best-effort diagnostics only
            return "no error message"

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        feedback = payload.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise ModelRuntimeError(f"Gemini blocked the prompt ({feedback['blockReason']})")
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ModelOutputError("Gemini returned no candidates")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict) and not p.get("thought")).strip()
        finish = candidate.get("finishReason")
        if not text:
            raise ModelOutputError(f"Gemini returned no text (finishReason={finish})")
        if finish == "MAX_TOKENS":
            raise ModelOutputError("Gemini output was cut off (MAX_TOKENS); the response may be incomplete")
        return text
