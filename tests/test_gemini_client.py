"""GeminiModel unit tests with a fake HTTP session: no network, no real key."""

from __future__ import annotations

import pytest
import requests

from models.base import ModelConfigError, ModelOutputError, ModelRuntimeError, parse_json_object
from models.gemini import DEFAULT_MODEL, GeminiModel, api_key_from_env

SECRET = "AIza-super-secret-key"


class FakeResponse:
    def __init__(self, status=200, payload=None, text_ok=True):
        self.status_code, self._payload, self._ok = status, payload if payload is not None else {}, text_ok

    def json(self):
        if not self._ok:
            raise ValueError("not json")
        return self._payload


def ok(text="{}", **extra):
    return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP", **extra}]})


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def model(*responses, **kw):
    kw.setdefault("retry_backoff_seconds", 0.0)
    return GeminiModel(api_key=SECRET, session=FakeSession(*responses), environ={}, **kw)


class TestConfiguration:
    def test_no_key_is_a_clear_config_error(self):
        with pytest.raises(ModelConfigError, match="GEMINI_API_KEY"):
            GeminiModel(environ={})

    def test_blank_key_counts_as_missing(self):
        with pytest.raises(ModelConfigError):
            GeminiModel(api_key="   ", environ={"GEMINI_API_KEY": " "})

    def test_key_from_environment_with_google_fallback(self):
        assert api_key_from_env({"GEMINI_API_KEY": "a", "GOOGLE_API_KEY": "b"}) == "a"
        assert api_key_from_env({"GOOGLE_API_KEY": "b"}) == "b"
        assert api_key_from_env({}) is None

    def test_model_name_precedence(self):
        assert GeminiModel(api_key="k", environ={}).name == DEFAULT_MODEL
        assert GeminiModel(api_key="k", model_name="from-config", environ={}).name == "from-config"
        assert GeminiModel(api_key="k", model_name="from-config", environ={"GEMINI_MODEL": "from-env"}).name == "from-env"

    def test_key_is_never_in_repr(self):
        assert SECRET not in repr(model())

    def test_invalid_settings_rejected(self):
        with pytest.raises(ModelConfigError):
            GeminiModel(api_key="k", timeout_seconds=0, environ={})


class TestRequests:
    def test_json_request_shape(self):
        m = model(ok('{"a": 1}'))
        assert m.generate_json("hello", system="sys", temperature=0.0, max_output_tokens=300) == {"a": 1}
        call = m._session.calls[0]
        assert call["url"].endswith(f"/models/{DEFAULT_MODEL}:generateContent")
        assert call["headers"]["x-goog-api-key"] == SECRET and SECRET not in call["url"]
        assert call["json"]["generationConfig"] == {"temperature": 0.0, "maxOutputTokens": 300, "responseMimeType": "application/json"}
        assert call["json"]["systemInstruction"]["parts"][0]["text"] == "sys"
        assert call["json"]["contents"][0]["parts"][0]["text"] == "hello" and call["timeout"] == 30.0

    def test_text_request_has_no_json_mode(self):
        m = model(ok("plain"))
        assert m.generate_text("hi") == "plain"
        assert "responseMimeType" not in m._session.calls[0]["json"]["generationConfig"]

    def test_response_schema_is_opt_in(self):
        schema = {"type": "OBJECT"}
        off, on = model(ok()), model(ok(), response_schema=True)
        off.generate_json("p", schema=schema)
        on.generate_json("p", schema=schema)
        assert "responseSchema" not in off._session.calls[0]["json"]["generationConfig"]
        assert on._session.calls[0]["json"]["generationConfig"]["responseSchema"] == schema

    def test_empty_prompt_rejected(self):
        with pytest.raises(ModelOutputError):
            model().generate_text("  ")


class TestResponses:
    def test_fenced_json_is_parsed(self):
        assert model(ok('```json\n{"a": 2}\n```')).generate_json("p") == {"a": 2}

    def test_invalid_json_is_an_output_error(self):
        with pytest.raises(ModelOutputError):
            model(ok("sorry, I cannot do that")).generate_json("p")

    def test_json_array_is_rejected(self):
        with pytest.raises(ModelOutputError):
            parse_json_object("[1, 2]")

    def test_thinking_parts_are_ignored(self):
        payload = {"candidates": [{"content": {"parts": [{"text": "hidden", "thought": True}, {"text": '{"a": 3}'}]}, "finishReason": "STOP"}]}
        assert model(FakeResponse(200, payload)).generate_json("p") == {"a": 3}

    def test_blocked_prompt(self):
        with pytest.raises(ModelRuntimeError, match="blocked"):
            model(FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}})).generate_text("p")

    def test_no_candidates_and_truncation(self):
        with pytest.raises(ModelOutputError):
            model(FakeResponse(200, {"candidates": []})).generate_text("p")
        truncated = FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "{"}]}, "finishReason": "MAX_TOKENS"}]})
        with pytest.raises(ModelOutputError, match="MAX_TOKENS"):
            model(truncated).generate_json("p")

    def test_non_json_http_body(self):
        with pytest.raises(ModelOutputError):
            model(FakeResponse(200, text_ok=False)).generate_text("p")


class TestErrorsAndRetries:
    def test_transient_errors_are_retried(self):
        m = model(FakeResponse(503), FakeResponse(429), ok('{"a": 1}'))
        assert m.generate_json("p") == {"a": 1} and len(m._session.calls) == 3

    def test_retries_are_bounded(self):
        m = model(FakeResponse(503), FakeResponse(503), FakeResponse(503), max_retries=2)
        with pytest.raises(ModelRuntimeError, match="3 attempt"):
            m.generate_text("p")

    def test_timeout_and_network_errors_are_runtime_errors(self):
        with pytest.raises(ModelRuntimeError, match="timed out"):
            model(requests.Timeout(), requests.Timeout(), requests.Timeout()).generate_text("p")
        with pytest.raises(ModelRuntimeError, match="network error"):
            model(requests.ConnectionError(), requests.ConnectionError(), requests.ConnectionError()).generate_text("p")

    def test_bad_key_is_a_config_error_and_not_retried(self):
        m = model(FakeResponse(403, {"error": {"message": "API key not valid"}}))
        with pytest.raises(ModelConfigError, match="API key"):
            m.generate_text("p")
        assert len(m._session.calls) == 1

    def test_unknown_model_gets_a_hint(self):
        with pytest.raises(ModelRuntimeError, match="GEMINI_MODEL"):
            model(FakeResponse(404, {"error": {"message": "model not found"}})).generate_text("p")

    def test_secret_never_appears_in_error_messages(self):
        with pytest.raises(ModelRuntimeError) as exc:
            model(FakeResponse(400, {"error": {"message": "bad request"}})).generate_text("p")
        assert SECRET not in str(exc.value)
