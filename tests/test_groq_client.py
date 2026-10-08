"""GroqModel unit tests with a fake HTTP session: no network, no real key."""

from __future__ import annotations

import pytest
import requests

from models.base import LanguageModel, ModelConfigError, ModelOutputError, ModelRuntimeError
from models.groq import DEFAULT_MODEL, ENDPOINT, GroqModel, groq_key_from_env

SECRET = "gsk-super-secret-key"


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload if payload is not None else {}, headers or {}

    def json(self):
        return self._payload


def ok(text="{}", finish="stop"):
    return FakeResponse(200, {"choices": [{"message": {"content": text, "reasoning": "hidden thoughts"}, "finish_reason": finish}]})


def limited(retry_after):
    return FakeResponse(429, {"error": {"message": "Rate limit reached"}}, {"retry-after": str(retry_after)})


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def model(*responses, sleeps=None, **kw):
    kw.setdefault("retry_backoff_seconds", 0.0)
    return GroqModel(api_key=SECRET, session=FakeSession(*responses), environ={}, sleep=(sleeps.append if sleeps is not None else lambda s: None), **kw)


def test_configuration_key_model_name_and_repr() -> None:
    with pytest.raises(ModelConfigError, match="GROQ_API_KEY"):
        GroqModel(environ={})
    assert groq_key_from_env({"GROQ_API_KEY": " k "}) == "k" and groq_key_from_env({}) is None
    assert GroqModel(api_key="k", environ={}).name == DEFAULT_MODEL
    assert GroqModel(api_key="k", model_name="cfg", environ={"GROQ_MODEL": "env"}).name == "env"
    assert SECRET not in repr(model()) and isinstance(model(), LanguageModel)


def test_json_request_shape_and_reasoning_kept_out_of_the_answer() -> None:
    m = model(ok('{"intent": "a"}'))
    assert m.generate_json("hello", system="Return JSON", temperature=0.0, max_output_tokens=300) == {"intent": "a"}
    call = m._session.calls[0]
    assert call["url"] == ENDPOINT and call["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert call["json"] == {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "hello"}],
        "temperature": 0.0, "max_completion_tokens": 300, "response_format": {"type": "json_object"}, "reasoning_effort": "low",
    }
    text = model(ok("plain"), model_name="qwen/qwen3.8-27b")
    assert text.generate_text("hi") == "plain"
    assert "response_format" not in text._session.calls[0]["json"] and "reasoning_effort" not in text._session.calls[0]["json"]


def test_output_problems_are_output_errors() -> None:
    with pytest.raises(ModelOutputError, match="cut off"):
        model(ok('{"a"', finish="length")).generate_json("p")
    with pytest.raises(ModelOutputError, match="no choices"):
        model(FakeResponse(200, {"choices": []})).generate_text("p")
    with pytest.raises(ModelOutputError):
        model(ok("not json at all")).generate_json("p")
    with pytest.raises(ModelOutputError):
        model().generate_text("  ")


def test_short_rate_limits_are_waited_out() -> None:
    sleeps: list[float] = []
    m = model(limited(7.5), limited(2), ok('{"a": 1}'), sleeps=sleeps)
    assert m.generate_json("p") == {"a": 1} and sleeps == [7.5, 2.0]


def test_daily_rate_limit_stops_immediately_without_leaking_the_key() -> None:
    sleeps: list[float] = []
    m = model(limited(3600), sleeps=sleeps)
    with pytest.raises(ModelRuntimeError, match="rate limit") as exc:
        m.generate_text("p")
    assert sleeps == [] and len(m._session.calls) == 1 and SECRET not in str(exc.value)
    with pytest.raises(ModelRuntimeError):
        model(limited(5), limited(5), max_rate_limit_waits=1).generate_text("p")


def test_transient_errors_retry_and_auth_errors_do_not() -> None:
    m = model(FakeResponse(503), requests.Timeout(), ok("fine"))
    assert m.generate_text("p") == "fine" and len(m._session.calls) == 3
    with pytest.raises(ModelRuntimeError, match="3 attempt"):
        model(FakeResponse(503), FakeResponse(503), FakeResponse(503)).generate_text("p")
    bad = model(FakeResponse(401, {"error": {"message": "Invalid API Key"}}))
    with pytest.raises(ModelConfigError):
        bad.generate_text("p")
    assert len(bad._session.calls) == 1
    with pytest.raises(ModelRuntimeError, match="GROQ_MODEL"):
        model(FakeResponse(404, {"error": {"message": "model not found"}})).generate_text("p")
