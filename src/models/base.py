"""The small, swappable interface the agent depends on, plus its errors and a tolerant JSON parser."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable


class ModelError(Exception):
    """Base class for every model-layer failure."""


class ModelConfigError(ModelError):
    """The model cannot be used as configured (e.g. no API key). A setup problem: do not hide it."""


class ModelRuntimeError(ModelError):
    """The call itself failed (timeout, HTTP error, blocked response). Possibly transient."""


class ModelOutputError(ModelError):
    """The model answered, but not with usable output (e.g. invalid JSON)."""


@runtime_checkable
class LanguageModel(Protocol):
    """What the classifier, generator and verifier need. A test double only has to implement these two methods."""

    name: str

    def generate_text(self, prompt: str, *, system: str | None = None, temperature: float = 0.0, max_output_tokens: int = 1024) -> str: ...

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model reply that should be one JSON object. Tolerates ```json fences and prose around the object."""
    if not isinstance(text, str) or not text.strip():
        raise ModelOutputError("the model returned an empty response")
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise ModelOutputError(f"the model response is not valid JSON: {body[:120]!r}") from None
        try:
            value = json.loads(body[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelOutputError(f"the model response is not valid JSON ({exc.msg}): {body[:120]!r}") from None
    if not isinstance(value, dict):
        raise ModelOutputError(f"expected a JSON object, got {type(value).__name__}")
    return value
