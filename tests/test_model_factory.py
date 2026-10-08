from types import SimpleNamespace

import pytest

from models.factory import KEY_ENV, PROVIDERS, build_model


def test_every_provider_names_its_key_variable():
    assert set(KEY_ENV) == set(PROVIDERS)


def test_unknown_provider_is_rejected():
    settings = SimpleNamespace(name="x", timeout_seconds=1, max_retries=0, response_schema=False)
    with pytest.raises(ValueError, match="unknown provider"):
        build_model("openai", settings)
