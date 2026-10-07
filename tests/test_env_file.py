"""The .env loader used by the agent scripts: real environment wins, missing file is fine, and .env is never tracked."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import _bootstrap  # noqa: E402


def test_loads_values_from_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HIVER_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("HIVER_TEST_KEY=from-file\n", encoding="utf-8")
    assert _bootstrap.load_env_file(env)
    assert os.environ["HIVER_TEST_KEY"] == "from-file"


def test_existing_environment_variables_take_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVER_TEST_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("HIVER_TEST_KEY=from-file\n", encoding="utf-8")
    _bootstrap.load_env_file(env)
    assert os.environ["HIVER_TEST_KEY"] == "from-shell"


def test_missing_file_is_not_an_error(tmp_path):
    assert _bootstrap.load_env_file(tmp_path / "absent.env") is False


@pytest.mark.parametrize("name, ignored", [(".env", True), (".env.local", True), (".env.example", False)])
def test_env_files_with_secrets_are_git_ignored(name, ignored):
    proc = subprocess.run(["git", "check-ignore", "-q", name], cwd=REPO, capture_output=True)
    if proc.returncode == 128:
        pytest.skip("not a git checkout")
    assert (proc.returncode == 0) is ignored
