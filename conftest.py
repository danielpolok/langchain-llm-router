"""Repo-wide pytest hooks."""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest
from dotenv import load_dotenv

# Loaded once at collection time so `requires_env`-gated tests (provider keys, LangSmith) see
# what's in .env without the shell having sourced it. Never loaded by `src/` — dev-only, and
# never overrides a variable the environment already sets.
load_dotenv()

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def _ollama_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{_OLLAMA_BASE_URL}/api/version", timeout=1)
    except (urllib.error.URLError, OSError):
        return False
    return True


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip, rather than fail, tests whose provider credentials or servers are absent."""
    for marker in item.iter_markers(name="requires_env"):
        missing = [name for name in marker.args if not os.environ.get(name)]
        if missing:
            pytest.skip(f"environment variables not set: {', '.join(missing)}")

    if any(item.iter_markers(name="requires_ollama")) and not _ollama_reachable():
        pytest.skip(f"no Ollama server reachable at {_OLLAMA_BASE_URL}")
