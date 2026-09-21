"""Repo-wide pytest hooks."""

from __future__ import annotations

import functools
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


@functools.cache
def _langsmith_unusable() -> str | None:
    """Why LangSmith cannot be used from here, or `None` if it can.

    A key that is set proves nothing: it may be revoked or belong to another workspace, and
    LangSmith answers that with a 403. So the credentials are tried for real, with the cheapest
    authenticated read there is — one project listing, which is lazy until iterated — the way
    `_ollama_reachable` pings the server instead of trusting that one is configured. Asked once
    per session: every live test would otherwise repeat the request.
    """
    if not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")):
        return "no LangSmith API key is set (LANGSMITH_API_KEY)"
    try:
        from langsmith import Client

        list(Client(timeout_ms=(3_000, 10_000)).list_projects(limit=1))
    except Exception as error:  # any failure, not only auth: a test that can't reach it skips
        return f"LangSmith did not accept the credentials: {type(error).__name__}"
    return None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip, rather than fail, tests whose provider credentials or servers are absent."""
    for marker in item.iter_markers(name="requires_env"):
        missing = [name for name in marker.args if not os.environ.get(name)]
        if missing:
            pytest.skip(f"environment variables not set: {', '.join(missing)}")

    if any(item.iter_markers(name="requires_ollama")) and not _ollama_reachable():
        pytest.skip(f"no Ollama server reachable at {_OLLAMA_BASE_URL}")

    if any(item.iter_markers(name="requires_langsmith")) and (why := _langsmith_unusable()):
        pytest.skip(why)
