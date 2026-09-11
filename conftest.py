"""Repo-wide pytest hooks."""

import os

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip, rather than fail, tests whose provider credentials are absent."""
    for marker in item.iter_markers(name="requires_env"):
        missing = [name for name in marker.args if not os.environ.get(name)]
        if missing:
            pytest.skip(f"environment variables not set: {', '.join(missing)}")
