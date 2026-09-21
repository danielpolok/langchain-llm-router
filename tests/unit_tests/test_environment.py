"""The test suite's own environment: nothing here may upload to a developer's LangSmith."""

import os

from langsmith.utils import tracing_is_enabled


def test_tracing_is_off_for_the_suite() -> None:
    """`.env` turns tracing on for interactive use; the root `conftest.py` turns it off, so a
    valid key never starts uploading a trace of every fake-model call. Only
    `LLM_ROUTER_TRACE_TESTS=1` opts the whole suite back in."""
    if os.environ.get("LLM_ROUTER_TRACE_TESTS"):
        return
    assert not tracing_is_enabled()
