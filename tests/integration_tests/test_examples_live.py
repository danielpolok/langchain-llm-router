"""Every example runs against the real models it names, the way its docstring says to run it.

Each script runs as `python examples/<name>.py` from the repository root and must exit zero. The
answers an example shows are real but illustrative, since model answers vary, so they aren't
compared. The warnings are, as on the documentation pages (`test_docs_live.py`): a router warning
the script raises must appear in the output it shows, and a warning the output shows must
actually be raised.

The examples use Gemini models. Skips without `GEMINI_API_KEY` (`requires_env`, root
`conftest.py`), which Gemini reads when `GOOGLE_API_KEY` is unset.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from langchain_llm_router import RoutingWarning
from tests.docs import EXAMPLES, ROOT, example_output, relative

ROUTER_WARNINGS = [cls.__name__ for cls in (RoutingWarning, *RoutingWarning.__subclasses__())]


@pytest.mark.requires_env("GEMINI_API_KEY")
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_example_runs(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    raised = {name for name in ROUTER_WARNINGS if f"{name}:" in result.stderr}
    shown = {name for name in ROUTER_WARNINGS if f"{name}:" in example_output(script)}
    assert raised == shown, f"{relative(script)} raised {sorted(raised)} but shows {sorted(shown)}"
