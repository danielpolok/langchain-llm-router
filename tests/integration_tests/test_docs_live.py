"""Every documented example runs against the real models it names.

Each page runs top to bottom in one namespace, the way a reader would paste it. The outputs a page
shows are real but illustrative, since model answers vary, so they aren't compared. The warnings
are compared: a router warning a block raises must appear in the output the page shows for it,
and a warning the output shows must actually be raised. That catches a policy that quietly stopped
deciding, such as an embedding threshold that no longer fits the model.

Skips without `GEMINI_API_KEY` (`requires_env`, root `conftest.py`); Gemini reads that variable
when `GOOGLE_API_KEY`, the one the pages name, is unset.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import pytest

from langchain_llm_router import RoutingWarning
from tests.docs import PAGES, python_blocks, relative

ROUTER_WARNINGS = [cls.__name__ for cls in (RoutingWarning, *RoutingWarning.__subclasses__())]


@pytest.mark.requires_env("GEMINI_API_KEY")
@pytest.mark.parametrize("page", [page for page in PAGES if python_blocks(page)], ids=relative)
def test_every_example_on_the_page_runs(page: Path) -> None:
    module = types.ModuleType("__docs__")  # dataclasses look their module up in `sys.modules`
    sys.modules[module.__name__] = module
    namespace: dict[str, Any] = module.__dict__
    try:
        for block in python_blocks(page):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = eval(block.compiled(), namespace)
                if inspect.iscoroutine(result):
                    asyncio.run(result)
            raised = {w.category.__name__ for w in caught if issubclass(w.category, RoutingWarning)}
            shown = {name for name in ROUTER_WARNINGS if f"{name}:" in block.output}
            assert raised == shown, (
                f"{block.location} raised {sorted(raised)} but its output shows {sorted(shown)}"
            )
    finally:
        sys.modules.pop(module.__name__, None)
