"""Tool binding, structured output and diversion, through the router, against a real
local Ollama model — adapted from the review's probe (`t115r_q7_ollama.py`), which first
measured this against `qwen3:8b`.

A small local model is flaky in what it *says*, so these assert on structure — that a tool call
happened, that the record is where it belongs — never on wording, and the prompts stay trivial.
Skips unless a local Ollama server answers (`requires_ollama`, root `conftest.py`); override the
model with `LLM_ROUTER_OLLAMA_MODEL`.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, cast

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel

from langchain_llm_router import (
    ChatRouter,
    RoutingChoice,
    RoutingRequest,
    RoutingStrategy,
    ToolSupportWarning,
    routing_decision,
)
from tests.fakes import FakeChatModel

OLLAMA_MODEL = os.environ.get("LLM_ROUTER_OLLAMA_MODEL", "ollama:qwen3:8b")

pytestmark = pytest.mark.requires_ollama


@tool
def get_weather(city: str) -> str:
    """Look up the current weather in a city."""
    return f"sunny in {city}"


class Person(BaseModel):
    """A person mentioned in the text."""

    name: str
    age: int


class ByText(RoutingStrategy):
    """The tool-incapable `cheap` route when the request says so (the diversion target is
    `local` either way), `local` otherwise — a policy in one line, real enough for this."""

    def decide(self, request: RoutingRequest) -> RoutingChoice:
        return RoutingChoice(route="cheap" if "cheap" in request.text else "local", reason="text")


def router() -> ChatRouter:
    """`local` is the only tool-capable route, and the default: a fresh one per test, so no
    state (bound routes, decision context) survives between them."""
    return ChatRouter(
        routes={
            "cheap": FakeChatModel(model_name="cheap", reply="a fake reply"),  # cannot use tools
            "local": init_chat_model(OLLAMA_MODEL, temperature=0, num_predict=300, reasoning=False),
        },
        default_route="local",
        strategy=ByText(),
    )


def test_bind_tools_is_replayed_on_ollama_and_the_record_rides_the_response() -> None:
    """The router's binding, replayed on `ChatOllama`'s own `bind_tools`,
    gets a real tool call back — and the response carries the routing record
    exactly as it does off a fake route."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ToolSupportWarning)
        bound = router().bind_tools([get_weather])

    message = bound.invoke("What is the weather in Paris? Use the get_weather tool.")

    assert message.tool_calls
    assert message.tool_calls[0]["name"] == "get_weather"
    record = routing_decision(message)
    assert record is not None
    assert record.route == "local"


def test_structured_output_json_schema_include_raw_carries_the_record_on_raw() -> None:
    """`method="json_schema"` forwards to `ChatOllama`'s own structured
    output (the base default would drop it, `chat_models.py:2530`), and the triple's `raw`
    message carries the record where the record belongs under `include_raw=True`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ToolSupportWarning)
        structured = router().with_structured_output(Person, method="json_schema", include_raw=True)

    result = cast("dict[str, Any]", structured.invoke("Ada Lovelace is 36 years old."))

    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert result["parsing_error"] is None
    assert isinstance(result["parsed"], Person)
    record = routing_decision(result["raw"])
    assert record is not None
    assert record.route == "local"


def test_a_diversion_from_the_incapable_route_emits_one_warning_and_records_it() -> None:
    """The strategy picks `cheap`, which can't use tools, and the request is
    diverted to `local` — the only tool-capable route — with exactly one `ToolSupportWarning`
    and `diverted_from` recorded, against a real model as against the fakes in `test_tools.py`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ToolSupportWarning)
        bound = router().bind_tools([get_weather])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = bound.invoke("cheap: what's the weather in Rome? Use the get_weather tool.")

    tool_support_warnings = [w for w in caught if issubclass(w.category, ToolSupportWarning)]
    assert len(tool_support_warnings) == 1
    record = routing_decision(message)
    assert record is not None
    assert record.route == "local"
    assert record.diverted_from == "cheap"
