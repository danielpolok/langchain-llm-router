"""Call a chat model through each entry point the router overrides (C2, D9).

Offline tests parametrise over `CONVENTIONS` to check that every entry point runs the same
pipeline — one decision step, one route call, one record. `ALL_CONVENTIONS` adds the ones that
do not take a `RunnableConfig` whole, or that a chat model gets from somewhere else:

- `batch` and `abatch` call `invoke` / `ainvoke`, and `astream_events` is built on `astream` —
  all three from `Runnable` (REQ-C2-1);
- `generate` and `agenerate` are `BaseChatModel`'s own, and take a config's `callbacks`, `tags`,
  `metadata`, `run_name` and `run_id` as separate arguments (REQ-C2-2). So they cannot carry
  `configurable`, and `respond` refuses a config that has one rather than drop it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, TypeAlias, cast

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    convert_to_messages,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableConfig

Convention: TypeAlias = Literal["invoke", "ainvoke", "stream", "astream"]
"""An entry point the router overrides itself (D9)."""

AnyConvention: TypeAlias = Literal[
    "invoke", "ainvoke", "stream", "astream", "batch", "abatch", "events", "generate", "agenerate"
]
"""Any calling convention REQ-C2-1 and REQ-C2-2 name, including the ones built on the above."""

CONVENTIONS: tuple[Convention, ...] = ("invoke", "ainvoke", "stream", "astream")

ALL_CONVENTIONS: tuple[AnyConvention, ...] = (
    *CONVENTIONS,
    "batch",
    "abatch",
    "events",
    "generate",
    "agenerate",
)
"""`"events"` is `astream_events(version="v2")`."""

_GENERATE_PARTS = frozenset({"callbacks", "tags", "metadata", "run_name", "run_id"})
"""The keys of a `RunnableConfig` that `generate` takes as arguments."""


async def respond(
    model: BaseChatModel,
    convention: AnyConvention,
    input_: LanguageModelInput,
    config: RunnableConfig | None = None,
) -> AIMessage:
    """The model's answer through `convention`, with streamed chunks merged into one message.

    `batch` and `abatch` are given the single input, so that every convention answers one
    request and assertions about a call's runs, warnings and record read the same for all.
    """
    if convention in ("generate", "agenerate"):
        result = await generated(model, convention, [messages_of(input_)], config)
        generation = result.generations[0][0]
        assert isinstance(generation, ChatGeneration)
        return cast("AIMessage", generation.message)
    if convention == "invoke":
        return model.invoke(input_, config)
    if convention == "ainvoke":
        return await model.ainvoke(input_, config)
    if convention == "batch":
        return model.batch([input_], config)[0]
    if convention == "abatch":
        return (await model.abatch([input_], config))[0]
    if convention == "stream":
        chunks = list(model.stream(input_, config))
    elif convention == "astream":
        chunks = [chunk async for chunk in model.astream(input_, config)]
    else:
        chunks = [chunk async for chunk in streamed_chunks(model, input_, config)]
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    return merged


def messages_of(input_: LanguageModelInput) -> list[BaseMessage]:
    """The transcript an input stands for, as a chat model reads it (`_convert_input`)."""
    if isinstance(input_, PromptValue):
        return input_.to_messages()
    if isinstance(input_, str):
        return [HumanMessage(content=input_)]
    return convert_to_messages(input_)


def prompts(*texts: str) -> list[list[BaseMessage]]:
    """One single-message prompt per text: what `generate` takes."""
    return [[HumanMessage(content=text)] for text in texts]


async def generated(
    model: BaseChatModel,
    convention: Literal["generate", "agenerate"],
    prompts: list[list[BaseMessage]],
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> LLMResult:
    """`generate` or `agenerate` over `prompts`, with the parts of `config` they take.

    `BaseChatModel.invoke` takes a config apart into these arguments (`chat_models.py:488`);
    this does the same, so a test can hand every convention the same config.
    """
    config = config or {}
    homeless = set(config) - _GENERATE_PARTS
    if homeless:
        msg = f"generate() has no argument for {sorted(homeless)}: it takes no full config"
        raise ValueError(msg)
    parts: dict[str, Any] = dict(config)
    if convention == "generate":
        return model.generate(prompts, **parts, **kwargs)
    return await model.agenerate(prompts, **parts, **kwargs)


async def streamed_chunks(
    model: BaseChatModel,
    input_: LanguageModelInput,
    config: RunnableConfig | None = None,
) -> AsyncIterator[AIMessageChunk]:
    """The chunks a v2 event stream reports for the call's own top-level run.

    Whatever a model is made of, `astream_events` reports its outermost run as the one with no
    parent — a chat model run for an ordinary model, the router's chain run for a router — and
    that run's chunks are what the other conventions return.
    """
    async for event in model.astream_events(input_, config, version="v2"):
        if event["event"].endswith("_stream") and not event["parent_ids"]:
            yield cast("AIMessageChunk", event["data"]["chunk"])
