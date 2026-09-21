"""Call a chat model through each entry point the router overrides (C2, D9).

Offline tests parametrise over `CONVENTIONS` to check that every entry point runs the same
pipeline — one decision step, one route call, one record. `ALL_CONVENTIONS` adds the three
REQ-C2-1 also names, which a chat model gets for free from `Runnable`: `batch` and `abatch`
call `invoke` / `ainvoke`, and `astream_events` is built on `astream`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, TypeAlias, cast

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig

Convention: TypeAlias = Literal["invoke", "ainvoke", "stream", "astream"]
"""An entry point the router overrides itself (D9)."""

AnyConvention: TypeAlias = Literal[
    "invoke", "ainvoke", "stream", "astream", "batch", "abatch", "events"
]
"""Any calling convention REQ-C2-1 names, including the ones `Runnable` builds on the above."""

CONVENTIONS: tuple[Convention, ...] = ("invoke", "ainvoke", "stream", "astream")

ALL_CONVENTIONS: tuple[AnyConvention, ...] = (*CONVENTIONS, "batch", "abatch", "events")
"""`"events"` is `astream_events(version="v2")`."""


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
