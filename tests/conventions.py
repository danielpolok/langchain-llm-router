"""Call a chat model through each entry point the router overrides (C2, D9).

Offline tests parametrise over `CONVENTIONS` to check that every entry point runs the same
pipeline — one decision step, one route call, one record.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

Convention = Literal["invoke", "ainvoke", "stream", "astream"]

CONVENTIONS: tuple[Convention, ...] = ("invoke", "ainvoke", "stream", "astream")


async def respond(
    model: BaseChatModel,
    convention: Convention,
    input_: LanguageModelInput,
    config: RunnableConfig | None = None,
) -> AIMessage:
    """The model's answer through `convention`, with streamed chunks merged into one message."""
    if convention == "invoke":
        return model.invoke(input_, config)
    if convention == "ainvoke":
        return await model.ainvoke(input_, config)
    if convention == "stream":
        chunks = list(model.stream(input_, config))
    else:
        chunks = [chunk async for chunk in model.astream(input_, config)]
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    return merged
