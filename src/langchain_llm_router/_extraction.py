"""Current-request extraction: the `RoutingRequest` a strategy sees.

The router calls `build_request` and nothing else from here, after turning whatever it was
invoked with into messages the way every chat model does (`_convert_input(...).to_messages()`),
so a string, a list of dicts, `BaseMessage`s and a `ChatPromptValue` all arrive the same way
.

**Which message is the current request.** The most recent user message, by
position. Trailing AI and tool messages are skipped, so every call in an agent's tool loop
routes on the request that started it. System prompts never count, wherever they
sit, and conversation length plays no part: what comes before the current request reaches a
strategy only if it opts in with `wants_full_context`.

- A user message is a `HumanMessage` — so a `HumanMessageChunk` too, its subclass — or a
  `ChatMessage` whose role is `"user"` or `"human"`: the roles LangChain itself turns into a
  `HumanMessage` (`_convert_to_message`), and that providers send as a user turn.
- Position alone decides. An empty user message is still the current request, with `text`
  `""` and no modalities; reaching back past it would route on a turn the user has moved on
  from.
- No user message at all (an empty transcript, a system prompt alone) is `None`: the strategy
  can't decide, and the router uses the default route.

**How it is read.** Through `BaseMessage.content_blocks`, LangChain's standard
view of content: string content, standard blocks, and the provider-native blocks LangChain
translates (OpenAI Chat Completions `image_url` / `input_audio` / `file`, Anthropic `image` /
`document` with a `source`, Google GenAI, Bedrock Converse, v0 `source_type` blocks). A block
LangChain can't translate stays `non_standard`; extraction doesn't guess at it.

- `text` is the text blocks' text, joined with newlines. `BaseMessage.text` concatenates with no
  separator — `"Write code"` and `"in Python"` become `"Write codein Python"` — and misses the
  untyped text blocks of Google GenAI and Bedrock Converse content that `content_blocks` reads.
  A newline is what LangChain uses when it collapses text blocks into one string for a model
  (`convert_to_openai_messages(text_format="string")`). For string content, the common case,
  the two agree.
- `content_blocks` is a deep copy, so a strategy can't edit the caller's message through it.
  Ids LangChain generated itself (the reserved `"lc_"` prefix, minted by `ensure_id` — a fresh
  uuid4 each time a provider-native block is translated) are dropped: they say nothing about the
  request, and would make the same message read twice, or reached through two input forms, two
  unequal requests. Ids the provider gave the content are kept.
- A `HumanMessage` carrying provider-native tool results — Anthropic sends tool output in a
  user turn — is the current request like any other user message, though its `tool_result`
  blocks stay `non_standard`, so no tool output reaches `text`. LangChain's own canonical form
  is a `ToolMessage`, which extraction skips.
- `modalities` names the kinds of content present, from a fixed vocabulary: `"text"` when `text`
  is not empty; `"image"`, `"audio"`, `"video"` and `"file"` for blocks of those types; `"file"`
  for a `text-plain` block too — a plain-text *document*, attached rather than typed, so its text
  is not part of `text`; and `"other"` for anything else: `non_standard` blocks LangChain
  couldn't translate, and block types that aren't request content (`reasoning`, tool calls).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.messages import (
    LC_AUTO_PREFIX,
    BaseMessage,
    ChatMessage,
    ContentBlock,
    HumanMessage,
)
from langchain_core.runnables import RunnableConfig

from langchain_llm_router.strategy import RoutingRequest

_USER_ROLES = frozenset({"human", "user"})
"""`ChatMessage` roles that make it a user message, as LangChain's own role mapping has them."""

_MODALITIES = {
    "image": "image",
    "audio": "audio",
    "video": "video",
    "file": "file",
    "text-plain": "file",
}
"""Non-text content block type → modality; any type not listed is `"other"`."""


def build_request(
    messages: Sequence[BaseMessage],
    *,
    routes: tuple[str, ...],
    tools_bound: bool,
    wants_full_context: bool,
    config: RunnableConfig,
) -> RoutingRequest | None:
    """The request a strategy decides on, or `None` when there is no user message.

    "Current request" is the most recent user message by position, ignoring trailing AI and
    tool messages; the module docstring has the exact rules. `None` means the
    strategy can't decide, and the router falls back to the default route.
    """
    current = next((m for m in reversed(messages) if _is_user_message(m)), None)
    if current is None:
        return None
    blocks = _read_blocks(current)
    text = "\n".join(
        block["text"]
        for block in blocks
        if block["type"] == "text" and isinstance(block.get("text"), str) and block["text"]
    )
    modalities = {
        _MODALITIES.get(block["type"], "other") for block in blocks if block["type"] != "text"
    }
    if text:
        modalities.add("text")
    return RoutingRequest(
        text=text,
        content_blocks=cast("list[ContentBlock]", blocks),
        modalities=frozenset(modalities),
        routes=routes,
        tools_bound=tools_bound,
        # A new list: the strategy may reorder or trim it without touching the caller's. The
        # messages in it are the caller's own — a transcript is not worth deep-copying.
        messages=list(messages) if wants_full_context else None,
        config=config,
    )


def _is_user_message(message: BaseMessage) -> bool:
    if isinstance(message, HumanMessage):
        return True
    return isinstance(message, ChatMessage) and message.role in _USER_ROLES


def _read_blocks(message: BaseMessage) -> list[dict[str, Any]]:
    """The message's content blocks, copied, without LangChain's own generated ids."""
    blocks = cast("list[dict[str, Any]]", copy.deepcopy(message.content_blocks))
    for block in blocks:
        block_id = block.get("id")
        if isinstance(block_id, str) and block_id.startswith(LC_AUTO_PREFIX):
            del block["id"]
    return blocks
