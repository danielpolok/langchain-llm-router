"""Which routes can take the content in a conversation: images, audio, video and PDFs.

A route receives the whole conversation, not only the message a strategy decided on. A
follow-up with no image of its own still carries the image from an earlier turn, so a strategy
that routes the follow-up as text would hand that image to a text-only route. The check is the
router's, after the decision, whatever the strategy chose.

**What the conversation needs.** Every message is read — user, AI, tool and system messages —
through `BaseMessage.content_blocks`, the same standard view current-request extraction uses, so
provider-native blocks LangChain translates count the same as standard ones. Each block names the
`ModelProfile` keys a route needs for it:

- an `image` block needs `image_inputs`; given by URL, `image_url_inputs` too;
- an `audio` block needs `audio_inputs`, and a `video` block `video_inputs`;
- a `file` block needs `pdf_inputs`, unless its `mime_type` names something other than a PDF.
  PDFs are the only files a profile describes, and a PDF often arrives without a type (an
  Anthropic `document` given by URL does), so an untyped file is read as one;
- an image or PDF inside a `ToolMessage` needs `image_tool_message` / `pdf_tool_message` as well:
  a provider that takes images from the user may still not take them in a tool result.

A plain-text document (`text-plain`) and any other file type need nothing a profile describes.

**What a route can take.** Only an explicit `False` in the route's `profile` rules it out. A
missing profile or a missing key says nothing, so it counts as capable: Ollama models report no
profile at all, and diverting every request with an image away from them would be wrong far more
often than right. A wrong profile is corrected on the route itself, with LangChain's own
`profile=`.

Nothing here warns or raises: what the router does with a route that can't take the content is
`router.py`'s, next to the tool check it is combined with.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage

__all__ = ["content_needs", "describe", "missing_content"]

_KINDS = {
    "image_inputs": "images",
    "image_url_inputs": "images given by URL",
    "image_tool_message": "images in tool results",
    "audio_inputs": "audio",
    "video_inputs": "video",
    "pdf_inputs": "PDFs",
    "pdf_tool_message": "PDFs in tool results",
}
"""Each profile key the check reads, and the content it is about, as warnings name it.

In this order wherever the keys are listed, so a message reads the same however the
conversation's content was ordered."""


def content_needs(messages: Sequence[BaseMessage]) -> frozenset[str]:
    """The profile keys a route needs to take every message in `messages`."""
    needs: set[str] = set()
    for message in messages:
        in_tool_result = isinstance(message, ToolMessage)
        for block in message.content_blocks:
            needs.update(_block_needs(dict(block), in_tool_result=in_tool_result))
    return frozenset(needs)


def _block_needs(block: dict[str, object], *, in_tool_result: bool) -> set[str]:
    kind = block.get("type")
    if kind == "image":
        needs = {"image_inputs"}
        if block.get("url"):
            needs.add("image_url_inputs")
        if in_tool_result:
            needs.add("image_tool_message")
        return needs
    if kind == "audio":
        return {"audio_inputs"}
    if kind == "video":
        return {"video_inputs"}
    if kind == "file" and block.get("mime_type") in (None, "application/pdf"):
        return {"pdf_inputs", "pdf_tool_message"} if in_tool_result else {"pdf_inputs"}
    return set()


def missing_content(route: BaseChatModel, needs: Iterable[str]) -> list[str]:
    """The keys in `needs` that `route`'s profile says it can't take, in `_KINDS` order."""
    profile = cast("dict[str, Any]", route.profile or {})
    return [key for key in _KINDS if key in needs and profile.get(key) is False]


def describe(keys: Iterable[str]) -> str:
    """The content `keys` are about, in words: `"images, audio"`."""
    return ", ".join(_KINDS[key] for key in keys)
