"""Current-request extraction: R4 (route on the current request) and C7 (messages as LangChain
defines them).

The fixture transcripts are the ones REQ-R4-1 names — single turn, multi-turn, agent tool loop,
multimodal, system prompt only. The agent-loop fixtures are built so the end-to-end version of
REQ-R4-2 through `ChatRouter` is a small addition: swap the model in
`test_create_agent_calls_all_route_on_the_originating_request` for a router over
`ToolCallingFakeChatModel(script=agent_loop_replies())` with a strategy that records requests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    ContentBlock,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.content import create_image_block
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langchain_llm_router import RoutingRequest
from langchain_llm_router._extraction import build_request
from tests.fakes import FakeChatModel, ToolCallingFakeChatModel

ROUTES = ("small", "frontier")

SYSTEM_PROMPT = (
    "You are a meticulous research assistant. Think step by step, cite every source you use, "
    "and prefer primary sources. Answer in the user's language."
)
SYSTEM = SystemMessage(SYSTEM_PROMPT)

QUESTION = "What's the weather in Paris and in Berlin today?"

# A tool result long enough that routing on the last message, or on length, would pick the
# frontier route — the misroute R4 exists to prevent.
FORECAST = "Hourly forecast for {city}: " + "; ".join(
    f"{hour:02d}:00 14C, light wind from the south-west, 10% chance of rain" for hour in range(24)
)


def by_length(request: RoutingRequest) -> str:
    """A length heuristic of the kind the PRD's survey found misroutes agent loops."""
    return "frontier" if len(request.text) > 200 else "small"


def extract(
    messages: Sequence[BaseMessage],
    *,
    wants_full_context: bool = False,
    tools_bound: bool = False,
    config: RunnableConfig | None = None,
) -> RoutingRequest | None:
    return build_request(
        messages,
        routes=ROUTES,
        tools_bound=tools_bound,
        wants_full_context=wants_full_context,
        config=config if config is not None else RunnableConfig(),
    )


def extract_input(model_input: LanguageModelInput, **kwargs: Any) -> RoutingRequest | None:
    """Extract from any chat model input, through the path every chat model converts it on."""
    # private API: `_convert_input` is the conversion REQ-C7-2 is about — what every chat
    # model does to its input before `_generate` sees it. The router calls it the same way.
    messages = FakeChatModel()._convert_input(model_input).to_messages()
    return extract(messages, **kwargs)


def request_for(
    text: str,
    *,
    content_blocks: list[ContentBlock] | None = None,
    modalities: frozenset[str] = frozenset({"text"}),
    messages: list[BaseMessage] | None = None,
) -> RoutingRequest:
    """The request a text-only user message is expected to produce."""
    return RoutingRequest(
        text=text,
        content_blocks=(
            content_blocks if content_blocks is not None else [{"type": "text", "text": text}]
        ),
        modalities=modalities,
        routes=ROUTES,
        tools_bound=False,
        messages=messages,
    )


# The agent tool loop: three model calls, the transcript growing between them.


def weather_call(city: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[ToolCall(name="get_weather", args={"city": city}, id=call_id)],
    )


def agent_loop_replies() -> list[AIMessage]:
    """What the model answers on each of the loop's three calls — a fresh script per run."""
    return [
        weather_call("Paris", "call_paris"),
        weather_call("Berlin", "call_berlin"),
        AIMessage("Paris and Berlin are both mild today, around 14C with little rain."),
    ]


@tool
def get_weather(city: str) -> str:
    """Today's hourly forecast for a city."""
    return FORECAST.format(city=city)


FIRST_CALL: list[BaseMessage] = [SYSTEM, HumanMessage(QUESTION)]
SECOND_CALL: list[BaseMessage] = [
    *FIRST_CALL,
    weather_call("Paris", "call_paris"),
    ToolMessage(FORECAST.format(city="Paris"), tool_call_id="call_paris"),
]
THIRD_CALL: list[BaseMessage] = [
    *SECOND_CALL,
    weather_call("Berlin", "call_berlin"),
    ToolMessage(FORECAST.format(city="Berlin"), tool_call_id="call_berlin"),
]
TOOL_LOOP_CALLS = [FIRST_CALL, SECOND_CALL, THIRD_CALL]

# The fixture transcripts REQ-R4-1 names.

SINGLE_TURN: list[BaseMessage] = [SYSTEM, HumanMessage("Summarise the plot of Hamlet.")]
MULTI_TURN: list[BaseMessage] = [
    SYSTEM,
    HumanMessage("Summarise the plot of Hamlet."),
    AIMessage("Prince Hamlet avenges his father's murder by his uncle, at great cost."),
    HumanMessage("Now write a haiku about it."),
]
AGENT_TOOL_LOOP: list[BaseMessage] = [*THIRD_CALL, agent_loop_replies()[-1]]
IMAGE_URL = "https://example.com/cat.png"
MULTIMODAL: list[BaseMessage] = [
    SYSTEM,
    HumanMessage(
        content_blocks=[
            {"type": "text", "text": "What breed is this cat?"},
            {"type": "image", "url": IMAGE_URL},
        ]
    ),
]
SYSTEM_PROMPT_ONLY: list[BaseMessage] = [SYSTEM]


# REQ-R4-1 — which message is the current request.


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        pytest.param(SINGLE_TURN, request_for("Summarise the plot of Hamlet."), id="single-turn"),
        pytest.param(MULTI_TURN, request_for("Now write a haiku about it."), id="multi-turn"),
        pytest.param(AGENT_TOOL_LOOP, request_for(QUESTION), id="agent-tool-loop"),
        pytest.param(
            MULTIMODAL,
            request_for(
                "What breed is this cat?",
                content_blocks=[
                    {"type": "text", "text": "What breed is this cat?"},
                    {"type": "image", "url": IMAGE_URL},
                ],
                modalities=frozenset({"text", "image"}),
            ),
            id="multimodal",
        ),
        pytest.param(SYSTEM_PROMPT_ONLY, None, id="system-prompt-only"),
    ],
)
def test_fixture_transcripts_route_on_the_most_recent_human_message(
    transcript: list[BaseMessage], expected: RoutingRequest | None
) -> None:
    """REQ-R4-1: the current request is the most recent `HumanMessage` — earlier turns,
    trailing AI and tool messages and the system prompt take no part in it, and a system
    prompt alone is no request at all."""
    assert extract(transcript) == expected


def test_trailing_ai_and_tool_messages_are_skipped() -> None:
    """REQ-R4-1: a transcript ending in tool output still routes on the user's request, even
    though routing on that last message would pick a different route."""
    request = extract(THIRD_CALL)

    assert request == request_for(QUESTION)
    assert by_length(request) == "small"
    assert isinstance(THIRD_CALL[-1], ToolMessage)
    assert len(THIRD_CALL[-1].text) > 200  # what a last-message router would have seen


def test_a_system_prompt_never_defines_the_request_wherever_it_sits() -> None:
    """REQ-R4-1: system prompts are not the request — neither before it nor after it (some
    applications append a reminder as a trailing system message)."""
    reminder = SystemMessage("Reminder: answer in under 50 words, and in French.")

    assert extract([SYSTEM, HumanMessage("Hi there"), reminder]) == request_for("Hi there")


def test_conversation_length_never_defines_the_request() -> None:
    """REQ-R4-1: the same current request after no history and after forty turns of it is the
    same request — nothing about the history's length reaches the strategy."""
    history: list[BaseMessage] = []
    for turn in range(20):
        history += [HumanMessage(f"Question {turn}: " + "x" * 500), AIMessage("y" * 2000)]
    current = HumanMessage("And what about tomorrow?")

    short = extract([SYSTEM, current])
    long = extract([SYSTEM, *history, current])

    assert short == long == request_for("And what about tomorrow?")


def test_a_human_message_chunk_is_a_human_message() -> None:
    """REQ-R4-1: `HumanMessageChunk` subclasses `HumanMessage`, and counts as one."""
    transcript = [HumanMessage("old"), AIMessage("reply"), HumanMessageChunk(content="new")]

    assert extract(transcript) == request_for("new")


def test_a_chat_message_with_a_user_role_is_a_human_message() -> None:
    """REQ-R4-1: `ChatMessage(role="user" | "human")` is a user turn — the roles LangChain
    converts to a `HumanMessage` — while any other role is not."""
    transcript: list[BaseMessage] = [
        HumanMessage("old"),
        AIMessage("reply"),
        ChatMessage(role="user", content="new"),
        ChatMessage(role="assistant", content="a trailing assistant turn"),
    ]

    assert extract(transcript) == request_for("new")
    assert extract([HumanMessage("old"), ChatMessage(role="human", content="new")]) == (
        request_for("new")
    )
    assert extract([ChatMessage(role="critic", content="not a user")]) is None


@pytest.mark.parametrize("content", ["", []], ids=["empty-string", "empty-list"])
def test_an_empty_human_message_is_still_the_current_request(content: str | list[Any]) -> None:
    """REQ-R4-1: position alone decides. An empty current message is not skipped in favour of
    an earlier, stale request; it is a request with no text and no modalities."""
    transcript = [HumanMessage("an earlier request"), AIMessage("reply"), HumanMessage(content)]

    assert extract(transcript) == request_for("", content_blocks=[], modalities=frozenset())


# REQ-R4-2 — agent loops.


def test_every_call_in_a_three_iteration_tool_loop_routes_on_the_originating_request() -> None:
    """REQ-R4-2: the loop's three model calls — after zero, one and two tool results — all see
    the request that started the loop, so a strategy routes them identically."""
    requests = [extract(call) for call in TOOL_LOOP_CALLS]

    assert requests == [request_for(QUESTION)] * 3
    assert [by_length(request) for request in requests if request] == ["small"] * 3
    # What routing on the last message would have done instead: the tool output wins.
    assert [len(call[-1].text) > 200 for call in TOOL_LOOP_CALLS] == [False, True, True]


class _ModelInputs(BaseCallbackHandler):
    """Records the messages every chat model call receives."""

    def __init__(self) -> None:
        self.calls: list[list[BaseMessage]] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.calls.extend(messages)


def test_create_agent_calls_all_route_on_the_originating_request() -> None:
    """REQ-R4-2: the transcripts a real `create_agent` tool loop hands its model — system
    prompt, tool calls, tool results and all — each extract the originating request."""
    model = ToolCallingFakeChatModel(script=agent_loop_replies())
    agent = create_agent(model=model, tools=[get_weather], system_prompt=SYSTEM_PROMPT)
    inputs = _ModelInputs()

    agent.invoke(
        {"messages": [{"role": "user", "content": QUESTION}]},
        config={"callbacks": [inputs]},
    )

    assert [type(call[-1]) for call in inputs.calls] == [HumanMessage, ToolMessage, ToolMessage]
    assert [extract(call, tools_bound=True) for call in inputs.calls] == [
        RoutingRequest(
            text=QUESTION,
            content_blocks=[{"type": "text", "text": QUESTION}],
            modalities=frozenset({"text"}),
            routes=ROUTES,
            tools_bound=True,
        )
    ] * 3


# REQ-R4-4 — no user message.


@pytest.mark.parametrize(
    "transcript",
    [
        pytest.param(SYSTEM_PROMPT_ONLY, id="system-prompt-only"),
        pytest.param([], id="empty"),
        pytest.param([SYSTEM, AIMessage("Hello! How can I help?")], id="ai-only"),
        pytest.param(
            [
                weather_call("Paris", "call_paris"),
                ToolMessage(FORECAST.format(city="Paris"), tool_call_id="call_paris"),
            ],
            id="tool-loop-without-a-user",
        ),
    ],
)
def test_no_user_message_means_the_strategy_cannot_decide(
    transcript: list[BaseMessage],
) -> None:
    """REQ-R4-4: with no user message at all there is no request, which the router treats as
    "can't decide" and sends to the default route (the warning is the router's, T-110)."""
    assert extract(transcript) is None


# REQ-C7-1 — content read as LangChain defines it.

TEXT_BLOCK: dict[str, Any] = {"type": "text", "text": "Describe this image."}


@pytest.mark.parametrize(
    ("image_block", "expected_block"),
    [
        pytest.param(
            {"type": "image", "url": IMAGE_URL},
            {"type": "image", "url": IMAGE_URL},
            id="standard",
        ),
        pytest.param(
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            {"type": "image", "url": IMAGE_URL},
            id="openai-chat-completions",
        ),
        pytest.param(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
            },
            {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"},
            id="anthropic",
        ),
        pytest.param(
            {"type": "image", "source_type": "url", "url": IMAGE_URL},
            {"type": "image", "url": IMAGE_URL},
            id="langchain-v0",
        ),
    ],
)
def test_a_text_and_image_request_exposes_its_text_and_the_image(
    image_block: dict[str, Any], expected_block: dict[str, Any]
) -> None:
    """REQ-C7-1: a text+image request yields its text in `text` and `"image"` in
    `modalities`, whichever format — standard or provider-native — the image is in."""
    request = extract([SYSTEM, HumanMessage(content=[TEXT_BLOCK, image_block])])

    assert request == request_for(
        "Describe this image.",
        content_blocks=[{"type": "text", "text": "Describe this image."}, expected_block],  # type: ignore[list-item]
        modalities=frozenset({"text", "image"}),
    )


@pytest.mark.parametrize(
    ("block", "modality"),
    [
        pytest.param({"type": "image", "url": IMAGE_URL}, "image", id="image"),
        pytest.param(
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
            "audio",
            id="openai-audio",
        ),
        pytest.param({"type": "video", "url": "https://example.com/v.mp4"}, "video", id="video"),
        pytest.param(
            {
                "type": "file",
                "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,JVBE"},
            },
            "file",
            id="openai-file",
        ),
        pytest.param(
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBE"},
            },
            "file",
            id="anthropic-pdf",
        ),
        pytest.param(
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "x"},
            },
            "file",
            id="anthropic-text-document",
        ),
        pytest.param(
            {"type": "text-plain", "text": "notes", "mime_type": "text/plain"},
            "file",
            id="text-plain",
        ),
        pytest.param({"type": "input_text", "text": "?"}, "other", id="non-standard"),
        pytest.param({"type": "reasoning", "reasoning": "hmm"}, "other", id="not-request-content"),
    ],
)
def test_modalities_come_from_a_fixed_vocabulary(block: dict[str, Any], modality: str) -> None:
    """REQ-C7-1: every block maps to one of text, image, audio, video, file or other. A
    plain-text document is a file, and its text is not the request's text."""
    request = extract([HumanMessage(content=[{"type": "text", "text": "Look:"}, block])])

    assert request is not None
    assert request.text == "Look:"
    assert request.modalities == frozenset({"text", modality})


def test_a_request_with_no_text_has_no_text_modality() -> None:
    """REQ-C7-1: `"text"` is present exactly when there is text — an image alone, or beside an
    empty text block, is an image-only request."""
    request = extract(
        [HumanMessage(content=[{"type": "text", "text": ""}, {"type": "image", "url": IMAGE_URL}])]
    )

    assert request is not None
    assert request.text == ""
    assert request.modalities == frozenset({"image"})


def test_text_joins_every_text_block_with_newlines() -> None:
    """REQ-C7-1: text is joined across blocks — plain strings, standard text blocks and the
    untyped text blocks of Google GenAI content, which `BaseMessage.text` doesn't read — with a
    newline, so words in separate blocks don't fuse."""
    message = HumanMessage(
        content=[
            "Write code",
            {"type": "text", "text": "in Python"},
            {"type": "text", "text": ""},
            {"text": "that sorts a list."},
        ]
    )

    request = extract([message])

    assert request is not None
    assert request.text == "Write code\nin Python\nthat sorts a list."
    assert message.text == "Write codein Python"  # LangChain's own accessor, for contrast
    assert request.modalities == frozenset({"text"})


def test_string_content_text_is_the_messages_own_text() -> None:
    """REQ-C7-1: for string content, the common case, `text` is exactly `message.text`."""
    message = HumanMessage("  Keep the whitespace\nand the newline.  ")

    request = extract([message])

    assert request is not None
    assert request.text == message.text


def test_reading_a_provider_native_block_twice_gives_equal_requests() -> None:
    """REQ-C7-1, REQ-C7-2: LangChain mints a fresh `lc_` id each time it translates an OpenAI
    image block; extraction drops every id LangChain generated, so a message reads the same
    whether it arrives provider-native or already stored as content blocks."""
    minted = HumanMessage(content=[{"type": "image_url", "image_url": {"url": IMAGE_URL}}])
    stored = HumanMessage(content_blocks=[create_image_block(url=IMAGE_URL)])
    blocks = [{"type": "image", "url": IMAGE_URL}]

    assert minted.content_blocks != minted.content_blocks  # LangChain's own reads differ
    assert extract([minted]) == extract([minted])
    assert extract([stored]) == extract([minted])
    for request in (extract([minted]), extract([stored])):
        assert request is not None
        assert request.content_blocks == blocks


def test_an_id_the_provider_gave_a_block_is_kept() -> None:
    """REQ-C7-1: only LangChain's own `lc_` ids are dropped — a provider's id is content."""
    message = HumanMessage(content=[{"type": "image", "url": IMAGE_URL, "id": "provider-1"}])

    request = extract([message])

    assert request is not None
    assert [block.get("id") for block in request.content_blocks] == ["provider-1"]


def test_the_strategy_cannot_edit_the_callers_messages_through_the_request() -> None:
    """REQ-C7-1, R4: the request's blocks and transcript are copies — changing them leaves the
    caller's message and list as they were."""
    message = HumanMessage(content=[{"type": "text", "text": "original"}])
    transcript: list[BaseMessage] = [SYSTEM, message]

    request = extract(transcript, wants_full_context=True)
    assert request is not None
    assert request.messages is not None
    request.content_blocks[0]["text"] = "edited"  # type: ignore[typeddict-unknown-key]
    request.messages.clear()

    assert message.content == [{"type": "text", "text": "original"}]
    assert transcript == [SYSTEM, message]


def test_the_transcript_is_included_only_when_the_strategy_wants_it() -> None:
    """R4: `messages` is the whole transcript under `wants_full_context`, and `None` otherwise;
    `config` passes through untouched."""
    config = RunnableConfig(tags=["strategy-run"])

    wide = extract(MULTI_TURN, wants_full_context=True, config=config)
    narrow = extract(MULTI_TURN, config=config)

    assert wide is not None
    assert narrow is not None
    assert wide.messages == MULTI_TURN
    assert wide.messages is not MULTI_TURN
    assert narrow.messages is None
    assert wide.config is config
    assert narrow.config is config


# REQ-C7-2 — every input form a chat model accepts.

HAIKU_REQUEST = "Write a haiku about the sea."


@pytest.mark.parametrize("wants_full_context", [False, True], ids=["current", "full-context"])
def test_every_input_form_produces_the_same_request(wants_full_context: bool) -> None:
    """REQ-C7-2: a string, a list of dicts, `BaseMessage` objects and a `ChatPromptValue`,
    converted the way a chat model converts them, produce the same `RoutingRequest`."""
    forms: dict[str, LanguageModelInput] = {
        "string": HAIKU_REQUEST,
        "dicts": [{"role": "user", "content": HAIKU_REQUEST}],
        "messages": [HumanMessage(HAIKU_REQUEST)],
        "prompt-value": ChatPromptValue(messages=[HumanMessage(HAIKU_REQUEST)]),
    }

    requests = {
        name: extract_input(value, wants_full_context=wants_full_context)
        for name, value in forms.items()
    }

    expected = request_for(
        HAIKU_REQUEST,
        messages=[HumanMessage(HAIKU_REQUEST)] if wants_full_context else None,
    )
    assert requests == dict.fromkeys(forms, expected)


def test_every_input_form_of_a_multimodal_conversation_produces_the_same_request() -> None:
    """REQ-C7-2: the forms that can carry a system prompt, history and an OpenAI-format image —
    dicts, messages and a prompt template's `ChatPromptValue` — agree too."""
    content: list[str | dict[str, Any]] = [
        {"type": "text", "text": "What breed is this cat?"},
        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
    ]
    template = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "Hi"), ("ai", "Hello!"), ("placeholder", "{q}")]
    )
    forms: dict[str, LanguageModelInput] = {
        "dicts": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": content},
        ],
        "messages": [SYSTEM, HumanMessage("Hi"), AIMessage("Hello!"), HumanMessage(content)],
        "prompt-value": template.invoke({"q": [HumanMessage(content)]}),
    }

    requests = {name: extract_input(value) for name, value in forms.items()}

    assert requests == dict.fromkeys(
        forms,
        request_for(
            "What breed is this cat?",
            content_blocks=[
                {"type": "text", "text": "What breed is this cat?"},
                {"type": "image", "url": IMAGE_URL},
            ],
            modalities=frozenset({"text", "image"}),
        ),
    )
