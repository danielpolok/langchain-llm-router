"""Routes that can't take the conversation's images, audio, video or PDFs.

The router checks what every message a route will receive holds against the route's `profile`,
after the strategy has decided, and diverts a request the way it diverts one for tools. What is
here, in order: which content needs which profile key, and that only an explicit `False` rules a
route out; the diversion under every calling convention; where a diverted request goes; the
error when no route can take the content, alone and together with tools; and forced routes.

Neighbours, not repeated here: diversion for tools alone is `test_tools.py`'s, and how a forced
route falls back is `test_forced_routes.py`'s.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel, ModelProfile
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langchain_model_router import (
    ChatRouter,
    ContentSupportWarning,
    ForcedRouteError,
    ForcedRouteWarning,
    NoContentCapableRouteError,
    RoutingChoice,
    RoutingDecision,
    RoutingRequest,
    RoutingWarning,
    ToolSupportWarning,
    routing_decision,
)
from tests.conventions import ALL_CONVENTIONS, CONVENTIONS, AnyConvention, Convention, respond
from tests.fakes import FakeChatModel, ToolCallingFakeChatModel, call_log

IMAGE = {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"}
IMAGE_URL = {"type": "image", "url": "https://example.com/car.png"}
AUDIO = {"type": "audio", "base64": "UklGRg==", "mime_type": "audio/wav"}
VIDEO = {"type": "video", "base64": "AAAAIGZ0eXA=", "mime_type": "video/mp4"}
PDF = {"type": "file", "base64": "JVBERi0=", "mime_type": "application/pdf"}
UNTYPED_FILE = {"type": "file", "url": "https://example.com/report.pdf"}
CSV_FILE = {"type": "file", "base64": "YSxi", "mime_type": "text/csv"}


@tool
def get_weather(city: str) -> str:
    """Look up the weather in a city."""
    return f"sunny in {city}"


def always(route: str) -> Any:
    """A strategy that sends every request to `route`, whatever it holds."""

    def decide(request: RoutingRequest) -> RoutingChoice:
        return RoutingChoice(route=route, reason=f"always {route!r}")

    return decide


def route(name: str, *, tools: bool = False, **profile: Any) -> BaseChatModel:
    """A fake route, with `profile` as given — or no profile at all when none is."""
    model = ToolCallingFakeChatModel if tools else FakeChatModel
    return model(
        model_name=f"model-{name}",
        reply=f"{name} answer",
        profile=cast("ModelProfile", profile) if profile else None,
    )


def follow_up(block: dict[str, Any]) -> list[BaseMessage]:
    """A conversation whose first turn carries `block`, and whose follow-up is text alone."""
    return [
        HumanMessage(content=[{"type": "text", "text": "Look at this."}, block]),
        AIMessage(content="I see it."),
        HumanMessage(content="What colour is the car on the left?"),
    ]


async def answered(
    router: Any,
    convention: AnyConvention,
    messages: list[BaseMessage],
    config: RunnableConfig | None = None,
) -> tuple[AIMessage, list[warnings.WarningMessage]]:
    """One request, and every routing warning it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        answer = await respond(router, convention, messages, config)
    return answer, [w for w in caught if issubclass(w.category, RoutingWarning)]


def diverted(route: str, to: str, cause: str) -> RoutingDecision:
    """The record for a request `always(route)` sent to `route` and was diverted to `to`."""
    return RoutingDecision(
        route=to,
        reason=f"always {route!r}; {route!r} {cause}, so it was diverted to {to!r}",
        strategy="decide",
        diverted_from=route,
    )


# --- what the conversation needs, and what a route can take ---

KINDS = [
    pytest.param(IMAGE, "image_inputs", "images", id="image"),
    pytest.param(AUDIO, "audio_inputs", "audio", id="audio"),
    pytest.param(VIDEO, "video_inputs", "video", id="video"),
    pytest.param(PDF, "pdf_inputs", "PDFs", id="pdf"),
    pytest.param(UNTYPED_FILE, "pdf_inputs", "PDFs", id="untyped file"),
    pytest.param(IMAGE_URL, "image_url_inputs", "images given by URL", id="image by url"),
]


@pytest.mark.parametrize(("block", "key", "kind"), KINDS)
@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_content_in_an_earlier_turn_never_reaches_a_route_that_can_t_take_it(
    convention: AnyConvention, block: dict[str, Any], key: str, kind: str
) -> None:
    """The follow-up is text, but the route receives the whole conversation: a route whose
    profile says it can't take what an earlier turn holds is diverted from, never called."""
    routes = {"vision": route("vision"), "text": route("text", **{key: False})}
    router = ChatRouter(routes=routes, default_route="vision", strategy=always("text"))

    answer, caught = await answered(router, convention, follow_up(block))

    assert routing_decision(answer) == diverted("text", "vision", f"can't take {kind}")
    assert answer.content == "vision answer"
    assert call_log(routes["text"]) == []
    (warning,) = caught
    assert warning.category is ContentSupportWarning
    assert str(warning.message) == f"'text' can't take {kind}; diverted to 'vision'"


NOT_RULED_OUT = [
    pytest.param(IMAGE, {}, id="no profile"),
    pytest.param(IMAGE, {"tool_calling": False}, id="no key for the content"),
    pytest.param(IMAGE, {"image_inputs": True}, id="profile says yes"),
    pytest.param(IMAGE, {"image_url_inputs": False}, id="image not given by url"),
    pytest.param(IMAGE, {"image_tool_message": False}, id="image not in a tool result"),
    pytest.param(PDF, {"pdf_tool_message": False}, id="pdf not in a tool result"),
    pytest.param(CSV_FILE, {"pdf_inputs": False}, id="a file that is not a pdf"),
    pytest.param({"type": "text-plain", "text": "a,b"}, {"pdf_inputs": False}, id="plain text"),
]


@pytest.mark.parametrize(("block", "profile"), NOT_RULED_OUT)
async def test_a_route_is_ruled_out_only_by_an_explicit_no(
    block: dict[str, Any], profile: dict[str, Any]
) -> None:
    """A missing profile or key counts as capable, and a key about some other way of giving the
    content says nothing about this one: the chosen route answers, with no warning."""
    routes = {"vision": route("vision"), "text": route("text", **profile)}
    router = ChatRouter(routes=routes, default_route="vision", strategy=always("text"))

    answer, caught = await answered(router, "invoke", follow_up(block))

    assert routing_decision(answer) == RoutingDecision(
        route="text", reason="always 'text'", strategy="decide"
    )
    assert caught == []


@pytest.mark.parametrize(
    ("block", "key", "kind"),
    [
        pytest.param(IMAGE, "image_tool_message", "images in tool results", id="image"),
        pytest.param(PDF, "pdf_tool_message", "PDFs in tool results", id="pdf"),
    ],
)
async def test_content_in_a_tool_result_needs_the_tool_message_key(
    block: dict[str, Any], key: str, kind: str
) -> None:
    """A provider that takes images from the user may not take them back from a tool, so content
    in a `ToolMessage` is checked against its own key too."""
    routes = {"vision": route("vision"), "text": route("text", **{key: False})}
    router = ChatRouter(routes=routes, default_route="vision", strategy=always("text"))
    messages: list[BaseMessage] = [
        HumanMessage(content="Show me the chart."),
        AIMessage(content="", tool_calls=[{"name": "chart", "args": {}, "id": "call_1"}]),
        ToolMessage(content=[block], tool_call_id="call_1"),
    ]

    answer, _ = await answered(router, "invoke", messages)

    assert routing_decision(answer) == diverted("text", "vision", f"can't take {kind}")


async def test_every_kind_a_route_can_t_take_is_named() -> None:
    """The reason lists all of them, in a fixed order, whatever order the conversation had."""
    routes = {
        "vision": route("vision"),
        "text": route("text", video_inputs=False, image_inputs=False),
    }
    router = ChatRouter(routes=routes, default_route="vision", strategy=always("text"))
    messages: list[BaseMessage] = [
        HumanMessage(content=[VIDEO]),
        HumanMessage(content=[IMAGE, {"type": "text", "text": "Compare them."}]),
    ]

    answer, _ = await answered(router, "invoke", messages)

    assert routing_decision(answer) == diverted("text", "vision", "can't take images, video")


# --- where a diverted request goes ---


async def test_when_the_default_can_t_take_it_the_first_route_that_can_does() -> None:
    """The same order as for tools: the default route, else the first in declaration order."""
    routes = {
        "text": route("text", image_inputs=False),
        "small": route("small", image_inputs=False),
        "vision": route("vision"),
        "big": route("big"),
    }
    router = ChatRouter(routes=routes, default_route="small", strategy=always("text"))

    answer, _ = await answered(router, "invoke", follow_up(IMAGE))

    assert routing_decision(answer) == diverted("text", "vision", "can't take images")


async def test_a_fallback_onto_a_default_that_can_t_take_it_is_diverted_too() -> None:
    """With no strategy the default route answers, and it is checked like any other."""
    routes = {"text": route("text", image_inputs=False), "vision": route("vision")}
    router = ChatRouter(routes=routes, default_route="text")

    answer, _ = await answered(router, "invoke", follow_up(IMAGE))

    assert routing_decision(answer) == RoutingDecision(
        route="vision",
        reason="no strategy configured; 'text' can't take images, so it was diverted to 'vision'",
        diverted_from="text",
    )


async def test_a_diverted_route_must_also_be_able_to_use_the_bound_tools() -> None:
    """A route that can take the image but not the tools is passed over for one that can do
    both, and the warning is the tool one when the chosen route couldn't use the tools."""
    routes = {
        "text": route("text", tools=True, image_inputs=False),
        "vision": route("vision", tool_calling=False),
        "both": route("both", tools=True),
    }
    router = ChatRouter(routes=routes, default_route="vision", strategy=always("text"))
    with pytest.warns(ToolSupportWarning):
        bound = router.bind_tools([get_weather])

    answer, caught = await answered(bound, "invoke", follow_up(IMAGE))

    assert routing_decision(answer) == diverted("text", "both", "can't take images")
    assert [w.category for w in caught] == [ContentSupportWarning]

    vision = ChatRouter(routes=routes, default_route="vision", strategy=always("vision"))
    with pytest.warns(ToolSupportWarning):
        bound = vision.bind_tools([get_weather])
    answer, caught = await answered(bound, "invoke", follow_up(IMAGE))

    assert routing_decision(answer) == diverted("vision", "both", "can't use the bound tools")
    assert [w.category for w in caught] == [ToolSupportWarning]


# --- when no route can take it ---


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_when_no_route_can_take_it_the_request_fails_before_any_route_runs(
    convention: AnyConvention,
) -> None:
    """`NoContentCapableRouteError` names each route and why, and no route is called."""
    routes = {
        "text": route("text", image_inputs=False),
        "small": route("small", image_inputs=False),
    }
    router = ChatRouter(routes=routes, default_route="small", strategy=always("text"))

    with pytest.raises(NoContentCapableRouteError) as raised:
        await respond(router, convention, follow_up(IMAGE))

    assert str(raised.value) == (
        "no route can take this conversation's content: 'text' can't take images; 'small' "
        "can't take images. A route can take it unless its profile says it can't, so a wrong "
        "profile is corrected on the route itself, with profile="
    )
    assert [call_log(model) for model in routes.values()] == [[], []]


async def test_content_and_tools_no_one_route_can_serve_together_is_an_error() -> None:
    """One route takes the image, another the tools: neither serves the request."""
    routes = {
        "text": route("text", tools=True, image_inputs=False),
        "vision": route("vision", tool_calling=False),
    }
    router = ChatRouter(routes=routes, default_route="vision", strategy=always("text"))
    with pytest.warns(ToolSupportWarning):
        bound = router.bind_tools([get_weather])

    with pytest.raises(NoContentCapableRouteError) as raised:
        await respond(bound, "invoke", follow_up(IMAGE))

    assert "'text' can't take images; 'vision' can't use the bound tools." in str(raised.value)


# --- forced routes ---


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_forced_route_that_can_t_take_the_content_is_refused(
    convention: Convention,
) -> None:
    """A forced route is never silently swapped: by default it is an error."""
    routes = {"vision": route("vision"), "text": route("text", image_inputs=False)}
    router = ChatRouter(routes=routes, default_route="vision")

    with pytest.raises(ForcedRouteError) as raised:
        await respond(router, convention, follow_up(IMAGE), {"configurable": {"route": "text"}})

    assert str(raised.value) == (
        "forced route 'text' can't take images; set on_unavailable_forced_route='fallback' to "
        "fall back to the default route instead"
    )
    assert call_log(routes["text"]) == []


async def test_a_refused_forced_route_falls_back_when_told_to() -> None:
    routes = {"vision": route("vision"), "text": route("text", image_inputs=False)}
    router = ChatRouter(
        routes=routes, default_route="vision", on_unavailable_forced_route="fallback"
    )

    answer, caught = await answered(
        router, "invoke", follow_up(IMAGE), {"configurable": {"route": "text"}}
    )

    assert routing_decision(answer) == RoutingDecision(
        route="vision",
        reason="forced route 'text' can't take images; fell back to the default route",
        forced=True,
        fallback=True,
    )
    assert [w.category for w in caught] == [ForcedRouteWarning]
