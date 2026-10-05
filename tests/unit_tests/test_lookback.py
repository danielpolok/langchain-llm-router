"""Lookback, end to end: a strategy that reads the user's previous messages as well.

A short follow-up has no signal of its own — "Is it enforceable in Germany?", "Add unit tests for
it" — so on its own it falls back to the default route or drops to the small model. With
`lookback=N` a strategy reads the user's previous N messages too.

The two conversations below are the specification: each turn's route and reason, today and with
`lookback=2`, through `ChatRouter` under every calling convention, for `KeywordStrategy`, the
equivalent `ConfigurableStrategy`, `HeuristicStrategy`, and the two strategies that call a model,
`EmbeddingStrategy` and `ClassifierStrategy`, on fakes. Then the interface a custom strategy
gets (`lookback`, `previous_requests`), an agent's tool loop, the router's own checks, and the
trace. Each strategy's own rules for reading earlier messages are tested with the strategy.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, NamedTuple, cast

import pytest
from langchain.agents import create_agent
from langchain_core.caches import InMemoryCache
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from langchain_model_router import (
    ChatRouter,
    ClassifierStrategy,
    ConfigurableStrategy,
    EmbeddingStrategy,
    FallbackWarning,
    HeuristicStrategy,
    KeywordStrategy,
    RoutingChoice,
    RoutingDecision,
    RoutingError,
    RoutingRequest,
    RoutingStrategy,
    RoutingWarning,
    routing_decision,
)
from langchain_model_router.strategies.configurable import Rule, keywords
from tests.conventions import ALL_CONVENTIONS, CONVENTIONS, AnyConvention, Convention, respond
from tests.fakes import (
    CountingEmbeddings,
    FakeChatModel,
    PromptReadingFakeChatModel,
    ToolCallingFakeChatModel,
    call_log,
)
from tests.tracing import model_runs, walk

# --- The two conversations, and what each turn does ---

KEYWORD_RULES = {
    "coder": ["python", "sql", "code"],
    "legal": ["contract", "gdpr", "clause", "liability"],
}

LEGAL_THEN_CODE = [
    "Review this contract clause on limitation of liability",
    "Is it enforceable in Germany?",
    "Which article of the civil code covers this?",
    "Now write a python script that flags such clauses",
    "Add unit tests for it",
]
"""A legal question, a follow-up, a false keyword match, a coding request, and its follow-up."""

HARD_THEN_EASY = [
    "Hi, can you help me with something?",
    "Compare the trade-offs of B-trees and LSM trees for databases.",
    "Thanks! Which one does SQLite use?",
    "ok",
    "Suggest a name for my cat",
]
"""Small talk, a hard question, two follow-ups that look easy, and a new, easy topic."""

DOMAIN_ROUTES = ("general", "coder", "legal")
TIER_ROUTES = ("small", "frontier")


def keyword_strategy(lookback: int) -> KeywordStrategy:
    return KeywordStrategy(KEYWORD_RULES, lookback=lookback)


def configurable_strategy(lookback: int) -> ConfigurableStrategy:
    """The same policy as `keyword_strategy`, as rules."""
    return ConfigurableStrategy(
        [
            Rule("coder", keywords(*KEYWORD_RULES["coder"]), name="coder"),
            Rule("legal", keywords(*KEYWORD_RULES["legal"]), name="legal"),
        ],
        lookback=lookback,
    )


def heuristic_strategy(lookback: int) -> HeuristicStrategy:
    return HeuristicStrategy(*TIER_ROUTES, lookback=lookback)


EMBEDDING_EXAMPLES = {"legal": [LEGAL_THEN_CODE[0]], "coder": [LEGAL_THEN_CODE[3]]}
"""One example per route, each a message of the conversation word for word: fake embeddings
carry no meaning, so a message matches an example only when it is that example."""


def embedding_strategy(lookback: int) -> EmbeddingStrategy:
    return EmbeddingStrategy(
        DeterministicFakeEmbedding(size=32), EMBEDDING_EXAMPLES, threshold=0.99, lookback=lookback
    )


def similar_to(route: str) -> str:
    """The reason `embedding_strategy` gives when a message is `route`'s example."""
    (example,) = EMBEDDING_EXAMPLES[route]
    return f"embedding similarity 1.00 >= 0.99 to {example!r} (route {route!r})"


ROUTE_DESCRIPTIONS = {
    "general": "anything else",
    "coder": "programming",
    "legal": "law",
}


def reads_keywords(prompt: str) -> str | None:
    """How the fake classifier chooses: the route of the last keyword in the messages it is
    shown, the newest message's own if it has one. Like a model, it reads every message the
    prompt shows; unlike one, it can be fooled by a keyword out of place."""
    _menu, shown = prompt.split("\n\n", 1)
    found = [
        (match.start(), route)
        for route, words in KEYWORD_RULES.items()
        for word in words
        for match in re.finditer(rf"\b{word}\b", shown, re.IGNORECASE)
    ]
    return max(found)[1] if found else None


def classifier_strategy(lookback: int) -> ClassifierStrategy:
    classifier = PromptReadingFakeChatModel(model_name="classifier", answer=reads_keywords)
    return ClassifierStrategy(classifier, ROUTE_DESCRIPTIONS, lookback=lookback)


def classified(route: str) -> str:
    return f"the classifier chose {route!r}: {ROUTE_DESCRIPTIONS[route]}"


class Turn(NamedTuple):
    """What one turn of a conversation should route to, and why."""

    route: str
    reason: str
    fallback: bool = False
    messages_back: int | None = 0
    """How far back the message that decided was: what the reason says in words."""


def fell_back(strategy: str) -> Turn:
    return Turn(
        "general", f"{strategy} could not decide; fell back to the default route", True, None
    )


def recorded(
    turns: Sequence[Turn], strategy: str, *, after: str | None = None
) -> list[RoutingDecision]:
    """The records a conversation of `turns` should carry, one per turn.

    Each turn's `previous_route` is the route the turn before it took — `after` for the first
    one: the conversation is passed back as messages, records and all, as a chat app passes it.
    """
    previous_routes = [after, *(turn.route for turn in turns[:-1])]
    return [
        RoutingDecision(
            route=turn.route,
            reason=turn.reason,
            strategy=strategy,
            fallback=turn.fallback,
            previous_route=previous_route,
            messages_back=turn.messages_back,
        )
        for turn, previous_route in zip(turns, previous_routes, strict=True)
    ]


EASY = "difficulty 0.00 < 1.00 (no signal fired)"
HARD = "difficulty 1.00 >= 1.00 (analysis 1.00)"


class Case(NamedTuple):
    build: Callable[[int], RoutingStrategy]
    lookback: int
    routes: tuple[str, ...]
    turns: list[str]
    expected: list[Turn]


CASES = [
    pytest.param(
        Case(
            keyword_strategy,
            0,
            DOMAIN_ROUTES,
            LEGAL_THEN_CODE,
            [
                Turn("legal", "matched keyword 'contract'"),
                fell_back("KeywordStrategy"),
                Turn("coder", "matched keyword 'code'"),
                Turn("coder", "matched keyword 'python'"),
                fell_back("KeywordStrategy"),
            ],
        ),
        id="keyword-today",
    ),
    pytest.param(
        Case(
            keyword_strategy,
            2,
            DOMAIN_ROUTES,
            LEGAL_THEN_CODE,
            [
                Turn("legal", "matched keyword 'contract'"),
                Turn("legal", "matched keyword 'contract' (1 message back)", messages_back=1),
                # A known false match: the newest match wins, and "code" is this message's own.
                Turn("coder", "matched keyword 'code'"),
                Turn("coder", "matched keyword 'python'"),
                Turn("coder", "matched keyword 'python' (1 message back)", messages_back=1),
            ],
        ),
        id="keyword-lookback-2",
    ),
    pytest.param(
        Case(
            configurable_strategy,
            0,
            DOMAIN_ROUTES,
            LEGAL_THEN_CODE,
            [
                Turn("legal", "rule 'legal' matched: keyword 'contract'"),
                fell_back("ConfigurableStrategy"),
                Turn("coder", "rule 'coder' matched: keyword 'code'"),
                Turn("coder", "rule 'coder' matched: keyword 'python'"),
                fell_back("ConfigurableStrategy"),
            ],
        ),
        id="configurable-today",
    ),
    pytest.param(
        Case(
            configurable_strategy,
            2,
            DOMAIN_ROUTES,
            LEGAL_THEN_CODE,
            [
                Turn("legal", "rule 'legal' matched: keyword 'contract'"),
                Turn(
                    "legal",
                    "rule 'legal' matched: keyword 'contract' (1 message back)",
                    messages_back=1,
                ),
                Turn("coder", "rule 'coder' matched: keyword 'code'"),
                Turn("coder", "rule 'coder' matched: keyword 'python'"),
                Turn(
                    "coder",
                    "rule 'coder' matched: keyword 'python' (1 message back)",
                    messages_back=1,
                ),
            ],
        ),
        id="configurable-lookback-2",
    ),
    pytest.param(
        Case(
            heuristic_strategy,
            0,
            TIER_ROUTES,
            HARD_THEN_EASY,
            [
                Turn("small", EASY),
                Turn("frontier", HARD),
                Turn("small", EASY),
                Turn("small", EASY),
                Turn("small", EASY),
            ],
        ),
        id="heuristic-today",
    ),
    pytest.param(
        Case(
            heuristic_strategy,
            2,
            TIER_ROUTES,
            HARD_THEN_EASY,
            [
                Turn("small", EASY),
                Turn("frontier", HARD),
                Turn("frontier", f"{HARD} (1 message back)", messages_back=1),
                Turn("frontier", f"{HARD} (2 messages back)", messages_back=2),
                # Back down: the hard question is three messages back now.
                Turn("small", EASY),
            ],
        ),
        id="heuristic-lookback-2",
    ),
    pytest.param(
        Case(
            embedding_strategy,
            0,
            DOMAIN_ROUTES,
            LEGAL_THEN_CODE,
            [
                Turn("legal", similar_to("legal")),
                fell_back("EmbeddingStrategy"),
                fell_back("EmbeddingStrategy"),
                Turn("coder", similar_to("coder")),
                fell_back("EmbeddingStrategy"),
            ],
        ),
        id="embedding-today",
    ),
    pytest.param(
        Case(
            embedding_strategy,
            2,
            DOMAIN_ROUTES,
            LEGAL_THEN_CODE,
            [
                Turn("legal", similar_to("legal")),
                Turn("legal", f"{similar_to('legal')} (1 message back)", messages_back=1),
                Turn("legal", f"{similar_to('legal')} (2 messages back)", messages_back=2),
                Turn("coder", similar_to("coder")),
                Turn("coder", f"{similar_to('coder')} (1 message back)", messages_back=1),
            ],
        ),
        id="embedding-lookback-2",
    ),
    pytest.param(
        Case(
            classifier_strategy,
            0,
            DOMAIN_ROUTES,
            LEGAL_THEN_CODE,
            [
                Turn("legal", classified("legal")),
                fell_back("ClassifierStrategy"),
                Turn("coder", classified("coder")),
                Turn("coder", classified("coder")),
                fell_back("ClassifierStrategy"),
            ],
        ),
        id="classifier-today",
    ),
    pytest.param(
        Case(
            classifier_strategy,
            2,
            DOMAIN_ROUTES,
            LEGAL_THEN_CODE,
            [
                Turn("legal", classified("legal")),
                # The model weighs every message it is shown, so no one of them decided alone.
                Turn("legal", classified("legal")),
                # The fake reads keywords, so "civil code" fools it; a real model (the live
                # test) reads the legal question before it.
                Turn("coder", classified("coder")),
                Turn("coder", classified("coder")),
                Turn("coder", classified("coder")),
            ],
        ),
        id="classifier-lookback-2",
    ),
]


def make_router(
    strategy: RoutingStrategy | Callable[[RoutingRequest], str],
    routes: tuple[str, ...] = DOMAIN_ROUTES,
) -> ChatRouter:
    """A router whose routes each answer with their own name; the first is the default."""
    return ChatRouter(
        routes={
            name: FakeChatModel(model_name=f"model-{name}", reply=f"{name} answer")
            for name in routes
        },
        default_route=routes[0],
        strategy=strategy,
    )


class Answered(NamedTuple):
    """One turn as the caller saw it."""

    decision: RoutingDecision | None
    text: str
    route_calls: int
    """How many route calls the turn made, across every route."""
    warnings: list[type[Warning]]


async def converse(
    router: ChatRouter, convention: AnyConvention, turns: Sequence[str]
) -> list[Answered]:
    """Send `turns` as one conversation, each with the history before it, as a chat app does."""

    def route_calls() -> int:
        return sum(len(call_log(route)) for route in router.routes.values())

    conversation: list[BaseMessage] = []
    answered: list[Answered] = []
    for turn in turns:
        conversation.append(HumanMessage(turn))
        before = route_calls()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            answer = await respond(router, convention, conversation)
        answered.append(
            Answered(
                routing_decision(answer),
                answer.text,
                route_calls() - before,
                [w.category for w in caught if issubclass(w.category, RoutingWarning)],
            )
        )
        conversation.append(answer)
    return answered


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
@pytest.mark.parametrize("case", CASES)
async def test_a_conversation_routes_as_specified(case: Case, convention: AnyConvention) -> None:
    """Each turn takes the route the table gives it, with that reason, through every calling
    convention. With `lookback=2` a follow-up goes where the message that set its topic went,
    and says how far back that was, in the reason and in `messages_back`; with `lookback=0`
    every turn routes as it always has. Every record names the route of the turn before it as
    its `previous_route`, so a switch shows. Every turn calls exactly one route: reading earlier
    messages costs no call."""
    strategy = case.build(case.lookback)

    answered = await converse(make_router(strategy, case.routes), convention, case.turns)

    assert [turn.decision for turn in answered] == recorded(case.expected, type(strategy).__name__)
    assert [turn.text for turn in answered] == [f"{turn.route} answer" for turn in case.expected]
    assert [turn.route_calls for turn in answered] == [1] * len(case.turns)
    assert [turn.warnings for turn in answered] == [
        [FallbackWarning] if turn.fallback else [] for turn in case.expected
    ]


# --- What a custom strategy gets ---


class Recording(RoutingStrategy):
    """Keeps every request it is handed, and sends each to the default route."""

    def __init__(self, *, lookback: int = 0) -> None:
        self.lookback = lookback
        self.requests: list[RoutingRequest] = []

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        self.requests.append(request)
        return RoutingChoice("general", "recorded")


class RecordingTranscripts(Recording):
    """`Recording`, with the whole transcript as well."""

    wants_full_context = True


def previous_texts(request: RoutingRequest) -> list[str]:
    return [previous.text for previous in request.previous_requests]


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_a_strategy_receives_up_to_lookback_previous_messages_newest_first(
    convention: AnyConvention,
) -> None:
    """A custom strategy sets `lookback` itself and gets that many of the user's previous
    messages in `previous_requests`, newest first — fewer while the conversation is shorter,
    and never more."""
    strategy = Recording(lookback=2)

    await converse(make_router(strategy), convention, LEGAL_THEN_CODE[:4])

    assert [(request.text, previous_texts(request)) for request in strategy.requests] == [
        (LEGAL_THEN_CODE[0], []),
        (LEGAL_THEN_CODE[1], [LEGAL_THEN_CODE[0]]),
        (LEGAL_THEN_CODE[2], [LEGAL_THEN_CODE[1], LEGAL_THEN_CODE[0]]),
        (LEGAL_THEN_CODE[3], [LEGAL_THEN_CODE[2], LEGAL_THEN_CODE[1]]),
    ]


async def test_without_lookback_a_strategy_gets_no_previous_messages() -> None:
    """`lookback` is the opt-in: the interface's default is `0`, today's behaviour."""
    strategy = Recording()

    await converse(make_router(strategy), "invoke", LEGAL_THEN_CODE[:3])

    assert strategy.lookback == 0
    assert [request.previous_requests for request in strategy.requests] == [()] * 3


async def test_a_plain_function_stays_current_request_only() -> None:
    """A function strategy has no `lookback` to set, so it sees the current request alone, as
    it always has; reading earlier messages takes a strategy class."""
    seen: list[RoutingRequest] = []

    def pick(request: RoutingRequest) -> str:
        seen.append(request)
        return "general"

    await converse(make_router(pick), "invoke", LEGAL_THEN_CODE[:3])

    assert [request.previous_requests for request in seen] == [()] * 3


class ByTopicSoFar(RoutingStrategy):
    """Asks a current-request-only strategy about each message in turn, newest first.

    The way a custom strategy reuses another one on the user's earlier messages: each previous
    request is a `RoutingRequest` of its own.
    """

    lookback = 3

    def __init__(self, inner: RoutingStrategy) -> None:
        self.inner = inner

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        for back, message in enumerate((request, *request.previous_requests)):
            choice = self.inner.decide(message)
            if choice is not None:
                return RoutingChoice(choice.route, f"{choice.reason}, {back} back", back)
        return None


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_a_custom_strategy_can_hand_previous_requests_to_another_strategy(
    convention: AnyConvention,
) -> None:
    """Each previous request can be passed to any strategy's `decide`. Its own
    `previous_requests` is empty, so the inner strategy decides on that one message. The
    strategy says how far back that message was in its choice, and the record carries it."""
    router = make_router(ByTopicSoFar(KeywordStrategy(KEYWORD_RULES)))

    answered = await converse(router, convention, LEGAL_THEN_CODE)

    assert [turn.decision for turn in answered] == recorded(
        [
            Turn("legal", "matched keyword 'contract', 0 back"),
            Turn("legal", "matched keyword 'contract', 1 back", messages_back=1),
            Turn("coder", "matched keyword 'code', 0 back"),
            Turn("coder", "matched keyword 'python', 0 back"),
            Turn("coder", "matched keyword 'python', 1 back", messages_back=1),
        ],
        "ByTopicSoFar",
    )


def test_a_previous_request_describes_its_message_and_shares_the_call() -> None:
    """Only what a previous request was read from is its own — text, content and modalities.
    The rest is the call's: the routes, whether tools are bound, and the config."""
    strategy = Recording(lookback=1)
    image: dict[str, Any] = {"type": "image", "url": "https://example.com/clause.png"}

    make_router(strategy).invoke(
        [
            HumanMessage(content=[{"type": "text", "text": "What does this say?"}, image]),
            AIMessage("It is a limitation of liability clause."),
            HumanMessage("Is it enforceable?"),
        ]
    )

    (request,) = strategy.requests
    (previous,) = request.previous_requests
    assert previous.text == "What does this say?"
    assert previous.content_blocks == [{"type": "text", "text": "What does this say?"}, image]
    assert previous.modalities == {"text", "image"}
    assert (previous.routes, previous.tools_bound) == (request.routes, request.tools_bound)
    assert previous.config is request.config
    assert previous.previous_requests == ()


@pytest.mark.parametrize("recording", [Recording, RecordingTranscripts])
@pytest.mark.parametrize("lookback", [0, 2])
def test_lookback_and_the_full_transcript_are_independent(
    recording: type[Recording], lookback: int
) -> None:
    """Each opt-in gives what it names and nothing else: `lookback` the user's previous
    messages, `wants_full_context` the transcript — the whole of it on the current request and,
    when both are on, on each previous request the part up to its own message."""
    strategy = recording(lookback=lookback)
    conversation = [HumanMessage(text) for text in LEGAL_THEN_CODE[:3]]

    make_router(strategy).invoke(conversation)

    (request,) = strategy.requests
    assert len(request.previous_requests) == lookback
    assert request.messages == (conversation if recording.wants_full_context else None)
    assert [previous.messages for previous in request.previous_requests] == [
        conversation[: len(conversation) - back] if recording.wants_full_context else None
        for back in range(1, lookback + 1)
    ]


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_an_earlier_request_is_the_one_read_when_that_message_arrived(
    convention: AnyConvention,
) -> None:
    """A previous request is the request the strategy was handed when that message was the
    current one — its text, its content and the conversation as it stood then — bar the
    previous requests of its own, which it doesn't carry. Handing it to a strategy that reads
    the transcript asks that strategy about that moment, not about now."""
    strategy = RecordingTranscripts(lookback=2)

    await converse(make_router(strategy), convention, LEGAL_THEN_CODE[:3])

    first, second, third = strategy.requests
    assert third.previous_requests == (
        replace(second, previous_requests=()),
        replace(first, previous_requests=()),
    )


def test_a_system_prompt_and_the_models_answers_never_take_a_place() -> None:
    """The N places are the user's: a system prompt, the model's answers and tool results are
    skipped however many of them sit between the user's messages."""
    strategy = Recording(lookback=2)

    make_router(strategy).invoke(
        [
            SystemMessage("You are a legal assistant."),
            HumanMessage("first"),
            AIMessage("an answer"),
            SystemMessage("Reminder: be brief."),
            AIMessage("a second answer"),
            HumanMessage("second"),
            AIMessage("", tool_calls=[ToolCall(name="lookup", args={}, id="call-1")]),
            ToolMessage("a tool result", tool_call_id="call-1"),
            HumanMessage("third"),
        ]
    )

    (request,) = strategy.requests
    assert (request.text, previous_texts(request)) == ("third", ["second", "first"])


# --- Inside an agent's tool loop ---


@tool
def get_weather(city: str) -> str:
    """Today's forecast for a city."""
    return f"Sunny in {city} and 14C, with an hourly breakdown nobody should route on. " * 20


def weather_call(city: str) -> AIMessage:
    return AIMessage(
        "", tool_calls=[ToolCall(name="get_weather", args={"city": city}, id=f"call-{city}")]
    )


def test_inside_a_tool_loop_the_previous_messages_are_the_ones_before_the_loop() -> None:
    """Every model call of an agent's tool loop reads the request that started the loop, and
    the user's messages before it — never the tool results or the model's own tool calls,
    however many the loop adds. The conversation is kept by a checkpointer, as a chat app
    keeps it."""
    strategy = Recording(lookback=2)
    route = ToolCallingFakeChatModel(
        script=[
            AIMessage("Hello! How can I help?"),
            weather_call("Paris"),
            weather_call("Berlin"),
            AIMessage("Both are sunny today."),
        ]
    )
    router = ChatRouter(routes={"general": route}, default_route="general", strategy=strategy)
    agent = create_agent(router, tools=[get_weather], checkpointer=InMemorySaver())
    thread = RunnableConfig(configurable={"thread_id": "chat-1"})

    agent.invoke({"messages": [HumanMessage("Hi there")]}, thread)
    result = agent.invoke({"messages": [HumanMessage("Weather in Paris and Berlin?")]}, thread)

    assert [type(message) for message in result["messages"]].count(ToolMessage) == 2
    assert [(request.text, previous_texts(request)) for request in strategy.requests] == [
        ("Hi there", []),
        ("Weather in Paris and Berlin?", ["Hi there"]),
        ("Weather in Paris and Berlin?", ["Hi there"]),
        ("Weather in Paris and Berlin?", ["Hi there"]),
    ]


# --- What the router checks ---

NOT_A_COUNT = [
    pytest.param(-1, id="negative"),
    pytest.param(True, id="bool"),
    pytest.param(1.5, id="float"),
    pytest.param("2", id="string"),
    pytest.param(None, id="none"),
]


def not_a_count(lookback: object) -> str:
    """Why the router refuses a strategy whose `lookback` is `lookback`."""
    return (
        f"Unreadable's lookback is {lookback!r}: it must be a non-negative integer, the number "
        "of the user's previous messages to read"
    )


@pytest.mark.parametrize("value", NOT_A_COUNT)
def test_a_lookback_that_is_not_a_count_fails_when_the_router_is_built(value: object) -> None:
    """A built-in checks its `lookback` when it is constructed; a custom strategy's class
    attribute is checked when the router is, the first time the package sees it — not on the
    first request."""

    class Unreadable(RoutingStrategy):
        lookback = cast("int", value)  # not a count, as untyped code can write it

        def decide(self, request: RoutingRequest) -> RoutingChoice | None:
            return RoutingChoice("general", "decided")

    with pytest.raises(ValidationError) as caught:
        make_router(Unreadable())

    [error] = caught.value.errors()
    assert error["loc"] == ("strategy",)
    assert isinstance(error["ctx"]["error"], RoutingError)
    assert str(error["ctx"]["error"]) == not_a_count(value)


def test_a_lookback_made_unreadable_after_the_router_was_built_falls_back() -> None:
    """`lookback` is an ordinary attribute, read on every request. One that stopped being a
    count after the router was built is reported like a strategy failure — the default route
    answers, one `FallbackWarning` says why — rather than failing the call. The strategy can't
    be consulted, so no strategy run opens, but the record names it: the fallback is down to its
    setting."""

    class Unreadable(Recording):
        pass

    strategy = Unreadable(lookback=1)
    router = make_router(strategy)
    strategy.lookback = -1
    collector = RunCollectorCallbackHandler()

    with pytest.warns(FallbackWarning) as caught:
        answer = router.invoke("Is it enforceable in Germany?", {"callbacks": [collector]})

    assert [str(warning.message) for warning in caught] == [
        f"{not_a_count(-1)}; falling back to the default route 'general'"
    ]
    assert routing_decision(answer) == RoutingDecision(
        route="general",
        reason=f"{not_a_count(-1)}; fell back to the default route",
        strategy="Unreadable",
        fallback=True,
    )
    assert strategy.requests == []
    (router_run,) = collector.traced_runs
    assert [run.name for run in router_run.child_runs] == ["FakeChatModel"]


# --- A strategy that wraps another ---


class Logged(RoutingStrategy):
    """Hands every request to another strategy: the shape of a wrapper that logs or measures.

    It reads as many previous messages as the strategy inside it, unless given a number of its
    own, which the strategy inside then reads too.
    """

    def __init__(self, inner: RoutingStrategy, *, lookback: int | None = None) -> None:
        self.inner = inner if lookback is None else inner.with_lookback(lookback)
        self.lookback = self.inner.lookback

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        return self.inner.decide(request)


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
@pytest.mark.parametrize(
    ("inner", "wrapper", "expected"),
    [
        pytest.param(
            2,
            None,
            Turn("legal", "matched keyword 'contract' (1 message back)", messages_back=1),
            id="inherits-the-inner-lookback",
        ),
        pytest.param(
            0,
            2,
            Turn("legal", "matched keyword 'contract' (1 message back)", messages_back=1),
            id="overrides-it-upwards",
        ),
        pytest.param(2, 0, fell_back("Logged"), id="overrides-it-downwards"),
        pytest.param(0, None, fell_back("Logged"), id="off-when-both-are-off"),
    ],
)
async def test_a_wrapper_reads_its_inner_strategys_lookback_unless_it_sets_its_own(
    inner: int, wrapper: int | None, expected: Turn, convention: AnyConvention
) -> None:
    """The router hands over as many previous messages as the wrapper asks for, and a built-in
    reads at most its own `lookback`. A wrapper that declares its inner strategy's `lookback`
    makes the two agree; one that sets its own gives the inner strategy a copy that reads it.
    Handing back the inner strategy's choice hands back how far back it says it decided."""
    strategy = Logged(keyword_strategy(inner), lookback=wrapper)

    _, follow_up = await converse(make_router(strategy), convention, LEGAL_THEN_CODE[:2])

    assert [follow_up.decision] == recorded([expected], "Logged", after="legal")


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
@pytest.mark.parametrize(
    ("build", "follow_up", "back"),
    [
        pytest.param(
            embedding_strategy, f"{similar_to('legal')} (1 message back)", 1, id="embedding"
        ),
        pytest.param(classifier_strategy, classified("legal"), 0, id="classifier"),
    ],
)
@pytest.mark.parametrize(
    ("inner", "wrapper", "reads_back"),
    [
        pytest.param(2, None, True, id="inherits-the-inner-lookback"),
        pytest.param(0, 2, True, id="overrides-it-upwards"),
        pytest.param(2, 0, False, id="overrides-it-downwards"),
    ],
)
async def test_a_wrapper_gives_a_model_calling_strategy_its_lookback_too(
    build: Callable[[int], RoutingStrategy],
    follow_up: str,
    back: int,
    inner: int,
    wrapper: int | None,
    reads_back: bool,
    convention: AnyConvention,
) -> None:
    """`with_lookback` works for the strategies that call a model as for the others: they read
    `self.lookback` when they decide, so the copy reads the number it was given."""
    strategy = Logged(build(inner), lookback=wrapper)

    _, answered = await converse(make_router(strategy), convention, LEGAL_THEN_CODE[:2])

    expected = Turn("legal", follow_up, messages_back=back) if reads_back else fell_back("Logged")
    assert [answered.decision] == recorded([expected], "Logged", after="legal")


@pytest.mark.parametrize(
    "build",
    [
        keyword_strategy,
        configurable_strategy,
        heuristic_strategy,
        embedding_strategy,
        classifier_strategy,
        Recording,
    ],
)
def test_with_lookback_is_a_copy_and_leaves_the_strategy_as_it_was(
    build: Callable[..., RoutingStrategy],
) -> None:
    """A copy of the same class, with the new `lookback` and everything else shared; the
    original keeps its own, so one strategy can sit in two routers with different windows."""
    original = build(lookback=1)

    copied = original.with_lookback(3)

    assert type(copied) is type(original)
    assert copied is not original
    assert (copied.lookback, original.lookback) == (3, 1)
    assert vars(copied) == {**vars(original), "lookback": 3}


async def test_a_copy_reads_its_own_lookback_and_the_original_keeps_reading_its() -> None:
    """Both strategies route as their own `lookback` says, side by side."""
    original = keyword_strategy(0)
    copied = original.with_lookback(2)

    [_, alone] = await converse(make_router(original), "invoke", LEGAL_THEN_CODE[:2])
    [_, with_lookback] = await converse(make_router(copied), "invoke", LEGAL_THEN_CODE[:2])

    assert alone.decision is not None
    assert with_lookback.decision is not None
    assert (alone.decision.route, with_lookback.decision.route) == ("general", "legal")


@pytest.mark.parametrize("value", NOT_A_COUNT)
def test_with_lookback_refuses_a_value_that_is_not_a_count(value: object) -> None:
    """The same check the built-ins make when constructed, made when the copy is asked for."""
    with pytest.raises(RoutingError) as caught:
        keyword_strategy(0).with_lookback(cast("int", value))

    assert str(caught.value) == (
        "lookback must be a non-negative integer, the number of the user's previous messages to "
        f"read, got {value!r}"
    )


# --- The trace ---


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_the_strategy_run_shows_the_previous_messages_it_read(
    convention: Convention,
) -> None:
    """The strategy run's inputs list the previous requests' text and modalities, so a reason
    that says "1 message back" can be read against them. Without any, the inputs are what they
    have always been."""
    router = make_router(keyword_strategy(2))
    collector = RunCollectorCallbackHandler()
    first, follow_up = LEGAL_THEN_CODE[:2]
    current = {"modalities": ["text"], "routes": list(DOMAIN_ROUTES), "tools_bound": False}

    await respond(router, convention, [HumanMessage(first)], {"callbacks": [collector]})
    await respond(
        router,
        convention,
        [HumanMessage(first), AIMessage("legal answer"), HumanMessage(follow_up)],
        {"callbacks": [collector]},
    )

    strategy_runs = [run for run in walk(collector.traced_runs) if run.name == "KeywordStrategy"]
    assert [run.inputs for run in strategy_runs] == [
        {"text": first, **current},
        {
            "text": follow_up,
            **current,
            "previous_requests": [{"text": first, "modalities": ["text"]}],
        },
    ]


class AsksAboutTheLastMessage(RoutingStrategy):
    """Asks a model about the user's previous message, with that request's config."""

    lookback = 1

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        (previous,) = request.previous_requests
        self.model.invoke(previous.text, config=previous.config)
        return RoutingChoice("general", "asked")

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        (previous,) = request.previous_requests
        await self.model.ainvoke(previous.text, config=previous.config)
        return RoutingChoice("general", "asked")


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_call_made_about_a_previous_request_nests_under_the_strategy_run(
    convention: Convention,
) -> None:
    """A previous request carries the strategy run's config, like the current one, so a model
    call a strategy makes about an earlier message is traced and costed under the strategy."""
    classifier = FakeChatModel(model_name="classifier", reply="legal")
    router = make_router(AsksAboutTheLastMessage(classifier))
    collector = RunCollectorCallbackHandler()

    await respond(
        router,
        convention,
        [HumanMessage(LEGAL_THEN_CODE[0]), AIMessage("..."), HumanMessage(LEGAL_THEN_CODE[1])],
        {"callbacks": [collector]},
    )

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    (classifier_run,) = model_runs([strategy_run])
    assert strategy_run.name == "AsksAboutTheLastMessage"
    assert classifier_run.parent_run_id == strategy_run.id
    assert [run.id for run in model_runs([router_run])] == [classifier_run.id, route_run.id]
    assert len(call_log(classifier)) == 1


def test_a_previous_requests_config_is_the_strategy_runs_child_config() -> None:
    """The config each previous request carries is the one the current request carries: the
    strategy run's child config, whose callbacks nest under that run."""
    strategy = Recording(lookback=1)
    collector = RunCollectorCallbackHandler()

    make_router(strategy).invoke(
        [HumanMessage("first"), AIMessage("..."), HumanMessage("second")],
        {"callbacks": [collector]},
    )

    (request,) = strategy.requests
    (strategy_run,) = [run for run in walk(collector.traced_runs) if run.name == "Recording"]
    for config in (request.config, request.previous_requests[0].config):
        callbacks = config.get("callbacks")
        assert isinstance(callbacks, BaseCallbackManager)
        assert callbacks.parent_run_id == strategy_run.id


# --- The strategies that call a model: what lookback costs ---


@pytest.mark.parametrize("convention", ALL_CONVENTIONS)
async def test_each_message_of_a_conversation_is_embedded_once(convention: AnyConvention) -> None:
    """With `lookback`, the embedding strategy reads earlier messages on every turn, but each
    was embedded on its own turn and is not embedded again: one call a turn, however far back
    the turns read."""
    embeddings = CountingEmbeddings(DeterministicFakeEmbedding(size=32))
    strategy = EmbeddingStrategy(embeddings, EMBEDDING_EXAMPLES, threshold=0.99, lookback=2)

    await converse(make_router(strategy), convention, LEGAL_THEN_CODE)

    assert embeddings.queries == LEGAL_THEN_CODE


def test_an_exact_cache_makes_a_tool_loop_one_classification() -> None:
    """The caching the docs advise for the classifier model. Each step of an agent's tool loop
    shows the classifier the same messages, since tool calls and results take no place among
    them, so with `cache=InMemoryCache()` the three steps of this loop classify once — and the
    route can't change halfway through it."""
    classifier = PromptReadingFakeChatModel(
        model_name="classifier", answer=lambda _: "general", cache=InMemoryCache()
    )
    strategy = ClassifierStrategy(classifier, {"general": "anything"}, lookback=2)
    route = ToolCallingFakeChatModel(
        script=[
            AIMessage("Hello! How can I help?"),
            weather_call("Paris"),
            weather_call("Berlin"),
            AIMessage("Both are sunny today."),
        ]
    )
    router = ChatRouter(routes={"general": route}, default_route="general", strategy=strategy)
    agent = create_agent(router, tools=[get_weather], checkpointer=InMemorySaver())
    thread = RunnableConfig(configurable={"thread_id": "chat-1"})

    agent.invoke({"messages": [HumanMessage("Hi there")]}, thread)
    result = agent.invoke({"messages": [HumanMessage("Weather in Paris and Berlin?")]}, thread)

    assert [type(message) for message in result["messages"]].count(ToolMessage) == 2
    assert len(call_log(route)) == 4
    assert [prompt.rsplit("\n", 1)[-1] for prompt in classifier.prompts] == [
        "Hi there",
        "Weather in Paris and Berlin?",
    ]
