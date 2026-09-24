"""The configurable strategy: the headline use cases from configuration alone.

The headline tests express both §4 use cases as configuration alone: `cost_tiering()` and
`domain_routing()` are module-level so the same two setups run on the strategy alone, through
`ChatRouter` on every entry point, and in the order-independent spelling `not_` allows. Then the
parts they are made of: each condition, priority and tie-breaking, everything that fails at
construction (and the dead-rule check's reach in both directions), and what fails only on a
request — narrowly.

Every test in this module runs under `model_calls`, the no-extra-calls counter, installed as a
LangChain configure hook the way `test_keyword_strategy.py` does it: it sees any LLM run started
anywhere, including one a strategy made without passing `request.config`. A test that
legitimately calls a route says so with `expect(...)`; every other test must end with no model
call at all.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import socket
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import FrozenInstanceError, replace
from typing import Any, NoReturn

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.context import register_configure_hook

import langchain_llm_router
from langchain_llm_router import (
    ChatRouter,
    ConfigurableStrategy,
    FallbackWarning,
    KeywordStrategy,
    RoutingChoice,
    RoutingDecision,
    RoutingError,
    RoutingRequest,
    RoutingStrategy,
    RoutingWarning,
    routing_decision,
)
from langchain_llm_router._extraction import build_request
from langchain_llm_router.strategies.configurable import (
    Condition,
    Rule,
    all_of,
    always,
    any_of,
    keywords,
    modality,
    not_,
    predicate,
    signal_at_least,
    tools_bound,
)
from langchain_llm_router.strategies.heuristic import Signal, code_signal, length_signal
from tests.conventions import CONVENTIONS, Convention, respond
from tests.fakes import FakeChatModel, call_log

ROUTES = ("coder", "frontier", "small")
IMAGE: dict[str, Any] = {"type": "image", "url": "https://example.com/cat.png"}

FENCED = "Why does this fail?\n```\nx = 1\n```"
TRACEBACK = (
    'Traceback (most recent call last):\n  File "app.py", line 3, in <module>\nValueError: x'
)


def words(count: int) -> str:
    """A request of exactly `count` words, none of them about anything."""
    return " ".join(["word"] * count)


def make_request(
    text: str = "",
    *,
    blocks: Sequence[dict[str, Any]] | None = None,
    routes: tuple[str, ...] = ROUTES,
    tools: bool = False,
) -> RoutingRequest:
    """The current request as the router hands it to a strategy."""
    message = HumanMessage(content=list(blocks)) if blocks is not None else HumanMessage(text)
    request = build_request(
        [message],
        routes=routes,
        tools_bound=tools,
        wants_full_context=False,
        config=RunnableConfig(),
    )
    assert request is not None
    return request


def strength(value: float) -> Signal:
    """A signal that scores every request `value`, so a test can sit exactly on a threshold."""
    return lambda request: value


def make_router(strategy: RoutingStrategy) -> tuple[ChatRouter, dict[str, BaseChatModel]]:
    """Three routes that each answer with their own name, and `small` as the default."""
    routes: dict[str, BaseChatModel] = {
        "coder": FakeChatModel(model_name="model-coder", reply="coder answer"),
        "frontier": FakeChatModel(model_name="model-frontier", reply="frontier answer"),
        "small": FakeChatModel(model_name="model-small", reply="small answer"),
    }
    return ChatRouter(routes=routes, default_route="small", strategy=strategy), routes


def routing_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [warning for warning in caught if issubclass(warning.category, RoutingWarning)]


def calls_per_route(routes: dict[str, BaseChatModel]) -> dict[str, int]:
    return {name: len(call_log(route)) for name, route in routes.items()}


# the counter every test in this module runs under


class ModelCalls(BaseCallbackHandler):
    """Every model run started while this handler is installed (the no-extra-calls counter)."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.expected: list[str] = []

    def expect(self, *names: str) -> None:
        """Declare the route calls this test makes; anything else is a call nobody asked for."""
        self.expected = list(names)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self.started.append(_model_name(serialized))

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        self.started.append(_model_name(serialized))


def _model_name(serialized: dict[str, Any]) -> str:
    """The model that started, as a trace names it: `dumpd`'s id ends in the class name."""
    path = serialized.get("id")
    return str(path[-1]) if isinstance(path, list) and path else "an unnamed model"


_model_calls: ContextVar[ModelCalls | None] = ContextVar("configurable_model_calls", default=None)
# A configure hook adds the current counter to every callback manager LangChain configures, so a
# call made with no config at all is still counted. Inert while the variable is unset — but the
# registry is global and per-process, and every strategy suite that needs one adds its own.
register_configure_hook(_model_calls, inheritable=True)


@contextmanager
def counting() -> Iterator[ModelCalls]:
    """Record every model run started inside the block."""
    calls = ModelCalls()
    token = _model_calls.set(calls)
    try:
        yield calls
    finally:
        _model_calls.reset(token)


@pytest.fixture(autouse=True)
def model_calls() -> Iterator[ModelCalls]:
    """No test here can pass by not looking — the calls it made are asserted for it."""
    with counting() as calls:
        yield calls
        assert calls.started == calls.expected


def _refuse(*args: object, **kwargs: object) -> NoReturn:
    msg = "the strategy opened a network connection"
    raise AssertionError(msg)


# The two §4 use cases, as configuration alone


def cost_tiering() -> ConfigurableStrategy:
    """Cost tiering: long or code-bearing requests to the frontier model, the rest to the small
    one. No strategy class — the rules are the whole of the policy."""
    long_request = signal_at_least(length_signal(20, 200), 0.5, name="length")
    carries_code = signal_at_least(code_signal, 0.5, name="code")
    return ConfigurableStrategy(
        [
            Rule("frontier", any_of(long_request, carries_code), name="long or code-bearing"),
            Rule("small", always(), name="short and simple"),
        ]
    )


def domain_routing() -> ConfigurableStrategy:
    """Domain routing: code requests to the code model; nothing says where the rest go."""
    carries_code = signal_at_least(code_signal, 0.5, name="code")
    return ConfigurableStrategy(
        [
            Rule(
                "coder",
                any_of(keywords("python", "regex", "stack trace"), carries_code),
                name="code",
            )
        ]
    )


def independent_cost_tiering() -> ConfigurableStrategy:
    """The same policy with each rule readable on its own: `not_` makes the rules disjoint, so
    the order they are written in — and any priority — no longer matters."""
    heavy = any_of(
        signal_at_least(length_signal(20, 200), 0.5, name="length"),
        signal_at_least(code_signal, 0.5, name="code"),
    )
    return ConfigurableStrategy(
        [
            Rule("small", not_(heavy), name="short and simple"),
            Rule("frontier", heavy, name="long or code-bearing"),
        ]
    )


TIERING_CASES: list[tuple[str, str, str, str]] = [
    # (id, text, route, reason)
    (
        "short and simple",
        "What is the capital of France?",
        "small",
        "rule 'short and simple' matched: always",
    ),
    (
        "long",
        words(150),
        "frontier",
        "rule 'long or code-bearing' matched: length 0.72 >= 0.50",
    ),
    (
        "code",
        FENCED,
        "frontier",
        "rule 'long or code-bearing' matched: code 0.50 >= 0.50",
    ),
    (
        "on the threshold",
        words(110),
        "frontier",
        "rule 'long or code-bearing' matched: length 0.50 >= 0.50",
    ),
    ("just under it", words(109), "small", "rule 'short and simple' matched: always"),
]
TIERING = [
    pytest.param(text, route, reason, id=case) for case, text, route, reason in TIERING_CASES
]
TIERING_TEXTS = [text for _, text, _, _ in TIERING_CASES]


@pytest.mark.parametrize(("text", "route", "reason"), TIERING)
def test_cost_tiering_is_configuration_alone(text: str, route: str, reason: str) -> None:
    """Cheap for short simple requests, frontier for long or code-bearing ones —
    built from rules and conditions, with no strategy class, and each decision's reason names the
    rule and the signal that decided it."""
    strategy = cost_tiering()

    assert type(strategy) is ConfigurableStrategy
    assert strategy.decide(make_request(text)) == RoutingChoice(route, reason)


DOMAIN = [
    pytest.param(
        "How do I do this in Python?",
        RoutingChoice("coder", "rule 'code' matched: keyword 'python'"),
        id="keyword",
    ),
    pytest.param(
        "why does my REGEX never match?",
        RoutingChoice("coder", "rule 'code' matched: keyword 'regex'"),
        id="keyword, whatever its case",
    ),
    pytest.param(
        TRACEBACK,
        RoutingChoice("coder", "rule 'code' matched: code 0.50 >= 0.50"),
        id="pasted traceback",
    ),
    pytest.param("What is the weather in Kraków?", None, id="everything else"),
]


@pytest.mark.parametrize(("text", "expected"), DOMAIN)
def test_domain_routing_is_configuration_alone(text: str, expected: RoutingChoice | None) -> None:
    """Code requests to a code route, built from keywords and a signal with no
    strategy class. Everything else is a request the rules have nothing to say about — `None`,
    which the router turns into its default route."""
    strategy = domain_routing()

    assert type(strategy) is ConfigurableStrategy
    assert strategy.decide(make_request(text)) == expected


def test_the_setups_define_no_strategy_class() -> None:
    """ "without a strategy class" is checked at the source — neither setup declares a
    class or a `decide`, so nothing in them is code a user would have to write and test."""
    for setup in (cost_tiering, domain_routing, independent_cost_tiering):
        body = inspect.getsource(setup).split('"""')[-1]  # past the docstring, which says so
        assert "class " not in body
        assert "decide" not in body


def test_a_reader_sees_from_the_configuration_what_will_happen() -> None:
    """The strategy prints as the rules it holds, each with the condition in words —
    what a user reads to know what a request will do, without running one."""
    assert repr(cost_tiering()) == (
        "ConfigurableStrategy(["
        "Rule(route='frontier', when=(length >= 0.5 or code >= 0.5), priority=0, "
        "name='long or code-bearing'), "
        "Rule(route='small', when=always, priority=0, name='short and simple')"
        "])"
    )


def route_of(strategy: ConfigurableStrategy, text: str) -> str | None:
    choice = strategy.decide(make_request(text))
    return choice.route if choice is not None else None


def test_the_order_independent_spelling_decides_the_same_in_any_order() -> None:
    """`not_` is what lets a rule be read without the rules above it. Both rules of
    the policy stand alone, so writing them in the other order — or ranking them any way at all —
    gives the same decisions as the ordered spelling above, on every request of the corpus."""
    ordered = cost_tiering()
    independent = independent_cost_tiering()
    reversed_ = ConfigurableStrategy(list(reversed(independent.rules)))
    ranked = ConfigurableStrategy(
        [
            replace(rule, priority=priority)
            for rule, priority in zip(independent.rules, (-2, 9), strict=True)
        ]
    )

    routes = [route_of(ordered, text) for text in TIERING_TEXTS]
    assert set(routes) == {"small", "frontier"}
    for strategy in (independent, reversed_, ranked):
        assert [route_of(strategy, text) for text in TIERING_TEXTS] == routes


# Through the router: the same two setups, on every entry point


@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize(("text", "route", "reason"), TIERING)
async def test_cost_tiering_through_the_router(
    convention: Convention, text: str, route: str, reason: str, model_calls: ModelCalls
) -> None:
    """The configured strategy plugged in through `strategy=` — the tier its rules
    picked answers on every entry point, nothing warns (an `always()` rule is a decision, not a
    fallback), and the record carries the reason that names the rule."""
    model_calls.expect("FakeChatModel")
    router, routes = make_router(cost_tiering())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, text)

    assert message.content == f"{route} answer"
    assert calls_per_route(routes) == {"coder": 0, "frontier": 0, "small": 0} | {route: 1}
    assert routing_warnings(caught) == []
    assert routing_decision(message) == RoutingDecision(
        route=route, reason=reason, strategy="ConfigurableStrategy"
    )


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_domain_routing_through_the_router(
    convention: Convention, model_calls: ModelCalls
) -> None:
    """A code request goes to the code route, silently, with the rule on the
    record."""
    model_calls.expect("FakeChatModel")
    router, routes = make_router(domain_routing())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "why is this regex wrong?")

    assert message.content == "coder answer"
    assert calls_per_route(routes) == {"coder": 1, "frontier": 0, "small": 0}
    assert routing_warnings(caught) == []
    assert routing_decision(message) == RoutingDecision(
        route="coder",
        reason="rule 'code' matched: keyword 'regex'",
        strategy="ConfigurableStrategy",
    )


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_request_no_rule_matches_takes_the_default_route(
    convention: Convention, model_calls: ModelCalls
) -> None:
    """ "everything else" left to the router is its own fallback — the default route answers,
    exactly one `FallbackWarning` fires, and the record says who couldn't decide. The strategy
    itself neither warns nor raises."""
    model_calls.expect("FakeChatModel")
    router, routes = make_router(domain_routing())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "what is the weather in Kraków?")

    assert message.content == "small answer"
    assert calls_per_route(routes) == {"coder": 0, "frontier": 0, "small": 1}
    assert [warning.category for warning in routing_warnings(caught)] == [FallbackWarning]
    assert str(routing_warnings(caught)[0].message) == (
        "ConfigurableStrategy could not decide; falling back to the default route 'small'"
    )
    assert routing_decision(message) == RoutingDecision(
        route="small",
        reason="ConfigurableStrategy could not decide; fell back to the default route",
        strategy="ConfigurableStrategy",
        fallback=True,
    )


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_final_always_rule_makes_everything_else_a_decision(
    convention: Convention, model_calls: ModelCalls
) -> None:
    """Domain routing with `Rule("small", always())` last — the same request that fell
    back above is now a decision with a reason of its own: no warning, `fallback` unset."""
    model_calls.expect("FakeChatModel")
    carries_code = signal_at_least(code_signal, 0.5, name="code")
    strategy = ConfigurableStrategy(
        [
            Rule("coder", any_of(keywords("regex"), carries_code), name="code"),
            Rule("small", always(), name="everything else"),
        ]
    )
    router, routes = make_router(strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "what is the weather in Kraków?")

    assert calls_per_route(routes) == {"coder": 0, "frontier": 0, "small": 1}
    assert routing_warnings(caught) == []
    assert routing_decision(message) == RoutingDecision(
        route="small",
        reason="rule 'everything else' matched: always",
        strategy="ConfigurableStrategy",
    )


# The conditions: what each one holds for, and what its reason says


def test_a_keyword_reads_a_request_as_keyword_strategy_does() -> None:
    """The docs say `keywords` reads a request "as `KeywordStrategy` reads it" — this holds them
    to it, keyword by keyword and text by text, whole words, case, phrases and literal
    punctuation included. Both directions occur, so the agreement is not vacuous."""
    cases: list[tuple[str | re.Pattern[str], str, bool]] = [
        ("python", "How do I do this in Python?", True),
        ("python", "write pythonic code", False),
        ("python", "monty-python", True),
        ("stack trace", "here is the stack\ntrace", True),
        ("stack trace", "a trace of the stack", False),
        ("c++", "rewrite it in c++", True),
        (".net", "the .net runtime", True),
        ("a.b", "axb", False),
        ("(beta)", "the beta build", False),
        ("cost*", "what is the cost of this", False),
        ("北京", "我在北京工作", False),
        (re.compile("SQL"), "this SQL query", True),
        (re.compile("SQL"), "this sql query", False),
        (re.compile(r"regexe?s?", re.IGNORECASE), "two Regexes", True),
    ]
    outcomes = set()

    for keyword, text, matches in cases:
        request = make_request(text, routes=("x",))
        ours = keywords(keyword).explain(request) is not None
        theirs = KeywordStrategy({"x": [keyword]}).decide(request) is not None
        assert (ours, theirs) == (matches, matches), (keyword, text)
        outcomes.add(matches)

    assert outcomes == {True, False}


def test_a_keyword_names_the_word_that_matched() -> None:
    """Several keywords are tried in the order given and the first that is in the text is the
    reason; a compiled pattern is written as its author wrote it, flags included."""
    request = make_request("a python regex")

    assert keywords("regex", "python").explain(request) == "keyword 'regex'"
    assert keywords("java", "python").explain(request) == "keyword 'python'"
    assert keywords(" python ").explain(request) == "keyword 'python'"
    assert keywords(re.compile(r"reg\w+", re.IGNORECASE)).explain(request) == (
        'pattern r"reg\\w+" with re.IGNORECASE'
    )
    assert keywords("java", "ruby").explain(request) is None


def test_a_signal_holds_at_its_threshold_and_above() -> None:
    """The threshold is the price of admission, as in `HeuristicStrategy`: a score exactly on it
    holds. The reason carries the score, so a trace shows how far over it the request was."""
    request = make_request("anything")

    assert signal_at_least(strength(0.5), 0.5, name="s").explain(request) == "s 0.50 >= 0.50"
    assert signal_at_least(strength(0.75), 0.5, name="s").explain(request) == "s 0.75 >= 0.50"
    assert signal_at_least(strength(0.49), 0.5, name="s").explain(request) is None
    assert signal_at_least(strength(1.0), 1, name="s").explain(request) == "s 1.00 >= 1.00"


def test_a_signal_is_named_after_its_function_unless_told_otherwise() -> None:
    """The reason calls a signal by `name=`, else by the function's own name — which for the
    closure `length_signal(...)` returns is not informative, hence `name=`."""
    request = make_request(FENCED)

    assert signal_at_least(code_signal, 0.5).explain(request) == "code_signal 0.50 >= 0.50"
    assert signal_at_least(code_signal, 0.5, name="code").explain(request) == "code 0.50 >= 0.50"
    assert str(signal_at_least(code_signal, 0.5, name="code")) == "code >= 0.5"


def test_a_signal_that_returns_no_number_fails_the_request_it_ran_for() -> None:
    """A signal is user code, so what it returns is checked when it runs: a clear error, which the
    router reports as any other strategy failure, rather than a comparison's own."""
    request = make_request("anything")
    broken: Signal = lambda request: None  # type: ignore[assignment,return-value]  # noqa: E731

    with pytest.raises(
        TypeError, match=r"^signal 'broken' returned NoneType; a signal returns a float$"
    ):
        signal_at_least(broken, 0.5, name="broken").explain(request)


def test_a_modality_holds_for_a_request_that_carries_it() -> None:
    """`modality` reads the request's modalities — here from a real text-and-image message."""
    picture = make_request(
        "what is this?", blocks=[{"type": "text", "text": "what is this?"}, IMAGE]
    )
    text_only = make_request("what is this?")

    assert modality("image").explain(picture) == "has image"
    assert modality("audio", "image").explain(picture) == "has image"
    assert modality("text").explain(picture) == "has text"
    assert modality("audio", "video").explain(picture) is None
    assert modality("image").explain(text_only) is None
    assert not_(modality("image", "audio", "video", "file", "other")).explain(text_only) == (
        "not has image or audio or video or file or other"
    )


def test_tools_bound_holds_when_tools_are_bound() -> None:
    """`request.tools_bound` as a condition, in both polarities."""
    with_tools = make_request("hello", tools=True)
    without_tools = make_request("hello")

    assert tools_bound().explain(with_tools) == "tools bound"
    assert tools_bound().explain(without_tools) is None
    assert not_(tools_bound()).explain(without_tools) == "not tools bound"
    assert not_(tools_bound()).explain(with_tools) is None


def mentions_pricing(request: RoutingRequest) -> bool:
    return "pricing" in request.text.lower()


class Shouting:
    """A callable object: it has no `__name__`, so a reason names its class."""

    def __call__(self, request: RoutingRequest) -> bool:
        return request.text.isupper()


def test_a_predicate_is_the_escape_hatch_for_anything_else() -> None:
    """A predicate is any function of the request that returns a `bool`; the reason names it,
    by `name=` if given, else by its function or — for a callable object — its class."""
    yes = make_request("What is your PRICING?")
    no = make_request("hello")

    assert predicate(mentions_pricing).explain(yes) == "mentions_pricing"
    assert predicate(mentions_pricing).explain(no) is None
    assert predicate(mentions_pricing, name="asks about price").explain(yes) == "asks about price"
    assert predicate(Shouting()).explain(make_request("HELLO")) == "Shouting"
    assert str(predicate(mentions_pricing)) == "mentions_pricing"


def test_a_predicate_must_return_a_bool() -> None:
    """A truthy non-bool is the silent mistake — a match object, or a coroutine from a function
    that wraps an `async` one — so it is an error naming the predicate, not a match."""
    request = make_request("hello")
    coroutines: list[Any] = []

    def sneaky(request: RoutingRequest) -> Any:
        coroutines.append(asyncio.sleep(0))
        return coroutines[-1]

    with pytest.raises(
        TypeError, match=r"^predicate '<lambda>' returned Match; a predicate returns a bool$"
    ):
        predicate(lambda request: re.search("hell", request.text)).explain(request)  # type: ignore[arg-type,return-value]
    with pytest.raises(
        TypeError, match=r"^predicate 'sneaky' returned coroutine; a predicate returns a bool$"
    ):
        predicate(sneaky).explain(request)
    coroutines[0].close()


def test_always_holds_for_every_request() -> None:
    assert always().explain(make_request("")) == "always"
    assert always().explain(make_request(FENCED, tools=True)) == "always"


def test_all_of_holds_when_every_condition_does_and_says_so() -> None:
    """The reason of an `all_of` is every part's, in order — a reader sees all that was needed."""
    both = make_request("what is this?", blocks=[{"type": "text", "text": "hi"}, IMAGE], tools=True)
    only_image = make_request("hi", blocks=[{"type": "text", "text": "hi"}, IMAGE])
    condition = all_of(modality("image"), tools_bound())

    assert condition.explain(both) == "has image and tools bound"
    assert condition.explain(only_image) is None
    assert all_of(tools_bound()).explain(both) == "tools bound"


def test_any_of_holds_when_a_condition_does_and_reports_the_first_that_did() -> None:
    """The reason of an `any_of` is the first part that held, in the order given — which keyword
    or signal it was, not only that one of them was."""
    request = make_request(FENCED, tools=True)

    assert any_of(tools_bound(), keywords("fail")).explain(request) == "tools bound"
    assert any_of(keywords("fail"), tools_bound()).explain(request) == "keyword 'fail'"
    assert any_of(keywords("java"), keywords("ruby")).explain(request) is None


def test_combinators_stop_at_the_first_part_that_decides() -> None:
    """Conditions are tried left to right and stop early, so a cheap check can guard a costly
    predicate — and a predicate that would fail is never reached."""
    seen: list[str] = []

    def expensive(request: RoutingRequest) -> bool:
        seen.append(request.text)
        return True

    request = make_request("hello")

    assert all_of(keywords("absent"), predicate(expensive)).explain(request) is None
    assert any_of(keywords("hello"), predicate(expensive)).explain(request) == "keyword 'hello'"
    assert seen == []
    assert all_of(keywords("hello"), predicate(expensive)).explain(request) == (
        "keyword 'hello' and expensive"
    )
    assert seen == ["hello"]


def test_not_inverts_a_condition_and_says_what_was_absent() -> None:
    """`not_` holds when its condition doesn't, and its reason is the condition in words."""
    request = make_request("hello")

    assert not_(keywords("hello")).explain(request) is None
    assert not_(keywords("absent")).explain(request) == "not keyword 'absent'"
    assert not_(any_of(keywords("a"), keywords("b"))).explain(request) == (
        "not (keyword 'a' or keyword 'b')"
    )


def test_a_condition_reads_as_the_words_of_its_configuration() -> None:
    """The text a reader sees when a rule is printed — and the wording `not_` explains itself in:
    each condition says what it asks, combinations are bracketed, and nothing needs decoding."""
    assert str(keywords("python")) == "keyword 'python'"
    assert str(keywords("python", "sql")) == "(keyword 'python' or keyword 'sql')"
    assert str(signal_at_least(code_signal, 0.75, name="code")) == "code >= 0.75"
    assert str(modality("image", "audio")) == "has image or audio"
    assert str(tools_bound()) == "tools bound"
    assert str(always()) == "always"
    assert str(all_of(modality("image"), tools_bound())) == "(has image and tools bound)"
    assert str(not_(all_of(keywords("a"), keywords("b")))) == "not (keyword 'a' and keyword 'b')"
    assert repr(any_of(keywords("a"), tools_bound())) == "(keyword 'a' or tools bound)"


def test_a_script_without_spaces_needs_a_predicate_to_be_called_long() -> None:
    """A limit the signals bring with them, pinned so nobody meets it in
    production: `length_signal` counts whitespace-separated words, so a long request in Chinese
    reads as one word and scores nothing. A predicate on the text's length is the way round —
    and it is a rule like any other."""
    request = make_request("北京是中华人民共和国的首都也是全国的政治和文化中心" * 20)
    by_words = signal_at_least(length_signal(20, 200), 0.5, name="length")
    by_characters = predicate(lambda request: len(request.text) > 300, name="over 300 characters")

    def tiering(long_request: Condition) -> ConfigurableStrategy:
        return ConfigurableStrategy(
            [
                Rule("frontier", long_request, name="long"),
                Rule("small", always(), name="short"),
            ]
        )

    assert tiering(by_words).decide(request) == RoutingChoice(
        "small", "rule 'short' matched: always"
    )
    assert tiering(by_characters).decide(request) == RoutingChoice(
        "frontier", "rule 'long' matched: over 300 characters"
    )


# Which rule wins


def test_the_highest_priority_rule_is_tried_first_whatever_its_position() -> None:
    """Priority, not position, orders the rules: two rules that both match the request give the
    same answer written either way round."""
    low = Rule("small", keywords("question"), priority=1, name="low")
    high = Rule("coder", keywords("python"), priority=5, name="high")
    request = make_request("a python question")

    for declared in ([low, high], [high, low]):
        assert ConfigurableStrategy(declared).decide(request) == RoutingChoice(
            "coder", "rule 'high' matched: keyword 'python'"
        )


def test_rules_of_equal_priority_are_tried_in_declaration_order() -> None:
    """The tie-break is the order the rules were written — so the same rules, swapped, answer
    the other way — and a set of rules with no priorities at all is "the first that matches"."""
    python = Rule("coder", keywords("python"), name="python")
    test = Rule("small", keywords("test"), name="test")
    request = make_request("a python test")

    assert ConfigurableStrategy([python, test]).decide(request) == RoutingChoice(
        "coder", "rule 'python' matched: keyword 'python'"
    )
    assert ConfigurableStrategy([test, python]).decide(request) == RoutingChoice(
        "small", "rule 'test' matched: keyword 'test'"
    )


def test_a_tie_is_broken_by_declaration_within_its_own_priority_only() -> None:
    """Ties are per priority level: of the two rules at priority 3 the first declared wins, but
    a higher rule declared last still beats both, and a lower one declared first loses to both."""
    rules = [
        Rule("small", keywords("test"), priority=0, name="lowest"),
        Rule("frontier", keywords("of"), priority=3, name="first of two"),
        Rule("coder", keywords("python"), priority=3, name="second of two"),
        Rule("coder", keywords("code"), priority=-4, name="below zero"),
        Rule("frontier", keywords("a"), priority=7, name="highest"),
    ]
    request = make_request("a test of python code")

    def winner(*dropped: str) -> str:
        kept = [rule for rule in rules if rule.name not in dropped]
        decision = ConfigurableStrategy(kept).decide(request)
        assert decision is not None
        return decision.reason

    assert ConfigurableStrategy(rules).decide(request) == RoutingChoice(
        "frontier", "rule 'highest' matched: keyword 'a'"
    )
    assert winner("highest") == "rule 'first of two' matched: keyword 'of'"
    assert winner("highest", "first of two") == "rule 'second of two' matched: keyword 'python'"
    assert winner("highest", "first of two", "second of two") == (
        "rule 'lowest' matched: keyword 'test'"
    )
    assert winner("highest", "first of two", "second of two", "lowest") == (
        "rule 'below zero' matched: keyword 'code'"
    )


def test_a_catch_all_can_be_declared_first_with_a_lower_priority() -> None:
    """Priority is what lets the order a rule set is written in differ from the order it is tried
    in: an `always()` rule declared first, but ranked last, shadows nothing."""
    strategy = ConfigurableStrategy(
        [
            Rule("small", always(), priority=-1, name="otherwise"),
            Rule("coder", keywords("python"), name="python"),
        ]
    )

    assert strategy.decide(make_request("a python question")) == RoutingChoice(
        "coder", "rule 'python' matched: keyword 'python'"
    )
    assert strategy.decide(make_request("a question")) == RoutingChoice(
        "small", "rule 'otherwise' matched: always"
    )


def test_an_unnamed_rule_goes_by_its_declaration_position() -> None:
    """A rule with no name is "rule #N" — N counting from 1 in the order written, not the
    order tried, so it points at the line a reader has open."""
    strategy = ConfigurableStrategy(
        [Rule("small", keywords("a"), name="named"), Rule("coder", keywords("b"), priority=9)]
    )

    assert strategy.decide(make_request("b")) == RoutingChoice(
        "coder", "rule #2 matched: keyword 'b'"
    )
    assert strategy.decide(make_request("a")) == RoutingChoice(
        "small", "rule 'named' matched: keyword 'a'"
    )


# Nothing matched, and what "can't decide" is not


def test_nothing_matched_is_no_decision_and_no_warning() -> None:
    """`None` means "can't decide", and the fallback with its warning is the router's — a
    strategy that warned too would double every notice the application sees."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        choice = domain_routing().decide(make_request("what is the weather in Kraków?"))

    assert choice is None
    assert caught == []


def test_a_request_with_no_text_matches_no_keyword() -> None:
    """An image on its own has no words, so a keyword rule abstains — while a `modality`
    rule is exactly the condition that does read it."""
    picture = make_request("", blocks=[IMAGE])

    assert domain_routing().decide(picture) is None
    assert ConfigurableStrategy([Rule("small", modality("image"), name="pictures")]).decide(
        picture
    ) == RoutingChoice("small", "rule 'pictures' matched: has image")


# What is checked at construction


BAD_CONFIGURATION: list[Any] = [
    pytest.param(
        lambda: ConfigurableStrategy([]),
        "rules is empty: a ConfigurableStrategy needs at least one rule",
        id="no rules",
    ),
    pytest.param(
        lambda: ConfigurableStrategy(Rule("small", always())),  # type: ignore[arg-type]
        "rules is Rule: a ConfigurableStrategy takes a list of Rule, "
        "such as [Rule('coder', keywords('python'))]",
        id="one rule not in a list",
    ),
    pytest.param(
        lambda: ConfigurableStrategy({"coder": ["python"]}),  # type: ignore[arg-type]
        "rules is dict: a ConfigurableStrategy takes a list of Rule, "
        "such as [Rule('coder', keywords('python'))]",
        id="a mapping",
    ),
    pytest.param(
        lambda: ConfigurableStrategy("small"),  # type: ignore[arg-type]
        "rules is str: a ConfigurableStrategy takes a list of Rule, "
        "such as [Rule('coder', keywords('python'))]",
        id="a string",
    ),
    pytest.param(
        lambda: ConfigurableStrategy([Rule("small", always()), "coder"]),  # type: ignore[list-item]
        "rule #2 is str: every rule is a Rule(route, when=...)",
        id="not a rule",
    ),
    pytest.param(
        lambda: ConfigurableStrategy(
            [Rule("coder", keywords("a"), name="x"), Rule("small", keywords("b"), name="x")]
        ),
        "two rules are named 'x': a name says which rule a decision's reason means, "
        "so each must be unique",
        id="duplicate names",
    ),
    pytest.param(
        lambda: Rule(" ", always()),
        "the route ' ' is not a route name: a rule names the route it serves",
        id="blank route",
    ),
    pytest.param(
        lambda: Rule(3, always()),  # type: ignore[arg-type]
        "the route 3 is not a route name: a rule names the route it serves",
        id="route not a string",
    ),
    pytest.param(
        lambda: Rule("small", lambda request: True),  # type: ignore[arg-type]
        "when= is function, not a condition: build one with keywords(), signal_at_least(), "
        "modality(), tools_bound(), predicate(), always(), all_of(), any_of() or not_()",
        id="a function for a condition",
    ),
    pytest.param(
        lambda: Rule("small", always(), priority=True),
        "priority must be an int, got True",
        id="priority a bool",
    ),
    pytest.param(
        lambda: Rule("small", always(), priority=1.5),  # type: ignore[arg-type]
        "priority must be an int, got 1.5",
        id="priority a float",
    ),
    pytest.param(
        lambda: Rule("small", always(), name=" "),
        "name must be a non-blank string, got ' '",
        id="blank rule name",
    ),
    pytest.param(
        lambda: keywords(),
        "keywords() needs at least one keyword or compiled pattern",
        id="no keywords",
    ),
    pytest.param(
        lambda: keywords("python", "  "),
        "the keyword '  ' is blank: a keyword needs a word",
        id="blank keyword",
    ),
    pytest.param(
        lambda: keywords(42),  # type: ignore[arg-type]
        "the keyword 42 is int: a keyword is a string, or a compiled regular expression "
        "— give several as separate arguments",
        id="not a keyword",
    ),
    pytest.param(
        lambda: keywords(["python", "regex"]),  # type: ignore[arg-type]
        "the keyword ['python', 'regex'] is list: a keyword is a string, or a compiled "
        "regular expression — give several as separate arguments",
        id="a list of keywords",
    ),
    pytest.param(
        lambda: keywords(re.compile(b"python")),  # type: ignore[arg-type]
        "the pattern b'python' is compiled from bytes: a request's text is a string, "
        "so compile the pattern from one",
        id="bytes pattern",
    ),
    pytest.param(
        lambda: signal_at_least(code_signal, 0),
        "signal_at_least needs a threshold above 0 and at most 1, got 0: a signal never "
        "scores above 1.0, and one that always reaches 0.0 is `always()`",
        id="threshold zero",
    ),
    pytest.param(
        lambda: signal_at_least(code_signal, 1.5),
        "signal_at_least needs a threshold above 0 and at most 1, got 1.5: a signal never "
        "scores above 1.0, and one that always reaches 0.0 is `always()`",
        id="threshold above one",
    ),
    pytest.param(
        lambda: signal_at_least(code_signal, float("nan")),
        "signal_at_least needs a threshold above 0 and at most 1, got nan: a signal never "
        "scores above 1.0, and one that always reaches 0.0 is `always()`",
        id="threshold not a number",
    ),
    pytest.param(
        lambda: signal_at_least(code_signal, True),
        "signal_at_least needs a threshold above 0 and at most 1, got True: a signal never "
        "scores above 1.0, and one that always reaches 0.0 is `always()`",
        id="threshold a bool",
    ),
    pytest.param(
        lambda: signal_at_least(code_signal, "high"),  # type: ignore[arg-type]
        "signal_at_least needs a threshold above 0 and at most 1, got 'high': a signal never "
        "scores above 1.0, and one that always reaches 0.0 is `always()`",
        id="threshold a string",
    ),
    pytest.param(
        lambda: signal_at_least("code", 0.5),  # type: ignore[arg-type]
        "the signal is str: it must be a function",
        id="signal not callable",
    ),
    pytest.param(
        lambda: modality(),
        "modality() needs at least one modality",
        id="no modality",
    ),
    pytest.param(
        lambda: modality("image", "imgae"),
        "unknown modality 'imgae': the modalities are 'text', 'image', 'audio', 'video', "
        "'file', 'other'",
        id="misspelt modality",
    ),
    pytest.param(
        lambda: predicate(42),  # type: ignore[arg-type]
        "the predicate is int: it must be a function",
        id="predicate not callable",
    ),
    pytest.param(
        lambda: predicate(mentions_pricing, name=""),
        "name must be a non-blank string, got ''",
        id="blank predicate name",
    ),
    pytest.param(
        lambda: all_of(),
        "all_of() has no conditions, so it would hold for every request",
        id="empty all_of",
    ),
    pytest.param(
        lambda: any_of(),
        "any_of() has no conditions, so it would hold for no request",
        id="empty any_of",
    ),
    pytest.param(
        lambda: any_of([keywords("a"), keywords("b")]),  # type: ignore[arg-type]
        "any_of() takes conditions, not list: give several as separate arguments, "
        "and wrap a function in predicate(...)",
        id="a list of conditions",
    ),
    pytest.param(
        lambda: not_(mentions_pricing),  # type: ignore[arg-type]
        "not_() takes conditions, not function: give several as separate arguments, "
        "and wrap a function in predicate(...)",
        id="a function where a condition goes",
    ),
]


@pytest.mark.parametrize(("build", "message"), BAD_CONFIGURATION)
def test_configuration_that_could_never_work_fails_at_construction(
    build: Any, message: str
) -> None:
    """Everything that can be judged from the configuration alone is judged when it is
    built — a `RoutingError` naming what is wrong, never a surprise on the first request."""
    with pytest.raises(RoutingError, match=rf"^{re.escape(message)}$"):
        build()


async def _async_signal(request: RoutingRequest) -> float:
    return 1.0


class AsyncPredicate:
    async def __call__(self, request: RoutingRequest) -> bool:
        return True


def test_an_async_function_is_refused_where_a_synchronous_one_is_called() -> None:
    """An `async def` returns a coroutine, which is truthy — so a rule built on one
    would match every request, silently. It is refused when built, as a function or as a
    callable object."""
    with pytest.raises(
        RoutingError,
        match=r"^the signal '_async_signal' is async: it must be a plain function, "
        r"because a strategy calls it while it decides$",
    ):
        signal_at_least(_async_signal, 0.5)  # type: ignore[arg-type]
    with pytest.raises(RoutingError, match=r"^the predicate 'AsyncPredicate' is async: "):
        predicate(AsyncPredicate())  # type: ignore[arg-type]


# Rules that can never fire


def signal_rules(*thresholds: float, signal: Signal = code_signal) -> list[Rule]:
    return [
        Rule(f"route-{index}", signal_at_least(signal, threshold, name="s"), name=f"at {threshold}")
        for index, threshold in enumerate(thresholds)
    ]


DEAD_RULES: list[Any] = [
    pytest.param(
        [Rule("small", always(), name="catch-all"), Rule("coder", keywords("python"))],
        "rule #2 (for 'coder') can never fire: rule 'catch-all' is tried first "
        "and matches every request",
        id="after a catch-all",
    ),
    pytest.param(
        [Rule("small", keywords("test")), Rule("coder", keywords("test"))],
        "rule #2 (for 'coder') can never fire: rule #1 is tried first and has the same "
        "condition, but sends its requests to 'small': the rules conflict, and one of them "
        "has to go",
        id="the same condition, different routes",
    ),
    pytest.param(
        [Rule("small", keywords("test")), Rule("small", keywords("test"))],
        "rule #2 (for 'small') can never fire: rule #1 is tried first and has the same "
        "condition, so one of the two is redundant",
        id="the same condition, the same route",
    ),
    pytest.param(
        [Rule("small", keywords("test")), Rule("coder", keywords("unit test"))],
        "rule #2 (for 'coder') can never fire: rule #1 is tried first, matches every "
        "request it does, and sends them to 'small' instead — give this rule a higher "
        "priority, or narrow the other",
        id="a phrase after the word it contains",
    ),
    pytest.param(
        [Rule("small", keywords("python")), Rule("small", keywords("unit test", "python"))],
        None,
        id="a broader rule after a narrower one is fine",
    ),
    pytest.param(
        [Rule("small", keywords("python")), Rule("small", keywords("python", "sql"))],
        None,
        id="an any_of that adds a keyword is not dead",
    ),
    pytest.param(
        [Rule("small", keywords("python", "sql")), Rule("small", keywords("sql"))],
        "rule #2 (for 'small') can never fire: rule #1 is tried first, matches every "
        "request it does, and sends them to the same route",
        id="one part of an any_of after the any_of",
    ),
    pytest.param(
        [
            Rule("small", keywords("python")),
            Rule("coder", all_of(keywords("python"), tools_bound())),
        ],
        "rule #2 (for 'coder') can never fire: rule #1 is tried first, matches every "
        "request it does, and sends them to 'small' instead — give this rule a higher "
        "priority, or narrow the other",
        id="an all_of after one of its parts",
    ),
    pytest.param(
        [
            Rule("coder", all_of(keywords("python"), tools_bound())),
            Rule("small", keywords("python")),
        ],
        None,
        id="the specific rule first is the way",
    ),
    pytest.param(
        signal_rules(0.3, 0.7),
        "rule 'at 0.7' (for 'route-1') can never fire: rule 'at 0.3' is tried first, "
        "matches every request it does, and sends them to 'route-0' instead — give this "
        "rule a higher priority, or narrow the other",
        id="thresholds out of order",
    ),
    pytest.param(signal_rules(0.7, 0.3), None, id="thresholds in order"),
    pytest.param(
        [
            Rule("small", signal_at_least(code_signal, 0.5, name="code")),
            Rule("coder", signal_at_least(strength(1.0), 0.9, name="other")),
        ],
        None,
        id="different signals are not comparable",
    ),
    pytest.param(
        [Rule("small", modality("image", "audio")), Rule("coder", modality("audio"))],
        "rule #2 (for 'coder') can never fire: rule #1 is tried first, matches every "
        "request it does, and sends them to 'small' instead — give this rule a higher "
        "priority, or narrow the other",
        id="a modality the earlier rule already covers",
    ),
    pytest.param(
        [Rule("small", modality("image")), Rule("coder", modality("audio"))],
        None,
        id="different modalities",
    ),
    pytest.param(
        [
            Rule("small", keywords("python")),
            Rule("coder", keywords("sql")),
            Rule("frontier", all_of(keywords("sql"), tools_bound()), name="sql with tools"),
        ],
        "rule 'sql with tools' (for 'frontier') can never fire: rule #2 is tried first, "
        "matches every request it does, and sends them to 'coder' instead — give this rule a "
        "higher priority, or narrow the other",
        id="shadowed by a rule that is not the first",
    ),
    pytest.param(
        [Rule("small", not_(keywords("a"))), Rule("coder", not_(keywords("a", "b")))],
        "rule #2 (for 'coder') can never fire: rule #1 is tried first, matches every "
        "request it does, and sends them to 'small' instead — give this rule a higher "
        "priority, or narrow the other",
        id="a narrower not_",
    ),
    pytest.param(
        [Rule("small", not_(keywords("a"))), Rule("coder", keywords("a"))],
        None,
        id="a condition and its negation",
    ),
    pytest.param(
        [Rule("small", keywords("test")), Rule("coder", keywords("unit test"), priority=1)],
        None,
        id="priority puts the specific rule first, wherever it was written",
    ),
    pytest.param(
        [Rule("coder", keywords("unit test"), priority=-1), Rule("small", keywords("test"))],
        "rule #1 (for 'coder') can never fire: rule #2 is tried first, matches every "
        "request it does, and sends them to 'small' instead — give this rule a higher "
        "priority, or narrow the other",
        id="priority can also put it last",
    ),
    pytest.param(
        [
            Rule("small", predicate(mentions_pricing)),
            Rule("coder", predicate(mentions_pricing, name="again")),
        ],
        "rule #2 (for 'coder') can never fire: rule #1 is tried first and has the same "
        "condition, but sends its requests to 'small': the rules conflict, and one of them "
        "has to go",
        id="the same predicate twice",
    ),
    pytest.param(
        [
            Rule("small", predicate(lambda request: True, name="p")),
            Rule("coder", predicate(lambda request: True, name="p")),
        ],
        None,
        id="two predicates are only the same if they are the same function",
    ),
    pytest.param(
        [
            Rule("small", keywords(re.compile("SQL"))),
            Rule("coder", keywords(re.compile("SQL", re.IGNORECASE))),
        ],
        None,
        id="a pattern with other flags is another pattern",
    ),
]


@pytest.mark.parametrize(("rules", "message"), DEAD_RULES)
def test_a_rule_that_can_never_fire_is_refused_and_one_that_can_is_not(
    rules: list[Rule], message: str | None
) -> None:
    """Conflicting and unreachable rules fail at construction. The check is sound, so a
    rule it reports is really dead — and the cases with `None` are the ones it must leave alone:
    the specific rule first, a threshold order that works, a priority that reorders."""
    if message is None:
        ConfigurableStrategy(rules)
        return

    with pytest.raises(RoutingError, match=rf"^{re.escape(message)}$"):
        ConfigurableStrategy(rules)


def test_a_rule_the_check_calls_dead_never_fires_and_the_reverse_order_does() -> None:
    """The check's claim, run rather than read: the refused order can never reach its second
    rule, and the accepted order reaches both — on a request the two rules both match."""
    request = make_request("write a unit test for this")
    specific = Rule("coder", keywords("unit test"), name="specific")
    general = Rule("small", keywords("test"), name="general")

    reached = ConfigurableStrategy([specific, general])
    assert reached.decide(request) == RoutingChoice(
        "coder", "rule 'specific' matched: keyword 'unit test'"
    )
    assert reached.decide(make_request("a test")) == RoutingChoice(
        "small", "rule 'general' matched: keyword 'test'"
    )
    with pytest.raises(RoutingError, match=r"rule 'specific' \(for 'coder'\) can never fire"):
        ConfigurableStrategy([general, specific])


# What only a request can tell: the router's route names — and failing narrowly


def test_one_mistyped_route_name_leaves_the_other_rules_working() -> None:
    """A typo costs the requests its own rule would have taken, and nothing else: matching comes
    before any check of the router's names, so the working rules keep working and only the
    mistyped one ends in a fallback. Validating every name up front would send every
    request to the default route over one bad key — a mistake a review once found."""
    strategy = ConfigurableStrategy(
        [
            Rule("coder", keywords("python"), name="code"),
            Rule("frontir", keywords("prove"), name="proofs"),
            Rule("small", keywords("hello"), name="greetings"),
        ]
    )

    assert strategy.decide(make_request("a question about python")) == RoutingChoice(
        "coder", "rule 'code' matched: keyword 'python'"
    )
    assert strategy.decide(make_request("prove there are infinitely many primes")) == RoutingChoice(
        "frontir", "rule 'proofs' matched: keyword 'prove'"
    )
    assert strategy.decide(make_request("nothing here matches a rule")) is None


def test_a_rule_set_that_names_no_route_the_router_has_raises_on_the_request() -> None:
    """The one case that can never decide anything: not a rule to fall back from, a rule set with
    no answer to give. A strategy is built before the router, so `request.routes` is its first
    sight of the real names — it says so there, and the router records it."""
    strategy = ConfigurableStrategy(
        [Rule("codr", keywords("python")), Rule("codr", keywords("sql")), Rule("frontir", always())]
    )

    with pytest.raises(
        RoutingError,
        match=(
            r"^no rule names a route this router has: the rules name 'codr', 'frontir'; "
            r"the routes are 'coder', 'frontier', 'small'$"
        ),
    ):
        # Raised although a rule matches: there is nothing this rule set could answer with.
        strategy.decide(make_request("a python question"))


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_phantom_route_reaches_the_caller_as_a_fallback_and_only_for_its_rule(
    convention: Convention, model_calls: ModelCalls
) -> None:
    """The router reports a route that doesn't exist, precisely. The mistyped rule's own
    requests fall back with the cause on the record; the requests another rule takes are
    untouched (no warning, the route answers)."""
    model_calls.expect("FakeChatModel")
    strategy = ConfigurableStrategy(
        [
            Rule("coder", keywords("python"), name="code"),
            Rule("frontir", keywords("prove"), name="proofs"),
        ]
    )
    router, routes = make_router(strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fallen_back = await respond(router, convention, "prove there are infinitely many primes")

    assert fallen_back.content == "small answer"
    assert [warning.category for warning in routing_warnings(caught)] == [FallbackWarning]
    assert str(routing_warnings(caught)[0].message) == (
        "ConfigurableStrategy chose 'frontir', which is not one of the routes; "
        "falling back to the default route 'small'"
    )
    assert routing_decision(fallen_back) == RoutingDecision(
        route="small",
        reason=(
            "ConfigurableStrategy chose 'frontir', which is not one of the routes; "
            "fell back to the default route"
        ),
        strategy="ConfigurableStrategy",
        fallback=True,
    )
    assert calls_per_route(routes) == {"coder": 0, "frontier": 0, "small": 1}

    model_calls.expect("FakeChatModel", "FakeChatModel")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        untouched = await respond(router, convention, "a question about python")

    assert untouched.content == "coder answer"
    assert routing_warnings(caught) == []


def test_rules_naming_no_route_at_all_reach_the_caller_as_a_fallback(
    model_calls: ModelCalls,
) -> None:
    """The loud case, end to end — the first request says so, with the cause and both
    name lists on the record for whoever fixes it."""
    model_calls.expect("FakeChatModel")
    strategy = ConfigurableStrategy([Rule("codr", keywords("python"))])
    router, _ = make_router(strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = router.invoke("a python question")

    assert [warning.category for warning in routing_warnings(caught)] == [FallbackWarning]
    assert routing_decision(message) == RoutingDecision(
        route="small",
        reason=(
            "ConfigurableStrategy raised RoutingError: no rule names a route this router has: "
            "the rules name 'codr'; the routes are 'coder', 'frontier', 'small'; "
            "fell back to the default route"
        ),
        strategy="ConfigurableStrategy",
        fallback=True,
    )


def test_a_predicate_that_raises_fails_only_the_request_it_ran_for(
    model_calls: ModelCalls,
) -> None:
    """A predicate is user code, checked when it runs. One that raises — or returns a
    non-bool — ends that request in the router's fallback with the cause on the record, and no
    other request: rules above it still decide theirs."""
    model_calls.expect("FakeChatModel", "FakeChatModel")
    strategy = ConfigurableStrategy(
        [
            Rule("coder", keywords("python"), name="code"),
            Rule("frontier", predicate(lambda request: 1 / 0 > 0, name="math"), name="hard"),
        ]
    )
    router, _ = make_router(strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decided = router.invoke("a python question")
        failed = router.invoke("something else")

    decision = routing_decision(failed)
    assert decision is not None
    assert decision.fallback is True
    assert decision.reason == (
        "ConfigurableStrategy raised ZeroDivisionError: division by zero; "
        "fell back to the default route"
    )
    assert routing_decision(decided) == RoutingDecision(
        route="coder",
        reason="rule 'code' matched: keyword 'python'",
        strategy="ConfigurableStrategy",
    )
    assert [warning.category for warning in routing_warnings(caught)] == [FallbackWarning]


# The transcript


def test_a_configuration_sees_the_current_request_only() -> None:
    """`wants_full_context` stays at the interface's default. A configuration can't switch it
    on — it is a class attribute the router reads before it builds the request."""
    assert ConfigurableStrategy.wants_full_context is False
    assert ConfigurableStrategy([Rule("small", always())]).wants_full_context is False


def conversation_is_long(request: RoutingRequest) -> bool:
    return request.messages is not None and len(request.messages) > 2


def test_a_predicate_sees_the_transcript_only_in_a_subclass_that_opts_in(
    model_calls: ModelCalls,
) -> None:
    """The documented way to let a rule read the conversation — a one-line subclass with
    the class attribute set. The same rules, on the same three-message conversation, route on the
    history in the subclass and don't see any in the base class."""

    class WithHistory(ConfigurableStrategy):
        wants_full_context = True

    rules = [
        Rule("frontier", predicate(conversation_is_long), name="long conversation"),
        Rule("small", always(), name="otherwise"),
    ]
    conversation = [HumanMessage("first"), HumanMessage("second"), HumanMessage("third")]
    model_calls.expect("FakeChatModel", "FakeChatModel")

    plain, _ = make_router(ConfigurableStrategy(rules))
    opted_in, _ = make_router(WithHistory(rules))

    assert routing_decision(plain.invoke(conversation)) == RoutingDecision(
        route="small",
        reason="rule 'otherwise' matched: always",
        strategy="ConfigurableStrategy",
    )
    assert routing_decision(opted_in.invoke(conversation)) == RoutingDecision(
        route="frontier",
        reason="rule 'long conversation' matched: conversation_is_long",
        strategy="WithHistory",
    )


# One more ordinary strategy


def test_the_component_is_an_ordinary_public_strategy() -> None:
    """One interface serves all three levels — the configured strategy is a concrete
    `RoutingStrategy` exported from the package, on the interface's defaults."""
    strategy = cost_tiering()

    assert isinstance(strategy, RoutingStrategy)
    assert langchain_llm_router.ConfigurableStrategy is ConfigurableStrategy
    assert "ConfigurableStrategy" in langchain_llm_router.__all__
    assert isinstance(keywords("python"), Condition)


def test_a_strategy_is_immutable_once_built() -> None:
    """The interface asks `decide` to be thread-safe: the strategy holds only its configuration,
    frozen at construction — the rules are a tuple of frozen rules, copied from the caller's
    list, so editing that list afterwards changes nothing."""
    rules = [Rule("small", keywords("a"), name="a"), Rule("coder", keywords("b"), name="b")]
    strategy = ConfigurableStrategy(rules)

    rules.clear()

    assert isinstance(strategy.rules, tuple)
    assert [rule.name for rule in strategy.rules] == ["a", "b"]
    assert strategy.decide(make_request("b")) == RoutingChoice(
        "coder", "rule 'b' matched: keyword 'b'"
    )
    with pytest.raises(FrozenInstanceError):
        strategy.rules[0].route = "coder"  # type: ignore[misc]


async def test_the_async_path_decides_the_same() -> None:
    """`adecide` is the interface's default — `decide` in a worker thread — which is safe
    because the configuration never changes; concurrent requests each get their own answer."""
    strategy = cost_tiering()
    requests = [make_request(text) for text in TIERING_TEXTS]

    answers = await asyncio.gather(*(strategy.adecide(request) for request in requests * 8))

    assert answers == [strategy.decide(request) for request in requests * 8]


# no model, API or network call


async def test_the_strategy_makes_no_model_or_network_call(
    model_calls: ModelCalls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deciding starts no LLM run and opens no socket, with every kind of condition — on
    a match, on a miss, on the async path and on a configuration that fails. The socket guard is
    what covers an embeddings or HTTP call, which LangChain's callbacks would not report at all.

    The counter is shown to be live on a real model call before it is trusted on none.
    """
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    strategy = ConfigurableStrategy(
        [
            Rule("coder", keywords("python"), name="keywords"),
            Rule("frontier", signal_at_least(length_signal(20, 200), 0.5, name="length")),
            Rule("frontier", modality("image"), name="pictures"),
            Rule("frontier", all_of(tools_bound(), not_(keywords("hello"))), name="tools"),
            Rule("small", predicate(mentions_pricing), name="pricing"),
        ]
    )

    with counting() as live:
        FakeChatModel(reply="seen").invoke("a call the counter must notice")

    assert live.started == ["FakeChatModel"]
    for request in (
        make_request("a question about python"),
        make_request(words(150)),
        make_request("what is this?", blocks=[{"type": "text", "text": "hi"}, IMAGE]),
        make_request("a question", tools=True),
        make_request("what is your pricing?"),
        make_request("nothing in the rules"),
    ):
        strategy.decide(request)
        await strategy.adecide(request)
    with pytest.raises(RoutingError):
        ConfigurableStrategy([Rule("gone", keywords("python"))]).decide(make_request("python"))
    assert model_calls.started == []


def test_the_no_call_check_catches_a_call(model_calls: ModelCalls) -> None:
    """The counter is not vacuous here — a predicate that does call a model (which the
    component itself never does; user code may) is caught, with no run around it at all."""
    model = FakeChatModel(reply="yes")
    strategy = ConfigurableStrategy(
        [Rule("small", predicate(lambda request: bool(model.invoke(request.text)), name="asks"))]
    )

    strategy.decide(make_request("hello"))

    assert model_calls.started == ["FakeChatModel"]
    # Deliberate: forgotten so this test's own call doesn't fail the module-wide check.
    model_calls.expect("FakeChatModel")
