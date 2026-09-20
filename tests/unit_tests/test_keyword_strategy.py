"""T-130: the keyword strategy — REQ-R6-1 and REQ-R7-1.

The rules themselves are checked on the strategy alone; that it is an ordinary strategy, and
that "no match" is the router's fallback rather than the strategy's business (R9), is checked
through `ChatRouter`.

Every test in this module runs under `model_calls`, the counter REQ-R7-1 asks for: it is
installed as a LangChain configure hook, so it sees any LLM run started anywhere — including one
a strategy made without passing `request.config`. A test that legitimately calls a route says so
with `expect(...)`; every other test must end with no model call at all.
"""

from __future__ import annotations

import inspect
import re
import socket
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, NoReturn

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tracers.context import register_configure_hook

from langchain_llm_router import (
    ChatRouter,
    FallbackWarning,
    KeywordStrategy,
    RoutingChoice,
    RoutingDecision,
    RoutingError,
    RoutingRequest,
    RoutingWarning,
    routing_decision,
)
from tests.conventions import CONVENTIONS, Convention, respond
from tests.fakes import FakeChatModel, call_log

ROUTES = ("coder", "frontier", "small")


def one_line_setup() -> KeywordStrategy:
    return KeywordStrategy({"coder": ["python", "regex", "stack trace"], "frontier": ["prove"]})


def request_for(text: str, *, routes: tuple[str, ...] = ROUTES) -> RoutingRequest:
    """The current request as the router hands it to a strategy (R4)."""
    return RoutingRequest(
        text=text,
        content_blocks=HumanMessage(text).content_blocks,
        modalities=frozenset({"text"} if text else set()),
        routes=routes,
        tools_bound=False,
    )


def keyword_router() -> tuple[ChatRouter, dict[str, BaseChatModel]]:
    """§4's domain routing, end to end: a code route, a default route, and one rule set."""
    routes: dict[str, BaseChatModel] = {
        "coder": FakeChatModel(model_name="model-coder", reply="coder answer"),
        "small": FakeChatModel(model_name="model-small", reply="small answer"),
    }
    router = ChatRouter(
        routes=routes,
        default_route="small",
        strategy=KeywordStrategy({"coder": ["python", "regex"]}),
    )
    return router, routes


def routing_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [warning for warning in caught if issubclass(warning.category, RoutingWarning)]


# REQ-R7-1 · the counter every test in this module runs under


class ModelCalls(BaseCallbackHandler):
    """Every model run started while this handler is installed (REQ-R7-1's counter)."""

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


_model_calls: ContextVar[ModelCalls | None] = ContextVar("keyword_model_calls", default=None)
# A configure hook adds the current counter to every callback manager LangChain configures, so
# a call made with no config at all is still counted. Registering it is a no-op while the
# context variable is unset, which it is outside this module — but the registry is global and
# per-process: it holds this entry for the rest of the session, and every strategy suite that
# registers one of its own (T-131 does) adds another. Inert, not free.
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
    """REQ-R7-1: no test here can pass by not looking — the calls it made are asserted for it."""
    with counting() as calls:
        yield calls
        assert calls.started == calls.expected


def _refuse(*args: object, **kwargs: object) -> NoReturn:
    msg = "the strategy opened a network connection"
    raise AssertionError(msg)


async def test_the_strategy_makes_no_model_or_network_call(
    model_calls: ModelCalls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-R7-1: deciding starts no LLM run and opens no socket — on a match, on a miss, on the
    async path and on a rule set that fails. The socket guard is what covers an embeddings or
    HTTP call, which LangChain's callbacks would not report at all.

    The counter is shown to be live on a real model call before it is trusted on none.
    """
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    strategy = one_line_setup()

    with counting() as live:
        FakeChatModel(reply="seen").invoke("a call the counter must notice")

    assert live.started == ["FakeChatModel"]
    assert strategy.decide(request_for("a question about python")) is not None
    assert strategy.decide(request_for("a question about nothing in the rules")) is None
    assert await strategy.adecide(request_for("a question about python")) is not None
    with pytest.raises(RoutingError):
        KeywordStrategy({"gone": ["python"]}).decide(request_for("about python"))
    assert model_calls.started == []


# The rules: one-line setup, matching, precedence


def test_a_one_line_setup_sends_a_code_request_to_the_code_route() -> None:
    """REQ-R6-1, §4: the fast start is the rules and nothing else — one line of user code, no
    other configuration, and a reason naming the keyword that decided it (R2)."""
    setup = inspect.getsource(one_line_setup).splitlines()[1:]

    assert len(setup) == 1
    assert one_line_setup().decide(request_for("how do I do this in Python?")) == RoutingChoice(
        route="coder", reason="matched keyword 'python'"
    )


WHOLE_WORD = [
    pytest.param("python", "How do I do this in Python?", True, id="case-insensitive"),
    pytest.param("python", "in python, please", True, id="punctuation around it"),
    pytest.param("python", "write pythonic code", False, id="not inside a longer word"),
    pytest.param("python", "monty-python", True, id="hyphen is a word boundary"),
    pytest.param("  python  ", "python, please", True, id="stripped of its own spaces"),
    pytest.param("stack trace", "here is the stack trace", True, id="several words"),
    pytest.param("stack trace", "here is the stack  trace", True, id="any run of whitespace"),
    pytest.param("stack trace", "here is the stack\ntrace", True, id="across a line break"),
    pytest.param("stack trace", "a trace of the stack", False, id="in the order written"),
    pytest.param("c++", "rewrite it in c++", True, id="trailing punctuation"),
    pytest.param(".net", "the .net runtime", True, id="leading punctuation"),
    # Every character is literal: without `re.escape` each of these would be a metacharacter,
    # and the rule would quietly match something else.
    pytest.param("a.b", "the a.b file", True, id="a dot is a dot"),
    pytest.param("a.b", "axb", False, id="a dot is not any character"),
    pytest.param("(beta)", "the (beta) build", True, id="brackets are literal"),
    pytest.param("(beta)", "the beta build", False, id="brackets are not a group"),
    pytest.param("[draft]", "a [draft] spec", True, id="a class is literal"),
    pytest.param("[draft]", "a draft spec", False, id="a class is not a choice of letters"),
    pytest.param("cost*", "what is the cost* of this", True, id="a star is literal"),
    pytest.param("cost*", "what is the cost of this", False, id="a star is not a repeat"),
]


@pytest.mark.parametrize(("keyword", "text", "matches"), WHOLE_WORD)
def test_a_keyword_matches_a_whole_word_whatever_its_case(
    keyword: str, text: str, *, matches: bool
) -> None:
    """The documented default: whole words, ignoring case, every character literal, the words
    of a keyword across any whitespace — and a keyword's own punctuation edge left open, so
    `'c++'` can be a keyword while `'pythonic'` is not `'python'`."""
    expected = RoutingChoice("coder", f"matched keyword {keyword.strip()!r}") if matches else None

    assert KeywordStrategy({"coder": [keyword]}).decide(request_for(text)) == expected


def test_a_keyword_in_an_unspaced_script_needs_a_pattern() -> None:
    """A documented limit, not a bug: a word boundary needs a non-word character beside the
    keyword, and running Chinese, Japanese or Thai text never offers one. The pattern is the
    way round, as it is for every other matching rule the default doesn't fit."""
    request = request_for("我在北京工作")

    assert KeywordStrategy({"coder": ["北京"]}).decide(request) is None
    assert KeywordStrategy({"coder": [re.compile("北京")]}).decide(request) == RoutingChoice(
        route="coder", reason='matched pattern r"北京"'
    )


def test_a_compiled_pattern_is_searched_exactly_as_compiled() -> None:
    """Compiling is how a user opts into regular expressions — so the flags are the caller's
    own, and a pattern compiled without `re.IGNORECASE` stays case-sensitive."""
    signature = KeywordStrategy({"coder": re.compile(r"\bdef\s+\w+\(")})
    sensitive = KeywordStrategy({"coder": re.compile("SQL")})

    assert signature.decide(request_for("why does def parse( fail?")) == RoutingChoice(
        route="coder", reason=r'matched pattern r"\bdef\s+\w+\("'
    )
    assert sensitive.decide(request_for("this SQL query")) == RoutingChoice(
        route="coder", reason='matched pattern r"SQL"'
    )
    assert sensitive.decide(request_for("this sql query")) is None


def test_a_pattern_is_the_way_past_whole_word_matching() -> None:
    """A keyword is literal on purpose; anything cleverer is the caller's pattern to write.

    The reason keeps the pattern as its author wrote it, flags included: `repr` would double
    every backslash and drop `re.IGNORECASE`, which is half of what the rule means (R2).
    """
    strategy = KeywordStrategy({"coder": [re.compile(r"regexe?s?", re.IGNORECASE)]})

    assert KeywordStrategy({"coder": ["regex"]}).decide(request_for("two regexes")) is None
    assert strategy.decide(request_for("two Regexes")) == RoutingChoice(
        route="coder", reason='matched pattern r"regexe?s?" with re.IGNORECASE'
    )


def test_the_first_rule_in_declaration_order_wins() -> None:
    """Several rules match, so order decides — route by route, and within a route keyword by
    keyword. The same two rules the other way round give the other answer, which is why the
    specific rule goes first."""
    specific_first = KeywordStrategy({"coder": ["unit test"], "small": ["test"]})
    general_first = KeywordStrategy({"small": ["test"], "coder": ["unit test"]})
    request = request_for("write a unit test for this")

    assert specific_first.decide(request) == RoutingChoice("coder", "matched keyword 'unit test'")
    assert general_first.decide(request) == RoutingChoice("small", "matched keyword 'test'")
    assert KeywordStrategy({"coder": ["python", "test"]}).decide(request_for("a python test")) == (
        RoutingChoice("coder", "matched keyword 'python'")
    )


def test_a_single_keyword_need_not_be_a_list() -> None:
    """`{"coder": "python"}` is one keyword, not six one-character ones — the difference
    between a rule that means what it says and one that matches nearly every request."""
    strategy = KeywordStrategy({"coder": "python"})

    assert strategy.decide(request_for("in Python")) == RoutingChoice(
        "coder", "matched keyword 'python'"
    )
    assert strategy.decide(request_for("not one of the letters, though")) is None


async def test_the_async_path_decides_the_same() -> None:
    """C2: `adecide` is the interface's default — `decide` in a worker thread, which is safe
    because a strategy's compiled rules never change after construction."""
    strategy = one_line_setup()
    request = request_for("a python question")

    assert await strategy.adecide(request) == RoutingChoice("coder", "matched keyword 'python'")
    assert await strategy.adecide(request_for("nothing here")) is None


# "Can't decide" (R9) and what is checked when


def test_nothing_matched_is_no_decision_and_no_warning() -> None:
    """R9: `None` means "can't decide", and the fallback with its warning is the router's —
    a strategy that warned too would double every notice the application sees."""
    strategy = one_line_setup()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        choice = strategy.decide(request_for("what is the weather in Kraków?"))

    assert choice is None
    assert caught == []


def test_a_request_with_no_text_matches_nothing() -> None:
    """R4, C7: an image on its own carries no words, so the strategy abstains and the default
    route answers (R9) — which is why `modalities` takes no part in the rules."""
    image = HumanMessage(content=[{"type": "image", "url": "https://example.com/cat.png"}])
    request = RoutingRequest(
        text="",
        content_blocks=image.content_blocks,
        modalities=frozenset({"image"}),
        routes=ROUTES,
        tools_bound=False,
    )

    assert one_line_setup().decide(request) is None


def test_one_mistyped_route_name_leaves_the_other_rules_working() -> None:
    """A typo costs the requests its own rule would have taken, and nothing else: matching
    comes before any check of the router's names, so the working rules keep working and only
    the mistyped one ends in a fallback (R9). Validating up front would send every request to
    the default route over one bad key."""
    strategy = KeywordStrategy({"coder": ["python"], "frontir": ["prove"], "small": ["hello"]})

    assert strategy.decide(request_for("a question about python")) == RoutingChoice(
        route="coder", reason="matched keyword 'python'"
    )
    assert strategy.decide(request_for("prove there are infinitely many primes")) == RoutingChoice(
        route="frontir", reason="matched keyword 'prove'"
    )
    assert strategy.decide(request_for("nothing in this request matches a rule")) is None


def test_rules_that_name_no_route_the_router_has_raise_on_the_first_request() -> None:
    """The one case that can never decide anything: not a rule to fall back from, a rule set
    with no answer to give. A strategy is built before the router, so `request.routes` is its
    first sight of the real names — it says so there, and the router records it (R9)."""
    strategy = KeywordStrategy({"codr": ["python"], "frontir": ["prove"]})

    with pytest.raises(
        RoutingError,
        match=(
            r"^no rule names a route this router has: the rules name 'codr', 'frontir'; "
            r"the routes are 'coder', 'frontier', 'small'$"
        ),
    ):
        # Raised although a rule matches: there is nothing this rule set could answer with.
        strategy.decide(request_for("a question about python"))


BAD_RULES = [
    pytest.param({}, "rules is empty: a KeywordStrategy needs at least one rule", id="no rules"),
    pytest.param(
        [("coder", ["python"])],
        "rules is list: a KeywordStrategy takes a mapping of route name to keywords, "
        "such as {'coder': ['python', 'regex']}",
        id="not a mapping",
    ),
    pytest.param(
        {"coder": {"python", "regex"}},
        "route 'coder' is given a set: the first rule in declaration order wins, and a set "
        "has no order that survives a restart — the same rules would route the same request "
        "differently. Use a list",
        id="a set of keywords",
    ),
    pytest.param(
        {" ": ["python"]},
        "the rule key ' ' is blank: a rule names the route it routes to",
        id="blank route name",
    ),
    pytest.param(
        {"coder": []}, "route 'coder' has no keywords: a rule needs something to match", id="empty"
    ),
    pytest.param(
        {"coder": ["python", "  "]},
        "the keyword '  ' for route 'coder' is blank: a keyword needs a word",
        id="blank keyword",
    ),
    pytest.param(
        {"coder": 42},
        "route 'coder' is given int: a rule takes a keyword, a compiled regular "
        "expression, or a list of them",
        id="not keywords at all",
    ),
    pytest.param(
        {"coder": [None]},
        "the keyword None for route 'coder' is NoneType: a keyword is a string, or a "
        "compiled regular expression",
        id="not a keyword",
    ),
    pytest.param(
        {"coder": [re.compile(b"python")]},
        "the pattern for route 'coder' is compiled from bytes: a request's text is a "
        "string, so compile the pattern from one",
        id="bytes pattern",
    ),
]


@pytest.mark.parametrize(("rules", "message"), BAD_RULES)
def test_a_rule_set_that_could_never_decide_fails_at_construction(rules: Any, message: str) -> None:
    """Everything that can be judged without the router's route names is judged here, so a
    strategy that could never decide doesn't reach a request. `RoutingError` is what the
    package raises for configuration, and the message says which rule is at fault."""
    with pytest.raises(RoutingError, match=rf"^{re.escape(message)}$"):
        KeywordStrategy(rules)


# Through the router: a built-in is an ordinary strategy (REQ-R6-1)


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_the_router_takes_the_route_the_matching_rule_names(
    convention: Convention, model_calls: ModelCalls
) -> None:
    """REQ-R6-1, R2: the router runs a built-in like any other strategy — the rule's route
    answers, no other route is called, nothing warns, and the record names the rule."""
    model_calls.expect("FakeChatModel")
    router, routes = keyword_router()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "why is this regex wrong?")

    assert message.content == "coder answer"
    assert (len(call_log(routes["coder"])), len(call_log(routes["small"]))) == (1, 0)
    assert routing_warnings(caught) == []
    assert routing_decision(message) == RoutingDecision(
        route="coder", reason="matched keyword 'regex'", strategy="KeywordStrategy"
    )


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_no_matching_rule_takes_the_default_route(
    convention: Convention, model_calls: ModelCalls
) -> None:
    """R9: "can't decide" reaches the caller as the router's own fallback — the default route
    answers, exactly one `FallbackWarning` fires, and the record says who couldn't decide."""
    model_calls.expect("FakeChatModel")
    router, routes = keyword_router()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "what is the weather in Kraków?")

    assert message.content == "small answer"
    assert (len(call_log(routes["coder"])), len(call_log(routes["small"]))) == (0, 1)
    assert [warning.category for warning in routing_warnings(caught)] == [FallbackWarning]
    assert str(routing_warnings(caught)[0].message) == (
        "KeywordStrategy could not decide; falling back to the default route 'small'"
    )
    assert routing_decision(message) == RoutingDecision(
        route="small",
        reason="KeywordStrategy could not decide; fell back to the default route",
        strategy="KeywordStrategy",
        fallback=True,
    )


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_phantom_route_reaches_the_caller_as_a_fallback(
    convention: Convention, model_calls: ModelCalls
) -> None:
    """R9, R2: the router is what reports a route that doesn't exist, and it does it precisely.
    The mistyped rule's own requests fall back with the cause on the record; the requests the
    other rules take are untouched, which is the point of matching first."""
    model_calls.expect("FakeChatModel")
    routes: dict[str, BaseChatModel] = {
        "coder": FakeChatModel(reply="coder answer"),
        "small": FakeChatModel(reply="small answer"),
    }
    rules = KeywordStrategy({"coder": ["python"], "frontir": ["prove"]})
    router = ChatRouter(routes=routes, default_route="small", strategy=rules)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "prove there are infinitely many primes")

    assert message.content == "small answer"
    assert [warning.category for warning in routing_warnings(caught)] == [FallbackWarning]
    assert str(routing_warnings(caught)[0].message) == (
        "KeywordStrategy chose 'frontir', which is not one of the routes; "
        "falling back to the default route 'small'"
    )
    assert routing_decision(message) == RoutingDecision(
        route="small",
        reason=(
            "KeywordStrategy chose 'frontir', which is not one of the routes; "
            "fell back to the default route"
        ),
        strategy="KeywordStrategy",
        fallback=True,
    )


def test_rules_naming_no_route_at_all_reach_the_caller_as_a_fallback(
    model_calls: ModelCalls,
) -> None:
    """R9, R2: the loud case, end to end — a rule set that can never decide says so on the
    first request, with the cause and both name lists on the record for whoever fixes it."""
    model_calls.expect("FakeChatModel")
    routes: dict[str, BaseChatModel] = {"coder": FakeChatModel(), "small": FakeChatModel()}
    router = ChatRouter(
        routes=routes, default_route="small", strategy=KeywordStrategy({"codr": ["python"]})
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = router.invoke("a python question")

    assert [warning.category for warning in routing_warnings(caught)] == [FallbackWarning]
    assert routing_decision(message) == RoutingDecision(
        route="small",
        reason=(
            "KeywordStrategy raised RoutingError: no rule names a route this router has: "
            "the rules name 'codr'; the routes are 'coder', 'small'; "
            "fell back to the default route"
        ),
        strategy="KeywordStrategy",
        fallback=True,
    )
