"""The strategy interface (T-111): REQ-R6-1, REQ-R6-2, REQ-R6-5 and REQ-R4-3.

Checked at the strategy level, without `ChatRouter` (T-110 builds it in parallel). The custom
strategies and cases are module-level so the same cases can run through `ChatRouter` once it
lands: the "custom strategy" use case as `strategy=`, REQ-R4-3 with the router passing
`wants_full_context` on, and REQ-R6-5's "the strategy run's child config" (T-117 opens that run).
"""

from __future__ import annotations

import ast
import asyncio
import functools
import importlib
import importlib.util
import inspect
import pkgutil
import threading
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, ensure_config, patch_config
from langchain_core.runnables.config import set_config_context
from langchain_core.runnables.utils import coro_with_context
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

import langchain_llm_router
from langchain_llm_router import RoutingCallable, RoutingChoice, RoutingRequest, RoutingStrategy
from langchain_llm_router._extraction import build_request
from langchain_llm_router.strategy import as_strategy, strategy_name
from tests.fakes import FakeChatModel
from tests.tracing import model_runs

PACKAGE = "langchain_llm_router"
ROUTES = ("small", "coder")


def make_request(text: str = "hello", *, config: RunnableConfig | None = None) -> RoutingRequest:
    """A current request as the router hands it over."""
    return RoutingRequest(
        text=text,
        content_blocks=HumanMessage(text).content_blocks,
        modalities=frozenset({"text"}),
        routes=ROUTES,
        tools_bound=False,
        config=config if config is not None else RunnableConfig(),
    )


# The "custom strategy" use case (PRD §4): a team's existing classifier, plugged in. User code,
# so its length is part of what REQ-R6-2 checks.


def existing_classifier(text: str) -> str:
    """Stands in for the team's own classifier."""
    return "code" if "def " in text else "chat"


def pick_route(request: RoutingRequest) -> str | None:
    return {"code": "coder", "chat": "small"}.get(existing_classifier(request.text))


class ClassifierStrategy(RoutingStrategy):
    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        label = existing_classifier(request.text)
        route = {"code": "coder", "chat": "small"}.get(label)
        return RoutingChoice(route, f"classifier said {label!r}") if route else None


CUSTOM_STRATEGIES = [
    pytest.param(pick_route, pick_route, "pick_route chose 'coder'", id="function"),
    pytest.param(ClassifierStrategy, ClassifierStrategy(), "classifier said 'code'", id="subclass"),
]


# REQ-R6-2 · a custom strategy is a few lines of user code


@pytest.mark.parametrize(("user_code", "strategy", "reason"), CUSTOM_STRATEGIES)
async def test_a_custom_strategy_is_five_lines(
    user_code: Callable[..., object], strategy: RoutingStrategy | RoutingCallable, reason: str
) -> None:
    """REQ-R6-2: the "custom strategy" use case fits in five lines, as a function and as a
    subclass, and decides the same on the sync and the async path."""
    assert len(inspect.getsource(user_code).splitlines()) <= 5

    coerced = as_strategy(strategy)
    code = make_request("def parse(): ...  why does this fail?")
    assert coerced.decide(code) == RoutingChoice("coder", reason)
    assert await coerced.adecide(code) == RoutingChoice("coder", reason)


def test_a_strategy_instance_is_used_as_given() -> None:
    """REQ-R6-2: only a plain function is wrapped; a `RoutingStrategy` passes through as is."""
    strategy = ClassifierStrategy()

    assert as_strategy(strategy) is strategy


def test_a_function_returns_a_choice_a_route_name_or_none() -> None:
    """REQ-R6-2: a plain function is coerced; a bare route name gets a reason written for it,
    a `RoutingChoice` passes through unchanged, and `None` abstains (R9)."""
    choice = RoutingChoice("coder", "mine")
    request = make_request()

    by_name = as_strategy(lambda request: "coder")
    assert isinstance(by_name, RoutingStrategy)
    assert by_name.decide(request) == RoutingChoice("coder", "<lambda> chose 'coder'")
    assert as_strategy(lambda request: choice).decide(request) is choice
    assert as_strategy(lambda request: None).decide(request) is None


def test_a_function_that_returns_anything_else_fails_clearly() -> None:
    """REQ-R6-2: an unusable return is a `TypeError` naming the function — which the router
    reports like any other strategy failure (R9) rather than routing on it."""
    strategy = as_strategy(lambda request: 42)  # type: ignore[arg-type,return-value]

    with pytest.raises(
        TypeError,
        match=r"^strategy '<lambda>' returned int; expected a RoutingChoice, a route name or None$",
    ):
        strategy.decide(make_request())


@pytest.mark.parametrize("not_a_strategy", ["coder", 42])
def test_what_is_neither_a_strategy_nor_a_function_is_rejected(not_a_strategy: object) -> None:
    """REQ-R6-2: `strategy=` takes a strategy or a function, nothing else."""
    kind = type(not_a_strategy).__name__

    with pytest.raises(
        TypeError, match=rf"^strategy must be a RoutingStrategy or a callable, not {kind}$"
    ):
        as_strategy(not_a_strategy)  # type: ignore[arg-type]


def test_a_strategy_class_must_be_instantiated() -> None:
    """REQ-R6-2: a class is callable, so without this check it would be called with each
    request and fail on every call instead of once, here."""
    with pytest.raises(TypeError, match=r"pass ClassifierStrategy\(\), not the class"):
        as_strategy(ClassifierStrategy)  # type: ignore[arg-type]


async def async_pick(request: RoutingRequest) -> str:
    return "coder"


class AsyncClassifier:
    async def __call__(self, request: RoutingRequest) -> str:
        return "coder"


@pytest.mark.parametrize(
    "func",
    [async_pick, functools.partial(async_pick), AsyncClassifier()],
    ids=["async-def", "partial", "async-call"],
)
def test_an_async_function_is_rejected_when_coerced(func: object) -> None:
    """REQ-R6-2: `RoutingCallable` is synchronous. An async function fails at coercion — at the
    router's construction — pointing at `adecide`, instead of on every sync call."""
    with pytest.raises(TypeError, match=r"subclass RoutingStrategy and override adecide instead$"):
        as_strategy(func)  # type: ignore[arg-type]


def pick_by_length(request: RoutingRequest, *, threshold: int) -> str:
    return "coder" if len(request.text) > threshold else "small"


class Classifier:
    def __call__(self, request: RoutingRequest) -> str:
        return "small"

    def route(self, request: RoutingRequest) -> str:
        return "small"


@pytest.mark.parametrize(
    ("strategy", "name"),
    [
        pytest.param(ClassifierStrategy(), "ClassifierStrategy", id="subclass"),
        pytest.param(pick_route, "pick_route", id="function"),
        pytest.param(lambda request: "small", "<lambda>", id="lambda"),
        pytest.param(
            functools.partial(pick_by_length, threshold=9), "pick_by_length", id="partial"
        ),
        pytest.param(
            functools.partial(functools.partial(pick_by_length), threshold=9),
            "pick_by_length",
            id="nested-partial",
        ),
        pytest.param(Classifier(), "Classifier", id="callable-object"),
        pytest.param(Classifier().route, "route", id="bound-method"),
    ],
)
def test_a_strategy_is_named_after_its_class_or_function(
    strategy: RoutingStrategy | RoutingCallable, name: str
) -> None:
    """REQ-R6-2, D8: the name the decision record and the strategy's run carry — a function's
    own name, seen through `functools.partial`, rather than `partial` or `_CallableStrategy`."""
    assert strategy_name(as_strategy(strategy)) == name


# Sync and async paths: `adecide`'s executor default (C2)


class BlockingStrategy(RoutingStrategy):
    """Blocks in `decide` until released by a coroutine, which runs only if the loop is free."""

    def __init__(self) -> None:
        self.released = threading.Event()
        self.thread: int | None = None

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        self.thread = threading.get_ident()
        released = self.released.wait(timeout=5)
        return RoutingChoice("small", "released" if released else "blocked the event loop")


async def test_adecide_runs_decide_off_the_event_loop() -> None:
    """C2: the default `adecide` runs a blocking `decide` in a worker thread, so a task running
    alongside it on the loop is not blocked."""
    strategy = BlockingStrategy()

    async def release() -> None:
        strategy.released.set()

    choice, _ = await asyncio.gather(strategy.adecide(make_request()), release())

    assert choice == RoutingChoice("small", "released")
    assert strategy.thread != threading.get_ident()


request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


class ContextSpy(RoutingStrategy):
    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        self.seen = {"request_id": request_id.get(), "tags": ensure_config().get("tags")}
        return None


async def test_adecide_carries_the_callers_context_into_decide() -> None:
    """D9: the context the router sets around `adecide` — LangChain's config context
    (`set_config_context`) and any other context variable — reaches `decide`'s thread."""
    strategy = ContextSpy()
    config = RunnableConfig(tags=["strategy-run"])

    with set_config_context(config) as context:
        context.run(request_id.set, "request-1")
        await coro_with_context(strategy.adecide(make_request(config=config)), context)

    assert strategy.seen == {"request_id": "request-1", "tags": ["strategy-run"]}


async def test_adecide_raises_what_decide_raises() -> None:
    """R9: the executor hands `decide`'s own exception back, so the router can record why the
    strategy failed."""

    class Broken(RoutingStrategy):
        def decide(self, request: RoutingRequest) -> RoutingChoice | None:
            raise LookupError("classifier offline")

    with pytest.raises(LookupError, match=r"^classifier offline$"):
        await Broken().adecide(make_request())


# REQ-R6-5 · the config a strategy's own calls must use


def test_config_takes_no_part_in_equality_or_repr() -> None:
    """REQ-R6-5: two requests that differ only in `config` compare equal, and neither repr
    shows it — it identifies a run, not a request."""
    first = make_request(
        config=RunnableConfig(
            tags=["run-1"], run_id=uuid4(), callbacks=[RunCollectorCallbackHandler()]
        )
    )
    second = make_request(config=RunnableConfig(tags=["run-2"]))

    assert first.config != second.config
    assert first == second
    assert repr(first) == repr(second)
    assert "config" not in repr(first)
    assert first != replace(first, text="goodbye")


def test_a_request_is_frozen_but_not_hashable() -> None:
    """REQ-R6-5, decision: a request is frozen, but its list fields (as pinned) make it
    unhashable — said plainly, not by a generated hash failing on a list. A choice hashes."""
    request = make_request()

    with pytest.raises(FrozenInstanceError):
        request.text = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match=r"^unhashable type: 'RoutingRequest'$"):
        hash(request)
    assert len({RoutingChoice("small", "short"), RoutingChoice("small", "short")}) == 1


class ModelBackedStrategy(RoutingStrategy):
    """Asks a model, passing `request.config` on as the interface says to (D9)."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        answer = self.model.invoke(request.text, config=request.config)
        return RoutingChoice(answer.text, "the model said so")


@pytest.mark.parametrize("path", ["decide", "adecide"])
async def test_a_call_given_request_config_nests_under_the_strategy_run(path: str) -> None:
    """REQ-R6-5: `config` is how a strategy's own model call joins the strategy's run (D9) —
    on the executor's async path too, with no context set around it (Python 3.10's case)."""
    collector = RunCollectorCallbackHandler()
    manager = CallbackManager.configure(inheritable_callbacks=[collector])
    strategy_run = manager.on_chain_start(None, {}, name="ModelBackedStrategy")
    request = make_request(config=patch_config(None, callbacks=strategy_run.get_child()))
    strategy = ModelBackedStrategy(FakeChatModel(reply="small"))

    if path == "decide":
        choice = strategy.decide(request)
    else:
        choice = await strategy.adecide(request)
    strategy_run.on_chain_end({})

    assert choice == RoutingChoice("small", "the model said so")
    (root,) = collector.traced_runs
    assert root.id == strategy_run.run_id
    assert [run.parent_run_id for run in model_runs([root])] == [root.id]


# REQ-R4-3 · wider context is an explicit opt-in


class FullContextStrategy(RoutingStrategy):
    wants_full_context = True

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        return None


TRANSCRIPT: list[BaseMessage] = [
    SystemMessage("You are terse."),
    HumanMessage("Why does my parser fail?"),
    AIMessage("", tool_calls=[{"name": "read_file", "args": {"path": "p.py"}, "id": "call-1"}]),
    ToolMessage("def parse(): ...", tool_call_id="call-1"),
]


@pytest.mark.parametrize(
    ("strategy", "messages"),
    [
        pytest.param(ClassifierStrategy(), None, id="subclass"),
        pytest.param(as_strategy(pick_route), None, id="function"),
        pytest.param(FullContextStrategy(), TRANSCRIPT, id="opted-in"),
    ],
)
def test_messages_only_for_a_strategy_that_opts_in(
    strategy: RoutingStrategy, messages: list[BaseMessage] | None
) -> None:
    """REQ-R4-3: `messages` is `None` unless the strategy sets `wants_full_context = True`."""
    request = build_request(
        TRANSCRIPT,
        routes=ROUTES,
        tools_bound=False,
        wants_full_context=strategy.wants_full_context,
        config=RunnableConfig(),
    )

    assert RoutingStrategy.wants_full_context is False
    assert request is not None
    assert request.messages == messages


# REQ-R6-1 · one interface for all three levels; the built-ins use no private hooks
#
# No built-in exists yet (T-130 to T-134), and "no private hooks" can't be checked by running
# one. It is approximated from both sides: the interface has nothing private to hook into, a
# built-in imports nothing from the package core that a custom strategy can't, and the core
# never names a built-in — so it can only reach one through `RoutingStrategy`.


def _published_strategies() -> list[type[RoutingStrategy]]:
    """Every public `RoutingStrategy` subclass the package ships. Built-ins join as they land."""
    modules = [langchain_llm_router]
    if importlib.util.find_spec(f"{PACKAGE}.strategies") is not None:
        package = importlib.import_module(f"{PACKAGE}.strategies")
        modules.append(package)
        modules += [
            importlib.import_module(info.name)
            for info in pkgutil.iter_modules(package.__path__, f"{package.__name__}.")
        ]
    found = {
        value
        for module in modules
        for name, value in vars(module).items()
        if not name.startswith("_")
        and inspect.isclass(value)
        and issubclass(value, RoutingStrategy)
        and value is not RoutingStrategy
    }
    return sorted(found, key=lambda cls: f"{cls.__module__}.{cls.__qualname__}")


BUILTINS = [pytest.param(cls, id=cls.__name__) for cls in _published_strategies()] or [
    pytest.param(
        None, id="none-yet", marks=pytest.mark.skip(reason="no built-in yet: T-130 to T-134")
    )
]


def _core_imports(source: str) -> set[str]:
    """What a `strategies/` module imports from the package outside `strategies/`."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            absolute = module.split(".")[0] == PACKAGE and not module.startswith(
                f"{PACKAGE}.strategies"
            )
            if node.level >= 2 or (node.level == 0 and absolute):
                imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(
                alias.name
                for alias in node.names
                if alias.name.startswith(f"{PACKAGE}.")
                and not alias.name.startswith(f"{PACKAGE}.strategies")
            )
    return imported


def _identifiers(source: str) -> set[str]:
    """The names a module's code uses — not its docstrings or comments."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.update(node.name.split("."))
    return names


def test_the_interface_has_no_private_hooks() -> None:
    """REQ-R6-1: `RoutingStrategy` is `decide`, `adecide` and `wants_full_context` — nothing
    hidden for a built-in to hook into that a custom strategy can't see."""
    members = {
        name
        for name in vars(RoutingStrategy)
        if not (name.startswith("__") and name.endswith("__")) and name != "_abc_impl"
    }

    assert members == {"decide", "adecide", "wants_full_context"}


@pytest.mark.parametrize("cls", BUILTINS)
def test_a_builtin_is_a_public_strategy_like_any_other(cls: type[RoutingStrategy]) -> None:
    """REQ-R6-1: every built-in is a concrete `RoutingStrategy`, exported from the package, and
    imports from the package core only what the package exports to anyone."""
    public = set(langchain_llm_router.__all__)

    assert issubclass(cls, RoutingStrategy)
    assert not inspect.isabstract(cls)
    assert cls.__name__ in public
    assert getattr(langchain_llm_router, cls.__name__) is cls
    assert _core_imports(Path(inspect.getfile(cls)).read_text(encoding="utf-8")) - public == set()


@pytest.mark.parametrize("cls", BUILTINS)
def test_the_core_never_names_a_builtin(cls: type[RoutingStrategy]) -> None:
    """REQ-R6-1: no module outside `strategies/` (bar the package's re-exports) refers to a
    built-in, so the router can only reach it through the `RoutingStrategy` interface."""
    core = Path(langchain_llm_router.__file__).parent

    naming = [
        path.name
        for path in sorted(core.glob("*.py"))
        if path.name != "__init__.py"
        and cls.__name__ in _identifiers(path.read_text(encoding="utf-8"))
    ]

    assert naming == []


def test_the_builtin_checks_catch_what_they_look_for() -> None:
    """REQ-R6-1: the two source checks above are not vacuous while no built-in exists."""
    source = (
        "from langchain_llm_router import RoutingChoice, RoutingStrategy\n"
        "from langchain_llm_router._extraction import build_request\n"
        "from ..strategy import as_strategy\n"
        "from .configurable import ConfigurableStrategy\n"
        "import langchain_llm_router.decision\n"
        "import langchain_llm_router.strategies.keyword as keyword\n"
        "isinstance(strategy, keyword.KeywordStrategy)\n"
    )

    assert _core_imports(source) - set(langchain_llm_router.__all__) == {
        "build_request",
        "as_strategy",
        "langchain_llm_router.decision",
    }
    assert {"KeywordStrategy", "ConfigurableStrategy"} <= _identifiers(source)
    assert "KeywordStrategy" not in _identifiers('"""Unlike KeywordStrategy, ..."""\n')
