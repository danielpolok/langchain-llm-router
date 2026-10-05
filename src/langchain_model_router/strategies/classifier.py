r"""`ClassifierStrategy`, opt-in because it makes a call: ask a small model which route fits.

`KeywordStrategy` needs the right word and `EmbeddingStrategy` needs the right idea in its
example set; `ClassifierStrategy` needs neither — it describes each route in plain language and
lets a chat model read the request and pick one. It makes an extra model call every request, so
— like `EmbeddingStrategy` — it is never on by default: construction takes the application's own
chat model, with no default. This is also the "custom strategy" template: a reader
who wants to plug in their own classifier can start from this module and change only the prompt
and the model.

Setting one up
--------------
```python
ClassifierStrategy(
    my_small_model,
    {
        "support": "account issues: password resets, billing, cancellations",
        "coder": "programming help: code, debugging, stack traces, refactors",
    },
)
```

Deciding
--------
The current request's text is the only thing classified: `request.text`, never tool output,
the system prompt or the model's answers — with `lookback`, the user's previous messages come
along as context (below). The prompt lists every route `request.routes`
and `route_descriptions` **both** name — in the router's own declaration order, not the mapping's
— together with its description, then the request; the model answers through
`with_structured_output`, the same mechanism `ChatRouter` itself uses (`_tools.py`), given a
Pydantic model built fresh for the request with one field, `route`, typed `Literal` over exactly
that list. **Unlike `KeywordStrategy` and `EmbeddingStrategy`, a route named in
`route_descriptions` that the router doesn't have is never offered to the model at all** — those
two strategies let a stray name through and have the router report the mismatch, which works
because their answer space isn't closed; here the schema *is* the answer space, so there is
nothing a stray name could contribute except an invalid choice the model was never given. The one
check that still happens is at the request, not construction, because only a request carries the
router's own routes: no overlap between `route_descriptions` and `request.routes` can never
decide anything, and `RoutingError` says so on the first `decide`, the same precedent
`KeywordStrategy` and `EmbeddingStrategy` set for a rule set that names none of the router's
routes.

The schema is rebuilt every request, not cached. Unlike `EmbeddingStrategy`'s route-example
vectors — genuine network calls, worth computing once — building a `Literal` and a `Field` is a
few microseconds of local Python; caching it would trade a line of clarity for nothing measurable
(don't build machinery a real cost doesn't justify).

**Its own call is traced for free.** A chat model *is* a `Runnable`: it takes a `config`
directly, so passing `request.config` to `invoke`/`ainvoke` is the whole of what tracing needs —
no manual `CallbackManager.configure(...)` run-opening, which is what `EmbeddingStrategy` has to
build because `Embeddings.embed_query` takes no `config` at all. LangChain's own callback
machinery reports the call as an `on_chat_model_start`/`on_chat_model_end` pair, nested wherever
the `Runnable` composition `with_structured_output` builds (a `RunnableMap` piped to an output
parser) lands it — a descendant of the strategy's run either way — and puts real `usage_metadata`
on it, so the classifier's cost falls out of an ordinary model call rather than being computed here
(contrast `EmbeddingStrategy`, whose `Embeddings` reports no usage at all and needs an estimate).
`adecide` is a native async implementation, not the base class's executor default, because that
default only nests a sync `decide` under the strategy's run via thread inheritance — an async
call needs its own `config` threaded through explicitly, exactly as `RoutingRequest.config`'s
docstring says. `tests/unit_tests/test_classifier_strategy.py` proves this empirically rather
than assuming it, the same way `tests/unit_tests/test_tracing.py`'s
`test_a_model_the_strategy_calls_nests_under_the_strategy_s_run` already does for any strategy
shaped like this one.

Deciding, and not deciding
---------------------------
- **An empty request** — no text — has nothing to classify, and abstains before any call is made,
  the same reasoning `HeuristicStrategy` and `EmbeddingStrategy` give for nothing to judge or
  embed.
- **A successful call with an answer that doesn't parse** — the model's tool call carries a
  `route` outside the `Literal`, or no tool call at all — is not an error: `with_structured_output`
  is called with `include_raw=True`, so a parse failure comes back as `parsed=None` in the result
  rather than a raised exception, and `decide`/`adecide` read that as `None` (abstain), the same
  outcome `EmbeddingStrategy` gives a score under threshold. This is deliberate, not merely what
  `include_raw` happens to do: it keeps this path a value to inspect instead of an exception to
  catch, and it is the one place this module's behaviour parts ways with `EmbeddingStrategy`'s
  "don't catch it here" — there is nothing to catch, because nothing was raised.
- **A genuine failure of the call itself** — a network error, a rate limit, a timeout, or
  `with_structured_output` raising `NotImplementedError` because the model given to this strategy
  never implements `bind_tools` — is not caught here, the same precedent `EmbeddingStrategy` sets
  for `Embeddings.embed_query`: it propagates out of `decide`/`adecide`, and the router's own
  `_conclude` is what turns it into the default route with a `FallbackWarning` naming the cause.
  A classifier model built with its own `timeout=`/`request_timeout=` needs nothing further
  from this module — the timeout surfaces as the model's own exception type, indistinguishable
  here from any other call failure, and the router's fallback already covers it.
- **A classifier model with no structured-output support at all** raises `NotImplementedError` on
  the very first request and every one after, since the schema — and so the `with_structured_output`
  call that needs it — can only be built once `request.routes` is known, at decide time, not at
  construction. That is a real, if noisy, `FallbackWarning` on every request rather than a single
  loud failure at construction; a plain-text-parse fallback for such a model is out of scope for
  v1 (the acceptance criteria ask for valid/invalid/failing coverage against a fake classifier,
  not universal real-model compatibility) and is a documented limitation, not an oversight.

Follow-ups: `lookback`
----------------------
With `lookback=N`, the user's previous N messages go into the prompt too, oldest first, marked as
earlier context, and the model classifies the latest request in their light. That is what the
rule-based strategies can't do: "this doesn't work, try again" has no signal words, and "Which
article of the civil code covers this?" has a misleading one, yet a model that reads the legal
question before it can route both. The schema is the same closed `Literal`, and a bad answer is
still `None`. Messages with no text are left out; if the latest request has none, the newest
earlier message with text is the one classified, in the light of those before it, and the reason
says how far back it was, `(1 message back)`, as every built-in strategy does. Otherwise the
reason is the same as without `lookback`: the model weighed every message, and no one of them
decided on its own. Either way `RoutingChoice.messages_back` is the distance of the message
classified, `0` for the latest request. With no earlier message to show, the prompt is exactly
the one without `lookback`.

**Cost.** The earlier messages are input tokens on every classification: up to N more messages
in each prompt, still one call per request.

**Caching.** Give the classifier model LangChain's exact cache, `cache=InMemoryCache()` (or any
`BaseCache`): the same prompt is then classified once. Every step of an agent's tool loop
classifies the same prompt, since the loop's tool calls and results take no place in it, so the
loop makes one classifier call, not one per step — and its route can't change halfway. Don't give
it a semantic cache: that would answer a request with the route of whichever earlier request read
most like it, which is `EmbeddingStrategy` done by a cache, without its threshold or its reasons.

Two things this deliberately leaves out of the prompt:

- **The route that answered the previous turn.** The model would tend to repeat it, so one
  misrouted turn would pull the next ones after it: the classifier's own past answer, fed back to
  it as evidence. Keeping to a route is a policy of its own, better written as one, such as a
  strategy that wraps this one and reads the previous route from the transcript.
- **The model's answers** (`wants_full_context`). Set on a subclass, it hands `decide` the
  transcript but changes nothing in the prompt. Arch-Router reads the whole conversation; here,
  an answer is long, so every classification would pay for it again, and it says what a route
  wrote, not what the user wants next. A subclass that wants it builds its own prompt from
  `request.messages`.

No dependency beyond `langchain-core` (and `pydantic`, which it depends on) is imported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

from pydantic import BaseModel, Field, create_model

from langchain_model_router.errors import RoutingError
from langchain_model_router.strategies._lookback import checked_lookback, considered, how_far_back
from langchain_model_router.strategy import RoutingChoice, RoutingRequest, RoutingStrategy

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = ["ClassifierStrategy"]

_SCHEMA_NAME = "RouteClassification"
"""The tool name a classifier call's structured-output binding carries in every trace and
request — fixed, since nothing about it needs to vary per request."""


class ClassifierStrategy(RoutingStrategy):
    """Routes by asking a chat model to classify the request.

    ```python
    ChatRouter(
        routes={"support": support_model, "coder": coder_model},
        default_route="support",
        strategy=ClassifierStrategy(
            my_small_model,
            {
                "support": "account issues: password resets, billing, cancellations",
                "coder": "programming help: code, debugging, stack traces, refactors",
            },
        ),
    )
    ```

    The module docstring has the reasoning behind every choice below: the prompt built from
    `route_descriptions` and the request's text, the per-request `Literal` schema, the call
    traced for free because a chat model is a `Runnable`, a bad answer turned into `None` rather
    than an exception, a genuine call failure left to propagate so the router's own
    fallback handles it, and, with `lookback`, the user's earlier messages in the prompt as
    context.

    Args:
        model: The application's own chat model, used only to classify — there is no
            default, so this strategy can never be enabled by accident. It needs no capability
            beyond `with_structured_output`; the module docstring covers what happens without one.
        route_descriptions: Each route's human-readable description, at least one. What the
            classifier prompt shows the model, and — intersected with `request.routes` at decide
            time — the closed set of answers the model is allowed to give.
        lookback: How many of the user's previous messages to put in the prompt as context,
            oldest first. Defaults to `0`, the current request alone. Each is more input tokens
            on every classification.

    Raises:
        RoutingError: for configuration that could never route — no routes, a blank route or
            description — or a `lookback` that isn't a non-negative integer. Raised here, at
            construction, not once per request.
    """

    def __init__(
        self, model: BaseChatModel, route_descriptions: Mapping[str, str], *, lookback: int = 0
    ) -> None:
        """Check `route_descriptions` and keep the classifier `model`.

        Args:
            model: The chat model used only to classify.
            route_descriptions: Each route's human-readable description.
            lookback: How many of the user's previous messages to put in the prompt.

        Raises:
            RoutingError: for configuration that could never route, or a `lookback` that isn't
                a non-negative integer.
        """
        self.model = model
        self.route_descriptions = _checked_route_descriptions(route_descriptions)
        self.lookback = checked_lookback(lookback)

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Classify `request.text`, or return `None` if there's nothing to classify.

        Also `None` when the model's answer doesn't parse. Thread-safe: the only state is
        configuration, fixed at construction; every request builds its own schema and prompt.

        Args:
            request: The current request; only its text is classified, and with `lookback` its
                previous requests' text is shown as context.

        Returns:
            The route the model chose and why, or `None` to abstain.

        Raises:
            RoutingError: if no route in `route_descriptions` is one the router has.
        """
        choices = self._choices(request.routes)
        asked = _asked(request, self.lookback)
        if asked is None:
            return None  # nothing to classify — better the default route than a guess
        classifier = self.model.with_structured_output(_schema(choices), include_raw=True)
        result = classifier.invoke(
            _prompt(choices, self.route_descriptions, asked), config=request.config
        )
        # `include_raw=True` always answers with the {"raw", "parsed", "parsing_error"} dict;
        # the return type is only a union because `with_structured_output` is typed for both.
        return self._finish(cast("dict[str, Any]", result), asked.distance)

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Async `decide`: a native implementation, as the interface asks of a model-calling one.

        `request.config` carries the tracing context an executor-wrapped `decide` would
        otherwise have to inherit from the calling thread.

        Args:
            request: The current request; only its text is classified, and with `lookback` its
                previous requests' text is shown as context.

        Returns:
            The route the model chose and why, or `None` to abstain.
        """
        choices = self._choices(request.routes)
        asked = _asked(request, self.lookback)
        if asked is None:
            return None
        classifier = self.model.with_structured_output(_schema(choices), include_raw=True)
        result = await classifier.ainvoke(
            _prompt(choices, self.route_descriptions, asked), config=request.config
        )
        return self._finish(cast("dict[str, Any]", result), asked.distance)

    def _choices(self, routes: tuple[str, ...]) -> tuple[str, ...]:
        """The routes this request can classify into.

        The router's own routes that `route_descriptions` also names, in the router's
        declaration order — the check only a request can make, same precedent as
        `KeywordStrategy`/`EmbeddingStrategy`.
        """
        choices = tuple(route for route in routes if route in self.route_descriptions)
        if choices:
            return choices
        msg = (
            f"no route in route_descriptions is one this router has: route_descriptions names "
            f"{_names(self.route_descriptions)}; the routes are {_names(routes)}"
        )
        raise RoutingError(msg)

    def _finish(self, result: dict[str, Any], distance: int) -> RoutingChoice | None:
        """Turn the classifier's answer into a choice.

        `result` is `with_structured_output(..., include_raw=True)`'s answer: `None` if
        `parsed` is `None` — a parse failure or no tool call at all, the module doc's "successful
        call, bad answer" case — otherwise the route it named, with the reason a trace shows,
        saying how far back the classified message was.
        """
        parsed = result.get("parsed")
        if parsed is None:
            return None
        route = cast("str", cast("Any", parsed).route)  # dynamic schema: no static attribute
        reason = (
            f"the classifier chose {route!r}: {self.route_descriptions[route]}"
            f"{how_far_back(distance)}"
        )
        return RoutingChoice(route=route, reason=reason, messages_back=distance)


def _checked_route_descriptions(route_descriptions: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(route_descriptions, Mapping):
        msg = (
            f"route_descriptions is {type(route_descriptions).__name__}: a ClassifierStrategy "
            "takes a mapping of route name to a human-readable description, such as "
            "{'coder': 'programming help: code, debugging, stack traces'}"
        )
        raise RoutingError(msg)
    if not route_descriptions:
        msg = "route_descriptions is empty: a ClassifierStrategy needs at least one route"
        raise RoutingError(msg)
    checked: dict[str, str] = {}
    for route, description in route_descriptions.items():
        if not isinstance(route, str) or not route.strip():
            msg = f"the route {route!r} is blank: every route needs a name"
            raise RoutingError(msg)
        if not isinstance(description, str) or not description.strip():
            msg = f"route {route!r} has a blank or non-string description: {description!r}"
            raise RoutingError(msg)
        checked[route.strip()] = description.strip()
    return MappingProxyType(checked)


def _names(names: Sequence[str] | Mapping[str, object]) -> str:
    return ", ".join(repr(name) for name in names)


# --- The per-request schema and prompt ---


def _schema(choices: Sequence[str]) -> type[BaseModel]:
    """A fresh structured-output schema for this request.

    One field, `route`, closed to `choices` — built per request because `choices` is only known
    once `request.routes` is (module doc); cheap enough that caching it would cost more than it
    saves.
    """
    return create_model(
        _SCHEMA_NAME,
        __doc__="The route that should answer this request.",
        route=(Literal[tuple(choices)], Field(description="One of the listed route names.")),
    )


class _Asked(NamedTuple):
    """The message to classify, how far back it is, and the earlier ones shown with it."""

    text: str
    distance: int
    earlier: tuple[str, ...]
    """The user's earlier messages with text, oldest first."""


def _asked(request: RoutingRequest, lookback: int) -> _Asked | None:
    """The message to classify: the newest one with text within `lookback`, with those before it.

    `None` when no message in reach has text: nothing to classify.
    """
    with_text = [
        (distance, message.text)
        for distance, message in considered(request, lookback)
        if message.text.strip()
    ]
    if not with_text:
        return None
    (distance, text), *earlier = with_text
    return _Asked(text, distance, tuple(text for _, text in reversed(earlier)))


def _prompt(choices: Sequence[str], route_descriptions: Mapping[str, str], asked: _Asked) -> str:
    """What the classifier reads: every candidate route and its description, then the request.

    With earlier messages, they come between the two, oldest first, marked as context; without
    them, the prompt is the one a strategy with no `lookback` sends.
    """
    menu = "\n".join(f"- {route}: {route_descriptions[route]}" for route in choices)
    if not asked.earlier:
        return (
            f"Classify the request below into exactly one of these routes:\n{menu}\n\n"
            f"Request:\n{asked.text}"
        )
    earlier = "\n".join(f"{number}. {text}" for number, text in enumerate(asked.earlier, 1))
    return (
        f"Classify the latest request below into exactly one of these routes:\n{menu}\n\n"
        "The user's earlier messages in this conversation, oldest first. They are context: the "
        "latest request may continue them, so read it in their light, but classify only the "
        f"latest request.\n{earlier}\n\nLatest request:\n{asked.text}"
    )
