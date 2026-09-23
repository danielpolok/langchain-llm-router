r"""`ClassifierStrategy` (R6, R7's opt-in level): ask a small model which route fits.

`KeywordStrategy` needs the right word and `EmbeddingStrategy` needs the right idea in its
example set; `ClassifierStrategy` needs neither — it describes each route in plain language and
lets a chat model read the request and pick one. It makes an extra model call every request, so
— like `EmbeddingStrategy` — it is never on by default: construction takes the application's own
chat model, with no default (REQ-R7-2). This is also §4's "custom strategy" template: a reader
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
The current request's text is the only thing classified (R4): `request.text`, never tool output,
the system prompt or the rest of the transcript. The prompt lists every route `request.routes`
and `route_descriptions` **both** name — in the router's own declaration order, not the mapping's
— together with its description, then the request; the model answers through
`with_structured_output`, the same mechanism `ChatRouter` itself uses (`_tools.py`), given a
Pydantic model built fresh for the request with one field, `route`, typed `Literal` over exactly
that list. **Unlike `KeywordStrategy` and `EmbeddingStrategy`, a route named in
`route_descriptions` that the router doesn't have is never offered to the model at all** — those
two strategies let a stray name through and have the router report the mismatch (R9), which works
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
(R8's spirit: don't build machinery a real cost doesn't justify).

**Its own call is traced for free (D9).** A chat model *is* a `Runnable`: it takes a `config`
directly, so passing `request.config` to `invoke`/`ainvoke` is the whole of what tracing needs —
no manual `CallbackManager.configure(...)` run-opening, which is what `EmbeddingStrategy` has to
build because `Embeddings.embed_query` takes no `config` at all. LangChain's own callback
machinery reports the call as an `on_chat_model_start`/`on_chat_model_end` pair, nested wherever
the `Runnable` composition `with_structured_output` builds (a `RunnableMap` piped to an output
parser) lands it — a descendant of the strategy's run either way — and puts real `usage_metadata`
on it, so REQ-R3-2 falls out of an ordinary model call rather than being computed here (contrast
`EmbeddingStrategy`, whose `Embeddings` reports no usage at all and needs an estimate).
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
  embed (R9).
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
  `_conclude` is what turns it into the default route with a `FallbackWarning` naming the cause
  (R9). A classifier model built with its own `timeout=`/`request_timeout=` needs nothing further
  from this module — the timeout surfaces as the model's own exception type, indistinguishable
  here from any other call failure, and R9's machinery already covers it.
- **A classifier model with no structured-output support at all** raises `NotImplementedError` on
  the very first request and every one after, since the schema — and so the `with_structured_output`
  call that needs it — can only be built once `request.routes` is known, at decide time, not at
  construction. That is a real, if noisy, `FallbackWarning` on every request rather than a single
  loud failure at construction; a plain-text-parse fallback for such a model is out of scope for
  v1 (the acceptance criteria ask for valid/invalid/failing coverage against a fake classifier,
  not universal real-model compatibility) and is a documented limitation, not an oversight.

No dependency beyond `langchain-core` (and `pydantic`, which it depends on) is imported (R8).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field, create_model

from langchain_llm_router.errors import RoutingError
from langchain_llm_router.strategy import RoutingChoice, RoutingRequest, RoutingStrategy

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = ["ClassifierStrategy"]

_SCHEMA_NAME = "RouteClassification"
"""The tool name a classifier call's structured-output binding carries in every trace and
request — fixed, since nothing about it needs to vary per request."""


class ClassifierStrategy(RoutingStrategy):
    """Routes by asking a chat model to classify the request (R6, R7, PRD §4).

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
    than an exception, and a genuine call failure left to propagate so the router's own R9
    machinery handles it.

    Args:
        model: The application's own chat model, used only to classify (REQ-R7-2) — there is no
            default, so this strategy can never be enabled by accident. It needs no capability
            beyond `with_structured_output`; the module docstring covers what happens without one.
        route_descriptions: Each route's human-readable description, at least one. What the
            classifier prompt shows the model, and — intersected with `request.routes` at decide
            time — the closed set of answers the model is allowed to give.

    Raises:
        RoutingError: for configuration that could never route — no routes, a blank route or
            description. Raised here, at construction, not once per request.
    """

    def __init__(self, model: BaseChatModel, route_descriptions: Mapping[str, str]) -> None:
        self.model = model
        self.route_descriptions = _checked_route_descriptions(route_descriptions)

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Classify `request.text`, or return `None` if there's nothing to classify or the
        model's answer doesn't parse (R9). Thread-safe: the only state is configuration, fixed
        at construction; every request builds its own schema and prompt."""
        choices = self._choices(request.routes)
        if not request.text.strip():
            return None  # nothing to classify — better the default route than a guess (R9)
        classifier = self.model.with_structured_output(_schema(choices), include_raw=True)
        result = classifier.invoke(
            _prompt(choices, self.route_descriptions, request.text), config=request.config
        )
        # `include_raw=True` always answers with the {"raw", "parsed", "parsing_error"} dict;
        # the return type is only a union because `with_structured_output` is typed for both.
        return self._finish(cast("dict[str, Any]", result))

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Async `decide`: a native implementation, as the interface asks of a strategy that
        calls a model (D9) — `request.config` carries the tracing context an executor-wrapped
        `decide` would otherwise have to inherit from the calling thread."""
        choices = self._choices(request.routes)
        if not request.text.strip():
            return None
        classifier = self.model.with_structured_output(_schema(choices), include_raw=True)
        result = await classifier.ainvoke(
            _prompt(choices, self.route_descriptions, request.text), config=request.config
        )
        return self._finish(cast("dict[str, Any]", result))

    def _choices(self, routes: tuple[str, ...]) -> tuple[str, ...]:
        """The routes this request can classify into: the router's own routes that
        `route_descriptions` also names, in the router's declaration order — the check only a
        request can make (R9), same precedent as `KeywordStrategy`/`EmbeddingStrategy`."""
        choices = tuple(route for route in routes if route in self.route_descriptions)
        if choices:
            return choices
        msg = (
            f"no route in route_descriptions is one this router has: route_descriptions names "
            f"{_names(self.route_descriptions)}; the routes are {_names(routes)}"
        )
        raise RoutingError(msg)

    def _finish(self, result: dict[str, Any]) -> RoutingChoice | None:
        """`result` is `with_structured_output(..., include_raw=True)`'s answer: `None` if
        `parsed` is `None` — a parse failure or no tool call at all, the module doc's "successful
        call, bad answer" case — otherwise the route it named, with the reason a trace shows."""
        parsed = result.get("parsed")
        if parsed is None:
            return None
        route = cast("str", cast("Any", parsed).route)  # dynamic schema: no static attribute
        reason = f"the classifier chose {route!r}: {self.route_descriptions[route]}"
        return RoutingChoice(route=route, reason=reason)


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
    """A fresh structured-output schema for this request: one field, `route`, closed to
    `choices` — built per request because `choices` is only known once `request.routes` is
    (module doc); cheap enough that caching it would cost more than it saves."""
    return create_model(
        _SCHEMA_NAME,
        __doc__="The route that should answer this request.",
        route=(Literal[tuple(choices)], Field(description="One of the listed route names.")),
    )


def _prompt(choices: Sequence[str], route_descriptions: Mapping[str, str], text: str) -> str:
    """What the classifier reads: every candidate route and its description, then the request."""
    menu = "\n".join(f"- {route}: {route_descriptions[route]}" for route in choices)
    return (
        f"Classify the request below into exactly one of these routes:\n{menu}\n\nRequest:\n{text}"
    )
