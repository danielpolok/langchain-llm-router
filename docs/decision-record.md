# The decision record, warnings and errors

Every routed call records **which route ran and why**. This page is the reference for that
record, for the three ways to read it back, and for every warning and error the router raises —
the default-route fallback, tool-aware routing and forced routes each show up here as one.

## The decision record

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    route: str  # the route that ran
    reason: str  # why, in words, for a human reading a trace
    strategy: str | None  # the strategy's class name, or None (no strategy, or a forced route)
    fallback: bool  # the default-route fallback was taken: the strategy couldn't decide
    forced: bool  # the route came from runtime config
    diverted_from: str | None  # the tool-incapable route this request was diverted from
```

`RoutingDecision.as_dict()` is what rides under `response_metadata["routing"]` and on the trace.

The examples on this page use this router:

```python
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_llm_router import ChatRouter, KeywordStrategy

small = GenericFakeChatModel(messages=cycle([AIMessage("hello")]), name="small")
frontier = GenericFakeChatModel(messages=cycle([AIMessage("a proof")]), name="frontier")
router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=KeywordStrategy({"frontier": ["prove"]}),
)
messages = [HumanMessage("Prove that there are infinitely many primes.")]
```

### Reading it back

Three ways, in the order they're worth reaching for:

1. **`routing_decision(response)`** — reads the record off a response's
   `response_metadata["routing"]`. Works for `invoke`, `stream` (merged chunks carry exactly one
   copy of the record, never a concatenation), `batch` and `generate`.

   ```python
   from langchain_llm_router import routing_decision

   response = router.invoke(messages)
   decision = routing_decision(response)  # RoutingDecision | None
   ```

2. **The trace.** The same record is placed three times: as the strategy run's output, as part
   of the router's own chain-run output, and in the selected route's run metadata — so it's
   visible in LangSmith, or to any callback handler, without touching the response at all.

3. **`last_routing_decision()`** — the escape hatch for the one path where nothing comes back
   that could carry a record: `with_structured_output(Schema)` **without** `include_raw=True`
   returns a parsed object with nowhere to put `response_metadata`. Ask for `include_raw=True`
   and read the record off the raw message when you can; reach for `last_routing_decision()`
   only for the parsed-only path, and only for one call at a time — concurrent or overlapping
   calls can read back a neighbour's record (see the function's own docstring in
   `langchain_llm_router.decision` for exactly which cases that covers and which it can't).

## Warnings

Every routing warning subclasses `RoutingWarning` (a `UserWarning`), so one filter sees them
all:

```python
import warnings

from langchain_llm_router import RoutingWarning

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", RoutingWarning)
    router.invoke("hello")  # no keyword matches, so the strategy can't decide

print([str(warning.message) for warning in caught])
assert caught[0].category.__name__ == "FallbackWarning"
```

| Warning | Fires when | From |
| --- | --- | --- |
| `FallbackWarning` | The strategy failed, abstained (`decide` returned `None`), or named a route the router doesn't have. The default route answers instead, with the cause in the recorded `reason`. | default-route fallback |
| `ToolSupportWarning` | Once at bind time, naming the routes that can't use the bound tools; again on each request the router diverts away from a tool-incapable route (the default route if it's tool-capable, else the first tool-capable route in declaration order). | tool-aware routing |
| `ForcedRouteWarning` | A forced route couldn't be used and the router was built with `on_unavailable_forced_route="fallback"` — the ordinary fallback and tool-diversion rules then apply to the route that follows. | forced routes |

## Errors

`RoutingError` subclasses `ValueError`, so a construction-time failure raised inside a pydantic
validator surfaces as `pydantic.ValidationError` (pydantic wraps `ValueError`), while a call-time
failure raises the subclass directly:

| Error | Raised when | From |
| --- | --- | --- |
| `NoToolCapableRouteError` | Tools or structured output are bound and **no** route can use them. | tool-aware routing |
| `ForcedRouteError` | A forced route (`config={"configurable": {"route": ...}}`) doesn't exist, or can't use the bound tools — and the router was built with the default `on_unavailable_forced_route="error"`. | forced routes |

## Forced routes

Runtime config pins one call to a named route, skipping the strategy entirely — useful for
comparing models on the same traffic:

```python
router.invoke(messages, config={"configurable": {"route": "frontier"}})
```

A forced route is **never silently swapped**: an unknown or tool-incapable forced route raises
`ForcedRouteError` by default. Build the router with `on_unavailable_forced_route="fallback"` to
have it fall back to the default route instead, with a `ForcedRouteWarning` and the reason
recorded. See [`examples/experimentation.py`](../examples/experimentation.py) for both paths.

## Tool-aware routing

When tools or structured output are bound, the router checks every route up front:

- A route can use tools when `profile["tool_calling"]` says so, or — absent a usable profile —
  when its class overrides `BaseChatModel.bind_tools`; `tool_support_overrides` (a per-route
  `dict[str, bool]` on the router) always wins over both.
- **Binding warns up front**, naming the routes that can't use the bound tools
  (`ToolSupportWarning`); if *none* can, binding raises `NoToolCapableRouteError` instead —
  there'd be nothing left to route to.
- **A request the strategy sends to a tool-incapable route is diverted** to the default
  route if it's tool-capable, otherwise the first tool-capable route in declaration order — with
  a `ToolSupportWarning` and `diverted_from` naming the route the strategy actually picked.

`with_structured_output` follows the same rules — structured output counts as tool binding.

## Always decides

A default route is mandatory (`ChatRouter(..., default_route=...)`), and every way a strategy
can fail to decide reaches it the same way: the strategy raised, abstained (`None`), or named an
unknown route. Each case is a `FallbackWarning`, `fallback=True` on the record, and a reason
naming the cause — never a guess, and never a silently wrong answer. A **route's own** failure
is not absorbed this way: if the selected route itself raises (a network error, a rate limit),
that propagates unchanged — only the strategy's own indecision falls back.
