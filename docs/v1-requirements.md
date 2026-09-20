# v1 requirements

The step v0 deferred: [PRD §3](../PRD.md#3-phase-v0-principles)'s 21 principles turned into
numbered, testable requirements (T-101). The PRD says *what* the router is; this says what v1
must build and how each requirement is checked.

**Checked against** `langchain-core` 1.6.3, `langchain` 1.4.0 — the versions the spike was built
on. Line references are to that `langchain_core`.

## How to read this

- **`REQ-<principle>-<n>`** — one requirement, owned by exactly one task. Tasks are GitHub issues
  titled `T-NNN · …`; [tasks/README.md](../tasks/README.md) maps each ID to its issue. The principle in the ID
  is the one it comes from; a requirement often serves others too.
- **Check** — phrased as the assertion a test makes. If it can't be written as one, it isn't a
  requirement yet.
- **Settled rules (D1–D9)** are the decisions the principles left open. They are recorded in
  [PRD §11](../PRD.md#11-key-decisions) and are not re-debated in the tasks that implement them.
- **Not repeated here:** the principles themselves (PRD §3), what the spike proved
  ([spike findings](spike-findings.md)), or the C3 and R3/C5 verdicts
  ([T-003](../tasks/T-003-spike-tools-structured-output.md),
  [T-004](../tasks/T-004-spike-cost-and-tracing.md)).

## Public API

The surface T-110 and T-111 build. Signatures are pinned here so the tasks after them can't drift.

```python
class ChatRouter(BaseChatModel):
    routes: dict[str, BaseChatModel]
    """Named routes, any number of them; declaration order is significant (D1)."""

    default_route: str
    """Mandatory (R9). Must name one of `routes`."""

    strategy: RoutingStrategy | RoutingCallable | None = None
    """None always uses the default route. A plain callable is coerced (REQ-R6-2)."""

    on_unavailable_forced_route: Literal["error", "fallback"] = "error"
    """R11: what happens when a forced route is unknown or can't use bound tools."""

    tool_support_overrides: dict[str, bool] = {}
    """Per-route override of capability detection (D5); always wins."""
```

The strategy interface (R6) — small and stable, one method:

```python
@dataclass(frozen=True)
class RoutingRequest:
    text: str  # the current request's text (R4, D6)
    content_blocks: list[ContentBlock]  # as LangChain defines them (C7)
    modalities: frozenset[str]  # "text", "image", "audio", "video", "file", "other" (T-112)
    routes: tuple[str, ...]  # available route names, declaration order
    tools_bound: bool  # tools or structured output are bound
    messages: list[BaseMessage] | None  # only when wants_full_context (R4)
    config: RunnableConfig  # pass on to any model or embeddings call (D9)


@dataclass(frozen=True)
class RoutingChoice:
    route: str
    reason: str


class RoutingStrategy(ABC):
    wants_full_context: ClassVar[bool] = False

    @abstractmethod
    def decide(self, request: RoutingRequest) -> RoutingChoice | None: ...

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Defaults to `decide` in an executor, so async callers never block (C2)."""


RoutingCallable = Callable[[RoutingRequest], RoutingChoice | str | None]
```

`config` is excluded from `RoutingRequest`'s equality and repr: it identifies a run, not a
request. A strategy that calls a model passes it on — `model.invoke(prompt, config=request.config)`
— so the call is traced and costed under the strategy's run (D9).

Exports from `langchain_llm_router`: `ChatRouter`, `RoutingStrategy`, `RoutingRequest`,
`RoutingChoice`, `RoutingDecision`, `routing_decision`, `last_routing_decision`, the warning and
error types below, and the built-in strategies (T-130–T-134).

## Decision record (D8)

```python
@dataclass(frozen=True)
class RoutingDecision:
    route: str  # the route that ran
    reason: str  # why, in words, for a human reading a trace
    strategy: str | None  # strategy class name, or None when none ran
    fallback: bool  # R9 path was taken
    forced: bool  # the route came from runtime config (R11)
    diverted_from: str | None  # the tool-incapable route it was diverted from (R10)
```

`as_dict()` is what rides under `response_metadata["routing"]`. The same dict is on the trace
three times (D9): as the strategy run's output, as part of the router run's output, and in the
selected route's run metadata. `routing_decision(message)` reads it back off a response.

## Warnings and errors

```
RoutingWarning(UserWarning)                 # base — catch this to see every routing warning
├── FallbackWarning        # R9  — the strategy failed, abstained, or named an unknown route
├── ToolSupportWarning     # R10 — the bind-time notice, and one per diverted request
└── ForcedRouteWarning     # R11 — a forced route gave way (only under "fallback")

RoutingError(ValueError)                    # base — configuration and forced-route failures
├── NoToolCapableRouteError                 # R10 — bind_tools when no route can use tools
└── ForcedRouteError                        # R11 — unknown or tool-incapable forced route
```

`RoutingError` subclasses `ValueError` so that construction-time failures raised inside pydantic
validators surface as `pydantic.ValidationError` (pydantic wraps `ValueError`), while call-time
failures raise the `RoutingError` subclass directly. Tests assert accordingly.

## Settled rules

| # | Rule | Why |
| --- | --- | --- |
| **D1** | A request diverted off a tool-incapable route goes to the **default route if it is tool-capable, otherwise the first tool-capable route in declaration order** (R10). | Deterministic and explainable. Re-running the strategy would spend a call for the opt-in strategies (R7) and could divert in a loop. |
| **D2** | A forced route **skips the strategy entirely** (R11). | Running a strategy whose answer is discarded costs money and trace space under T-133/T-134, and R11 exists because a forced route is not a suggestion. |
| **D3** | The decision always reaches the **trace** (placement per D9); on messages it reaches `response_metadata`, and under `with_structured_output` only when `include_raw=True`. The parsed-object path carries no in-band record — `last_routing_decision()` (a context variable set per call) is the escape hatch. | A parsed Pydantic object has nowhere to put it (T-003 caveat 2). The trace is the one placement that is always available. |
| **D4** | **The route owns response caching.** The router delegates from `invoke`/`stream`, so its own `_generate_with_cache` never runs; the selected route's cache applies, keyed on that route's identity. The router's own `cache=` is rejected at construction rather than silently ignored. | `_generate_with_cache` (`chat_models.py:1892`) looks up *before* `_generate`, keyed by `_get_llm_string` (`:1578`) — the model's serialized repr plus call kwargs. Because the key is the route's, C10's "never reused for a different route" holds structurally instead of by arithmetic we maintain. |
| **D5** | Tool capability is `profile["tool_calling"]` when the route reports a profile; otherwise whether the route's class overrides `BaseChatModel.bind_tools`; `tool_support_overrides` always wins. | `BaseChatModel.bind_tools` raises `NotImplementedError` (`chat_models.py:2383`) at *call* time only — the late failure R10 exists to pre-empt. `profile` is beta and may be absent, so it can't be the only signal. |
| **D6** | One strategy interface: `decide` / `adecide` over a `RoutingRequest`, returning a `RoutingChoice` or `None` for "can't decide". Wider context only when the strategy sets `wants_full_context`. The request also carries the run config its calls must use (D9). | R6's three levels must be one interface (ready-made, configured and custom alike), and R4's default must be the *current request*, with more context an opt-in the strategy declares. |
| **D7** | Forced routes travel under the configurable key **`"route"`**, declared through `config_specs`. | Makes `with_config`, `config={"configurable": …}` and config-schema introspection all work through the standard mechanism (C4); `ConfigurableFieldSpec`, `runnables/utils.py:655`. It is what `README.md` already advertises. |
| **D8** | The decision record is the six-field schema above, and exactly **one** streamed chunk carries it. Where it sits in the trace is D9's. | `merge_dicts` concatenates strings that repeat across chunks, so a record on every chunk aggregates to `"frontierfrontier…"` (spike surprise 6). |
| **D9** | **The router opens its run before it decides, and the strategy runs in a child run of its own.** The strategy run's output is the decision; any model or embeddings call the strategy makes nests under it. The strategy gets that run's child config explicitly as `RoutingRequest.config`, and the router also runs `decide` / `adecide` inside `set_config_context` (sync: the returned context's `run`; async: `coro_with_context`), so a call that doesn't pass the config still nests wherever context propagation works. Built-in strategies always pass it. No strategy run when there is no strategy or the route is forced (D2). Because deciding now happens after the router run starts, the record can't ride in that run's start metadata: it goes on the strategy run's output, the router run's output (`on_chain_end`) and the route run's inherited metadata. | Found after T-101 closed (2026-09-19). A strategy that calls a model needs a parent run, and the spike's design only opened one *after* deciding, so the classifier's call would land as a sibling or root run and R3's "costed to the strategy's own model" couldn't be checked in a trace. Context propagation alone isn't enough: LangChain's docs say that below Python 3.11 an async call must be given its `RunnableConfig`, or its callbacks don't propagate, and the package supports 3.10. Passing the config explicitly works on every supported version; setting the context covers custom code that forgets to. |

## Requirements

### C1 · It is a chat model

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-C1-1 | `ChatRouter` is a `BaseChatModel` and passes `langchain_tests`' standard unit suite. | `ChatModelUnitTests` passes; every skipped standard test has a documented reason. | T-120 |
| REQ-C1-2 | The router works in each place a chat model goes. | Offline placement tests with fake routes: LCEL chain, `create_agent(model=router)`, LangGraph node, message-history wrapper. | T-120 |
| REQ-C1-3 | Integration suite passes with real routes. | `ChatModelIntegrationTests` against Gemini + Ollama routes, skipped without credentials. | T-120 |

### C2 · Same calling conventions

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-C2-1 | `invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch` and `astream_events` all work, with no router-specific call form. | Each convention returns the selected route's output; merged stream chunks equal the `invoke` output for a deterministic fake route. | T-113 |
| REQ-C2-2 | `generate()` / `agenerate()` route, record and cost exactly as `invoke` does. | Usage totals and decision record match `invoke`; no second LLM run. *(Spike caveat: these still take the base path and would double count.)* | T-117 |
| REQ-C2-3 | Async paths await both the strategy and the route; neither blocks the event loop. | A strategy whose `adecide` sleeps does not block a concurrently running task; `abatch` of N requests overlaps. | T-113 |
| REQ-C2-4 | A route without native streaming still streams through the router. | Streaming a fake route that implements only `_generate` yields one chunk equal to the full message. | T-113 |

### C3 · Tools and structured output

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-C3-1 | `bind_tools` keeps tools unconverted until a route is chosen; the *route's* `bind_tools` converts at call time. `tool_choice` and binding kwargs (`strict=`) are replayed on the route's binder, not passed as call kwargs. | Two fake routes with different conversions each receive the tools in their own form; `strict=` reaches the route's `bind_tools`, not its call kwargs. | T-115 |
| REQ-C3-2 | The raw `tools` list is also bound in its usual place, so LangChain's own checks behave as on any chat model. | `disable_streaming="tool_calling"` on a route behaves identically bound through the router. | T-115 |
| REQ-C3-3 | `with_structured_output` overrides the base default and forwards per request to the selected route's own implementation. | `method=` and `strict=` reach the route (the base default drops them, `chat_models.py:2530`); `include_raw=True` returns the usual `{"raw", "parsed", "parsing_error"}`. | T-115 |
| REQ-C3-4 | The router reports a `profile` that is the intersection of its routes' profiles. | Booleans AND-ed, ints minimised, equal values kept, differing values dropped, `None` when any route reports none; `create_agent(response_format=…)` picks a strategy every route can serve. | T-121 |

### C4 · Runtime configuration

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-C4-1 | The forced route is exposed through standard runtime configuration under the key `route` (D7). | `config_specs` lists the spec; `router.invoke(msgs, config={"configurable": {"route": "x"}})` and `router.with_config(configurable={"route": "x"})` both force it. | T-116 |
| REQ-C4-2 | All other runtime configuration reaches the route unchanged. | `tags`, `metadata`, `run_name`, `callbacks` and `max_concurrency` set on the router's config are observed on the route's run. | T-116 |

### C5 · Traces

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-C5-1 | The router's own run is a **chain** run whose output carries the decision; the selected route's call is the only LLM run outside the strategy run, nested inside the router run. | Offline tracer, zero-call strategy: exactly one `on_chat_model_start`, its parent the router's chain run; the router run's outputs and the route run's metadata contain the record. | T-117 |
| REQ-C5-2 | That shape holds for every calling convention. | The run-tree assertion is parametrised over invoke, ainvoke, stream, astream, batch and generate. | T-117 |
| REQ-C5-3 | LangGraph's `stream_mode="messages"` yields the route's tokens exactly once. | A real graph with the router as a node streams one token stream, not two or zero. *(The router's run being a chain run changes what registers on `on_chat_model_start`.)* | T-113 |
| REQ-C5-4 | The shape is confirmed live, not only against the offline tracer. | One LangSmith trace per convention shows the decision and the real call. | T-117 |
| REQ-C5-5 | A strategy runs in a child run of the router's run, and any model or embeddings call it makes nests under that strategy run (D9). | Offline tracer with a strategy that calls a fake chat model: router run → strategy run → strategy's LLM run, and router run → route's LLM run; parametrised over sync and async, and run on Python 3.10 in CI. A custom strategy that omits `config` still nests on sync paths and on Python ≥ 3.11. | T-117 |

### C6 · Errors, retries and fallbacks stay LangChain's job

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-C6-1 | A route's exception reaches the caller as the same exception, with no router-level retry or model fallback. | `pytest.raises(RouteSpecificError)` with identical type and args; the route was called once. | T-118 |
| REQ-C6-2 | `with_retry` and `with_fallbacks` behave on the router as on a model, and routes may themselves be wrapped. | `router.with_retry()` retries; `router.with_fallbacks([other])` falls over; a route wrapped in `with_retry` retries inside the router. | T-118 |
| REQ-C6-3 | A route failure closes the router's chain run as an error. | The tracer records `on_chain_error` on the router's run, and no dangling open run. | T-118 |

### C7 · Messages as LangChain defines them

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-C7-1 | Extraction reads string content and content blocks, joins text, and exposes non-text modalities so a strategy can route on them. | A text+image request yields the text in `RoutingRequest.text` and `"image"` in `modalities`. | T-112 |
| REQ-C7-2 | Every input form a chat model accepts reaches extraction identically. | A string, a list of dicts, `BaseMessage` objects and a `ChatPromptValue` produce the same `RoutingRequest`. | T-112 |

### C8 · Complements agent middleware

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-C8-1 | The router coexists with `@wrap_model_call` middleware inside `create_agent`. | An agent with both runs to completion; the middleware sees the router as its model and routing still happens. | T-120 |
| REQ-C8-2 | The documentation says when to use middleware instead. | The docs page exists and its example runs in CI. | T-150 |

### C9 · LangChain 1.x

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-C9-1 | Supported range is `langchain-core>=1.1,<2`, tested at both ends. | CI matrix runs the suite on the minimum supported and the latest 1.x. | T-120 |
| REQ-C9-2 | No private `langchain-core` API is used. | The implementation imports only documented, public names; any exception is listed with its justification. | T-151 |

### C10 · Response cache

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-C10-1 | A repeated identical request is a hit; anything that would change which route answers is a miss. | With `InMemoryCache`: identical request hits; the same prompt forced to another route, under a changed strategy config, or with different bound tools, misses. | T-119 |
| REQ-C10-2 | The router's own `cache=` never silently no-ops. | Constructing `ChatRouter(cache=…)` raises, with a message pointing at per-route caching (D4). | T-119 |
| REQ-C10-3 | Tool binding contributes nothing process-unstable to the cache key. | The key for the same tools is equal across two processes — bare functions must not reach it as `<function f at 0x…>`. | T-119 |
| REQ-C10-4 | A cached response still carries a correct decision record. | A hit's `response_metadata["routing"]` matches the miss that filled it. | T-119 |

### R1 · Transparent to the caller

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R1-1 | The response is the selected model's own — content, tool calls, `usage_metadata`, `response_metadata`, `id` — with only the decision record added. | Field-by-field comparison of the router's output against the route called directly; the sole difference is `response_metadata["routing"]`. | T-114 |

### R2 · Every decision is inspectable

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R2-1 | Every response carries the D8 record. | Present for invoke, stream, batch and generate, and on every fallback, diversion and forced path. | T-114 |
| REQ-R2-2 | The same record is on the trace, placed per D9. | The offline tracer finds it as the strategy run's output, in the router run's outputs and in the route run's metadata; a live LangSmith trace shows it. | T-114 |
| REQ-R2-3 | Exactly one streamed chunk carries the record. | Merging all chunks yields one record, not a concatenation (D8). | T-113 |
| REQ-R2-4 | Structured output has an answer for R2 (D3). | With `include_raw=True` the raw message carries it; the parsed-only path is covered by the chain run and `last_routing_decision()`, and its limit is documented. | T-114 |

### R3 · Cost counted once

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R3-1 | `UsageMetadataCallbackHandler` totals equal the routes' direct usage, keyed by the model that ran. | Parametrised over invoke, ainvoke, stream, batch and generate. | T-117 |
| REQ-R3-2 | An opt-in strategy's own calls are costed to the strategy's model, never folded into the router or the answering route. | A classifier's LLM run appears inside the strategy run with its own `usage_metadata`; the answering route's usage is unchanged. Embedding calls appear as child runs the strategy opens itself — LangChain's `Embeddings` emits no callbacks and reports no usage, so their cost is estimated from input length (T-133, T-140), not measured. | T-117 |

### R4 · Routes on the current request

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R4-1 | "Current request" is the most recent `HumanMessage`, ignoring trailing AI and tool messages; system prompts and conversation length never define it. | Fixture transcripts: single turn, multi-turn, agent tool loop, system prompt only. | T-112 |
| REQ-R4-2 | In an agent loop, every model call routes on the originating user request. | A three-iteration tool loop routes identically on all three calls. | T-112 |
| REQ-R4-3 | Wider context is an explicit opt-in. | `messages` is `None` unless the strategy sets `wants_full_context = True`. | T-111 |
| REQ-R4-4 | No user message at all means the strategy can't decide. | A system-prompt-only transcript goes to the default route with one warning and a recorded reason. | T-112 |

### R5 · Any number of named routes

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R5-1 | Any number of named routes, each an ordinary chat model, used as given. | Works with one, two and many routes; the router never mutates or reconfigures a route (identity check). | T-110 |
| REQ-R5-2 | Route names are validated at construction. | Empty `routes`, or a `default_route` that names none of them, fails at construction with a message naming the offender. | T-110 |

### R6 · Strategies at three levels

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R6-1 | One interface (D6) serves all three levels; the built-ins use no private hooks. | Every built-in strategy subclasses `RoutingStrategy` and is exercised through the public interface alone. | T-111 |
| REQ-R6-2 | A custom strategy is a few lines of user code. | A plain function is accepted as `strategy=` and coerced; the "custom strategy" use case fits in five lines in a test. | T-111 |
| REQ-R6-3 | The configurable component expresses the §4 use cases from configuration alone. | Cost tiering and domain routing each built without a strategy class; invalid configuration fails at construction, not per request. | T-132 |
| REQ-R6-4 | The interface's stability promise is written down. | The docs state what may change and under what version bump. | T-151 |
| REQ-R6-5 | A strategy receives the config its own calls must use (D9). | `RoutingRequest.config` is the strategy run's child config; it is excluded from the request's equality; every built-in strategy that makes a call passes it on. | T-111 |

### R7 · No hidden costs

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R7-1 | Ready-made heuristic strategies make no model or API call. | A callback counter records zero LLM/embedding starts across the strategies' test suites. | T-130, T-131 |
| REQ-R7-2 | Strategies that do make calls cannot be enabled implicitly. | Construction requires the caller to pass the embeddings instance or classifier model; there is no default. | T-133, T-134 |

### R8 · In-process and self-contained

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R8-1 | Installing the package pulls in only `langchain-core` and its transitive dependencies. | A clean-environment install test lists the resolved set. | T-151 |
| REQ-R8-2 | `src/` imports nothing beyond the standard library and `langchain-core`. | A test walks the package's imports and fails on anything else — including `spike/`. | T-151 |

### R9 · Always decides

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R9-1 | A default route is mandatory. | Construction without one is an error (REQ-R5-2). | T-110 |
| REQ-R9-2 | Each way a strategy can fail to decide falls back to the default route, with one warning and a recorded reason. | Three cases — abstained, raised, named an unknown route — each: default route used, exactly one `FallbackWarning`, `fallback=True` and a reason naming the cause. | T-110 |
| REQ-R9-3 | The fallback never swallows a *route* failure. | A route that raises propagates (REQ-C6-1); only strategy failures are absorbed. | T-118 |

### R10 · Tool-aware routing

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R10-1 | Tool capability is detected per D5, with a per-route override that wins. | A route with `profile["tool_calling"] = False`, a route that doesn't override `bind_tools`, and an override are each detected correctly. | T-115 |
| REQ-R10-2 | Binding warns up front, naming the routes that can't use tools; if none can, it is an error. | One `ToolSupportWarning` listing the incapable routes; `NoToolCapableRouteError` when none are capable. | T-115 |
| REQ-R10-3 | A request the strategy sends to a tool-incapable route is diverted per D1, with a warning each time and the diversion recorded. | The diverted request runs on the D1 target; one `ToolSupportWarning` per request; `diverted_from` names the original route. | T-115 |
| REQ-R10-4 | `with_structured_output` follows the same rules. | The three cases of REQ-R10-2 and REQ-R10-3 repeat through `with_structured_output`. | T-115 |

### R11 · Forced routes are honoured

| ID | Requirement | Check | Task |
| --- | --- | --- | --- |
| REQ-R11-1 | A forced route skips the strategy (D2) and is never silently swapped. | The forced route runs even when the strategy would choose otherwise; the strategy's `decide` is not called. | T-116 |
| REQ-R11-2 | An unavailable forced route errors by default and falls back only when configured to. | Unknown and tool-incapable forced routes raise `ForcedRouteError`; with `on_unavailable_forced_route="fallback"` they follow R9/R10 and warn once. | T-116 |
| REQ-R11-3 | Forcing and any subsequent fallback are recorded. | `forced=True`; under fallback, `fallback=True` with a reason naming the forced route. | T-116 |

## Traceability

Every principle has at least one requirement; this table is T-101's first acceptance criterion.

| Principle | Requirements | Tasks |
| --- | --- | --- |
| C1 | REQ-C1-1…3 | T-120 |
| C2 | REQ-C2-1…4 | T-113, T-117 |
| C3 | REQ-C3-1…4 | T-115, T-121 |
| C4 | REQ-C4-1, REQ-C4-2 | T-116 |
| C5 | REQ-C5-1…5 | T-113, T-117 |
| C6 | REQ-C6-1…3 | T-118 |
| C7 | REQ-C7-1, REQ-C7-2 | T-112 |
| C8 | REQ-C8-1, REQ-C8-2 | T-120, T-150 |
| C9 | REQ-C9-1, REQ-C9-2 | T-120, T-151 |
| C10 | REQ-C10-1…4 | T-119 |
| R1 | REQ-R1-1 | T-114 |
| R2 | REQ-R2-1…4 | T-113, T-114 |
| R3 | REQ-R3-1, REQ-R3-2 | T-117 |
| R4 | REQ-R4-1…4 | T-111, T-112 |
| R5 | REQ-R5-1, REQ-R5-2 | T-110 |
| R6 | REQ-R6-1…5 | T-111, T-132, T-151 |
| R7 | REQ-R7-1, REQ-R7-2 | T-130, T-131, T-133, T-134 |
| R8 | REQ-R8-1, REQ-R8-2 | T-151 |
| R9 | REQ-R9-1…3 | T-110, T-118 |
| R10 | REQ-R10-1…4 | T-115 |
| R11 | REQ-R11-1…3 | T-116 |

## Spike caveats answered

Every row of the spike's "Carried into v1" table, and the four C3 caveats from T-003.

| Caveat | Answered by |
| --- | --- |
| `generate()` / `agenerate()` still take the base path and would double count | REQ-C2-2, REQ-R3-1 |
| R2 has no answer for structured output: a parsed object cannot carry the decision | D3, REQ-R2-4 |
| Cache ownership — the *route's* cache applies and the router's own `cache=` is ignored | D4, REQ-C10-2 |
| A route that cannot use tools fails at call time, only when selected | D5, REQ-R10-1, REQ-R10-2 |
| The chain run changes what `stream_mode="messages"` sees; confirm against a real graph | REQ-C5-3 |
| Raw tool objects ride in the invocation params, which are traced and used as a cache key | REQ-C10-3 |
| `with_structured_output`'s default discards `method=` and `strict=` | REQ-C3-3 |
| `tool_choice` / `strict=` must be replayed at bind time, not passed as call kwargs | REQ-C3-1 |
| `create_agent` asks the model what it can do (`model.profile`) | REQ-C3-4, T-121 |
| A decision record on every streamed chunk aggregates to `"frontierfrontier…"` | D8, REQ-R2-3 |

## What this does not decide

- **Which heuristic signals** T-131 scores on — T-140's benchmark results settle the defaults.
- **Whether the ready-made strategies are presets** of the configurable component — T-132's call
  (REQ-R6-3 holds either way).
- **Whether embedding and classifier strategies are required or optional** — T-140 decides,
  recorded in PRD §11 (§8).
- **Package naming and the release process** — T-151; the import path in the API section above is
  already fixed by `pyproject.toml`.
