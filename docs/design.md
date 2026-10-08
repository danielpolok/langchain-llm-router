# Design notes

Why `ChatRouter` is shaped the way it is. These decisions were each settled against evidence — a
spike, a measurement, or a LangChain behaviour that turned out to differ from the documentation —
and the code relies on them. If you need to change one, read its reasoning first and replace it
deliberately rather than working around it.

For what the router does *not* do, see the [README's scope](../README.md#scope). For the public
surface these decisions produce, see [strategies.md](strategies.md) and [guide.md](guide.md).

## A chat model, not middleware

`ChatRouter` is a `BaseChatModel`, so it works anywhere a chat model does — chains, agents, graph
nodes. Middleware only works inside `create_agent`. The consequence is the project's core rule:
**anything that works on a chat model must work identically on the router**, so the router uses
LangChain's own mechanisms (`bind_tools`, `with_structured_output`, `with_config`, callbacks, the
cache) instead of inventing parallel ones. It depends on `langchain-core` only (`>=1.2.21,<2`),
uses only its public names (never private ones), and needs no proxy or service.

## Routes and policy

The router takes any number of **named routes** — fixed tier counts (two, four) don't fit domain or
mixed policies — and a **default route** that is mandatory. The routing *policy* lives in the
application's code; a *strategy* applies it. Policy is static: learning from traffic needs a
feedback signal that does not exist yet, so it is a non-goal.

The router always answers. A strategy that raises, or returns `None` ("can't decide"), degrades to
the default route with a warning and the reason recorded, instead of failing the request.

## Route on the current request

By default a strategy sees the user's **current request**, not the last message of the history, tool
output, system prompts or conversation length. Routing on the last message or on length misroutes
agent loops: after a tool call the "last message" is a tool result, and a long conversation says
nothing about how hard the next question is. More context (the user's earlier messages through
`lookback`, the full messages, the config) is an opt-in the strategy declares.

There is one strategy interface — `decide` / `adecide` over a `RoutingRequest` — and the three
levels (ready-made, configurable, custom code) are all implementations of it.

## The router's own run is a chain run

This is the decision the rest of the tracing and cost behaviour depends on. If the router emitted a
*model* run of its own, the same tokens would be counted twice — by `UsageMetadataCallbackHandler`
and by LangSmith, which prices each model run from its own usage — and nothing would error; the
number would just be wrong. So the router opens a **chain** run that delegates, leaving the selected
route's call as the only model run in the trace. Cost is counted once, against the model that ran,
and the real call is still visible.

Things that follow from it, each learned the hard way:

- A model run cannot have child runs, so a strategy that itself calls a model needs a parent that
  can: the chain run. The router opens its run *before* it decides, runs the strategy in a child
  run of its own, and hands the strategy that run's config so its nested call is costed to the
  strategy's own model and shows up in the trace.
- On Python below 3.11, async callbacks only propagate through an explicit config, and the package
  supports 3.10 — so the config is passed explicitly rather than left to context propagation.
- Streaming hands `_stream` no run manager, so a router built on `_stream` cannot pass its route any
  callbacks and the real call would vanish from the trace. Delegating from `invoke`/`stream`/`astream`
  avoids that.
- `generate()` / `agenerate()` still take the base path, which would double-count, so they are
  handled separately and scoped to message-level parity with `invoke`.
- The beta v3 streaming protocol is out of scope; it raises `NotImplementedError` before any run
  opens.

## Tools and structured output

When tools are bound, routes that can't use them are skipped **with warnings** rather than
refused, so tool-calling requests keep working and every diversion stays visible; if no route can
use tools, binding is an error. Structured output counts as tool binding.

- **Capability:** a route can use tools if its `profile["tool_calling"]` says so; when the profile
  is silent (`None` or `{}`) the router falls back to whether the route's class overrides
  `BaseChatModel.bind_tools`, because the base implementation only fails at call time — the late
  failure this check exists to pre-empt. The deprecated `tool_support_overrides` wins over both;
  the route's own `profile=` is the way to correct it.
- **Diversion order:** a request diverted off a tool-incapable route goes to the default route if
  it can use tools, otherwise to the first tool-capable route in declaration order. It is
  deterministic and explainable, and it does not re-run the strategy, which under the opt-in
  strategies would spend another call and could loop.
- **Nothing extra is bound:** the router binds no `tools` argument of its own, so a pre-bound
  router works in `create_agent` and a later bare `bind(tools=…)` is honoured.
- **Profile:** the router reports the *intersection* of its routes' profiles (booleans AND-ed,
  numbers minimised, differing values dropped, `None` if any route reports none), because
  `create_agent` picks a structured-output strategy from `model.profile` and must not pick one a
  route can't serve. This is why the package needs `langchain-core>=1.2.21`: that is the first
  release whose `BaseChatModel` asks a subclass for its profile (`_resolve_model_profile`).

## Images, audio, video and PDFs

A route receives the whole conversation, so what it must be able to take is decided by every
message in it, not by the request the strategy routed on. A follow-up with no image of its own still
carries the earlier image. The check belongs to the router, after the decision, because it is a hard
requirement whatever the strategy chose, and `lookback` sees only a few messages.

- **Signal:** LangChain's own `profile` keys — `image_inputs`, `audio_inputs`, `video_inputs` and
  `pdf_inputs` — read against every message's `content_blocks`. Only an explicit `False` rules a
  route out. Profiles are beta and often missing (Ollama models report none), and skipping every
  model that is silent would divert far more requests wrongly than rightly.
- **The narrower keys apply too.** An image given by URL also needs `image_url_inputs`, and an image
  or PDF inside a tool result needs `image_tool_message` or `pdf_tool_message`: a provider can take
  an image from the user and still reject one in a tool result. They cost nothing when they are
  missing, by the rule above. A `file` block counts as a PDF unless its `mime_type` says otherwise,
  because PDFs are the only files a profile describes and they often arrive untyped.
- **Diversion** works as for tools, and the two are combined: the default route if it can serve the
  request, else the first route in declaration order that can, where serving means both using the
  bound tools and taking the content. The warning is `ContentSupportWarning`, or
  `ToolSupportWarning` when the chosen route couldn't use the tools.
- **No capable route is an error,** `NoContentCapableRouteError`, raised before any route is called.
  The alternative, warning and sending the request anyway, gives the outcome the check exists to
  prevent: a provider error or, worse, an answer that silently ignored the image. Routers that check
  capabilities, such as OpenRouter, refuse in the same case. The error is a `RoutingError`, so
  `with_fallbacks` can catch it.
- **A wrong profile is corrected on the route,** with `profile=`, LangChain's own mechanism, which
  also reaches the router's own profile. For the same reason `tool_support_overrides` is deprecated:
  `profile={"tool_calling": ...}` does its job, and one mechanism covers tools and content alike.
- **Not checked:** conversation length against `max_input_tokens`, which needs token counting.

## Forced routes

A route pinned through runtime config is never silently swapped. It travels under the configurable
key `route`, declared through `config_specs`, so `with_config`, `config={"configurable": …}` and
config-schema introspection all work through LangChain's standard mechanism. A forced route skips
the strategy entirely — its answer would be discarded, and running it costs money under the opt-in
strategies. If the forced route doesn't exist or can't serve the request — it can't use the bound
tools or take the conversation's content — the call errors by default; falling back is an explicit
setting.

## The decision record

Every routed call records which route ran and why — eight fields, carried on exactly one streamed
chunk (LangChain merges a string repeated across chunks by concatenating it, so a record on every
chunk would aggregate to `"frontierfrontier…"`).

The record always reaches the trace. On messages it also reaches `response_metadata`. Under
`with_structured_output` a parsed object has nowhere to carry it, so it is in the raw message only
with `include_raw=True`, and `last_routing_decision()` covers the parsed-only path. That function
publishes to both a context variable and the calling thread, because LangChain runs a sequence's
steps in a copy of the caller's context — a context variable alone never reaches the caller.

Two of the fields describe the conversation rather than the choice, and each has an owner:

- **`previous_route` is the router's.** It reads it from the newest AI message in the input whose
  record names one of its own routes, so it is there whatever the strategy, and on forced and
  fallen-back turns too. Nothing decides on it: keeping a conversation on one route is a policy,
  and a policy is a strategy. Inside a router that is itself a route, the history holds only the
  outer router's records, so the inner router finds none of its own routes there and records
  `None`, unless the two routers share a route name — a record names its route and nothing else.
- **`messages_back` is the strategy's.** Only the strategy knows which message decided, so it says
  so in `RoutingChoice.messages_back`, a field added under the
  [stability promise](strategies.md#stability-promise), and every built-in strategy fills it.
  The alternative, reading `(1 message back)` out of the reason, would have made the reason's
  wording part of the interface. It is `None` when no one message decided: a forced route, a
  fallback, a catch-all rule, or a strategy that doesn't say.

## Caching

The **route owns response caching**; the router rejects a `cache=` of its own at construction.
The router delegates from `invoke`/`stream`, so its own cache path never runs, and an accepted
`cache=` would silently do nothing. Because the cache key is the route's, an answer cached for one
route can never be returned for another — structurally, not by arithmetic the router has to keep
right. Two consequences the tests hold to: the key for the same tools must be equal across
processes (a bare function must not reach it as `<function f at 0x…>`), and a cached response still
carries a correct decision record.

Provider-side prompt caching is out of the router's control; switching routes mid-conversation can
lose the cached prefix. A strategy that keeps a conversation on one route avoids that; see
[strategies.md](strategies.md#a-strategy-class).

## Errors, retries and fallbacks

These stay LangChain's job. The selected route's errors surface unchanged. `with_retry` and
`with_fallbacks` behave on the router as on any model. A route is a chat model, so retries belong on
the router or inside the provider client; a route that is already wrapped is rejected at
construction rather than half-working.

## Which strategies are optional

The embedding and classifier strategies are **optional**, not required. The cost/quality benchmark
(see [`../benchmark/`](../benchmark/README.md)) found the free strategies, keyword and heuristic,
already meet the target — well over 30% cheaper at no quality loss on the mixed workload — while
the opt-in ones saved less on that run. The classifier routed items correctly far more often than
the heuristic did and still saved less, because its own call is a real, billed request whose
overhead outpaces the accuracy it buys. The embedding strategy's threshold is a placeholder, not
its ceiling. This is one run against one model pair; it is evidence that the free tier is enough to
start with, not a bound on what a tuned opt-in strategy could do.
