# Changelog

Every release of `langchain-model-router` and what changed in it, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions follow [Semantic Versioning](https://semver.org/). Until 1.0, the minor version plays the
role of the major one: a `0.x` minor release may break the public API, and a patch release never
does. The routing strategy interface carries a stricter
[stability promise](docs/strategies.md#stability-promise).

## [Unreleased]

### Added

- Requests are diverted away from a route that can't take the images, audio, video or PDFs in
  the conversation, including ones from earlier turns, as its `profile` reports. A
  `ContentSupportWarning` says so, and `NoContentCapableRouteError` is raised when no route can
  take them.

### Deprecated

- `tool_support_overrides`. Set the route's own `profile`, for example
  `init_chat_model(..., profile={"tool_calling": True})`.

## [0.1.0] - 2026-10-05

The first release.

### Added

- `ChatRouter`, a LangChain chat model that sends each request to one of several named chat
  models and returns that model's response unchanged. It works with `invoke`, `stream`, `batch`,
  their async forms, tools, structured output, `create_agent` and LangGraph.
- Five routing strategies: `HeuristicStrategy` (difficulty), `KeywordStrategy` (words),
  `ConfigurableStrategy` (combined rules), and the opt-in `EmbeddingStrategy` (similarity to
  examples) and `ClassifierStrategy` (a small model chooses). A plain function works as a
  strategy too.
- The routing strategy interface (`RoutingStrategy`, `RoutingRequest`, `RoutingChoice`,
  `RoutingCallable`) for writing your own strategy, with a stability promise.
- `lookback`, so a strategy can read earlier user messages to route short follow-up questions.
- A record of each routing decision, read with `routing_decision()` and
  `last_routing_decision()`, and shown in LangSmith traces. It includes the route the previous
  turn of the conversation took, so you can see when a conversation switches models, and how
  many messages back the message that decided the route was.
- Forcing a route per call, for A/B tests and debugging.
- Tools and structured output bound once on the router and applied to whichever route answers,
  with requests diverted away from routes that can't use them.
- Warnings, all under `RoutingWarning`, when the default route answers because the strategy
  couldn't decide, when a request is diverted for tools, or when a forced route gives way. Errors
  for misconfiguration and impossible forced routes, all under `RoutingError`.

[Unreleased]: https://github.com/danielpolok/langchain-model-router/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/danielpolok/langchain-model-router/releases/tag/v0.1.0
