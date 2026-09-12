# PRD — `RouterChatModel`: in-process model routing for LangChain

**Owner:** Daniel Polok · **Phase:** v0 — principles · **Status:** Draft · **Updated:** 2026-09-11

---

## 1. Why

Teams want easy requests on cheap models and hard ones on frontier models — typically 30–70% lower
inference cost at similar quality. Today's options fall short:

- **Routing services and proxies** (LiteLLM, Not Diamond, Martian, OpenRouter, Bedrock prompt
  routing, RouteLLM's server) run outside the application: another hop, another system to operate,
  and a decision invisible to the application's graph and traces.
- **LangChain agent middleware** (`@wrap_model_call`) is in-process, but only works inside
  `create_agent` — not in chains, graph nodes, or anywhere else a chat model is used.
- **Hand-written routing** is rebuilt by every team, with no shared strategies or conventions.

Missing: a reusable, in-process router that works everywhere a LangChain chat model does.

## 2. What it is

A LangChain chat model that, for each request, picks one of several candidate chat models and
returns that model's answer.

The routing **policy** — which request goes to which model — lives in the application's own code.
A **strategy** is the component that applies the policy to each request.

To the rest of the application it is simply a chat model: it goes wherever a LangChain chat model
goes.

## 3. Phase v0: principles

v0 sets out **what** the router is and the principles it holds to. It says nothing about
implementation; detailed requirements come in v1 (§9).

The guiding principle: **the router works with LangChain's model abstractions, not against them.**
Anything that works with a LangChain chat model works the same way with the router.

### 3.1 Consistency with LangChain

| # | Principle | LangChain concept it follows |
| --- | --- | --- |
| C1 | **It is a chat model.** Usable anywhere a chat model is — chains, agents, graph nodes, message-history wrappers. | `BaseChatModel`, `create_agent(model=…)`, LangGraph nodes |
| C2 | **Same calling conventions.** Sync, async, streaming and batch all work; no new ways of calling it. | Runnable interface |
| C3 | **Tools and structured output behave as on any chat model**, applied to whichever model is selected. | `bind_tools`, `with_structured_output` |
| C4 | **Runtime configuration works as usual**, including forcing a specific route for one call (see R11). | `configurable_fields`, `with_config` |
| C5 | **Traces show the routing decision and the real model call**, in LangSmith's normal shape. | Callbacks, nested runs, LangSmith |
| C6 | **Errors, retries and fallbacks stay LangChain's job.** The selected model's errors surface unchanged. | `with_retry`, `with_fallbacks` |
| C7 | **Messages are read as LangChain defines them**, including multimodal content. | Message types, content blocks |
| C8 | **Complements agent middleware, doesn't replace it.** State-driven model selection inside `create_agent` remains a middleware concern. | `@wrap_model_call` |
| C9 | **Targets LangChain 1.x.** | `langchain-core` 1.x |
| C10 | **LangChain's response cache works as on any chat model** — and a cached answer is never reused for a request that would have gone to a different model or configuration. | `set_llm_cache`, a model's `cache` |

### 3.2 Routing behaviour

| # | Principle |
| --- | --- |
| R1 | **Transparent to the caller.** The response is the selected model's response — content, tool calls, metadata — with the routing decision added. Nothing is dropped. |
| R2 | **Every decision is inspectable.** Each response records which route was taken and why. |
| R3 | **Cost is counted once and attributed to the model that actually ran** — never to the router, never twice. |
| R4 | **Routes on the user's current request by default** — not on tool output, system prompts, or conversation length. Wider conversation context is an explicit opt-in. |
| R5 | **Any number of named routes**, each an ordinary chat model the application already builds. |
| R6 | **Strategies at three levels.** Ready-made heuristic strategies with sensible defaults; a configurable component for defining a strategy from scratch; or a fully custom strategy through one small, stable interface. |
| R7 | **No hidden costs.** Ready-made heuristic strategies make no extra model or API calls; strategies that do (embedding, classifier) are explicit opt-ins. |
| R8 | **In-process and self-contained.** No proxy, sidecar, external service or extra credentials; no dependencies beyond LangChain's core. |
| R9 | **Always decides.** A default route is always configured. Whenever the strategy can't decide or fails, the request goes to the default route, with a warning and the reason recorded (R2). |
| R10 | **Tool-aware routing.** When tools are bound, a route whose model can't use them is skipped and the request goes to a route that can. This covers structured output too, which LangChain builds on tool binding. An up-front warning names the routes that can't use tools, and a warning is raised each time a request is diverted (recorded per R2). If no route can use tools, binding them is an explicit error. |
| R11 | **Forced routes are honoured.** A route forced for one call (C4) is never silently swapped. If it can't serve the request — it doesn't exist, or can't use bound tools — a setting decides what happens: an explicit error (**default**), or falling back as R9 and R10 would, with a warning. |

## 4. Use cases

- **Cost tiering** — simple questions to a small model, complex ones to a frontier model, with the
  saving visible in LangSmith.
- **Domain routing** — e.g. code requests to a code-strong model.
- **Agent backbone** — the router as the model behind `create_agent`.
- **Custom strategy** — a team's existing classifier plugged in as the strategy.
- **Experimentation** — force a route per call to compare models on the same traffic.

## 5. Non-goals

- Learning from traffic (bandits, LLM-as-judge feedback, retraining).
- Post-generation escalation (cascades).
- Automatic optimisation against a cost budget — the application's policy decides.
- A hosted service, proxy or gateway.
- Replacing agent middleware for in-agent model selection.
- Packaging and distribution — planned for v1 (§9).

## 6. v0 exit criteria

v0 is done when:

1. **A small spike shows the two riskiest principles are achievable:**
   - **C3** — tool binding and structured output work through the router. If not, C1 ("it is a
     chat model") is revisited before v1.
   - **R3** — cost is counted exactly once while the real model call still appears in traces
     (C5). If both can't hold, decide which gives way.
2. **§3 still holds after the spike** — reread it once the spike is built; it stands as written.
3. **The open questions in §10 are answered.**

## 7. Success metrics

Exit criteria say v0 is *done*; these say the project *worked*.

- **Cost:** the v1 benchmark shows **≥ 30% lower inference cost at ≥ 95% of always-frontier
  quality** on a mixed workload.
- **Use:** it replaces hand-written routing in at least one real application.

## 8. Risks & assumptions

Recorded so the same doubts aren't re-argued later.

- **LangChain 1.x stays stable.** Assumes `langchain-core` 1.x keeps its chat-model contract —
  tool binding, structured output, callbacks — stable. A breaking change there lands directly on
  C1–C5.
- **C3/R3 is a real fork point.** If C3 fails, the chat-model approach (C1) gives way, most likely
  to the agent-middleware form planned for v1.x. If R3 conflicts with C5, exact cost reporting and
  a traced model call can't both hold. The §6 spike settles this — it is not re-debated.
- **Simple strategies may not save enough.** The ≥ 30% target comes from published results with
  *trained* routers; keyword or heuristic strategies alone may fall short. The v1 benchmark
  decides whether embedding or classifier strategies are required rather than optional.
- **Switching models can forfeit provider prompt caching.** Prompt caching happens on the
  provider's side and the router can't control it — but routing consecutive turns to different
  models discards the cached prompt prefix, and cached input is billed at a large discount. In long
  conversations and agent loops this can eat into the savings. The v1 benchmark should measure it;
  keeping a conversation on one route is a candidate later feature.

## 9. Later phases

| Phase | Focus |
| --- | --- |
| **v1** | Detailed requirements; built-in strategies (keyword, heuristic, embedding, small-LLM classifier); documentation; a benchmark demonstrating the cost saving; packaging and publishing for other LangChain developers. |
| **v1.x** | Adapters: an agent-middleware form of the router; RouteLLM and other trained routers as strategies; LangGraph helpers. |
| **v2** | Learning from traffic; cascade / escalation mode. |

## 10. Open questions

_None open._ Q1–Q3 were answered on 2026-09-11 and folded into **R10** (non-tool routes), **C10**
(response cache) and **R6** (strategy levels).

## 11. Key decisions

| Decision | Why |
| --- | --- |
| A chat model, not middleware | Middleware works only inside agents; a chat model works everywhere (§1). |
| Any number of named routes | Fixed tier counts (2 in RouteLLM, 4 in LiteLLM) don't fit domain or mixed policies. |
| Static policy first | Learning from traffic needs a feedback signal that doesn't exist yet. |
| Route on the current user request by default | Surveyed RouteLLM, LiteLLM, semantic-router and LangChain's middleware example: routing on the last message or on conversation length misroutes agent conversations (R4). |
| Divert non-tool routes, with warnings — don't refuse | Keeps tool-calling requests working instead of failing them; the warnings keep every diversion visible (R10). |
| Response cache is the router's concern; provider prompt caching is not | LangChain's cache is in-process and keyed on the router, so the router must keep it correct (C10). Provider prompt caching is outside its control — only its hit rate is affected (§8). |
| Ready-made strategies plus from-scratch configuration | A fast start for common setups, full control when needed, custom code as the escape hatch (R6). |
| A default route is mandatory | The router always answers: a strategy that fails or can't decide degrades to the default model instead of failing the request (R9). |
| Forced routes error by default, configurable | An explicit choice shouldn't be silently overridden — otherwise experiments compare the wrong model. Falling back instead is available as a setting (R11). |
| The router's own run is a chain run, not a model run | While the router emits a model run too, the same tokens are billed twice (R3) and the streamed route call vanishes from the trace. A chain run that delegates leaves the selected route's call as the only model run, so cost stays exact and the real call stays traced (C5). Settled by the T-004 spike; R3 and C5 do not conflict. |
