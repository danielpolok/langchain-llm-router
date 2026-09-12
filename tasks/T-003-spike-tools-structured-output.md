---
id: T-003
title: "Spike: tools and structured output through the router (C3)"
phase: v0
status: blocked
principles: [C3, C1, R10]
depends_on: [T-002]
prd: ["§6", "§8"]
---

# T-003 · Spike: tools and structured output through the router (C3)

## Goal

Show that `bind_tools` and `with_structured_output` work through the router exactly as on the
selected model. If they can't, C1 ("it is a chat model") is revisited before v1 (§8).

## Questions to answer

1. **Binding.** `BaseChatModel.bind_tools` raises by default, so the router must implement it.
   Can it keep tools in their original form and let the *selected* route's own `bind_tools` do
   the provider-specific conversion at call time? Do `tool_choice` and other binding kwargs pass
   through?
2. **Structured output.** The `BaseChatModel` default for `with_structured_output` builds on
   `bind_tools` and rejects provider-specific kwargs such as `method=`. Is the default enough, or
   must the router override it and forward to the selected route (e.g. for native JSON-schema
   modes)? Does `include_raw=True` still work?
3. **Agents.** Does `create_agent(model=router, tools=[...])` run a full tool loop? With
   `response_format`, `create_agent` picks a structured-output strategy from the model's
   capabilities (`profile`) — what should the router report when its routes differ?
4. **Non-tool routes (preview of R10).** What happens when the selected route can't bind tools?
   Observe only; R10 is built in T-115.

## Acceptance criteria

- [x] Offline tests with fake routes — one that implements `bind_tools`, one that doesn't — cover
      questions 1–4. (`spike/tests/test_tools.py`, 18 tests.)
- [ ] One real-provider run with two different models: a tool call and structured output both
      succeed on each route. **Blocked:** written as `spike/tests/test_real_providers.py`
      (OpenAI and Anthropic, a tool call and structured output on each), but it skips —
      `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are not set in this environment.
- [x] Written verdict in this file: **C3 holds / holds with caveats / fails**, caveats listed. A
      fail triggers the §8 fork, recorded in PRD §11.

## Verdict — C3 holds with caveats (offline; one real-provider run outstanding)

Nothing here disproves C1, so the §8 fork is not triggered. The caveats are v1 requirements,
not obstacles.

**1. Binding.** The router implements `bind_tools` and keeps the tools exactly as given — a
`ToolBinding` carried as a call kwarg — until a route is chosen; the *route's* own `bind_tools`
then does the provider-specific conversion. `tool_choice` and binding kwargs are replayed on the
route's binder, not passed as call kwargs: `strict=` changes how OpenAI builds the tool schema at
bind time, so where it is applied matters. The raw `tools` list is also bound in its usual place
so LangChain's own checks (`disable_streaming="tool_calling"`) behave as on any chat model.

- *Caveat:* raw tool objects then ride in the router's invocation params, which are what the
  tracer records and what `_get_llm_string` builds the cache key from. v1 decides what is traced
  (T-117) and what the cache key covers (T-119).

**2. Structured output.** The inherited default is **not** enough. `BaseChatModel.with_structured_output`
builds on *this* model's `bind_tools` and silently discards `method=` and `strict=`
(`chat_models.py:2530`), so every route would be forced through function calling and no route's
native JSON-schema mode would ever be used. The router overrides it and forwards per request to
the selected route's own implementation. `include_raw=True` works unchanged.

- *Caveat:* the decision record is lost. Forwarding bypasses the router's `_generate`, and a
  parsed Pydantic object has nowhere to carry the record — even with `include_raw=True` the raw
  message has none. R2 needs a separate answer for structured output (T-114).

**3. Agents.** `create_agent(model=router, tools=[...])` runs a full tool loop. With
`response_format`, `create_agent` chooses `ProviderStrategy` or `ToolStrategy` from
`model.profile` (`_supports_provider_strategy`, `langchain/agents/factory.py:560`), so the router
must report one. **What it should report: the intersection of its routes' profiles** — booleans
AND-ed, numbers reduced to the minimum, equal values kept. A router claiming a capability only
some routes have would have a strategy picked for it that a route cannot serve. With a mixed
profile the agent settles on `ToolStrategy` and the structured response comes back.

**4. Non-tool routes (R10 preview).** `bind_tools` on the router always succeeds — it cannot know
which route will run. The failure is `NotImplementedError` from `BaseChatModel.bind_tools`, raised
at *call* time and only when such a route is selected, so a policy can hide it until live traffic
reaches that route. This is the case R10 turns into an up-front warning naming the routes that
cannot use tools, plus a diversion per request (T-115).
