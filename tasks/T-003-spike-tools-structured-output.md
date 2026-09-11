---
id: T-003
title: "Spike: tools and structured output through the router (C3)"
phase: v0
status: todo
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

- [ ] Offline tests with fake routes — one that implements `bind_tools`, one that doesn't — cover
      questions 1–4.
- [ ] One real-provider run with two different models: a tool call and structured output both
      succeed on each route.
- [ ] Written verdict in this file: **C3 holds / holds with caveats / fails**, caveats listed. A
      fail triggers the §8 fork, recorded in PRD §11.
