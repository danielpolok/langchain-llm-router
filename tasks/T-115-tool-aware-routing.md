---
id: T-115
title: Tool-aware routing
phase: v1
status: todo
principles: [R10, C3, R2]
depends_on: [T-110, T-114]
prd: ["§3.2", "§11"]
---

# T-115 · Tool-aware routing

## Goal

When tools are bound, never send a request to a route that can't use them (R10). Structured output
counts too — LangChain builds it on tool binding.

Requirements: **REQ-C3-1, REQ-C3-2, REQ-C3-3, REQ-R10-1…4**
([v1 requirements](../docs/v1-requirements.md)).

## Scope

- **Binding (C3, from T-003's verdict).** `bind_tools` keeps the tools as given until a route is
  chosen; the route's own `bind_tools` does the provider-specific conversion at call time.
  `tool_choice` and binding kwargs such as `strict=` are replayed on the route's binder, not
  passed as call kwargs — `strict=` changes how the tool schema is built at bind time. The raw
  `tools` list is also bound in its usual place so LangChain's own checks
  (`disable_streaming="tool_calling"`) behave as on any chat model.
- **Structured output.** Override `with_structured_output` and forward per request to the selected
  route's own implementation; the inherited default drops `method=` and `strict=`
  (`chat_models.py:2530`).
- **Capability detection (D5).** `profile["tool_calling"]` when the route reports a profile, else
  whether the route's class overrides `BaseChatModel.bind_tools`; `tool_support_overrides` always
  wins. (The base `bind_tools` raises `NotImplementedError` at *call* time, `:2383` — the late
  failure this task exists to pre-empt.)
- **At bind time.** One warning naming the routes that can't use tools; `NoToolCapableRouteError`
  when none can.
- **Per request (D1).** A request the strategy sends to a tool-incapable route is diverted to the
  default route if it is tool-capable, otherwise to the first tool-capable route in declaration
  order; one warning per diverted request; `diverted_from` recorded (R2).

## Acceptance criteria

- [ ] Tests for: all routes tool-capable; mixed (up-front warning + per-request diversion); none
      (error at bind time).
- [ ] Capability detection tested per signal: profile, `bind_tools` override, explicit override.
- [ ] `with_structured_output` follows the same rules, and `method=`, `strict=` and
      `include_raw=True` all survive to the route.
- [ ] Diversions appear in the decision record.
