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

## Scope

- Detect tool capability per route. Candidates: the model's `profile` (`tool_calling`; a beta
  feature, may be missing), or whether `bind_tools` is implemented / succeeds. Decide, and allow a
  per-route override.
- At bind time: warn once, naming the routes that can't use tools; if none can, binding is an
  error.
- Per request: if the strategy picks a non-tool route, divert to a tool-capable one (rule from
  T-101), warn, and record the diversion (R2).
- Build on the C3 findings from T-003.

## Acceptance criteria

- [ ] Tests for: all routes tool-capable; mixed (up-front warning + per-request diversion); none
      (error at bind time).
- [ ] `with_structured_output` follows the same rules.
- [ ] Diversions appear in the decision record.
