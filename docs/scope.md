# Scope: middleware, non-goals and prompt caching

What `ChatRouter` deliberately does not do, and where each edge is documented elsewhere.

## When to use agent middleware instead (C8)

`ChatRouter` and LangChain's [`@wrap_model_call` middleware](https://docs.langchain.com/oss/python/langchain/middleware)
solve overlapping problems from different places, and they compose (REQ-C8-1) rather than
compete:

| | `ChatRouter` | `@wrap_model_call` |
| --- | --- | --- |
| Sees | The current request only, by default (R4) — text, content blocks, modalities, whether tools are bound | The whole `create_agent` call: agent state, the loop iteration, anything an earlier middleware stored |
| Works | Anywhere a chat model goes — chains, agents, LangGraph nodes (C1) | Only inside `create_agent` |
| Does | Picks one of several named chat models and returns its response, with a recorded reason (R1, R2) | Anything: rewrite the request, swap the model outright, retry, short-circuit |

Reach for the router when your policy is a function of **the request you'd hand any chat
model** — that's the case §4's use cases (cost tiering, domain routing, forced experimentation)
all share, and it's why the router works identically whether or not an agent is involved.

Reach for `@wrap_model_call` **instead of** the router when the decision needs something the
router's strategy interface deliberately never sees — agent state, which node is calling, a
running tool-call count — because R4 exists precisely to keep that kind of thing from misrouting
an agent's own tool loop. Nothing stops you from also passing a `ChatRouter` as that middleware's
model, if you want both: request-shaped routing inside a state-aware decision.

[`examples/agent_middleware.py`](../examples/agent_middleware.py) runs the coexistence case —
the router as the model behind `create_agent`, with `@wrap_model_call` middleware observing it —
which is exactly what `tests/unit_tests/test_placement.py`'s
`test_wrap_model_call_middleware_coexists_with_routing` checks (REQ-C8-1): the middleware sees
the router itself as `request.model`, never bypassed and never some inner route, and routing
still happens underneath it.

An agent-middleware **form** of the router itself — letting `@wrap_model_call` drive route
selection from agent state, rather than composing with the router as it stands — is tracked as a
later-phase adapter (PRD §9, `v1.x`), not built here.

## Non-goals (PRD §5)

The router is deliberately narrow. It does not:

- **Learn from traffic.** No bandits, no LLM-as-judge feedback, no retraining — the router
  applies the policy your strategy encodes; it never updates that policy from what happened.
- **Escalate after generation (cascades).** A route is chosen before the call, not after seeing
  a weak answer and trying again on a stronger model.
- **Optimise against a cost budget.** The application's policy decides what's worth spending;
  the router has no notion of a budget to spend against.
- **Run as a hosted service, proxy or gateway.** It's in-process — a `BaseChatModel`, like any
  other, with no server of its own (R8).
- **Replace agent middleware for in-agent model selection** — see the comparison above.

Each is a real feature other tools provide; none of them is what this package is for. Trained
routers (RouteLLM and similar) and LangGraph-specific helpers are tracked as later-phase
strategies and adapters (PRD §9), not non-goals — they're simply not built yet.

## The prompt-caching caveat (PRD §8)

Provider-side prompt caching happens on the **provider's** side, and the router has no way to
control it. Routing consecutive turns of one conversation to different models discards
whichever provider's cached prompt prefix that conversation had built up — each route is a
different provider (or a different deployment), so the cache each one built for that
conversation's earlier turns doesn't carry over.

Cached input is typically billed at a steep discount, so in long conversations and agent loops
this can eat into the savings a routing policy is chasing in the first place — worth measuring
for your own traffic mix before treating "smaller model when possible" as a straightforward win.
Keeping a whole conversation pinned to one route to preserve provider caching is a candidate
later feature (PRD §8), not something the router does today; forcing a route for a whole session
via runtime config (R11, [`decision-record.md`](decision-record.md#forced-routes-r11)) is the
closest tool available for that now.
