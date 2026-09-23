"""`ChatRouter` and `@wrap_model_call` middleware together (C8, REQ-C8-1, REQ-C8-2).

The two are complementary, not competing: the router picks a route from the **current
request** (R4) — the same policy wherever the router goes, agent or not. `@wrap_model_call`
sees the whole `create_agent` call — agent **state**, the loop iteration, anything the
middleware stack put there — and can rewrite the request or swap the model outright before the
call happens.

Put the router behind an agent when the policy only needs to see the incoming request. Reach for
`@wrap_model_call` instead — with or without the router as its `request.model` — when the
decision needs agent state the router's strategy interface deliberately doesn't see: which
`create_agent` node is calling, how many tool round trips have run, something a previous
middleware stored. See `docs/scope.md` for the fuller comparison, LangChain's own
[middleware docs](https://docs.langchain.com/oss/python/langchain/middleware) for what
`@wrap_model_call` can do on its own, and `tests/unit_tests/test_placement.py`'s
`test_wrap_model_call_middleware_coexists_with_routing` for the requirement this example backs
(REQ-C8-1: the middleware sees the router itself as `request.model`, and routing still happens).

Run it:

    uv run python examples/agent_middleware.py

Offline, against a small scripted fake (`examples/_fakes.py`) — swap in real routes and the
wiring below is unchanged.
"""

from __future__ import annotations

from typing import Any

from _fakes import ScriptedToolChatModel
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, KeywordStrategy, routing_decision


@wrap_model_call
def log_model_calls(request: Any, handler: Any) -> Any:
    """Middleware that only observes — it never has to know a router is involved at all.

    `request.model` is the router itself, not whichever route answers (REQ-C8-1): from here,
    `ChatRouter` is indistinguishable from any other chat model.
    """
    print(f"agent is calling {type(request.model).__name__}")
    return handler(request)


def main() -> AIMessage:
    small = ScriptedToolChatModel(messages=iter([AIMessage(content="It's sunny in Warsaw.")]))
    frontier = ScriptedToolChatModel(messages=iter([AIMessage(content="It's sunny in Warsaw.")]))

    router = ChatRouter(
        routes={"small": small, "frontier": frontier},
        default_route="small",
        strategy=KeywordStrategy({"small": ["weather"], "frontier": ["urgent"]}),
    )

    agent = create_agent(model=router, middleware=[log_model_calls])
    result: dict[str, Any] = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Warsaw?"}]}
    )

    final = result["messages"][-1]
    decision = routing_decision(final)
    assert decision is not None
    print(f"final answer: {final.text!r} (route: {decision.route})")
    return final


if __name__ == "__main__":
    main()
