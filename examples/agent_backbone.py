"""Agent backbone (PRD §4): the router as the model behind `create_agent`.

`ChatRouter` is a `BaseChatModel` (C1), so it goes anywhere a chat model goes, `create_agent`
included: the strategy routes each model call the agent makes on that call's current request
(R4), and every tool-calling round trip goes through the same routing pipeline as a plain
`invoke`. See `examples/agent_middleware.py` for combining this with `@wrap_model_call`, and
`docs/scope.md` for when agent-state-aware selection calls for middleware instead.

Run it:

    uv run python examples/agent_backbone.py

Offline, against a small scripted fake (`examples/_fakes.py`) standing in for two real routes —
swap in `init_chat_model(...)` for each and the router and agent wiring are unchanged.
"""

from __future__ import annotations

from typing import Any

from _fakes import ScriptedToolChatModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langchain_llm_router import ChatRouter, KeywordStrategy, routing_decision


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}."


def main() -> AIMessage:
    small = ScriptedToolChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_weather", "args": {"city": "Warsaw"}, "id": "1"}],
                ),
                AIMessage(content="It's sunny in Warsaw."),
            ]
        ),
        name="small",
    )
    frontier = ScriptedToolChatModel(messages=iter([]), name="frontier")

    router = ChatRouter(
        routes={"small": small, "frontier": frontier},
        default_route="small",
        # A weather lookup doesn't need the frontier model; only "urgent" traffic does. R4:
        # every call in the tool-calling round trip below routes on this same original request,
        # not on the tool's own output.
        strategy=KeywordStrategy({"small": ["weather"], "frontier": ["urgent"]}),
    )

    agent = create_agent(model=router, tools=[get_weather])
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
