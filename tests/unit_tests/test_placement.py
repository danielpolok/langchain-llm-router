"""The router works everywhere a chat model goes, and coexists with agent
middleware rather than being bypassed by it.

Four placements, offline with fake routes: an LCEL chain (`prompt | router`), the "agent
backbone" use case — `create_agent(model=router)`, unbound, letting `create_agent`
call `router.bind_tools` itself, the way it would on a bare model (the *pre-bound* form,
`create_agent(model=router.bind_tools(...))`, is the placement test for a pre-bound router,
`tests/unit_tests/test_tools.py::test_create_agent_builds_and_answers_with_a_pre_bound_router`) —
a LangGraph node, and a message-history wrapper.

The last one is not `RunnableWithMessageHistory`: `langchain_core.runnables.history` (installed
1.6.3) deprecated it in 1.3.3 — its own `warn_deprecated` call says "Use LangGraph's built-in
persistence instead" — so this exercises the idiom that replaces it, a `StateGraph` compiled
with a checkpointer, on the router as its one node.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, ToolCall
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from langchain_llm_router import (
    ChatRouter,
    RoutingChoice,
    RoutingDecision,
    RoutingRequest,
    RoutingStrategy,
    routing_decision,
)
from tests.fakes import FakeChatModel, ToolCallingFakeChatModel

# --- A strategy that actually decides, so the LangGraph and message-history placements below
# exercise routing and not just delegation to a single route ---


class ByKeyword(RoutingStrategy):
    """Names `match` when `keyword` is in the request's text, `other` otherwise."""

    def __init__(self, keyword: str, match: str, other: str) -> None:
        self.keyword = keyword
        self.match = match
        self.other = other

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        if self.keyword in request.text:
            return RoutingChoice(route=self.match, reason=f"text names {self.keyword!r}")
        return RoutingChoice(route=self.other, reason="the other route otherwise")


class _ModelInputs(BaseCallbackHandler):
    """Records the messages every chat model call receives, across separate `graph.invoke`s."""

    def __init__(self) -> None:
        self.calls: list[list[BaseMessage]] = []

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        self.calls.extend(messages)


class _GraphState(TypedDict):
    """The one-key state every graph below carries."""

    messages: Annotated[list[AnyMessage], add_messages]


def _graph_with(router: BaseChatModel, **compile_kwargs: Any) -> Any:
    """A one-node graph that answers with `router` — `Any`, not `CompiledStateGraph[...]`: its
    generic `invoke`/`stream` overloads reject the informal message shapes (`("human", ...)`
    tuples, a bare `dict`) every test below uses, the same way `tests/unit_tests/test_
    conventions.py`'s own `graph_with` sidesteps it for the streaming tests."""

    def answer(state: _GraphState) -> dict[str, list[AnyMessage]]:
        return {"messages": [router.invoke(state["messages"])]}

    builder = StateGraph(_GraphState)
    builder.add_node("model", answer)
    builder.add_edge(START, "model")
    return builder.compile(**compile_kwargs)


# --- 1. An LCEL chain: `prompt | router` ---


def test_prompt_pipe_router_is_an_ordinary_lcel_chain() -> None:
    """`prompt | router` works as `prompt | any_chat_model` does — an LCEL chain that
    formats a prompt and hands it to the router, whose own answer (record included) comes back
    unchanged through the pipe."""
    router = ChatRouter(
        routes={"a": FakeChatModel(model_name="a", reply="hi from a")}, default_route="a"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are helpful."), ("human", "{question}")]
    )
    chain = prompt | router

    answer = chain.invoke({"question": "hello"})

    assert isinstance(answer, AIMessage)
    assert answer.content == "hi from a"
    assert routing_decision(answer) == RoutingDecision(route="a", reason="no strategy configured")


# --- 2. `create_agent(model=router)`, unbound: the agent backbone ---


@tool
def lookup(city: str) -> str:
    """Look up something about a city."""
    return f"info about {city}"


def test_create_agent_with_an_unbound_router_completes_its_tool_loop() -> None:
    """`create_agent(model=router)` — the router passed as given, not pre-bound —
    builds and completes a real tool loop, with `create_agent` calling `router.bind_tools`
    itself exactly as it would on a bare model."""
    from langchain.agents import create_agent

    route = ToolCallingFakeChatModel(
        model_name="a",
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="lookup", args={"city": "Lyon"}, id="c1", type="tool_call")
                ],
            ),
            AIMessage(content="Lyon is nice."),
        ],
    )
    router = ChatRouter(routes={"a": route}, default_route="a")

    agent = create_agent(model=router, tools=[lookup])
    out = agent.invoke({"messages": [{"role": "user", "content": "tell me about Lyon"}]})

    final = out["messages"][-1]
    assert final.content == "Lyon is nice."
    assert routing_decision(final) == RoutingDecision(route="a", reason="no strategy configured")


# --- 3. A LangGraph node: a plain function that calls the router ---


def test_a_langgraph_node_routes_and_returns_the_record() -> None:
    """The router works as an ordinary LangGraph node — a plain function that calls
    it and puts its answer into the graph's state, record included."""
    routes: dict[str, BaseChatModel] = {
        "a": FakeChatModel(model_name="a", reply="a answers"),
        "b": FakeChatModel(model_name="b", reply="b answers"),
    }
    router = ChatRouter(routes=routes, default_route="a", strategy=ByKeyword("special", "b", "a"))

    out = _graph_with(router).invoke({"messages": [("human", "a special request")]})

    final = out["messages"][-1]
    assert final.content == "b answers"
    assert routing_decision(final) == RoutingDecision(
        route="b", reason="text names 'special'", strategy="ByKeyword"
    )


# --- 4. A message-history wrapper: LangGraph's checkpointer (RunnableWithMessageHistory's
# replacement, per its own deprecation notice) ---


def test_a_checkpointed_graph_persists_the_conversation_across_turns() -> None:
    """The current message-history idiom — a checkpointer, not the deprecated
    `RunnableWithMessageHistory` — works with the router as the graph's model. The second turn's
    route call sees both the first turn's messages and the second's, proving the history
    persisted across two separate `graph.invoke` calls on the same thread, through the router."""
    route = FakeChatModel(
        model_name="a",
        script=[AIMessage(content="Hi Dan!"), AIMessage(content="Your name is Dan.")],
    )
    router = ChatRouter(routes={"a": route}, default_route="a")
    graph = _graph_with(router, checkpointer=InMemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "conversation-1"}}
    inputs = _ModelInputs()

    graph.invoke({"messages": [("human", "Hi, I'm Dan")]}, {**config, "callbacks": [inputs]})
    out = graph.invoke(
        {"messages": [("human", "What's my name?")]}, {**config, "callbacks": [inputs]}
    )

    final = out["messages"][-1]
    assert final.content == "Your name is Dan."
    assert routing_decision(final) == RoutingDecision(route="a", reason="no strategy configured")
    # The second call's own transcript, not just the graph's accumulated state, carried the
    # first turn forward: proof the checkpointer — not the router — is what persisted it.
    first_call, second_call = inputs.calls
    assert len(first_call) == 1
    assert len(second_call) == 3  # first human + first AI + second human


# --- 5. `@wrap_model_call` middleware coexists with routing, inside `create_agent` ---


def test_wrap_model_call_middleware_coexists_with_routing() -> None:
    """An agent with both the router as its model and `@wrap_model_call` middleware
    runs to completion — the middleware sees the router itself as `request.model` (not
    bypassed, and not some inner route), and routing still happens: the decision record is on
    the final answer, naming the route `ByKeyword` picked."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call

    seen_models: list[Any] = []

    @wrap_model_call
    def observe(request: Any, handler: Any) -> Any:
        seen_models.append(request.model)
        return handler(request)

    routes: dict[str, BaseChatModel] = {
        "a": ToolCallingFakeChatModel(model_name="a", reply="a's answer"),
        "b": ToolCallingFakeChatModel(model_name="b", reply="b's answer"),
    }
    router = ChatRouter(routes=routes, default_route="a", strategy=ByKeyword("urgent", "b", "a"))

    agent = create_agent(model=router, middleware=[observe])
    out = agent.invoke({"messages": [{"role": "user", "content": "an urgent question"}]})

    final = out["messages"][-1]
    assert final.content == "b's answer"
    assert seen_models == [router]  # the middleware saw the router itself as its model
    assert routing_decision(final) == RoutingDecision(
        route="b", reason="text names 'urgent'", strategy="ByKeyword"
    )
