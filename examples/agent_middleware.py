"""Agent middleware: `@wrap_model_call` chooses from agent state, and the router does the rest.

The router's strategy sees only the current request, which keeps routing stable inside an agent.
Some choices depend on the agent's state instead: how long the conversation has grown, how many
tools have run, something another middleware stored. Those belong in `@wrap_model_call`
middleware, and the two work together. The router is the agent's model, and the middleware
steps in only when its own rule applies.

The rule here follows LangChain's own example of dynamic model selection: a long conversation
goes to the frontier model. Every other call is left to the router.

Run it with a Gemini API key in `GOOGLE_API_KEY`:

    uv run python examples/agent_middleware.py
"""

from textwrap import shorten

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.chat_models import init_chat_model

from langchain_llm_router import ChatRouter, HeuristicStrategy, routing_decision

small = init_chat_model("google_genai:gemini-3.5-flash-lite")
frontier = init_chat_model("google_genai:gemini-3.8-flash")

router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),
)


@wrap_model_call
def long_conversations_to_frontier(request, handler):
    """After six messages the frontier model answers. Until then, the router decides."""
    if len(request.messages) > 6:
        return handler(request.override(model=frontier))
    return handler(request)


agent = create_agent(router, middleware=[long_conversations_to_frontier])

question = {"role": "user", "content": "What should I pack for Lisbon in October?"}
earlier_turns = [
    {"role": "user", "content": "I'm planning a trip to Portugal."},
    {"role": "assistant", "content": "Lovely! How long are you staying?"},
    {"role": "user", "content": "Ten days, mostly in Lisbon and Porto."},
    {"role": "assistant", "content": "Good choice. Will you take the train between them?"},
    {"role": "user", "content": "Yes, and I'd like a day trip to Sintra."},
    {"role": "assistant", "content": "Sintra is an easy train ride from Lisbon."},
]

for chat, messages in [("new chat", [question]), ("long chat", [*earlier_turns, question])]:
    answer = agent.invoke({"messages": messages})["messages"][-1]
    decision = routing_decision(answer)  # None when the middleware chose the model
    chosen = f"the router chose {decision.route}" if decision else "the middleware chose frontier"
    print(f"{chat:<9}  {chosen}")
    print(f"           {shorten(answer.text, 72, placeholder=' …')}")

# It prints (the answers come from real models, so their wording changes from run to run):
#
#   new chat   the router chose small
#              October is one of the best months to visit Lisbon. The intense summer …
#   long chat  the middleware chose frontier
#              October in Lisbon is a transitional month—you'll get a mix of warm, …
#
# The same question went to the small model in a new chat, where the router decided, and to the
# frontier model after six earlier messages, where the middleware's rule applied first. The
# second answer has no routing decision, because the router didn't choose its model. See
# LangChain's middleware docs for everything `@wrap_model_call` can read and change:
# https://docs.langchain.com/oss/python/langchain/middleware
