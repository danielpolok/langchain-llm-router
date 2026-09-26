"""Agent backbone: the router as the model behind a LangChain agent.

`ChatRouter` is a chat model, so `create_agent` takes it like any other. Each customer message
picks its model once, from what the customer asked, and every step of the agent's tool loop
stays on that model. A tool result never moves the conversation to a different model partway
through, so a simple lookup stays cheap and a hard question keeps the model it started with.

Run it with a Gemini API key in `GOOGLE_API_KEY`:

    uv run python examples/agent_backbone.py
"""

from textwrap import shorten

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langchain_llm_router import ChatRouter, HeuristicStrategy, routing_decision

small = init_chat_model("google_genai:gemini-3.5-flash-lite")
frontier = init_chat_model("google_genai:gemini-3.8-flash")

router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),
)


@tool
def get_order(order_id: str) -> str:
    """Look up an order by its ID, such as A-1001."""
    orders = {
        "A-1001": "shipped on 22 September, arriving on 26 September",
        "A-1002": "delivered on 20 September, reported damaged by the customer",
    }
    return orders.get(order_id, f"there is no order {order_id}")


agent = create_agent(router, tools=[get_order])

for question in [
    "Where is my order A-1001?",
    "My order A-1002 arrived damaged. Compare a refund and a replacement, with the trade-offs.",
]:
    print(question)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    for message in result["messages"]:
        if message.type == "ai":
            calls = [call["name"] for call in message.tool_calls]
            answer = calls or shorten(message.text, 72, placeholder=" …")
            print(f"  {routing_decision(message).route:<8} {answer}")

# It prints (the answers come from real models, so their wording changes from run to run):
#
#   Where is my order A-1001?
#     small    ['get_order']
#     small    Your order A-1001 was shipped on September 22 and is scheduled to …
#   My order A-1002 arrived damaged. Compare a refund and a replacement, with the trade-offs.
#     frontier ['get_order']
#     frontier I'm sorry to hear that your order **A-1002** arrived damaged. Here is …
#
# Each message picked its model once. The status question stayed on the small model for both
# steps: the tool call and the answer. The complaint asks for a comparison, so it went to the
# frontier model and stayed there after the tool result came back. The router routes on the
# customer's message, never on a tool result, so an agent's loop can't change models halfway
# through. See docs/guide.md for agents, and examples/agent_middleware.py for choosing a model
# from the agent's state.
