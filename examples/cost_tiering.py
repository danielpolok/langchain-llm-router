"""Cost tiering: easy questions go to a small model, hard ones to a frontier model.

Most of what an assistant is asked is easy, and a small model answers it as well as a frontier
model would, for a fraction of the price. `HeuristicStrategy` judges how hard each question looks
without calling a model: how long it is, whether it carries code, how many things it asks, and
whether it asks for reasoning, with words such as "compare" or "why does". Easy questions go to
the small model, and only the hard ones pay for the frontier model.

Run it with a Gemini API key in `GOOGLE_API_KEY`:

    uv run python examples/cost_tiering.py
"""

from textwrap import shorten

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback

from langchain_llm_router import ChatRouter, HeuristicStrategy, routing_decision

small = init_chat_model("google_genai:gemini-3.5-flash-lite")
frontier = init_chat_model("google_genai:gemini-3.8-flash")

router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),  # cheapest first
)

questions = [
    "What's the capital of France?",
    "Translate 'Thank you for your order' into Spanish.",
    "Compare the trade-offs of quicksort and mergesort on linked lists.",
    "Why does SELECT * FROM orders JOIN customers return duplicate rows?",
]

with get_usage_metadata_callback() as usage:
    for question in questions:
        response = router.invoke(question)
        decision = routing_decision(response)
        print(f"{decision.route:<8} {question}")
        print(f"         {decision.reason}")
        print(f"         {shorten(response.text, 72, placeholder=' …')}")

print()
for model, tokens in usage.usage_metadata.items():
    print(f"{model:<22} {tokens['total_tokens']:>5} tokens")

# It prints (the answers come from real models, so their wording changes from run to run):
#
#   small    What's the capital of France?
#            difficulty 0.00 < 1.00 (no signal fired)
#            The capital of France is Paris.
#   small    Translate 'Thank you for your order' into Spanish.
#            difficulty 0.00 < 1.00 (no signal fired)
#            "Thank you for your order" translates to Spanish as: **"Gracias por su …
#   frontier Compare the trade-offs of quicksort and mergesort on linked lists.
#            difficulty 1.00 >= 1.00 (analysis 1.00)
#            When sorting **arrays**, Quicksort is usually favored over Mergesort …
#   frontier Why does SELECT * FROM orders JOIN customers return duplicate rows?
#            difficulty 1.00 >= 1.00 (code 0.50, analysis 0.50)
#            When a `JOIN` produces duplicate (or more rows than expected), it …
#
#   gemini-3.5-flash-lite    104 tokens
#   gemini-3.8-flash        4410 tokens
#
# The two quick questions went to the small model. The comparison asks for reasoning, and the SQL
# question carries code and asks why, so both went to the frontier model. Each reason shows the
# score and the signals behind it. The token counts show where the money goes: the two in-depth
# answers used thousands of tokens on the pricier model, the quick ones about a hundred on the
# cheap one. See docs/strategies.md for how the score works and how to tune it.
