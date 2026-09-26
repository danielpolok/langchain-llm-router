"""Experimentation: force a route to compare two models on the same questions.

Before you change which model answers a kind of question, you want to see what the change does.
Setting `route` in the runtime config pins one call to that route and skips the strategy, so
both models answer exactly the same question and you can compare their answers and what they
cost. The same setting runs an A/B test on live traffic, one arm per route.

Run it with a Gemini API key in `GOOGLE_API_KEY`:

    uv run python examples/experimentation.py
"""

from textwrap import shorten

from langchain.chat_models import init_chat_model

from langchain_llm_router import ChatRouter, ForcedRouteError, HeuristicStrategy, routing_decision

small = init_chat_model("google_genai:gemini-3.5-flash-lite")
frontier = init_chat_model("google_genai:gemini-3.8-flash")

router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=HeuristicStrategy("small", "frontier"),
)

for question in [
    "Summarise the plot of Hamlet in two sentences.",
    "Write a subject line for an email announcing our new pricing.",
]:
    print(question)
    for route in ["small", "frontier"]:
        response = router.invoke(question, config={"configurable": {"route": route}})
        decision = routing_decision(response)
        tokens = response.usage_metadata["total_tokens"]
        preview = shorten(response.text, 60, placeholder=" …")
        print(f"  {decision.route:<8} {tokens:>4} tokens  {preview}")

# A forced route is never swapped for another, so a typo fails instead of running the wrong model.
try:
    router.invoke("Hello!", config={"configurable": {"route": "fronteir"}})
except ForcedRouteError as error:
    print(f"ForcedRouteError: {error}")

# It prints (the answers come from real models, so their wording changes from run to run):
#
#   Summarise the plot of Hamlet in two sentences.
#     small      72 tokens  Prince Hamlet of Denmark seeks revenge against his uncle …
#     frontier  588 tokens  After learning from his father's ghost that his uncle …
#   Write a subject line for an email announcing our new pricing.
#     small     287 tokens  Here are a few options for your new pricing announcement, …
#     frontier  993 tokens  Because the right subject line depends heavily on whether …
#   ForcedRouteError: forced route 'fronteir' is not one of the routes: 'small', 'frontier'; set
#   on_unavailable_forced_route='fallback' to fall back to the default route instead
#
# Both models answered the same questions, so you can judge the answers side by side, and the
# token counts show what each would cost: here the frontier model used several times as many.
# The misspelt route failed rather than quietly running on the default model, which would spoil a
# comparison. Each response's decision also records `forced=True`, so forced calls are easy to
# tell apart in a trace. To run this as an A/B test on live traffic, with each user pinned to one
# arm, see docs/guide.md#forcing-a-route.
