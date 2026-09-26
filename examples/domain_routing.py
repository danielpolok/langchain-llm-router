"""Domain routing: programming questions go to a stronger model, everything else to a fast one.

A team assistant is asked about everything, but only the programming questions need the stronger
model. `KeywordStrategy` spots them by the words they use, such as "Python", "SQL" or "regex".
It matches whole words, ignoring case, and makes no model call of its own, so routing adds
nothing to the bill.

Run it with a Gemini API key in `GOOGLE_API_KEY`:

    uv run python examples/domain_routing.py
"""

from textwrap import shorten

from langchain.chat_models import init_chat_model

from langchain_llm_router import ChatRouter, KeywordStrategy, routing_decision

small = init_chat_model("google_genai:gemini-3.5-flash-lite")
frontier = init_chat_model("google_genai:gemini-3.8-flash")

router = ChatRouter(
    routes={"general": small, "coder": frontier},
    default_route="general",
    strategy=KeywordStrategy({"coder": ["python", "sql", "regex", "stack trace"]}),
)

for question in [
    "Write a regex that matches a UK postcode.",
    "In Python, how do I read a CSV file into a list of dicts?",
    "Draft a two-line out-of-office reply for the holidays.",
]:
    response = router.invoke(question)
    decision = routing_decision(response)
    print(f"{decision.route:<8} {question}")
    print(f"         {decision.reason}")
    print(f"         {shorten(response.text, 72, placeholder=' …')}")

# It prints (the answers come from real models, so their wording changes from run to run):
#
#   coder    Write a regex that matches a UK postcode.
#            matched keyword 'regex'
#            Here is the standard regex for matching UK postcodes, which balances …
#   coder    In Python, how do I read a CSV file into a list of dicts?
#            matched keyword 'python'
#            The standard and most Pythonic way to do this is using the built-in …
#   FallbackWarning: KeywordStrategy could not decide; falling back to the default route 'general'
#   general  Draft a two-line out-of-office reply for the holidays.
#            KeywordStrategy could not decide; fell back to the default route
#            I am currently out of the office for the holidays and will return on …
#
# The regex and Python questions matched a keyword and went to the coder route. The out-of-office
# reply matched none, so the strategy abstained: the default route answered, and the router
# warned, so a request no rule covers never goes unnoticed. If "everything else goes to general"
# is your policy, write it as a rule with `ConfigurableStrategy` and `always()`, and there is
# nothing to warn about. See docs/strategies.md for both.
