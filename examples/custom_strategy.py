"""Custom strategy: plug a classifier you already run in as the routing policy.

A support team already labels every ticket with its own triage classifier. Incidents should get
the frontier model, because a wrong answer during an outage is expensive, and everything else
can go to the small one. A strategy can be a plain function: it receives the request and returns
a route name, or a `RoutingChoice` that also says why. Wiring the classifier in takes a few lines,
and no subclass.

Run it with a Gemini API key in `GOOGLE_API_KEY`:

    uv run python examples/custom_strategy.py
"""

from textwrap import shorten

from langchain.chat_models import init_chat_model

from langchain_llm_router import ChatRouter, RoutingChoice, routing_decision

small = init_chat_model("google_genai:gemini-3.5-flash-lite")
frontier = init_chat_model("google_genai:gemini-3.8-flash")


def triage(ticket):
    """Stands in for the team's own classifier: a trained model, a rules engine or a service."""
    text = ticket.lower()
    if any(phrase in text for phrase in ("outage", "is down", "been down", "not loading")):
        return "incident"
    if any(phrase in text for phrase in ("invoice", "refund", "charged")):
        return "billing"
    return "how-to"


def by_triage(request):
    label = triage(request.text)
    route = "frontier" if label == "incident" else "small"
    return RoutingChoice(route, f"triage labelled it {label!r}")


router = ChatRouter(
    routes={"small": small, "frontier": frontier},
    default_route="small",
    strategy=by_triage,
)

for ticket in [
    "Checkout has been down for every customer since 9am. What should we check first?",
    "I was charged twice for my March invoice. How do I get a refund?",
    "How do I export my contacts to a CSV file?",
]:
    response = router.invoke(ticket)
    decision = routing_decision(response)
    print(f"{decision.route:<8} {ticket}")
    print(f"         {decision.reason}")
    print(f"         {shorten(response.text, 72, placeholder=' …')}")

# It prints (the answers come from real models, so their wording changes from run to run):
#
#   frontier Checkout has been down for every customer since 9am. What should we check first?
#            triage labelled it 'incident'
#            Because checkout is down for **100% of customers starting at a …
#   small    I was charged twice for my March invoice. How do I get a refund?
#            triage labelled it 'billing'
#            I can certainly help you get that sorted out and make sure you get …
#   small    How do I export my contacts to a CSV file?
#            triage labelled it 'how-to'
#            Because contacts can be stored in many different places (Google, …
#
# The outage went to the frontier model, and the billing and how-to tickets to the small one.
# Each reason is the one `by_triage` wrote, so a trace shows the classifier's label. A function
# can also return just a route name, or `None` to leave the ticket to the default route. When you
# need state, an async call or the whole conversation, subclass `RoutingStrategy` instead; see
# docs/strategies.md.
