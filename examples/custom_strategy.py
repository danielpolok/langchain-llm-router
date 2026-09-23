"""Custom strategy (PRD §4): plug an existing classifier in as the routing policy.

R6's third level needs no subclassing at all for the common case — a plain function is coerced
into a `RoutingStrategy` (REQ-R6-2). This is the shape a team's own classifier takes: something
that already turns a request into a label, wrapped in a few lines that turn the label into a
`RoutingChoice`.

    def pick(request: RoutingRequest) -> RoutingChoice | str | None:
        label = my_existing_classifier(request.text)
        return label  # a bare route name is accepted; the reason is filled in for you

    ChatRouter(routes=..., default_route=..., strategy=pick)

For anything the function shape can't express — state across requests, an async model call, the
whole transcript instead of just the current request (`wants_full_context`) — subclass
`RoutingStrategy` directly; see `docs/strategies.md` for the full interface reference and
`langchain_llm_router.strategies.classifier.ClassifierStrategy` for a worked subclass.

Run it:

    uv run python examples/custom_strategy.py

Offline, against `GenericFakeChatModel` (`langchain_core`) — the "existing classifier" here is
a plain function standing in for a team's own code (a trained model, a rules engine, a call to
another service); nothing about wiring it into `strategy=` depends on what it's not.
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, RoutingChoice, RoutingRequest, routing_decision


def existing_classifier(text: str) -> str:
    """Stands in for a team's own classifier — a trained model, a rules engine, a REST call.

    All `strategy=` needs from it is: given the request's text, return a route name.
    """
    return "urgent" if "asap" in text.lower() or "urgent" in text.lower() else "normal"


def pick_route(request: RoutingRequest) -> RoutingChoice:
    """The few lines REQ-R6-2 asks for: adapt the classifier's answer to `RoutingChoice`.

    A bare route name (`return existing_classifier(request.text)`) works too — the router
    fills in a reason itself. Returning a `RoutingChoice` explicitly just lets the trace say
    *why*, which is worth the one extra line when the classifier's answer alone wouldn't.
    """
    label = existing_classifier(request.text)
    return RoutingChoice(route=label, reason=f"existing_classifier labelled it {label!r}")


def main() -> list[AIMessage]:
    urgent = GenericFakeChatModel(
        messages=iter([AIMessage(content="Escalating now — on it.")]), name="urgent"
    )
    normal = GenericFakeChatModel(
        messages=iter([AIMessage(content="I'll get to this within a day.")]), name="normal"
    )

    router = ChatRouter(
        routes={"urgent": urgent, "normal": normal},
        default_route="normal",
        strategy=pick_route,  # a plain function — no RoutingStrategy subclass needed
    )

    replies = []
    for question in ["The prod database is down, need this ASAP.", "Any tips for a Friday demo?"]:
        response = router.invoke(question)
        decision = routing_decision(response)
        assert decision is not None
        print(f"{question!r} -> {decision.route} ({decision.reason})")
        replies.append(response)
    return replies


if __name__ == "__main__":
    main()
