"""Experimentation: force a route per call to compare models on the same traffic.

Runtime config pins one call to a named route — the strategy is skipped
entirely, and a forced route is never silently swapped: an unknown or tool-incapable
forced route raises `ForcedRouteError` by default, and only falls back when the router is built
with `on_unavailable_forced_route="fallback"`. See `docs/decision-record.md` for the full
warning and error reference.

Run it:

    uv run python examples/experimentation.py

Offline, against `GenericFakeChatModel` (`langchain_core`) — swap in real routes and the
`config={"configurable": {"route": ...}}` call is unchanged.
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, ForcedRouteError, KeywordStrategy, routing_decision


def main() -> list[AIMessage]:
    model_a = GenericFakeChatModel(
        messages=iter([AIMessage(content="(model A) Warsaw is the capital of Poland.")] * 2),
        name="model-a",
    )
    model_b = GenericFakeChatModel(
        messages=iter([AIMessage(content="(model B) Warsaw, Poland.")] * 2),
        name="model-b",
    )

    router = ChatRouter(
        routes={"model-a": model_a, "model-b": model_b},
        default_route="model-a",
        strategy=KeywordStrategy({"model-b": ["urgent"]}),
    )

    question = "What's the capital of Poland?"
    replies = []
    for forced_route in ["model-a", "model-b"]:
        # Pin this one call, regardless of what the strategy would otherwise choose — the
        # strategy never even runs.
        response = router.invoke(question, config={"configurable": {"route": forced_route}})
        decision = routing_decision(response)
        assert decision is not None
        assert decision.forced
        print(f"forced {forced_route!r} -> {decision.route}: {response.text}")
        replies.append(response)

    # Forcing an unknown route errors by default rather than silently falling back —
    # a comparison that quietly ran on the wrong model would be worse than no comparison.
    try:
        router.invoke(question, config={"configurable": {"route": "model-c"}})
    except ForcedRouteError as error:
        print(f"forcing an unknown route raised: {error}")

    return replies


if __name__ == "__main__":
    main()
