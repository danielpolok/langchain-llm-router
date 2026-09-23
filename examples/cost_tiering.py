"""Cost tiering (PRD §4): simple requests to a small model, hard ones to a frontier model.

`HeuristicStrategy` (R6, R7) scores each request from signals it can compute on the spot —
length, code, how many things it asks for, and so on — with no extra model or API call. See
`docs/strategies.md` for the three strategy levels and `langchain_llm_router.strategies.heuristic`
for what the score is built from.

Run it:

    uv run python examples/cost_tiering.py

Everything here runs offline, against `GenericFakeChatModel` (`langchain_core`, no API key
needed) — swap in real routes, e.g. `init_chat_model("ollama:qwen3:8b")` and
`init_chat_model("google_genai:gemini-3-flash-preview")`, and the router usage is unchanged.
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, HeuristicStrategy, routing_decision


def main() -> list[AIMessage]:
    small = GenericFakeChatModel(messages=iter([AIMessage(content="2 + 2 is 4.")]), name="small")
    frontier = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="A proof by contradiction: suppose there are finitely many primes..."
                )
            ]
        ),
        name="frontier",
    )

    router = ChatRouter(
        routes={"small": small, "frontier": frontier},
        default_route="small",
        # Cheapest tier first (R6). The default threshold is tuned for the two-tier case;
        # HeuristicStrategy("small", "mid", "frontier", thresholds=[...]) adds a middle tier.
        strategy=HeuristicStrategy("small", "frontier"),
    )

    replies = []
    for question in [
        "what's 2 + 2?",
        "Prove that there are infinitely many primes, and explain why the proof works.",
    ]:
        response = router.invoke(question)
        decision = routing_decision(response)
        assert decision is not None
        print(f"{question!r} -> {decision.route} ({decision.reason})")
        replies.append(response)
    return replies


if __name__ == "__main__":
    main()
