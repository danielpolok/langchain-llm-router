"""Domain routing (PRD §4): code requests to a code-strong model, everything else to the
general one.

`KeywordStrategy` (R6, R7) matches whole words in the current request's text against rules you
write — no extra model or API call. See `docs/strategies.md` for the three strategy levels and
`langchain_llm_router.strategies.keyword` for exactly how matching works (whole words, case
folded; a compiled `re.Pattern` for anything a word list can't express).

Run it:

    uv run python examples/domain_routing.py

Offline, against `GenericFakeChatModel` (`langchain_core`) — swap in real routes and the router
usage is unchanged.
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, KeywordStrategy, routing_decision


def main() -> list[AIMessage]:
    coder = GenericFakeChatModel(
        messages=iter([AIMessage(content="The traceback means the list is empty at line 12.")]),
        name="coder",
    )
    general = GenericFakeChatModel(
        messages=iter([AIMessage(content="Warsaw is the capital of Poland.")]),
        name="general",
    )

    router = ChatRouter(
        routes={"coder": coder, "general": general},
        default_route="general",
        strategy=KeywordStrategy(
            {
                # Specific rules first (D1): a rule set is read top to bottom, route by route,
                # keyword by keyword — the first match wins.
                "coder": ["python", "regex", "stack trace", "traceback"],
            }
        ),
    )

    replies = []
    for question in [
        "Why does this Python stack trace mention an IndexError?",
        "What's the capital of Poland?",
    ]:
        response = router.invoke(question)
        decision = routing_decision(response)
        assert decision is not None
        print(f"{question!r} -> {decision.route} ({decision.reason})")
        replies.append(response)
    return replies


if __name__ == "__main__":
    main()
