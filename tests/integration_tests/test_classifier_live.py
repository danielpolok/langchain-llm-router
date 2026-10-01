"""The classifier strategy with `lookback`, against a real Gemini classifier.

What only a real model can show: a follow-up that a keyword list gets wrong, or can't route at
all, is routed by what the conversation is about once the classifier reads the user's earlier
messages. "Which article of the civil code covers this?" contains "code", and "this doesn't work,
try again" contains nothing. The routes are fakes, since only the classifier's choice is under
test. Skips without `GEMINI_API_KEY` (`requires_env`, root `conftest.py`); override the model with
`LLM_ROUTER_GEMINI_MODEL`.
"""

from __future__ import annotations

import os

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_llm_router import ChatRouter, ClassifierStrategy, routing_decision
from tests.fakes import FakeChatModel

GEMINI_MODEL = os.environ.get("LLM_ROUTER_GEMINI_MODEL", "google_genai:gemini-3-flash-preview")

ROUTE_DESCRIPTIONS = {
    "general": "anything that isn't programming or law",
    "coder": "programming: code, bugs, errors, SQL, tooling",
    "legal": "law: contracts, clauses, liability, statutes, regulations",
}


def conversation(*turns: str) -> list[BaseMessage]:
    """The user's `turns`, each followed by a stand-in answer, and the last left unanswered."""
    messages: list[BaseMessage] = []
    for turn in turns:
        messages += [HumanMessage(turn), AIMessage("(an answer)")]
    return messages[:-1]


@pytest.mark.requires_env("GEMINI_API_KEY")
@pytest.mark.parametrize(
    ("turns", "route"),
    [
        pytest.param(
            [
                "Review this contract clause on limitation of liability",
                "Is it enforceable in Germany?",
                "Which article of the civil code covers this?",
            ],
            "legal",
            id="a-false-keyword-match",
        ),
        pytest.param(
            [
                "My Python deploy script fails with KeyError: 'user'",
                "I added the key to the config file",
                "this doesn't work, try again",
            ],
            "coder",
            id="a-follow-up-with-no-signal",
        ),
    ],
)
def test_the_classifier_routes_a_follow_up_by_the_conversation(
    turns: list[str], route: str
) -> None:
    strategy = ClassifierStrategy(init_chat_model(GEMINI_MODEL), ROUTE_DESCRIPTIONS, lookback=3)
    router = ChatRouter(
        routes={
            name: FakeChatModel(model_name=f"model-{name}", reply=f"{name} answer")
            for name in ROUTE_DESCRIPTIONS
        },
        default_route="general",
        strategy=strategy,
    )

    decision = routing_decision(router.invoke(conversation(*turns)))

    assert decision is not None
    assert (decision.route, decision.fallback) == (route, False)
