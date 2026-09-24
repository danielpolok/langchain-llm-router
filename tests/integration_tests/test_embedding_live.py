"""The embedding-similarity strategy against a real embeddings provider (acceptance
criterion 3 — "one integration test with real embeddings").

Gemini, not Ollama: the local Ollama server this suite's other live tests use answers chat
requests but was not started with `--embeddings` and has no embedding model pulled — its
`/api/embed` refuses every request in this environment. Gemini's Developer API always serves
embeddings, so it is the real provider available here; override the model with
`LLM_ROUTER_GEMINI_EMBED_MODEL`. Skips without `GEMINI_API_KEY` (`requires_env`, root
`conftest.py`).

Real embeddings are semantically meaningful, unlike `DeterministicFakeEmbedding`'s unit-test
vectors, so this checks what only a real provider can prove — a request lands on the route whose
examples actually read closest to it, at a generous threshold, without pinning an exact score —
and that the call is where every offline test already proves it should be: its own child run of
the strategy's.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_llm_router import ChatRouter, routing_decision
from langchain_llm_router.strategies.embedding import EmbeddingStrategy
from tests.fakes import FakeChatModel

EMBED_MODEL = os.environ.get("LLM_ROUTER_GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")

EXAMPLES = {
    "coder": [
        "why is my Python function raising a KeyError",
        "refactor this SQL query for performance",
        "write a unit test for this class",
    ],
    "support": [
        "I forgot my account password",
        "my order hasn't shipped yet",
        "please cancel my subscription",
    ],
}


@pytest.mark.requires_env("GEMINI_API_KEY")
def test_a_real_embeddings_call_routes_to_the_closest_domain_and_is_traced() -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    strategy = EmbeddingStrategy(embeddings, EXAMPLES, threshold=0.3)
    router = ChatRouter(
        routes={
            "coder": FakeChatModel(model_name="model-coder", reply="coder answer"),
            "support": FakeChatModel(model_name="model-support", reply="support answer"),
        },
        default_route="support",
        strategy=strategy,
    )
    collector = RunCollectorCallbackHandler()

    message = router.invoke(
        "my code throws a null pointer exception, can you help me debug it",
        config={"callbacks": [collector]},
    )

    record = routing_decision(message)
    assert record is not None
    assert record.route == "coder"
    assert record.fallback is False

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    (embed_run,) = strategy_run.child_runs
    assert embed_run.name == "embed_query"
    assert embed_run.run_type == "chain"
    assert embed_run.error is None
    outputs = embed_run.outputs or {}
    assert outputs["input_length"] > 0
    assert outputs["estimated_tokens"] > 0
    assert "cost_estimate_method" in outputs
    assert route_run.child_runs == []
