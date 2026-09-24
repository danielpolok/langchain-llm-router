"""Cost math: real usage priced correctly per model, and the embedding estimate matches the
embedding strategy's own chars-per-token method.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.messages.ai import UsageMetadata

from benchmark.costing import chat_cost, cost_of, estimate_embedding_cost
from benchmark.pricing import EMBEDDING_PRICE_PER_MILLION_TOKENS, TokenPrice, price_for
from langchain_llm_router.strategies.embedding import _CHARS_PER_TOKEN


def _usage(input_tokens: int, output_tokens: int) -> UsageMetadata:
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def test_price_for_matches_the_provider_qualified_model_name() -> None:
    price = price_for("google_genai:gemini-3-flash-preview")
    assert price == TokenPrice(input_per_million=0.50, output_per_million=3.00)


def test_price_for_matches_ollama_qualified_names_too() -> None:
    assert price_for("ollama:qwen3:8b").input_per_million == 0.0


def test_price_for_an_unpriced_model_raises() -> None:
    with pytest.raises(KeyError, match="no-such-model"):
        price_for("google_genai:no-such-model")


def test_chat_cost_sums_every_priced_model() -> None:
    usage = {
        "google_genai:gemini-3-flash-preview": _usage(1_000_000, 1_000_000),
        "ollama:qwen3:8b": _usage(1_000_000, 1_000_000),
    }
    # $0.50 + $3.00 for the Gemini line, $0.00 for the free local one.
    assert chat_cost(usage) == pytest.approx(3.50)


def test_chat_cost_of_no_usage_is_zero() -> None:
    assert chat_cost({}) == 0.0


def test_embedding_estimate_uses_the_strategy_s_own_chars_per_token() -> None:
    text = "x" * (_CHARS_PER_TOKEN * 10)
    expected_tokens = math.ceil(len(text) / _CHARS_PER_TOKEN)
    expected_cost = expected_tokens * EMBEDDING_PRICE_PER_MILLION_TOKENS / 1_000_000
    assert estimate_embedding_cost(text) == pytest.approx(expected_cost)


def test_embedding_estimate_of_empty_text_is_at_least_one_token() -> None:
    assert estimate_embedding_cost("") > 0.0


def test_cost_of_without_an_embedding_query_has_zero_embedding_cost() -> None:
    breakdown = cost_of({"ollama:qwen3:8b": _usage(10, 10)})
    assert breakdown.embedding_usd == 0.0
    assert breakdown.total_usd == breakdown.chat_usd


def test_cost_of_with_an_embedding_query_adds_the_estimate() -> None:
    breakdown = cost_of({}, embedding_query_text="hello world")
    assert breakdown.chat_usd == 0.0
    assert breakdown.embedding_usd > 0.0
    assert breakdown.total_usd == breakdown.embedding_usd
