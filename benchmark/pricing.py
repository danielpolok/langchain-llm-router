"""Pinned $ pricing for the models this benchmark calls (T-140).

Prices drift; these are a snapshot, not a live lookup — `PRICING_SOURCE` and `PRICING_ASOF` say
where they came from and when, so a reader can tell whether to re-check them before trusting an
old report. Override a figure with `LLM_ROUTER_BENCHMARK_PRICE_OVERRIDES` (see below) rather than
editing this file for a one-off re-run with different assumptions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_ASOF = "2026-09-23"


@dataclass(frozen=True)
class TokenPrice:
    """$ per 1M tokens. `input` and `output` only — no cached-input discount modelled."""

    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


# Keyed by the `init_chat_model` identifier the harness uses, matching what `usage_metadata`'s
# model name reports for the route that ran (REQ-R3-1's "keyed by the model that ran").
CHAT_PRICES: dict[str, TokenPrice] = {
    # Gemini 3.8 Flash (the "frontier" route), PRICING_ASOF: standard tier through 2026-12-31.
    "gemini-3.8-flash": TokenPrice(input_per_million=0.75, output_per_million=3.75),
    # Gemini 3.5 Flash-Lite (the "small" route), PRICING_ASOF.
    "gemini-3.5-flash-lite": TokenPrice(input_per_million=0.30, output_per_million=2.50),
    # Gemini 3 Flash Preview — this repo's usual default route, not one of this benchmark's
    # arms, kept priced for anyone who overrides FRONTIER_MODEL/SMALL_MODEL back to it.
    "gemini-3-flash-preview": TokenPrice(input_per_million=0.50, output_per_million=3.00),
    # Gemini 3.1 Pro Preview (the judge model — never a route), prompts <=200k tokens.
    "gemini-3.1-pro-preview": TokenPrice(input_per_million=2.00, output_per_million=12.00),
    # Ollama runs locally: no per-token $ charge. Real infrastructure cost (electricity, the
    # machine) isn't zero, but isn't priced in $/token either, so it is out of scope here — an
    # assumption worth restating in the report, not silently baked into a number that looks
    # like it means the same thing as Gemini's. Not one of this benchmark's arms either (see
    # `arms.ollama_smoke_model`) — priced so the live smoke test's own cost accounting doesn't
    # raise on an unpriced model.
    "qwen3:8b": TokenPrice(input_per_million=0.0, output_per_million=0.0),
}

# Gemini Embedding 2, PRICING_ASOF: $0.20/1M text input tokens. `EmbeddingStrategy` reports no
# usage (embedding.py's docstring), so this prices T-133's chars-per-token estimate, matching
# what the T-140 issue asks for ("the input-length estimate from T-133").
EMBEDDING_PRICE_PER_MILLION_TOKENS = 0.20


def _apply_overrides(prices: dict[str, TokenPrice]) -> dict[str, TokenPrice]:
    """`LLM_ROUTER_BENCHMARK_PRICE_OVERRIDES`: a JSON object of
    `{"model": {"input_per_million": x, "output_per_million": y}}`, merged over the defaults.
    """
    raw = os.environ.get("LLM_ROUTER_BENCHMARK_PRICE_OVERRIDES")
    if not raw:
        return prices
    overrides = json.loads(raw)
    merged = dict(prices)
    for model, fields in overrides.items():
        merged[model] = TokenPrice(**fields)
    return merged


def chat_prices() -> dict[str, TokenPrice]:
    """`CHAT_PRICES`, with any environment overrides applied."""
    return _apply_overrides(CHAT_PRICES)


def price_for(model_name: str, prices: dict[str, TokenPrice] | None = None) -> TokenPrice:
    """The price for `model_name`, matched by suffix so `"google_genai:gemini-3-flash-preview"`
    and `"ollama:qwen3:8b"` (the `usage_metadata` model names LangChain reports) find their
    provider-qualified entries without this table having to spell every provider prefix out.

    Raises `KeyError` naming the model for one this table has no price for — a silent $0.00 for
    an unpriced model would understate cost rather than fail loudly.
    """
    table = prices if prices is not None else chat_prices()
    for key, price in table.items():
        if model_name == key or model_name.endswith(f":{key}"):
            return price
    msg = f"no price for model {model_name!r}; add it to benchmark.pricing.CHAT_PRICES"
    raise KeyError(msg)
