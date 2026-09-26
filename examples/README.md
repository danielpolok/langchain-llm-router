# Examples

One runnable script per use case, each a small real-world scenario: an assistant, a support
queue, an order-support agent, a model comparison. Every script calls the same two real models as
the rest of the documentation, a small one and a frontier one, and prints which of them answered
each request, with the start of its answer. Each opens with what it's for and ends with what it
printed, so you can read one without running it.

## Running them

The examples call Gemini, so they need a Gemini API key in `GOOGLE_API_KEY` (or
`GEMINI_API_KEY`); LangChain's
[Google GenAI integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
explains how to get one. From a clone of this repository:

```bash
uv sync
export GOOGLE_API_KEY="..."
uv run python examples/cost_tiering.py
```

The routes are the same on every run. The answers come from real models, so their wording
changes from run to run. To try other models, change the two `init_chat_model(...)` lines at the
top of a script; nothing else in it depends on the provider.

## The examples

| Example | Scenario | Shows |
| --- | --- | --- |
| [`cost_tiering.py`](cost_tiering.py) | An assistant sends easy questions to the small model and hard ones to the frontier model | [`HeuristicStrategy`](../docs/strategies.md#heuristicstrategy-route-on-difficulty) |
| [`domain_routing.py`](domain_routing.py) | A team assistant sends programming questions to the stronger model | [`KeywordStrategy`](../docs/strategies.md#keywordstrategy-route-on-words) |
| [`custom_strategy.py`](custom_strategy.py) | A support team's own ticket classifier picks the model | [A function as the strategy](../docs/strategies.md#a-function) |
| [`agent_backbone.py`](agent_backbone.py) | An order-support agent keeps each customer's tool loop on one model | [The router inside an agent](../docs/guide.md#inside-an-agent) |
| [`agent_middleware.py`](agent_middleware.py) | Middleware sends long conversations to the frontier model, and the router picks for the rest | [`@wrap_model_call` middleware](https://docs.langchain.com/oss/python/langchain/middleware) with the router |
| [`experimentation.py`](experimentation.py) | Both models answer the same questions, to compare answers and cost before switching | [Forcing a route](../docs/guide.md#forcing-a-route) |
