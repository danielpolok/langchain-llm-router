# Examples

One runnable script per use case. Each runs offline against fake models, so no API key is needed,
and `tests/unit_tests/test_examples.py` runs every one of them in CI.

```bash
uv run python examples/cost_tiering.py
```

| Example | Use case | Strategy shown |
| --- | --- | --- |
| [`cost_tiering.py`](cost_tiering.py) | Simple requests to a small model, hard ones to a frontier model | `HeuristicStrategy` |
| [`domain_routing.py`](domain_routing.py) | Code requests to a code-strong model | `KeywordStrategy` |
| [`agent_backbone.py`](agent_backbone.py) | The router as the model behind `create_agent` | `KeywordStrategy`, tool calling |
| [`custom_strategy.py`](custom_strategy.py) | An existing classifier plugged in as the strategy | a plain function |
| [`experimentation.py`](experimentation.py) | Force a route per call to compare models on the same traffic | forced routes |
| [`agent_middleware.py`](agent_middleware.py) | `@wrap_model_call` middleware coexisting with the router | `KeywordStrategy` + middleware |

To use real models, replace each fake in `routes=` with `init_chat_model("provider:model")` or any
other chat model. Nothing else in the example changes.

The documentation shows the same use cases with real models: [strategies](../docs/strategies.md)
and [using the router](../docs/guide.md).
