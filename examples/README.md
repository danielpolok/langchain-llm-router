# Examples

One runnable script per use case, plus the agent-middleware case (see
[`../docs/scope.md`](../docs/scope.md)). Each runs offline —
against `GenericFakeChatModel` (`langchain_core`) or a small scripted fake in
[`_fakes.py`](_fakes.py), never a real provider — so no API key is needed, and
`tests/unit_tests/test_examples.py` runs every one of them in CI.

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

Every route in these examples is a fake — swap in `init_chat_model("provider:model")` (or any
other `BaseChatModel`) for each `routes=` entry and the rest of the example is unchanged; that
substitutability is the point of building against `langchain-core`'s own chat-model interface.

See also:

- [`../README.md`](../README.md) — installation, the public API, and a walkthrough of the same
  ground these examples cover.
- [`../docs/strategies.md`](../docs/strategies.md) — the three strategy levels and the full
  strategy interface reference.
- [`../docs/decision-record.md`](../docs/decision-record.md) — the decision record, and every
  warning and error the router raises.
- [`../docs/scope.md`](../docs/scope.md) — when to reach for agent middleware instead, what
  the router deliberately doesn't do, and the prompt-caching caveat.
