# Domain routing

**Goal:** send each kind of request to the model that's strongest at it — code questions to a
code-strong model, everything else to a general one — without a classifier to train or a service to
call.

## The approach

Use `KeywordStrategy`. Whole words in the request's text are matched against rules you write; the
first rule that matches wins, and the default route answers when none does.

```python
import warnings
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_llm_router import ChatRouter, FallbackWarning, KeywordStrategy, routing_decision

# "Everything else" is meant to reach the default route; silence the warning that says so.
warnings.filterwarnings("ignore", category=FallbackWarning)

coder = GenericFakeChatModel(
    messages=cycle([AIMessage("The traceback means the list is empty at line 12.")]), name="coder"
)
general = GenericFakeChatModel(
    messages=cycle([AIMessage("Warsaw is the capital of Poland.")]), name="general"
)

router = ChatRouter(
    routes={"coder": coder, "general": general},
    default_route="general",
    strategy=KeywordStrategy({"coder": ["python", "regex", "stack trace", "traceback"]}),
)

for question in [
    "Why does this Python stack trace mention an IndexError?",
    "What's the capital of Poland?",
]:
    decision = routing_decision(router.invoke(question))
    print(f"{question!r} -> {decision.route} ({decision.reason})")
```

For the second question nothing matches, so the default route answers — the intended path for
"everything else". The router reports it with a `FallbackWarning`, which the code above silences;
the decision record still says `fallback=True` (see
[fallbacks](../capabilities/fallbacks-and-retries.md)).

## Run the full example

[`examples/domain_routing.py`](../../examples/domain_routing.py) is the same story as a script,
tested in CI:

```bash
uv run python examples/domain_routing.py
```

## What to look at

- **`decision.reason`** names the keyword that matched, so a misroute points straight at the rule.
- **Rule order.** Rules are read top to bottom, route by route; put the specific ones first.
- **Whole-word matching.** `"python"` matches `"Python?"` but not `"pythonic"`. Use a compiled
  `re.Pattern` for what a word list can't say, such as `re.compile(r"regexe?s?")` or scripts with
  no word boundaries.

## Variations

- The meaning matters more than the words:
  [`EmbeddingStrategy`](../capabilities/strategies.md#embeddingstrategy) routes on similarity to
  example requests.
- Combine a domain rule with a difficulty rule:
  [`ConfigurableStrategy`](../capabilities/strategies.md#configurablestrategy).
