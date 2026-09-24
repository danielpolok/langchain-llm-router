# Calling conventions

The router is a chat model, so everything you can do with one works the same way here: `invoke`,
`stream`, `batch`, the async forms of each, `generate`, and use inside chains and graphs. Each
call is routed on its own request; the response is the selected route's own.

```python
import asyncio
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_llm_router import ChatRouter, KeywordStrategy, routing_decision


def fake(name: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=cycle([AIMessage(f"({name}) hello there")]), name=name)


router = ChatRouter(
    routes={"small": fake("small"), "coder": fake("coder")},
    default_route="small",
    strategy=KeywordStrategy({"coder": ["python"], "small": ["capital"]}),
)
```

## `invoke` and `ainvoke`

```python
response = router.invoke("Why is my Python loop slow?")
assert routing_decision(response).route == "coder"

response = asyncio.run(router.ainvoke("What's the capital of Poland?"))
assert routing_decision(response).route == "small"
```

Anything a chat model accepts as input works: a string, a list of messages or a list of
`{"role": ..., "content": ...}` dicts. The strategy sees the **current request** — the latest user
message — however the conversation is passed.

```python
history = [
    HumanMessage("What's the capital of Poland?"),
    AIMessage("Warsaw."),
    HumanMessage("Now fix this Python error."),
]
assert routing_decision(router.invoke(history)).route == "coder"
```

## `stream` and `astream`

Streaming delegates to the selected route's own streaming, so the chunks are the route's chunks.
Exactly one chunk carries the decision record; merged chunks hold one copy of it, never a
concatenation.

```python
chunks = list(router.stream("Why is my Python loop slow?"))
merged = chunks[0]
for chunk in chunks[1:]:
    merged += chunk
print(merged.text, routing_decision(merged).route)
assert routing_decision(merged).route == "coder"


async def collect() -> list[str]:
    return [chunk.text async for chunk in router.astream("What's the capital of Poland?")]


print("".join(asyncio.run(collect())))
```

A consumer that abandons a stream part-way is not an error; the record it already received stays.

## `batch` and `abatch`

Each input is routed independently, so one batch can fan out across routes.

```python
answers = router.batch(["Why is my Python loop slow?", "What's the capital of Poland?"])
assert [routing_decision(a).route for a in answers] == ["coder", "small"]
```

## `generate`

`generate` and `agenerate` (and the prompt forms LangChain builds on them) run one prompt at a
time through `invoke`, so a call returns a normal `LLMResult` and each generation's message
carries its own decision.

## In a chain

Put the router where a chat model goes.

```python
prompt = ChatPromptTemplate.from_messages([("human", "Answer briefly: {question}")])
chain = prompt | router
answer = chain.invoke({"question": "Why is my Python loop slow?"})
assert routing_decision(answer).route == "coder"
```

## Events and callbacks

`astream_events` works as on any runnable, and callbacks and `RunnableConfig` (`tags`, `metadata`,
`callbacks`, `run_name`) pass through to the routed call. See
[tracing and cost](tracing-and-cost.md) for what a trace looks like.

## Not supported

LangChain's beta v3 streaming protocol isn't supported and raises `NotImplementedError` before any
run starts. Use `stream` or `astream_events`.
