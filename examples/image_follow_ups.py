"""Image follow-ups: a question about a picture never reaches a model that can't see it.

An analyst shares a chart and asks about it, then asks follow-ups in plain text. A strategy that
routes on the newest message sees only text in a follow-up, and would send it to the local
text-only model, along with the chart from the first turn, which that model can't read. The router
checks every message a model will receive against the model's `profile`, so the follow-ups go to
the model that can see the chart, and the decision records which model was skipped.

The local model is Ollama's `qwen3:8b`, which reads only text. Ollama models report no profile,
so the script says so with LangChain's own `profile=`. Every turn here goes to Gemini, so Ollama
is never called and doesn't need to be running.

Run it with a Gemini API key in `GOOGLE_API_KEY`:

    uv run python examples/image_follow_ups.py
"""

from textwrap import shorten

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_model_router import ChatRouter, ConfigurableStrategy, routing_decision
from langchain_model_router.strategies.configurable import Rule, always, modality

local = init_chat_model("ollama:qwen3:8b", profile={"image_inputs": False})
vision = init_chat_model("google_genai:gemini-3.5-flash-lite")

router = ChatRouter(
    routes={"local": local, "vision": vision},
    default_route="local",
    strategy=ConfigurableStrategy(
        [
            Rule("vision", modality("image"), name="has an image"),
            Rule("local", always(), name="text"),
        ]
    ),
)

# A small bar chart: a blue, an orange and a green bar, the orange one tallest.
CHART = (
    "iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAAAiElEQVR42u3QQQ3AIBAAQRRgoQbwx7NWKgJ/YIBAcuVDMptVMKlr"
    "WUIACBAgQIAAATrXm+MDAgQIECBAgAABAgQIECBAgAABAgQI0CGg8pXwgAABAgQIECBAgAABAgQI0P1AT23hAQECBAgQIECAAAEC"
    "BAgQIECAAAECBAgQoBmQNnUtAwToXwPy6rxhQn3gugAAAABJRU5ErkJggg=="
)

questions = [
    "Here is this quarter's chart. Which bar is the tallest?",
    "What colour is the shortest bar?",
    "Is the green bar more than half as tall as the tallest one?",
]

conversation: list[BaseMessage] = []
for question in questions:
    content: list[str | dict] = [{"type": "text", "text": question}]
    if not conversation:  # the chart comes with the first question only
        content.append({"type": "image", "base64": CHART, "mime_type": "image/png"})
    conversation.append(HumanMessage(content))
    response = router.invoke(conversation)
    conversation.append(response)
    decision = routing_decision(response)
    print(f"{decision.route:<7} {question}")
    print(f"        {decision.reason}")
    print(f"        {shorten(response.text, 72, placeholder=' …')}")

# It prints (the answers come from real models, so their wording changes from run to run):
#
#   vision  Here is this quarter's chart. Which bar is the tallest?
#           rule 'has an image' matched: has image
#           Based on the chart, the **orange** bar in the middle is the tallest.
#   ContentSupportWarning: 'local' can't take images; diverted to 'vision'
#   vision  What colour is the shortest bar?
#           rule 'text' matched: always; 'local' can't take images, so it was diverted to 'vision'
#           Based on the chart, the shortest bar is **blue** (the first bar on the …
#   ContentSupportWarning: 'local' can't take images; diverted to 'vision'
#   vision  Is the green bar more than half as tall as the tallest one?
#           rule 'text' matched: always; 'local' can't take images, so it was diverted to 'vision'
#           Based on a visual estimate, yes. The green bar reaches a little over …
#
# The first question carries the chart, so its rule chose the vision model. The follow-ups are
# plain text, so the strategy chose the local model, but each one still sends the chart along with
# the conversation. The local model's profile says it can't take images, so the router sent the
# follow-ups to the vision model instead, warned, and recorded the skipped model in the decision's
# `diverted_from`. A conversation with no image never needs this, and the local model answers it.
