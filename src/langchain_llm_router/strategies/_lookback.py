"""What the built-in strategies share for `lookback`: its check, the messages, and the reason.

A strategy with `lookback=N` reads the current request and then up to N of the user's previous
messages, newest first — at most its own N, whatever the request carries, so a strategy handed a
request by another one still reads what it was configured to. A message that decides from further
back says so in the reason, with the same words in every strategy: `(1 message back)`.
"""

from __future__ import annotations

from collections.abc import Iterator

from langchain_llm_router.errors import RoutingError
from langchain_llm_router.strategy import RoutingRequest


def checked_lookback(lookback: object) -> int:
    """`lookback` itself, if it is a non-negative integer — it counts messages.

    A `bool` is refused although Python counts it as an `int`: `True` is a flag where a count
    was meant.
    """
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 0:
        msg = (
            f"lookback must be a non-negative integer, the number of the user's previous "
            f"messages to read, got {lookback!r}"
        )
        raise RoutingError(msg)
    return lookback


def considered(request: RoutingRequest, lookback: int) -> Iterator[tuple[int, RoutingRequest]]:
    """The current request, then up to `lookback` previous ones: each with how far back it is."""
    yield 0, request
    yield from enumerate(request.previous_requests[:lookback], start=1)


def how_far_back(distance: int) -> str:
    """What a reason ends with when the message `distance` user messages back decided it.

    Nothing for the current request, so a strategy's reasons are the same as without lookback
    whenever the current request decides.
    """
    if distance == 0:
        return ""
    return f" ({distance} message{'' if distance == 1 else 's'} back)"
