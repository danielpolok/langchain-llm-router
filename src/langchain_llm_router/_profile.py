"""What the router reports about itself: the intersection of its routes' capabilities (C1, C3).

`create_agent` reads `model.profile` to decide whether a chat model can serve structured output
through the provider's own strategy rather than a bound tool (`_supports_provider_strategy`,
`langchain/agents/factory.py:560` in the installed `langchain` 1.4.0: it checks
`model.profile.get("structured_output")` and, if that is not true, falls back to a name-pattern
match that a router — which has no `model`/`model_name`/`model_id` of its own — never hits
either way). A router that claimed a capability only some of its routes have would have a
strategy picked for it that a route can't serve (R10); REQ-C3-4 is the router answering only for
what every route can do.

This module holds the reduction (`resolve_profile`); `ChatRouter._resolve_model_profile`
(`router.py`) is the one line that calls it. `BaseChatModel._resolve_model_profile`
(`chat_models.py:398`) is the hook a chat model overrides to report its own profile, and the
base class's `_set_model_profile` validator (`chat_models.py:417`) calls it and assigns the
result to `self.profile` only when `profile` was not already given — so a `ChatRouter(...,
profile=...)` supplied explicitly wins over this module without any code here needing to check
for it (confirmed in `tests/unit_tests/test_profile.py`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from langchain_core.language_models import BaseChatModel, ModelProfile

__all__ = ["resolve_profile"]


def resolve_profile(routes: Mapping[str, BaseChatModel]) -> ModelProfile | None:
    """The intersection of every route's own `.profile` (REQ-C3-4).

    `None` when any route reports none at all (`route.profile is None`) — a route `create_agent`
    could not ask either, so the router cannot answer for it. Otherwise, per key present in
    *every* route's profile (a key one route doesn't report is dropped, not defaulted to a worst
    case it never claimed):

    - Both `bool`: AND-ed. Checked before the `int` case below — `bool` is a subclass of `int`
      in Python, so a bare `isinstance(value, int)` test would catch a boolean too, and a
      boolean pair would take the `int` branch's `min()` instead of this one's `all()` (the two
      happen to agree for a genuine boolean pair, since Python orders `False < True`, but the
      point is to compute the claim the spec names, not to rely on that coincidence).
    - Both `int` (and not `bool`, by the `elif` above already having failed): the minimum — the
      more conservative of the two claims.
    - Otherwise (a `str`, a `list`, or a route pair that disagrees on a value neither `bool` nor
      `int` reduces): kept if every route's value is literally equal, dropped if they differ — a
      router can't average a `name` or reconcile two different `reasoning_effort_levels` lists.

    A single-route router falls out of this for free: the "intersection" of one profile with
    itself, key by key, is that profile, unchanged.
    """
    profiles = [route.profile for route in routes.values()]
    if not profiles or any(profile is None for profile in profiles):
        # No route at all — unreachable through `ChatRouter`, whose `_check_routes` validator
        # requires at least one (REQ-R5-2) — or a route that has not resolved a profile of its
        # own. Either way, nothing can be claimed on that route's behalf.
        return None
    known = cast("list[dict[str, Any]]", profiles)  # `ModelProfile` is a `TypedDict`: a dict.
    shared: dict[str, Any] = {}
    for key in set.intersection(*(set(profile) for profile in known)):
        values = [profile[key] for profile in known]
        if all(isinstance(value, bool) for value in values):
            shared[key] = all(values)
        elif all(isinstance(value, int) for value in values):
            shared[key] = min(values)
        elif all(value == values[0] for value in values):
            shared[key] = values[0]
        # else: the routes disagree on a value this reduction has no claim to make for —
        # dropped, not guessed at.
    return cast("ModelProfile", shared)
