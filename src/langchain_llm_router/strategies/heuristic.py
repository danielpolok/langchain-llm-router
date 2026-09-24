"""`HeuristicStrategy`: cost tiering from cheap local signals, with no extra call.

The "cost tiering" use case: simple questions to a small model, complex ones to a frontier
model. The request's difficulty is *scored* from signals computable on the spot — how much text
there is, whether it carries code, how many things it asks for, whether it asks for reasoning,
and whether it carries anything but text — and the score picks a tier. Nothing here calls a
model, embeddings or a network service, and nothing here is private to the package:
it is an ordinary `RoutingStrategy` a user could have written.

Setting one up
--------------
Cheapest tier first; the thresholds ascend with difficulty.

```python
HeuristicStrategy("small", "frontier")  # the two-tier case, on the default threshold
HeuristicStrategy("small", "frontier", thresholds=[2.0])  # a stricter bar for the frontier
HeuristicStrategy("small", "mid", "frontier", thresholds=[1.0, 3.0])  # three tiers, two bands
```

Tier *i* answers when `thresholds[i - 1] <= score < thresholds[i]`; the last tier takes
everything from its threshold up. A score exactly on a threshold takes the *upper* tier — the
threshold is the price of admission to it.

The score
---------
Every signal returns a strength in `[0.0, 1.0]`, which its weight scales; the score is their
sum, so with the default weights it runs from `0.0` to `5.0`. The signals, and what each is for:

| Signal | Fully on when |
| --- | --- |
| `length` | the request is `DEFAULT_LENGTH_RANGE[1]` words or longer; silent below `[0]` |
| `code` | two of three code markers are present: a fence, code syntax, a traceback |
| `parts` | the request asks `_PARTS_FOR_FULL` separate things (question marks or list items) |
| `analysis` | `_ANALYSIS_TERMS_FOR_FULL` distinct analysis words appear ("compare", "why does") |
| `modalities` | the request carries anything but text — an image, audio, a document |

Each is a plain function of the request, exported and testable on its own, and the whole set is
replaceable through `signals=`. `weights=` scales individual signals without replacing any;
weight `0.0` silences one.

What a benchmark result retunes
-------------------------------
The defaults are meant to be settled by the benchmark (`benchmark/`), not guessed. Everything it
should need to change is a module-level constant, so the change is mechanical and the tests it
argues with are the tier tests in `tests/unit_tests/test_heuristic_strategy.py`:

- `DEFAULT_THRESHOLD` — the two-tier bar, and with it how much traffic reaches the frontier tier.
- `DEFAULT_WEIGHTS` — how much each signal counts (currently all equal).
- `DEFAULT_LENGTH_RANGE` — where "long" starts and tops out.
- `_CODE_MARKERS`, `_ANALYSIS_TERMS` and the `_..._FOR_FULL` counts — the marker vocabulary.

Nothing else about the strategy depends on these values, and `signals=` lets the benchmark try a
signal set that isn't in this module at all.

What the score cannot see
-------------------------
Measured, not guessed — the cases below are pinned in
`tests/unit_tests/test_heuristic_strategy.py::test_the_defaults_blind_spots_are_the_ones_we_know_of`,
and are the evidence a benchmark argues with:

- **Scripts that don't separate words with spaces.** `length_signal` counts whitespace-separated
  words, so Chinese, Japanese and Thai read as a handful of words however much they say; the
  analysis vocabulary is English, so it never fires on them either. A hard request in those
  languages scores `0.00` and takes the cheapest tier. Replace `length` through `signals=` if
  that is your traffic.
- **Length stands in for difficulty, and sometimes it is wrong in both directions.** A pasted log
  with "what does this mean?" reads as hard (`length 1.00`); "prove that P != NP implies one-way
  functions exist" reads as easy (`analysis 0.50`, one term of the two a full signal needs).
  Local signals cannot read intent — that is the trade a strategy with no extra call makes.
  The embedding and classifier strategies are where a router buys its way out.

Deciding, and not deciding
--------------------------
- **A request with nothing to judge** — no text and no other modality — scores nothing, so the
  strategy abstains with `None` and the router falls back to the default route. It does not
  warn or raise: that is the router's to report, once.
- **A tier that isn't one of the router's routes** is named anyway, never quietly swapped for a
  neighbouring tier. A strategy sees `request.routes` but cannot know which name the application
  meant, so it hands the router the route its policy chose and lets the router report the
  mismatch through the one path that already exists for it: a `FallbackWarning`, the default
  route, and a record saying which route was missing.
- **The reason** carries the score, the band it fell in and the signals that made it, so a
  human reading a trace sees why the request was judged easy or hard:
  `"difficulty 2.50 >= 1.00 (code 1.00, length 1.00, analysis 0.50)"`.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TypeAlias

from langchain_llm_router.errors import RoutingError
from langchain_llm_router.strategy import RoutingChoice, RoutingRequest, RoutingStrategy

__all__ = [
    "DEFAULT_LENGTH_RANGE",
    "DEFAULT_SIGNALS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WEIGHTS",
    "HeuristicStrategy",
    "Signal",
    "analysis_signal",
    "code_signal",
    "length_signal",
    "modality_signal",
    "parts_signal",
]

Signal: TypeAlias = Callable[[RoutingRequest], float]
"""How hard one aspect of a request makes it look, from `0.0` (silent) to `1.0` (fully on)."""


# --- The signals ---


def _ramp(value: float, floor: float, ceiling: float) -> float:
    """`0.0` at `floor` or below, `1.0` at `ceiling` or above, linear in between."""
    return min(1.0, max(0.0, (value - floor) / (ceiling - floor)))


DEFAULT_LENGTH_RANGE = (20, 200)
"""Words at which the length signal starts to rise, and at which it is fully on.

Twenty words is about a sentence of context; two hundred is a page of pasted material."""


def length_signal(
    floor: int = DEFAULT_LENGTH_RANGE[0], ceiling: int = DEFAULT_LENGTH_RANGE[1]
) -> Signal:
    """How long the request is, in whitespace-separated words, as a ramp between two bounds.

    A factory rather than a plain signal: the bounds are the one part of the default signal set
    that is a number rather than a vocabulary, so `signals=` can retune it in a line.
    """
    if floor < 0 or ceiling <= floor:
        msg = f"length_signal needs 0 <= floor < ceiling, got floor={floor}, ceiling={ceiling}"
        raise RoutingError(msg)

    def signal(request: RoutingRequest) -> float:
        return _ramp(len(request.text.split()), floor, ceiling)

    return signal


_CODE_MARKERS: Mapping[str, re.Pattern[str]] = MappingProxyType(
    {
        "fence": re.compile(r"```|~~~"),
        # Deliberately narrow: each alternative needs punctuation a sentence wouldn't carry, so
        # "a class of problems" or "import duties" don't read as code.
        "syntax": re.compile(
            r"\b(?:def|fn|func|function)\s+\w+\s*\("
            r"|\bclass\s+\w+\s*[(:{]"
            r"|\b(?:const|let|var)\s+\w+\s*="
            r"|#include\s*<"
            r"|\bSELECT\b[\s\S]{0,400}?\bFROM\b"
            r"|=>|::|\);"
        ),
        "traceback": re.compile(
            r"Traceback \(most recent call last\)"
            r'|^\s+(?:File "|at \w)'
            r"|\b[A-Z]\w*(?:Error|Exception)\b:",
            re.MULTILINE,
        ),
    }
)
"""Marker family → the pattern that spots it. Two families present is fully on."""

_CODE_MARKERS_FOR_FULL = 2


def code_signal(request: RoutingRequest) -> float:
    """Whether the request carries code: a fenced block, code syntax, or a traceback.

    Counted by *family*, not by hits, so one long paste of Python doesn't outweigh a short
    snippet that also comes with an error.
    """
    found = sum(1 for pattern in _CODE_MARKERS.values() if pattern.search(request.text))
    return _ramp(found, 0, _CODE_MARKERS_FOR_FULL)


_ENUMERATED = re.compile(r"^\s*(?:[-*•]|\(?\d{1,2}[.)])\s+\S", re.MULTILINE)
"""A bullet or numbered item: one line of a list of sub-requests."""

_PARTS_FOR_FULL = 3


def parts_signal(request: RoutingRequest) -> float:
    """How many separate things the request asks for: questions, or enumerated items.

    The larger of the two counts, not their sum — a numbered list of questions is one list, not
    two — and one part is what every request has, so the signal only rises from the second.
    """
    parts = max(request.text.count("?"), len(_ENUMERATED.findall(request.text)))
    return _ramp(parts, 1, _PARTS_FOR_FULL)


_ANALYSIS_TERMS: tuple[str, ...] = (
    r"analy[sz]",
    r"architect",
    r"compar",
    r"contrast",
    r"critique",
    r"debug",
    r"deriv",
    r"design",
    r"evaluat",
    r"explain why",
    r"implication",
    r"justif",
    r"optimi[sz]",
    r"prove\b",
    r"rational",
    r"refactor",
    r"root cause",
    r"synthes",
    r"trade-?offs?",
    r"why (?:does|is|are|do)",
)
"""Words that ask for reasoning rather than recall. Two distinct ones is fully on."""

_ANALYSIS = tuple(re.compile(rf"\b{term}", re.IGNORECASE) for term in _ANALYSIS_TERMS)

_ANALYSIS_TERMS_FOR_FULL = 2


def analysis_signal(request: RoutingRequest) -> float:
    """How much reasoning the request asks for, by how many distinct analysis terms appear.

    Distinct terms, so "compare … and compare …" counts once: repetition is emphasis, not a
    second thing to reason about.
    """
    hits = sum(1 for pattern in _ANALYSIS if pattern.search(request.text))
    return _ramp(hits, 0, _ANALYSIS_TERMS_FOR_FULL)


def modality_signal(request: RoutingRequest) -> float:
    """Whether the request carries anything but text.

    Binary: an image, audio, video, a document or content LangChain couldn't translate all need
    a model that can read them, and "how much of it" says nothing about difficulty.
    """
    return 1.0 if request.modalities - {"text"} else 0.0


DEFAULT_SIGNALS: Mapping[str, Signal] = MappingProxyType(
    {
        "length": length_signal(),
        "code": code_signal,
        "parts": parts_signal,
        "analysis": analysis_signal,
        "modalities": modality_signal,
    }
)
"""The signal set a `HeuristicStrategy` scores with unless `signals=` replaces it."""

DEFAULT_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {"length": 1.0, "code": 1.0, "parts": 1.0, "analysis": 1.0, "modalities": 1.0}
)
"""What each default signal counts for. Equal until a benchmark gives evidence for anything else."""

DEFAULT_THRESHOLD = 1.0
"""The two-tier bar: one signal fully on, or two half on, earns the stronger tier."""


# --- The strategy ---


class HeuristicStrategy(RoutingStrategy):
    """Routes on a difficulty score built from cheap local signals.

    ```python
    ChatRouter(
        routes={"small": small, "frontier": frontier},
        default_route="small",
        strategy=HeuristicStrategy("small", "frontier"),
    )
    ```

    Args:
        tiers: The route names, cheapest first; at least two.
        thresholds: The score at which each tier gives way to the next, ascending — one fewer
            than there are tiers. Defaults to `DEFAULT_THRESHOLD` for two tiers; with more,
            there is nothing sensible to default to, so it is required.
        weights: What a signal counts for, overriding `DEFAULT_WEIGHTS` name by name. `0.0`
            silences a signal.
        signals: The whole signal set, replacing `DEFAULT_SIGNALS`. A signal is any
            `Callable[[RoutingRequest], float]` returning `0.0` to `1.0`; one not in
            `DEFAULT_WEIGHTS` weighs `1.0` unless `weights` says otherwise.

    Raises:
        RoutingError: for configuration that could never route — too few tiers, the wrong
            number of thresholds, thresholds out of order, a weight for a signal that doesn't
            exist. Raised here, at construction, not once per request.
    """

    def __init__(
        self,
        *tiers: str,
        thresholds: Sequence[float] | None = None,
        weights: Mapping[str, float] | None = None,
        signals: Mapping[str, Signal] | None = None,
    ) -> None:
        self.tiers = _checked_tiers(tiers)
        self.signals = _checked_signals(signals)
        self.weights = _checked_weights(weights, self.signals)
        self.thresholds = _checked_thresholds(thresholds, len(self.tiers))

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        """The tier this request's difficulty score falls in, or `None` if there is none.

        Thread-safe, as the interface asks: the strategy holds only its configuration, and
        scoring touches nothing but the request.
        """
        if not request.text.strip() and not request.modalities - {"text"}:
            # Nothing to score — an empty user turn. Better the default route than a guess.
            return None
        contributions = {
            name: self.weights[name] * signal(request) for name, signal in self.signals.items()
        }
        score = sum(contributions.values())
        tier = next(
            (index for index, bar in enumerate(self.thresholds) if score < bar),
            len(self.thresholds),
        )
        return RoutingChoice(
            route=self.tiers[tier], reason=self._reason(score, tier, contributions)
        )

    def _reason(self, score: float, tier: int, contributions: Mapping[str, float]) -> str:
        """The score, the band it fell in, and the signals that made it."""
        if tier == 0:
            band = f"difficulty {score:.2f} < {self.thresholds[0]:.2f}"
        elif tier == len(self.thresholds):
            band = f"difficulty {score:.2f} >= {self.thresholds[-1]:.2f}"
        else:
            band = (
                f"difficulty {score:.2f} in "
                f"[{self.thresholds[tier - 1]:.2f}, {self.thresholds[tier]:.2f})"
            )
        # Largest contribution first, so the reason leads with what drove the decision; a stable
        # sort leaves equal contributions in the order the signals were declared.
        fired = sorted(
            ((name, value) for name, value in contributions.items() if value),
            key=lambda contribution: -contribution[1],
        )
        detail = ", ".join(f"{name} {value:.2f}" for name, value in fired) or "no signal fired"
        return f"{band} ({detail})"


def _checked_tiers(tiers: Sequence[str]) -> tuple[str, ...]:
    if len(tiers) < 2:
        msg = (
            "HeuristicStrategy needs at least two tiers, cheapest first, as in "
            'HeuristicStrategy("small", "frontier")'
        )
        raise RoutingError(msg)
    blank = [name for name in tiers if not name.strip()]
    if blank:
        msg = f"tier {blank[0]!r} is blank: every tier names a route"
        raise RoutingError(msg)
    return tuple(tiers)


def _checked_signals(signals: Mapping[str, Signal] | None) -> Mapping[str, Signal]:
    if signals is None:
        return DEFAULT_SIGNALS
    if not signals:
        msg = "signals is empty: a HeuristicStrategy needs at least one signal to score with"
        raise RoutingError(msg)
    return MappingProxyType(dict(signals))


def _checked_weights(
    weights: Mapping[str, float] | None, signals: Mapping[str, Signal]
) -> Mapping[str, float]:
    given = dict(weights or {})
    unknown = sorted(name for name in given if name not in signals)
    if unknown:
        msg = (
            f"weights names signals that don't exist: {_names(unknown)}; "
            f"the signals are {_names(signals)}"
        )
        raise RoutingError(msg)
    negative = sorted(name for name, weight in given.items() if weight < 0)
    if negative:
        msg = f"weights must not be negative: {_names(negative)}"
        raise RoutingError(msg)
    return MappingProxyType(
        {name: given.get(name, DEFAULT_WEIGHTS.get(name, 1.0)) for name in signals}
    )


def _checked_thresholds(thresholds: Sequence[float] | None, tiers: int) -> tuple[float, ...]:
    if thresholds is None:
        if tiers > 2:
            msg = (
                f"{tiers} tiers need {tiers - 1} thresholds: there is a default "
                "only for the two-tier case"
            )
            raise RoutingError(msg)
        return (DEFAULT_THRESHOLD,)
    if len(thresholds) != tiers - 1:
        wanted = tiers - 1
        msg = (
            f"{tiers} tiers need {wanted} threshold{'' if wanted == 1 else 's'}, "
            f"got {len(thresholds)}"
        )
        raise RoutingError(msg)
    bars = tuple(float(bar) for bar in thresholds)
    if any(later <= earlier for earlier, later in itertools.pairwise(bars)):
        msg = f"thresholds must ascend with difficulty, got {list(bars)}"
        raise RoutingError(msg)
    return bars


def _names(names: Sequence[str] | Mapping[str, object]) -> str:
    return ", ".join(repr(name) for name in names)
