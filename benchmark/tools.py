"""The fixed toolkit agent-kind workload items are bound against.

Five tools, one per dataset domain plus a general-purpose one, each deterministic — canned or
computed from its arguments, never random or time-dependent — so a run is reproducible and
grading isn't chasing a moving target. Every agent item binds the full set (`ALL_TOOLS`): the
question a routed model has to answer isn't just "can it call a tool" but "can it call the
*right* one" among plausible distractors, which is the harder and more realistic case.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from langchain_core.tools import BaseTool, tool

_ORDER_STATUS = {
    "A1001": "shipped 2026-09-18, arriving 2026-09-24",
    "A1002": "processing, expected to ship within 2 business days",
    "A1003": "delivered 2026-09-15",
}

_DOCS = {
    "refund": "Refunds post to the original payment method 5-7 business days after approval.",
    "retry": "The SDK retries a request up to 3 times with exponential backoff on a 429 or 5xx.",
    "cache": "The response cache key includes the route name, so two routes never share an entry.",
}

_BINARY_OPERATORS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type, Callable[[float], float]] = {ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        binary_op = _BINARY_OPERATORS[type(node.op)]
        return binary_op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        unary_op = _UNARY_OPERATORS[type(node.op)]
        return unary_op(_eval_node(node.operand))
    msg = f"unsupported expression: {ast.dump(node)}"
    raise ValueError(msg)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "12 * (3 + 4)", and return the numeric result."""
    try:
        value = _eval_node(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return f"error: could not evaluate {expression!r} ({error})"
    return f"{value:g}"


@tool
def get_weather(city: str) -> str:
    """Look up the current weather in a city."""
    # Deterministic from the city name so the same city always answers the same way.
    conditions = ["sunny, 21C", "cloudy, 14C", "light rain, 11C", "clear, 27C"]
    return f"{city}: {conditions[sum(map(ord, city)) % len(conditions)]}"


@tool
def lookup_order_status(order_id: str) -> str:
    """Look up the shipping status of a customer's order by its order ID."""
    return _ORDER_STATUS.get(order_id, f"no order found with ID {order_id!r}")


@tool
def search_docs(query: str) -> str:
    """Search internal documentation for a short answer to a policy or technical question."""
    for keyword, answer in _DOCS.items():
        if keyword in query.lower():
            return answer
    return "no matching documentation found"


@tool
def lint_code(snippet: str) -> str:
    """Check a Python code snippet for syntax errors and report the first one found."""
    try:
        ast.parse(snippet)
    except SyntaxError as error:
        return f"SyntaxError: {error.msg} at line {error.lineno}"
    return "no syntax errors found"


ALL_TOOLS: tuple[BaseTool, ...] = (
    calculator,
    get_weather,
    lookup_order_status,
    search_docs,
    lint_code,
)
"""Bound in full to every agent-kind item (module docstring explains why)."""

BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
