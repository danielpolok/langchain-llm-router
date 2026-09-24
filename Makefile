.PHONY: test integration_tests lint format type-check

# Offline runs: without this every test tries to trace.
export LANGSMITH_TRACING ?= false

TEST_FILE ?= tests/unit_tests

test:
	uv run pytest $(TEST_FILE)

integration_tests:
	uv run pytest tests/integration_tests

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff check --fix .
	uv run ruff format .

type-check:
	uv run mypy
