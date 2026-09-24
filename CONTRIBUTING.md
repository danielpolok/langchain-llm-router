# Contributing

Read [AGENTS.md](AGENTS.md) for the layout, conventions and commands, and
[docs/design.md](docs/design.md) before changing behaviour it describes.

```bash
uv sync
make test          # offline unit tests
make lint          # ruff check, ruff format --check, mypy
make format        # apply ruff fixes and formatting
make integration_tests   # real providers; skipped without GEMINI_API_KEY / a local Ollama
```

- Open an issue before a large feature or refactor.
- Add tests with every change, through the public API.
- Public docstrings are Google style; `ruff` enforces this on `src/`.
- Reference the issue in the PR with a closing keyword (`Closes #NNN`).
