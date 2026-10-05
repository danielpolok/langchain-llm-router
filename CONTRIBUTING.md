# Contributing

Thanks for helping. Bug reports, ideas, documentation fixes and code are all welcome.

- **Found a bug?** Open a [bug report](https://github.com/danielpolok/langchain-model-router/issues/new/choose)
  with a small example that shows it.
- **Have a question or an idea?** Start a
  [discussion](https://github.com/danielpolok/langchain-model-router/discussions), or open a feature
  request. The README's [scope](README.md#scope) says what the router deliberately doesn't do.
- **Found a security problem?** Follow [SECURITY.md](SECURITY.md) instead of opening an issue.

## Making a change

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
- Add a line under **Unreleased** in [CHANGELOG.md](CHANGELOG.md) for anything a user would notice.
- Reference the issue in the PR with a closing keyword (`Closes #NNN`).

## Releasing

Maintainers release from `main`:

1. Set the new version in `pyproject.toml` (`uv version 0.2.0`). Until 1.0, a release that breaks
   the public API bumps the minor version, and the strategy interface follows its
   [stability promise](docs/strategies.md#stability-promise).
2. In `CHANGELOG.md`, rename **Unreleased** to the version and date, add a new empty
   **Unreleased** above it, and update the compare links at the bottom.
3. Merge that to `main`, then tag the merge commit and push the tag:

   ```bash
   git tag v0.2.0 && git push origin v0.2.0
   ```

The [release workflow](.github/workflows/release.yml) checks that the tag, `pyproject.toml` and
`CHANGELOG.md` agree, runs CI on the tagged commit, publishes the tested distributions to PyPI
and creates the GitHub release with the changelog section as its notes. It publishes through
PyPI's trusted publishing, configured once for this repository, its `release.yml` workflow and
its `pypi` environment, so no PyPI token is stored anywhere.
