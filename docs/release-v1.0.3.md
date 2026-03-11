# Epstein Pipeline v1.0.3

This release is the production handoff for the current branch state, with the CLI, packaging, and release workflow validated as one unit instead of a pile of “should work” assumptions.

## What changed

- `epstein_pipeline.__version__` is now `1.0.3`, matching the release tag and package metadata.
- `status` remains the operator entrypoint and now has a documented `--check-database` mode for live Neon smoke checks when credentials are present.
- `search` now embeds the user query before calling Neon semantic search, and missing dependency stacks fail with a clean operator error instead of a Python traceback.
- `validate` accepts either a single JSON file or a directory tree, which fixes shell-loop and CI data validation flows.
- The built wheel and sdist now carry the bundled `persons-registry.json` fallback, and `scripts/release_check.py` enforces that in release validation.
- CI/release workflows now include packaging checks (`twine check`, artifact content verification, status smoke) instead of only source-level tests.

## Validation used for this release

```bash
ruff check .
pytest -q
python -m build
python scripts/release_check.py dist
python -m twine check dist/*
python -m epstein_pipeline.cli status --json
python -m epstein_pipeline.cli status --fail-on-unhealthy
pytest tests/test_cli.py -q
```

## Known limits

- Heavy OCR, embeddings, and audit paths still depend on optional installs. `status` tells you what is missing; it does not provision those stacks.
- A live database check only becomes meaningful when `EPSTEIN_NEON_DATABASE_URL` and the Neon extras are installed in the target environment.
- There are older non-critical typing gaps in deeper processor modules, so CI scopes mypy to the release-critical surface instead of pretending the entire optional stack is fully typed.
