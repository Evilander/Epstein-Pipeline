# Production Readiness

This repo is release-ready when the same commands below pass locally and in CI.

## Fast Smoke

```bash
python -m epstein_pipeline.cli --version
python -m epstein_pipeline.cli status --json
python -m epstein_pipeline.cli status --fail-on-unhealthy
pytest tests/test_cli.py -q
```

Use `python -m epstein_pipeline.cli status --check-database --fail-on-unhealthy` when `EPSTEIN_NEON_DATABASE_URL` is set and you want a live Neon reachability check before ingest, export, or audit work.

## Release Validation

```bash
ruff check src tests
python -m mypy src/epstein_pipeline/config.py src/epstein_pipeline/state.py src/epstein_pipeline/utils/parallel.py src/epstein_pipeline/utils/paths.py --ignore-missing-imports
pytest -q
python -m build
python scripts/release_check.py dist
```

`scripts/release_check.py` verifies that both the wheel and sdist include the bundled `persons-registry.json` fallback required for clean installs outside the repo checkout.

## Runtime Expectations

- `EPSTEIN_DATA_DIR`, `EPSTEIN_OUTPUT_DIR`, and `EPSTEIN_CACHE_DIR` must be writable.
- If `EPSTEIN_PERSONS_REGISTRY_PATH` does not exist, the package falls back to the bundled registry file.
- `search` requires the embeddings stack plus Neon connectivity; it now embeds the user query before calling pgvector search.
- `validate` accepts either a single JSON file or a directory tree, which keeps shell and CI loops simple.

## GitHub Release Flow

1. Run the release validation commands above.
2. Commit the release branch state.
3. Tag the release as `vX.Y.Z`.
4. Push the branch and tag.
5. Confirm the `CI` workflow and tag-triggered publish workflow complete successfully.
