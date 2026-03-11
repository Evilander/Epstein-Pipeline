# Epstein Pipeline v1.0.4

This release closes the two gaps that showed up immediately after `v1.0.3` landed: a stale CI schema validation step and a PyPI publish job that depended on unconfigured trusted publishing.

## What changed

- `epstein_pipeline.__version__` is now `1.0.4`, matching the recovery release tag and package metadata.
- The CI `schema-validate` job now validates the current Neon migration contract by importing `MIGRATION_SQL` and `SCHEMA_VERSION` from `epstein_pipeline.exporters.neon_schema`.
- The release workflow now treats `workflow_dispatch` as a safe validation path and only creates GitHub releases or uploads to PyPI on version tags.
- PyPI publishing now uses the `PYPI_API_TOKEN` secret from the GitHub `pypi` environment instead of failing on missing trusted-publisher configuration.

## Validation used for this release

```bash
python -m ruff check .
python -m mypy src/epstein_pipeline/config.py src/epstein_pipeline/state.py src/epstein_pipeline/utils/parallel.py src/epstein_pipeline/utils/paths.py --ignore-missing-imports
python -m pytest -q
python -m build
python -m twine check dist/*
python scripts/release_check.py dist
python -m epstein_pipeline.cli status --fail-on-unhealthy
```

## Known limits

- PyPI token publishing works immediately, but it is less desirable long-term than restoring trusted publishing on the PyPI project.
- Release automation still assumes GitHub-hosted Ubuntu runners for packaging and publish jobs.
