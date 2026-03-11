# Epstein Pipeline v1.0.2

This patch release finishes the production pass by shipping the last verified fix instead of leaving it as an uncommitted local delta.

## What changed

- `config.runtime_summary()` now satisfies the scoped mypy gate used by CI and the release workflow.
- The `v1.0.1` hardening work remains the main payload: release-safe `status`, package-data fallback for the persons registry, safer remote downloads, fail-fast parallel batch handling, honest export behavior, and validated release automation.
- The final release process now reflects the state that actually passed lint, scoped types, tests, build, artifact checks, and GitHub auth/publish checks.

## Validation used for this release

```bash
python -m ruff check src tests
python -m mypy src/epstein_pipeline/config.py src/epstein_pipeline/state.py src/epstein_pipeline/utils/parallel.py src/epstein_pipeline/utils/paths.py --ignore-missing-imports
python -m pytest -q
python -m build
python -m twine check dist/*
python scripts/release_check.py dist
python -m epstein_pipeline.cli status --fail-on-unhealthy
```
