# Epstein Pipeline v1.0.1

This release turns the repo into something you can actually ship without hand-waving around the last mile.

## What changed

- Package versioning now comes from `epstein_pipeline.__version__`, so the CLI, build metadata, and release tag stop drifting apart.
- Packaging metadata was updated to current setuptools expectations and the wheel now bundles a fallback `persons-registry.json`, which means `epstein-pipeline status --json` works from a clean install.
- The CLI now exposes a release-safe `status` contract with JSON output, `--fail-on-unhealthy`, writable path checks, state DB visibility, and optional dependency reporting.
- First-contact command drift was fixed: the main `export` command now supports `neon`, and `download doj --dataset N` matches the README instead of failing on the first copy-paste.
- CI now validates packaging plus the health command on both Ubuntu and Windows instead of assuming Linux is enough.
- Type-checking is now explicit about scope: CI gates the release-critical CLI/config/state surface cleanly, while the older importer/processor typing backlog remains a follow-up item instead of a hidden failing job.
- Tagged releases now publish GitHub release assets as well as PyPI artifacts.

## Validation used for this release

```bash
python -m ruff check .
python -m mypy src
python -m pytest -q
python -m build
python -m epstein_pipeline.cli status --json --fail-on-unhealthy
python -m epstein_pipeline.cli --version
```

## Known limits

- Heavy OCR, embedding, and audit features still depend on optional model/runtime installs. `status` reports what is missing, but it does not install them for you.
- Database reachability is still an operator responsibility. The release verifies config surface and packaging; Neon connectivity depends on the target environment and credentials.
