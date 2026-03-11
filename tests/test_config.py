"""Configuration tests for runtime path resolution."""

from __future__ import annotations

from pathlib import Path

from epstein_pipeline.config import Settings


def test_resolve_persons_registry_path_uses_bundled_copy(tmp_path: Path) -> None:
    """Installed users should still have a working registry path without repo data."""
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / ".cache",
        persons_registry_path=tmp_path / "missing" / "persons-registry.json",
    )

    resolved = settings.resolve_persons_registry_path()

    assert resolved.exists()
    assert resolved.name == "persons-registry.json"
    assert resolved != settings.persons_registry_path
