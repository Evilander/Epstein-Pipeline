"""Safe path helpers for filesystem writes driven by remote metadata."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def safe_join(root: Path, remote_path: str) -> Path:
    """Join a remote relative path under *root* without allowing traversal."""
    candidate = PurePosixPath(remote_path.replace("\\", "/"))
    if candidate.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {remote_path!r}")

    parts = [part for part in candidate.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Path escapes target directory: {remote_path!r}")

    root_resolved = root.resolve()
    destination = root_resolved.joinpath(*parts).resolve()
    try:
        destination.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes target directory: {remote_path!r}") from exc

    return destination
