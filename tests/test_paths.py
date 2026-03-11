"""Tests for safe remote-path handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from epstein_pipeline.utils.paths import safe_join


def test_safe_join_accepts_nested_relative_paths(tmp_path: Path) -> None:
    dest = safe_join(tmp_path, "nested/file.json")

    assert dest == tmp_path.resolve() / "nested" / "file.json"


@pytest.mark.parametrize(
    "remote_path",
    [
        "../escape.txt",
        "nested/../../escape.txt",
        "/absolute.txt",
        r"..\\escape.txt",
    ],
)
def test_safe_join_rejects_path_traversal(tmp_path: Path, remote_path: str) -> None:
    with pytest.raises(ValueError):
        safe_join(tmp_path, remote_path)
