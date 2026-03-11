"""CLI smoke tests for release-critical commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from epstein_pipeline.cli import cli
from epstein_pipeline.exporters.neon_export import SemanticSearchResult
from epstein_pipeline.models.document import Document


def test_status_json_is_machine_readable(tmp_path: Path) -> None:
    """`status --json` should emit clean JSON without banner noise."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["status", "--json"],
        env={
            "EPSTEIN_DATA_DIR": str(tmp_path / "data"),
            "EPSTEIN_OUTPUT_DIR": str(tmp_path / "output"),
            "EPSTEIN_CACHE_DIR": str(tmp_path / ".cache"),
        },
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["healthy"] is True
    assert report["persons_registry"]["exists"] is True
    assert report["paths"]["data_dir"]["exists"] is True


def test_export_help_exposes_neon_path() -> None:
    """The main export help should show Neon support and its required option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["export", "--help"])

    assert result.exit_code == 0, result.output
    assert "neon" in result.output
    assert "--database-url" in result.output


def test_download_help_exposes_dataset_option() -> None:
    """README examples should map to the actual DOJ download flags."""
    runner = CliRunner()
    result = runner.invoke(cli, ["download", "--help"])

    assert result.exit_code == 0, result.output
    assert "--dataset" in result.output
    assert "--list-datasets" in result.output


def test_validate_accepts_single_json_file(tmp_path: Path) -> None:
    """Single-file validation keeps the data workflow from shell-looping directories."""
    runner = CliRunner()
    input_file = tmp_path / "document.json"
    input_file.write_text(
        Document(
            id="doc-001", title="Test Document", source="other", category="other"
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = runner.invoke(cli, ["validate", str(input_file)])

    assert result.exit_code == 0, result.output
    assert "Valid:" in result.output
    assert "Invalid:" in result.output


@patch("epstein_pipeline.processors.embeddings.EmbeddingProcessor")
@patch("epstein_pipeline.exporters.neon_export.NeonExporter")
def test_search_embeds_query_before_database_lookup(
    mock_exporter_cls: MagicMock,
    mock_embedder_cls: MagicMock,
) -> None:
    """The CLI should embed user text before calling Neon semantic search."""
    runner = CliRunner()
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    mock_embedder_cls.return_value = mock_embedder

    mock_exporter = MagicMock()
    mock_exporter.semantic_search = AsyncMock(
        return_value=[
            SemanticSearchResult(
                document_id="doc-123",
                chunk_text="Financial records and offshore transfers",
                chunk_index=0,
                similarity=0.91,
                title="Financial Records",
            )
        ]
    )
    mock_exporter_cls.return_value = mock_exporter

    result = runner.invoke(
        cli,
        [
            "search",
            "offshore transfers",
            "--database-url",
            "postgresql://user:pass@example.com/epstein",
            "--limit",
            "5",
            "--threshold",
            "0.8",
        ],
    )

    assert result.exit_code == 0, result.output
    mock_embedder.embed_texts.assert_called_once_with(["offshore transfers"])
    mock_exporter.semantic_search.assert_awaited_once_with([0.1, 0.2, 0.3], top_k=5, threshold=0.8)
    assert "doc-123" in result.output
