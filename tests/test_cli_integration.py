"""CLI integration tests covering real command/file flows."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from epstein_pipeline.cli import cli
from epstein_pipeline.exporters.neon_export import SemanticSearchResult
from epstein_pipeline.models.document import Document, ProcessingResult


def _cli_env(tmp_path: Path, registry_path: Path) -> dict[str, str]:
    return {
        "EPSTEIN_DATA_DIR": str(tmp_path / "data"),
        "EPSTEIN_OUTPUT_DIR": str(tmp_path / "output"),
        "EPSTEIN_CACHE_DIR": str(tmp_path / ".cache"),
        "EPSTEIN_PERSONS_REGISTRY_PATH": str(registry_path),
    }


def _write_registry(path: Path, sample_persons: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([person.model_dump(exclude_none=True) for person in sample_persons], indent=2),
        encoding="utf-8",
    )
    return path


def _write_processing_results(
    output_dir: Path,
    documents: list[Document],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, document in enumerate(documents, start=1):
        result = ProcessingResult(
            source_path=f"/tmp/{document.id}.pdf",
            document=document,
            errors=[],
            warnings=[],
            processing_time_ms=index,
        )
        (output_dir / f"{document.id}.json").write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
    return output_dir


def test_validate_directory_reports_mixed_results(
    tmp_path: Path,
    sample_documents,
    sample_persons,
) -> None:
    runner = CliRunner()
    registry_path = _write_registry(tmp_path / "runtime" / "persons-registry.json", sample_persons)
    input_dir = tmp_path / "validate"
    _write_processing_results(input_dir, sample_documents[:2])
    (input_dir / "broken.json").write_text('{"not":"a document"}', encoding="utf-8")

    result = runner.invoke(cli, ["validate", str(input_dir)], env=_cli_env(tmp_path, registry_path))

    assert result.exit_code == 0, result.output
    assert "Validating" in result.output
    assert "Valid:" in result.output
    assert "Invalid:" in result.output
    assert "broken.json" in result.output


def test_export_json_creates_document_array(tmp_path: Path, sample_documents, sample_persons) -> None:
    runner = CliRunner()
    registry_path = _write_registry(tmp_path / "runtime" / "persons-registry.json", sample_persons)
    input_dir = _write_processing_results(tmp_path / "processed-json", sample_documents)
    output_path = tmp_path / "artifacts" / "documents.json"

    result = runner.invoke(
        cli,
        ["export", "json", str(input_dir), "--output", str(output_path)],
        env=_cli_env(tmp_path, registry_path),
    )

    assert result.exit_code == 0, result.output
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(exported) == len(sample_documents)
    assert exported[0]["id"] == sample_documents[0].id


def test_export_csv_creates_flat_rows(tmp_path: Path, sample_documents, sample_persons) -> None:
    runner = CliRunner()
    registry_path = _write_registry(tmp_path / "runtime" / "persons-registry.json", sample_persons)
    input_dir = _write_processing_results(tmp_path / "processed-csv", sample_documents)
    output_path = tmp_path / "artifacts" / "documents.csv"

    result = runner.invoke(
        cli,
        ["export", "csv", str(input_dir), "--output", str(output_path)],
        env=_cli_env(tmp_path, registry_path),
    )

    assert result.exit_code == 0, result.output
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(sample_documents)
    assert rows[0]["id"] == sample_documents[0].id
    assert "p-0001" in rows[0]["personIds"]


def test_export_sqlite_creates_database_with_links(
    tmp_path: Path,
    sample_documents,
    sample_persons,
) -> None:
    runner = CliRunner()
    registry_path = _write_registry(tmp_path / "runtime" / "persons-registry.json", sample_persons)
    input_dir = _write_processing_results(tmp_path / "processed-sqlite", sample_documents)
    output_path = tmp_path / "artifacts" / "epstein.sqlite"

    result = runner.invoke(
        cli,
        ["export", "sqlite", str(input_dir), "--output", str(output_path)],
        env=_cli_env(tmp_path, registry_path),
    )

    assert result.exit_code == 0, result.output
    conn = sqlite3.connect(str(output_path))
    try:
        documents_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        persons_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        links_count = conn.execute("SELECT COUNT(*) FROM document_persons").fetchone()[0]
    finally:
        conn.close()

    assert documents_count == len(sample_documents)
    assert persons_count == len(sample_persons)
    assert links_count > 0


@patch("epstein_pipeline.exporters.neon_export.NeonExporter")
def test_export_neon_loads_documents_and_upserts(
    mock_exporter_cls: MagicMock,
    tmp_path: Path,
    sample_documents,
    sample_persons,
) -> None:
    runner = CliRunner()
    registry_path = _write_registry(tmp_path / "runtime" / "persons-registry.json", sample_persons)
    input_dir = _write_processing_results(tmp_path / "processed-neon", sample_documents)

    mock_exporter = MagicMock()
    mock_exporter.upsert_documents = AsyncMock(return_value=len(sample_documents))
    mock_exporter_cls.return_value = mock_exporter

    result = runner.invoke(
        cli,
        [
            "export",
            "neon",
            str(input_dir),
            "--database-url",
            "postgresql://user:pass@example.com/epstein",
            "--batch-size",
            "2",
        ],
        env=_cli_env(tmp_path, registry_path),
    )

    assert result.exit_code == 0, result.output
    loaded_documents = mock_exporter.upsert_documents.await_args.args[0]
    assert len(loaded_documents) == len(sample_documents)
    assert loaded_documents[0].id == sample_documents[0].id
    assert "Export complete." in result.output


@patch("epstein_pipeline.processors.embeddings.EmbeddingProcessor")
@patch("epstein_pipeline.exporters.neon_export.NeonExporter")
def test_search_renders_results_table(
    mock_exporter_cls: MagicMock,
    mock_embedder_cls: MagicMock,
    tmp_path: Path,
    sample_persons,
) -> None:
    runner = CliRunner()
    registry_path = _write_registry(tmp_path / "runtime" / "persons-registry.json", sample_persons)

    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    mock_embedder_cls.return_value = mock_embedder

    mock_exporter = MagicMock()
    mock_exporter.semantic_search = AsyncMock(
        return_value=[
            SemanticSearchResult(
                document_id="doc-001",
                chunk_text="Jeffrey Epstein financial records",
                chunk_index=0,
                similarity=0.92,
                title="Financial Records",
            )
        ]
    )
    mock_exporter_cls.return_value = mock_exporter

    result = runner.invoke(
        cli,
        [
            "search",
            "financial records",
            "--database-url",
            "postgresql://user:pass@example.com/epstein",
            "--limit",
            "3",
        ],
        env=_cli_env(tmp_path, registry_path),
    )

    assert result.exit_code == 0, result.output
    assert 'Searching for: "financial records"' in result.output
    assert "doc-001" in result.output
    mock_embedder.embed_texts.assert_called_once_with(["financial records"])


def test_build_graph_writes_json_and_gexf(tmp_path: Path, sample_documents, sample_persons) -> None:
    runner = CliRunner()
    registry_path = _write_registry(tmp_path / "runtime" / "persons-registry.json", sample_persons)
    input_dir = _write_processing_results(tmp_path / "processed-graph", sample_documents)
    output_root = tmp_path / "artifacts" / "graph"

    result = runner.invoke(
        cli,
        ["build-graph", str(input_dir), "--output", str(output_root), "--format", "both"],
        env=_cli_env(tmp_path, registry_path),
    )

    assert result.exit_code == 0, result.output
    json_path = output_root.with_suffix(".json")
    gexf_path = output_root.with_suffix(".gexf")
    assert json_path.exists()
    assert gexf_path.exists()
    graph_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(graph_data["nodes"]) > 0
    assert len(graph_data["links"]) > 0


def test_investigate_non_interactive_prints_top_entities(
    tmp_path: Path,
    sample_documents,
    sample_persons,
) -> None:
    runner = CliRunner()
    registry_path = _write_registry(tmp_path / "runtime" / "persons-registry.json", sample_persons)
    input_dir = _write_processing_results(tmp_path / "processed-investigate", sample_documents)

    result = runner.invoke(
        cli,
        ["investigate", str(input_dir), "--non-interactive"],
        env=_cli_env(tmp_path, registry_path),
    )

    assert result.exit_code == 0, result.output
    assert "Top 20 entities by connection count" in result.output
    assert "p-0001" in result.output


def test_sync_site_writes_site_json_and_sqlite(
    tmp_path: Path,
    sample_documents,
    sample_persons,
) -> None:
    runner = CliRunner()
    registry_path = _write_registry(tmp_path / "runtime" / "persons-registry.json", sample_persons)
    input_dir = _write_processing_results(tmp_path / "processed-site", sample_documents)
    site_dir = tmp_path / "site"
    site_dir.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(
        cli,
        ["sync-site", "--site-dir", str(site_dir), "--input-dir", str(input_dir)],
        env=_cli_env(tmp_path, registry_path),
    )

    assert result.exit_code == 0, result.output
    assert (site_dir / "data" / "pipeline-documents.json").exists()
    assert (site_dir / "data" / "epstein.sqlite").exists()


@patch("epstein_pipeline.downloaders.opensanctions.download_opensanctions")
def test_check_sanctions_passes_registry_and_threshold(
    mock_download: MagicMock,
    tmp_path: Path,
    sample_persons,
) -> None:
    runner = CliRunner()
    registry_path = _write_registry(tmp_path / "runtime" / "persons-registry.json", sample_persons)

    result = runner.invoke(
        cli,
        ["check-sanctions", "--threshold", "0.3", "--use-search"],
        env={
            **_cli_env(tmp_path, registry_path),
            "EPSTEIN_OPENSANCTIONS_API_KEY": "test-key",
        },
    )

    assert result.exit_code == 0, result.output
    kwargs = mock_download.call_args.kwargs
    assert kwargs["persons_registry_path"] == registry_path
    assert kwargs["match_threshold"] == 0.3
    assert kwargs["use_match_api"] is False
