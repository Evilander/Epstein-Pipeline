"""Import GhostCrawl HuggingFace datasets into Pipeline Document format.

Handles five dataset schemas:
- mega  (post-train/Epstein-Files) — 4.11M rows, parquet
- 20k   (teyler/epstein-files-20k) — 20K OCR docs, parquet
- emails (notesbymuneeb/Epstein-emails) — email corpus, parquet
- fbi   (svetfm/Epstein-Investigation-Fbi-Archive) — FBI files, parquet

Each importer reads parquet files from a local directory and yields
Document dicts compatible with the Pipeline's NDJSON export format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

logger = logging.getLogger(__name__)
console = Console()


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


def _write_chunk(output_dir: Path, prefix: str, chunk_num: int, data: list[dict]) -> None:
    """Write a chunk of documents as NDJSON."""
    path = output_dir / f"{prefix}_{chunk_num:04d}.ndjson"
    with open(path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False))
            f.write("\n")


def _infer_category(text: str | None) -> str:
    """Infer DocumentCategory from text content."""
    if not text:
        return "other"
    t = text.lower()[:2000]
    if any(w in t for w in ("court", "deposition", "plaintiff", "defendant", "motion")):
        return "legal"
    if any(w in t for w in ("account", "wire transfer", "bank", "financial")):
        return "financial"
    if any(w in t for w in ("flight", "tail number", "passenger", "aircraft")):
        return "travel"
    if any(w in t for w in ("email", "subject:", "from:", "to:")):
        return "communications"
    if any(w in t for w in ("fbi", "investigation", "agent", "interview")):
        return "investigation"
    return "other"


class GhostCrawlImporter:
    """Import GhostCrawl's HuggingFace datasets into Pipeline format."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def _find_parquets(self) -> list[Path]:
        """Find all parquet files in data_dir (recursive)."""
        return sorted(self.data_dir.rglob("*.parquet"))

    def import_mega(self, output_dir: Path | None = None, limit: int | None = None) -> int:
        """Import the mega dataset (post-train/Epstein-Files, 4.11M rows).

        Expects parquet files with columns like: text, id, file_name, page, etc.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            console.print("[red]pyarrow required: pip install pyarrow[/red]")
            return 0

        parquets = self._find_parquets()
        if not parquets:
            console.print(f"[red]No parquet files found in {self.data_dir}[/red]")
            return 0

        console.print(f"  Found [bold]{len(parquets)}[/bold] parquet files")
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        imported = 0
        chunk: list[dict] = []
        chunk_num = 0
        chunk_size = 10_000

        with _progress() as progress:
            task = progress.add_task("Importing mega dataset", total=limit or None)

            for pq_path in parquets:
                table = pq.read_table(pq_path)
                cols = set(table.column_names)

                for i in range(table.num_rows):
                    if limit and imported >= limit:
                        break

                    row = {c: table.column(c)[i].as_py() for c in cols}

                    text = row.get("text") or row.get("text_content") or ""
                    raw_id = row.get("id") or row.get("doc_id") or str(imported)
                    title = row.get("file_name") or row.get("title") or f"MEGA-{raw_id}"

                    doc = {
                        "id": f"gc-mega-{raw_id}",
                        "title": title,
                        "source": "ghostcrawl-mega",
                        "category": _infer_category(text),
                        "ocrText": text if text else None,
                        "tags": ["ghostcrawl", "mega"],
                        "pageCount": row.get("page") or row.get("num_pages"),
                    }

                    chunk.append(doc)
                    imported += 1
                    progress.advance(task)

                    if output_dir and len(chunk) >= chunk_size:
                        _write_chunk(output_dir, "gc-mega", chunk_num, chunk)
                        chunk_num += 1
                        chunk = []

                if limit and imported >= limit:
                    break

        if output_dir and chunk:
            _write_chunk(output_dir, "gc-mega", chunk_num, chunk)

        console.print(f"  [green]Imported {imported:,} documents from mega dataset[/green]")
        return imported

    def import_20k(self, output_dir: Path | None = None, limit: int | None = None) -> int:
        """Import the 20K OCR documents (teyler/epstein-files-20k)."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            console.print("[red]pyarrow required: pip install pyarrow[/red]")
            return 0

        parquets = self._find_parquets()
        if not parquets:
            console.print(f"[red]No parquet files found in {self.data_dir}[/red]")
            return 0

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        imported = 0
        chunk: list[dict] = []
        chunk_num = 0

        with _progress() as progress:
            task = progress.add_task("Importing 20k dataset", total=limit or None)

            for pq_path in parquets:
                table = pq.read_table(pq_path)
                cols = set(table.column_names)

                for i in range(table.num_rows):
                    if limit and imported >= limit:
                        break

                    row = {c: table.column(c)[i].as_py() for c in cols}
                    text = row.get("text") or row.get("text_content") or ""
                    raw_id = row.get("id") or row.get("doc_id") or str(imported)
                    title = row.get("file_name") or row.get("title") or f"20K-{raw_id}"

                    doc = {
                        "id": f"gc-20k-{raw_id}",
                        "title": title,
                        "source": "ghostcrawl-20k",
                        "category": _infer_category(text),
                        "ocrText": text if text else None,
                        "tags": ["ghostcrawl", "20k"],
                        "pageCount": row.get("page") or row.get("num_pages"),
                    }

                    chunk.append(doc)
                    imported += 1
                    progress.advance(task)

                if limit and imported >= limit:
                    break

        if output_dir and chunk:
            _write_chunk(output_dir, "gc-20k", chunk_num, chunk)

        console.print(f"  [green]Imported {imported:,} documents from 20k dataset[/green]")
        return imported

    def import_emails(self, output_dir: Path | None = None, limit: int | None = None) -> int:
        """Import emails (notesbymuneeb/Epstein-emails)."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            console.print("[red]pyarrow required: pip install pyarrow[/red]")
            return 0

        parquets = self._find_parquets()
        if not parquets:
            console.print(f"[red]No parquet files found in {self.data_dir}[/red]")
            return 0

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        imported = 0
        chunk: list[dict] = []

        with _progress() as progress:
            task = progress.add_task("Importing emails", total=limit or None)

            for pq_path in parquets:
                table = pq.read_table(pq_path)
                cols = set(table.column_names)

                for i in range(table.num_rows):
                    if limit and imported >= limit:
                        break

                    row = {c: table.column(c)[i].as_py() for c in cols}
                    text = row.get("text") or row.get("body") or row.get("content") or ""
                    raw_id = row.get("id") or str(imported)
                    subject = row.get("subject") or row.get("title") or f"Email-{raw_id}"

                    doc = {
                        "id": f"gc-email-{raw_id}",
                        "title": subject,
                        "source": "ghostcrawl-emails",
                        "category": "communications",
                        "ocrText": text if text else None,
                        "tags": ["ghostcrawl", "emails"],
                    }

                    chunk.append(doc)
                    imported += 1
                    progress.advance(task)

                if limit and imported >= limit:
                    break

        if output_dir and chunk:
            _write_chunk(output_dir, "gc-emails", 0, chunk)

        console.print(f"  [green]Imported {imported:,} email documents[/green]")
        return imported

    def import_fbi(self, output_dir: Path | None = None, limit: int | None = None) -> int:
        """Import FBI archive (svetfm/Epstein-Investigation-Fbi-Archive)."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            console.print("[red]pyarrow required: pip install pyarrow[/red]")
            return 0

        parquets = self._find_parquets()
        if not parquets:
            console.print(f"[red]No parquet files found in {self.data_dir}[/red]")
            return 0

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        imported = 0
        chunk: list[dict] = []

        with _progress() as progress:
            task = progress.add_task("Importing FBI archive", total=limit or None)

            for pq_path in parquets:
                table = pq.read_table(pq_path)
                cols = set(table.column_names)

                for i in range(table.num_rows):
                    if limit and imported >= limit:
                        break

                    row = {c: table.column(c)[i].as_py() for c in cols}
                    text = row.get("text") or row.get("text_content") or ""
                    raw_id = row.get("id") or row.get("doc_id") or str(imported)
                    title = row.get("file_name") or row.get("title") or f"FBI-{raw_id}"

                    doc = {
                        "id": f"gc-fbi-{raw_id}",
                        "title": title,
                        "source": "ghostcrawl-fbi",
                        "category": "investigation",
                        "ocrText": text if text else None,
                        "tags": ["ghostcrawl", "fbi"],
                        "pageCount": row.get("page") or row.get("num_pages"),
                    }

                    chunk.append(doc)
                    imported += 1
                    progress.advance(task)

                if limit and imported >= limit:
                    break

        if output_dir and chunk:
            _write_chunk(output_dir, "gc-fbi", 0, chunk)

        console.print(f"  [green]Imported {imported:,} FBI documents[/green]")
        return imported
