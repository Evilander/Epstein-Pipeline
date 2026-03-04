"""GhostCrawl dataset downloader — five HuggingFace Epstein datasets.

Datasets range from small (~12 MB emails) to massive (~200 GB mega corpus).
Small datasets download directly via huggingface_hub; the mega dataset
prints download instructions similar to the DOJ downloader pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass(frozen=True)
class _GCDataset:
    """Metadata for a GhostCrawl HuggingFace dataset."""

    key: str
    repo_id: str
    description: str
    format: str
    approx_size: str
    auto_download: bool  # False = too large, print instructions instead


DATASETS: list[_GCDataset] = [
    _GCDataset(
        key="mega",
        repo_id="post-train/Epstein-Files",
        description="4.11M document rows — full OCR corpus",
        format="parquet",
        approx_size="~200 GB",
        auto_download=False,
    ),
    _GCDataset(
        key="20k",
        repo_id="teyler/epstein-files-20k",
        description="20K OCR documents (curated subset)",
        format="parquet",
        approx_size="~500 MB",
        auto_download=True,
    ),
    _GCDataset(
        key="embeddings",
        repo_id="svetfm/epstein-files-20k-embeddings",
        description="Pre-computed 384-dim sentence embeddings",
        format="parquet",
        approx_size="~200 MB",
        auto_download=True,
    ),
    _GCDataset(
        key="emails",
        repo_id="notesbymuneeb/Epstein-emails",
        description="Email corpus (structured)",
        format="parquet",
        approx_size="~12 MB",
        auto_download=True,
    ),
    _GCDataset(
        key="fbi",
        repo_id="svetfm/Epstein-Investigation-Fbi-Archive",
        description="FBI investigation archive",
        format="parquet",
        approx_size="~100 MB",
        auto_download=True,
    ),
]

DATASET_BY_KEY: dict[str, _GCDataset] = {ds.key: ds for ds in DATASETS}


class GhostCrawlDownloader:
    """Download GhostCrawl-sourced Epstein datasets from HuggingFace.

    Small datasets are fetched automatically via ``huggingface_hub``.
    The mega dataset (~200 GB) prints download instructions with URLs.
    """

    def __init__(self) -> None:
        self._console = Console()

    def list_datasets(self) -> None:
        """Print a formatted table of all GhostCrawl datasets."""
        table = Table(
            title="GhostCrawl Epstein Datasets (HuggingFace)",
            title_style="bold cyan",
            show_lines=True,
        )
        table.add_column("Key", style="bold", width=12)
        table.add_column("Repository", min_width=35)
        table.add_column("Description", min_width=30)
        table.add_column("Format", justify="center", style="green")
        table.add_column("Size", justify="right", style="yellow")
        table.add_column("Auto", justify="center")

        for ds in DATASETS:
            table.add_row(
                ds.key,
                ds.repo_id,
                ds.description,
                ds.format,
                ds.approx_size,
                "[green]yes[/green]" if ds.auto_download else "[yellow]manual[/yellow]",
            )

        self._console.print()
        self._console.print(table)
        self._console.print()

    def download(self, output_dir: Path, subset: str | None = None) -> None:
        """Download one or all GhostCrawl datasets.

        Parameters
        ----------
        output_dir:
            Root directory for downloaded files.
        subset:
            If provided, download only this dataset key (mega, 20k, etc.).
            If None, downloads all auto-downloadable datasets and prints
            instructions for the mega set.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if subset:
            ds = DATASET_BY_KEY.get(subset)
            if ds is None:
                self._console.print(
                    f"[red]Unknown subset '{subset}'. "
                    f"Valid: {', '.join(DATASET_BY_KEY)}[/red]"
                )
                return
            targets = [ds]
        else:
            targets = DATASETS

        for ds in targets:
            self._console.print()
            self._console.rule(f"[bold cyan]{ds.key}[/bold cyan] — {ds.repo_id}")
            self._console.print(f"  {ds.description}")
            self._console.print(f"  Size: [yellow]{ds.approx_size}[/yellow]  Format: {ds.format}")

            if ds.auto_download:
                self._download_hf(ds, output_dir)
            else:
                self._print_instructions(ds, output_dir)

    def _download_hf(self, ds: _GCDataset, output_dir: Path) -> None:
        """Download a dataset using huggingface_hub."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            self._console.print(
                "[yellow]huggingface_hub not installed. "
                "Install with: pip install huggingface_hub[/yellow]"
            )
            self._print_instructions(ds, output_dir)
            return

        local_dir = output_dir / ds.key
        self._console.print(f"  Downloading to [green]{local_dir}[/green] ...")

        try:
            snapshot_download(
                repo_id=ds.repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
            )
            file_count = sum(1 for f in local_dir.rglob("*") if f.is_file())
            self._console.print(
                f"  [bold green]Done — {file_count:,} files downloaded[/bold green]"
            )
        except Exception as exc:
            self._console.print(f"  [red]Download failed: {exc}[/red]")

    def _print_instructions(self, ds: _GCDataset, output_dir: Path) -> None:
        """Print manual download instructions for large datasets."""
        hf_url = f"https://huggingface.co/datasets/{ds.repo_id}"
        self._console.print()
        self._console.print(f"  [bold]This dataset is too large for automatic download ({ds.approx_size}).[/bold]")
        self._console.print()
        self._console.print("  [bold]Option 1 — huggingface-cli:[/bold]")
        self._console.print(
            f"    [dim]huggingface-cli download --repo-type dataset "
            f"{ds.repo_id} --local-dir {output_dir / ds.key}[/dim]"
        )
        self._console.print()
        self._console.print("  [bold]Option 2 — Browser / git:[/bold]")
        self._console.print(f"    [link={hf_url}]{hf_url}[/link]")
        self._console.print()
        self._console.print("  [bold]After downloading:[/bold]")
        self._console.print(
            f"    epstein-pipeline import ghostcrawl-mega --data-dir {output_dir / ds.key}"
        )
        self._console.print()
