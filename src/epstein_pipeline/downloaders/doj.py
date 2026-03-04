"""DOJ EFTA dataset downloader.

The DOJ released Epstein-related documents in 12 data sets through the
Epstein Files Transparency Act (EFTA).  Individual volumes range from
61 MB (Data Set 5) to ~181 GB (Data Set 9).

DOJ direct URLs require age verification (JavaScript gate), so this module
provides multiple download options: DOJ links, Archive.org mirrors,
GeekenDev mirrors, and BitTorrent magnets.

Source catalogue: github.com/yung-megafone/Epstein-Files
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class _DatasetMeta:
    """Metadata for a single DOJ EFTA data set."""

    description: str
    size_label: str
    sha256: str
    doj_url: str
    torrent_magnet: str
    mirror_urls: list[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Dataset catalogue — verified against yung-megafone/Epstein-Files
# DOJ URL pattern: https://www.justice.gov/epstein/files/DataSet%20{N}.zip
# DOJ index page:  https://www.justice.gov/epstein/doj-disclosures
# Individual pages: https://www.justice.gov/epstein/doj-disclosures/data-set-{N}-files
# ---------------------------------------------------------------------------

_DOJ_BASE = "https://www.justice.gov/epstein/files"
_GEEKEN_BASE = "https://doj-files.geeken.dev/doj_zips/original_archives"
_IA_BASE = "https://archive.org/download"

_DATASETS: dict[int, _DatasetMeta] = {
    1: _DatasetMeta(
        description="Data Set 1 (VOL00001)",
        size_label="1.23 GB",
        sha256="598F4D2D71F0D183CF898CD9D6FB8EC1F6161E0E71D8C37897936AEF75F860B4",
        doj_url=f"{_DOJ_BASE}/DataSet%201.zip",
        torrent_magnet="magnet:?xt=urn:btih:10d451428ac43a98f2eff06ce256231b8eba7bed&dn=VOL00001&xl=1329247376",
        mirror_urls=[
            f"{_IA_BASE}/data-set-1/DataSet%201.zip",
            f"{_GEEKEN_BASE}/DataSet%201.zip",
        ],
    ),
    2: _DatasetMeta(
        description="Data Set 2 (VOL00002)",
        size_label="630 MB",
        sha256="24CEBBAEFE9D49BCA57726B5A4B531FF20E6A97C370BA87A7593DD8DBDB77BFF",
        doj_url=f"{_DOJ_BASE}/DataSet%202.zip",
        torrent_magnet="magnet:?xt=urn:btih:dd6b5cac7991d34625e4eea1fb2c295c6fbd3adc&dn=VOL00002&xl=662334369",
        mirror_urls=[
            f"{_IA_BASE}/data-set-1/DataSet%202.zip",
            f"{_GEEKEN_BASE}/DataSet%202.zip",
        ],
    ),
    3: _DatasetMeta(
        description="Data Set 3 (VOL00003)",
        size_label="595 MB",
        sha256="160231C8C689C76003976B609E55689530FC4832A1535CE13BFCD8F871C21E65",
        doj_url=f"{_DOJ_BASE}/DataSet%203.zip",
        torrent_magnet="magnet:?xt=urn:btih:3f5923fefc496e394fc1fd553f9d3a1c4242789c&dn=VOL00003&xl=628519331",
        mirror_urls=[
            f"{_IA_BASE}/data-set-1/DataSet%203.zip",
            f"{_GEEKEN_BASE}/DataSet%203.zip",
        ],
    ),
    4: _DatasetMeta(
        description="Data Set 4 (VOL00004)",
        size_label="351 MB",
        sha256="979154842BAC356EF36BB2D0E72F78E0F6B771D79E02DD6934CFF699944E2B71",
        doj_url=f"{_DOJ_BASE}/DataSet%204.zip",
        torrent_magnet="magnet:?xt=urn:btih:6fc5aede157615f08335568efbf459b537001756&dn=VOL00004&xl=375905556",
        mirror_urls=[
            f"{_IA_BASE}/data-set-1/DataSet%204.zip",
            f"{_GEEKEN_BASE}/DataSet%204.zip",
        ],
    ),
    5: _DatasetMeta(
        description="Data Set 5 (VOL00005)",
        size_label="61.4 MB",
        sha256="7317E2AD089C82A59378A9C038E964FEAB246BE62ECC24663B741617AF3DA709",
        doj_url=f"{_DOJ_BASE}/DataSet%205.zip",
        torrent_magnet="magnet:?xt=urn:btih:9e3dd82b77f2c7264b0ef13f87f31a5c3c291046&dn=VOL00005&xl=64579973",
        mirror_urls=[
            f"{_IA_BASE}/data-set-1/DataSet%205.zip",
            f"{_GEEKEN_BASE}/DataSet%205.zip",
        ],
    ),
    6: _DatasetMeta(
        description="Data Set 6 (VOL00006)",
        size_label="51.2 MB",
        sha256="D54D26D94127B9A277CF3F7D9EEAF9A7271F118757997EDAC3BC6E1039ED6555",
        doj_url=f"{_DOJ_BASE}/DataSet%206.zip",
        torrent_magnet="magnet:?xt=urn:btih:6d1792e4d04d814bbbf0ed06e70ee72b92b5544f&dn=VOL00006&xl=55600717",
        mirror_urls=[
            f"{_IA_BASE}/data-set-1/DataSet%206.zip",
            f"{_GEEKEN_BASE}/DataSet%206.zip",
        ],
    ),
    7: _DatasetMeta(
        description="Data Set 7 (VOL00007)",
        size_label="96.9 MB",
        sha256="51E1961B3BCF18A21AFD9BCF697FDB54DAC97D1B64CF88297F4C5BE268D26B8E",
        doj_url=f"{_DOJ_BASE}/DataSet%207.zip",
        torrent_magnet="magnet:?xt=urn:btih:101b10571e0b2edc2496f72a831c865ec6a3c070&dn=VOL00007&xl=103060624",
        mirror_urls=[
            f"{_IA_BASE}/data-set-1/DataSet%207.zip",
            f"{_GEEKEN_BASE}/DataSet%207.zip",
        ],
    ),
    8: _DatasetMeta(
        description="Data Set 8 (VOL00008)",
        size_label="10.67 GB",
        sha256="8CB7345BF7A0B32F183658AC170FB0B6527895C95F0233D7B99D544579567294",
        doj_url=f"{_DOJ_BASE}/DataSet%208.zip",
        torrent_magnet="magnet:?xt=urn:btih:8dacaa3a16be77a51db1cc21fe3b0ffaca0ab116&dn=VOL00008&xl=11465535175",
        mirror_urls=[
            "https://archive.org/details/data-set-8",
            f"{_GEEKEN_BASE}/DataSet%208.zip",
        ],
    ),
    9: _DatasetMeta(
        description="Data Set 9 (VOL00009) — LARGEST",
        size_label="~181 GB",
        sha256="EB3C112A62D326E302BCFC6224CC1D31A5FFF6C5F3BDF9F4EC32089511191922",
        doj_url=f"{_DOJ_BASE}/DataSet%209.zip",
        torrent_magnet=(
            "magnet:?xt=urn:btih:50fc8133084864e15440b53dfa89cd43cda6c934"
            "&xt=urn:btmh:1220482c89a1b7e09185fdf01b2fb7e15d60fe4518425e8db6f75a5a5a2f45054f40"
            "&dn=DS9_181GB&xl=186445201408"
            "&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
        ),
        mirror_urls=[
            f"{_GEEKEN_BASE}/DataSet%209.zip",
        ],
        notes="~99.9% reconstructed; original DOJ release was incomplete",
    ),
    10: _DatasetMeta(
        description="Data Set 10 (VOL00010)",
        size_label="78.65 GB",
        sha256="7D6935B1C63FF2F6BCABDD024EBC2A770F90C43B0D57B646FA7CBD4C0ABCF846",
        doj_url=f"{_DOJ_BASE}/DataSet%2010.zip",
        torrent_magnet=(
            "magnet:?xt=urn:btih:d509cc4ca1a415a9ba3b6cb920f67c44aed7fe1f"
            "&dn=DataSet%2010.zip&xl=84439381640"
        ),
        mirror_urls=[
            f"{_IA_BASE}/data-set-10/DataSet%2010.zip",
            f"{_GEEKEN_BASE}/DataSet%2010.zip",
        ],
    ),
    11: _DatasetMeta(
        description="Data Set 11 (VOL00011)",
        size_label="27.5 GB",
        sha256="9714273B9E325F0A1F406063C795DB32F5DA2095B75E602D4C4FBABA5DE3ED80",
        doj_url=f"{_DOJ_BASE}/DataSet%2011.zip",
        torrent_magnet=(
            "magnet:?xt=urn:btih:59975667f8bdd5baf9945b0e2db8a57d52d32957"
            "&xt=urn:btmh:12200ab9e7614c13695fe17c71baedec717b6294a34dfa243a614602b87ec06453ad"
            "&dn=DataSet%2011.zip&xl=27441913130"
            "&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
        ),
        mirror_urls=[
            f"{_GEEKEN_BASE}/DataSet%2011.zip",
        ],
    ),
    12: _DatasetMeta(
        description="Data Set 12 (VOL00012)",
        size_label="114.1 MB",
        sha256="B5314B7EFCA98E25D8B35E4B7FAC3EBB3CA2E6CFD0937AA2300CA8B71543BBE2",
        doj_url=f"{_DOJ_BASE}/DataSet%2012.zip",
        torrent_magnet="magnet:?xt=urn:btih:3db3fc05e2481513675a50e313333692995e19ca&dn=VOL00012&xl=125711730",
        mirror_urls=[
            "https://archive.org/details/data-set-12_202601",
            f"{_GEEKEN_BASE}/DataSet%2012.zip",
        ],
    ),
}


class DojDownloader:
    """Download (or print instructions for) DOJ EFTA datasets.

    The DOJ hosts files behind a JavaScript age verification gate that
    blocks curl/wget.  This downloader provides verified URLs for DOJ
    direct links, Archive.org mirrors, GeekenDev mirrors, and BitTorrent
    magnets with SHA256 checksums for integrity verification.
    """

    DATASETS = _DATASETS

    def __init__(self) -> None:
        self._console = Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, dataset_num: int, output_dir: Path) -> None:
        """Print download instructions for the specified DOJ data set."""
        if dataset_num not in self.DATASETS:
            self._console.print(
                f"[red]Unknown dataset number {dataset_num}. Valid range is 1-12.[/red]"
            )
            return

        meta = self.DATASETS[dataset_num]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_file = output_dir / f"DataSet_{dataset_num}.zip"

        self._console.print()
        self._console.rule(f"[bold cyan]DOJ EFTA Data Set {dataset_num}[/bold cyan]")
        self._console.print(f"[bold]{meta.description}[/bold]")
        self._console.print(f"Size: [yellow]{meta.size_label}[/yellow]")
        if meta.notes:
            self._console.print(f"Note: [dim]{meta.notes}[/dim]")
        self._console.print(f"SHA256: [dim]{meta.sha256}[/dim]")
        self._console.print()

        # DOJ direct (behind age gate)
        self._console.print("[bold]1. DOJ direct[/bold] (requires age verification in browser):")
        self._console.print(f"   [link={meta.doj_url}]{meta.doj_url}[/link]")
        self._console.print()

        # Community mirrors (best for programmatic access)
        self._console.print("[bold]2. Mirrors[/bold] (recommended for wget/curl):")
        for url in meta.mirror_urls:
            self._console.print(f"   [link={url}]{url}[/link]")
        self._console.print()
        if meta.mirror_urls:
            best = meta.mirror_urls[0]
            self._console.print(f'   [dim]wget -O "{dest_file}" "{best}"[/dim]')
            self._console.print()

        # Torrent
        self._console.print("[bold]3. BitTorrent[/bold] (best for large sets):")
        self._console.print(f"   [dim]{meta.torrent_magnet[:120]}...[/dim]")
        self._console.print()

        # Verification
        self._console.print("[bold]After downloading:[/bold]")
        self._console.print(f"   1. Verify: [dim]sha256sum {dest_file}[/dim]")
        self._console.print(f"      Expected: [dim]{meta.sha256}[/dim]")
        self._console.print(f"   2. Extract to: [green]{output_dir.resolve()}[/green]")
        self._console.print()

    def download_all(self, output_dir: Path) -> None:
        """Print download instructions for all 12 data sets."""
        self.list_datasets()
        self._console.print()
        self._console.rule("[bold cyan]Download Instructions[/bold cyan]")
        self._console.print()
        self._console.print("[bold]Batch download via mirrors (direct links):[/bold]")
        self._console.print()
        for num in sorted(self.DATASETS):
            meta = self.DATASETS[num]
            dest = output_dir / f"DataSet_{num}.zip"
            # Pick first mirror that's a direct file link, not an archive.org details page
            best_mirror = meta.doj_url
            for url in meta.mirror_urls:
                if "/download/" in url or "geeken.dev" in url:
                    best_mirror = url
                    break
            self._console.print(f'  wget -O "{dest}" "{best_mirror}"')
        self._console.print()
        self._console.print("[bold]DOJ index page:[/bold]")
        self._console.print("  https://www.justice.gov/epstein/doj-disclosures")
        self._console.print()

    def list_datasets(self) -> None:
        """Print a formatted table of all available DOJ EFTA datasets."""
        table = Table(
            title="DOJ EFTA Datasets (12 Data Sets)",
            title_style="bold cyan",
            show_lines=True,
        )
        table.add_column("#", style="bold", justify="right", width=4)
        table.add_column("Description", min_width=30)
        table.add_column("Size", justify="right", style="yellow")
        table.add_column("SHA256", style="dim", max_width=20, overflow="ellipsis")
        table.add_column("Mirrors", justify="center", style="green")

        for num, meta in sorted(self.DATASETS.items()):
            table.add_row(
                str(num),
                meta.description,
                meta.size_label,
                meta.sha256[:16] + "...",
                str(len(meta.mirror_urls)),
            )

        self._console.print()
        self._console.print(table)
        self._console.print()
        self._console.print(
            "  [dim]DOJ files require browser age verification. "
            "Use mirrors or torrents for programmatic access.[/dim]"
        )
        self._console.print()
