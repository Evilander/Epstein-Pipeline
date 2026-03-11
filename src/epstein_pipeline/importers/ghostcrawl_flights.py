"""Import parsed Epstein flight logs from GitHub sources.

Fetches flight log CSVs and maps them to Pipeline's Flight model.
Searches multiple known repositories for flight data — sources may
become unavailable as repos are reorganised.

Also imports the persons registry from rhowardstone/Epstein-research-data
(1,614 categorised persons) which can cross-reference passenger names.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, cast

import httpx
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Known flight log sources — checked in order, failures are non-fatal
_FLIGHT_SOURCES = [
    {
        "name": "rhowardstone/Epstein-research-data flights",
        "url": "https://raw.githubusercontent.com/rhowardstone/Epstein-research-data/main/data/flights.csv",
    },
    {
        "name": "rhowardstone/Epstein flights (legacy)",
        "url": "https://raw.githubusercontent.com/rhowardstone/Epstein/main/data/flights.csv",
    },
]

_PERSONS_REGISTRY_URL = (
    "https://raw.githubusercontent.com/rhowardstone/"
    "Epstein-research-data/main/persons_registry.json"
)


def _fetch_text(url: str) -> str:
    """Fetch text content from a URL."""
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _fetch_json(url: str) -> list[dict[str, Any]] | dict[str, Any]:
    """Fetch and parse JSON from a URL."""
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list):
            return cast(list[dict[str, Any]], payload)
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        return {}


class GhostCrawlFlightImporter:
    """Import parsed flight logs into Pipeline Flight format."""

    def import_flights(self, output_dir: Path | None = None) -> list[dict]:
        """Download and import flight logs.

        Returns a list of Flight-compatible dicts.
        """
        console.print()
        console.rule("[bold cyan]GhostCrawl Flight Log Import[/bold cyan]")

        all_flights: list[dict] = []

        for source in _FLIGHT_SOURCES:
            console.print(f"  Fetching: {source['name']}...")
            try:
                text = _fetch_text(source["url"])
            except Exception as exc:
                console.print(f"  [yellow]Unavailable: {exc}[/yellow]")
                continue

            flights = self._parse_csv(text, source["name"])
            console.print(f"  [green]Parsed {len(flights)} flight records[/green]")
            all_flights.extend(flights)

        if not all_flights:
            console.print("  [yellow]No flight data available from known sources.[/yellow]")
            console.print(
                "  [dim]Flight CSVs may have been removed during repo reorganisation.[/dim]"
            )
            console.print(
                "  [dim]The persons registry (1,614 entries) is still available "
                "via the knowledge graph importer.[/dim]"
            )

        if output_dir and all_flights:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "ghostcrawl-flights.json"
            out_path.write_text(
                json.dumps(all_flights, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            console.print(f"  [green]Saved to {out_path}[/green]")

        console.print()
        console.rule("[bold green]Flight Import Complete[/bold green]")
        console.print(f"  Total flights: {len(all_flights):,}")

        return all_flights

    def import_persons_registry(self, output_dir: Path | None = None) -> list[dict]:
        """Download the persons registry (1,614 categorised persons).

        This cross-references multiple sources: epstein-pipeline (1,195),
        knowledge graph (285), la-rana-chicana (237), wikipedia, bondi letter, etc.

        Returns a list of person dicts with name, slug, aliases, category, sources.
        """
        console.print()
        console.rule("[bold cyan]Persons Registry Import[/bold cyan]")

        console.print("  Fetching persons registry from GitHub...")
        try:
            persons_payload = _fetch_json(_PERSONS_REGISTRY_URL)
        except Exception as exc:
            console.print(f"  [red]Failed to fetch persons registry: {exc}[/red]")
            return []

        if not isinstance(persons_payload, list):
            console.print("  [red]Persons registry payload was not a list.[/red]")
            return []
        persons = persons_payload

        console.print(f"  [green]Got {len(persons)} persons[/green]")

        # Category breakdown
        from collections import Counter

        cats = Counter(p.get("category", "unknown") for p in persons)
        for cat, count in cats.most_common(8):
            console.print(f"    {cat}: {count}")

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "ghostcrawl-persons-registry.json"
            out_path.write_text(
                json.dumps(persons, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            console.print(f"  [green]Saved to {out_path}[/green]")

        console.print()
        console.rule("[bold green]Persons Registry Import Complete[/bold green]")
        console.print(f"  Total persons: {len(persons):,}")

        return persons

    def _parse_csv(self, text: str, source_name: str) -> list[dict]:
        """Parse a flight CSV into Flight-compatible dicts.

        Handles various column naming conventions:
        - date/Date/DATE
        - aircraft/Aircraft/tail_number
        - origin/departure/from
        - destination/arrival/to
        - passengers/Passengers/passenger_list
        """
        reader = csv.DictReader(io.StringIO(text))
        flights: list[dict] = []

        for i, row in enumerate(reader):
            r = {k.lower().strip(): v for k, v in row.items() if v}

            date = r.get("date") or r.get("flight_date")
            aircraft = r.get("aircraft") or r.get("aircraft_type")
            tail = r.get("tail_number") or r.get("tailnumber") or r.get("tail")
            origin = r.get("origin") or r.get("departure") or r.get("from")
            dest = r.get("destination") or r.get("arrival") or r.get("to")

            passengers_raw = r.get("passengers") or r.get("passenger_list") or ""
            if passengers_raw:
                passenger_names = [p.strip() for p in passengers_raw.split(",") if p.strip()]
            else:
                passenger_names = []

            flight = {
                "id": f"gc-flight-{i:05d}",
                "date": date,
                "aircraft": aircraft,
                "tailNumber": tail,
                "origin": origin,
                "destination": dest,
                "passengerIds": [],
                "pilotIds": [],
                "tags": ["ghostcrawl", source_name],
                "_passengerNames": passenger_names,
            }

            flights.append(flight)

        return flights
