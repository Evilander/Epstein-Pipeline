"""Import OpenSanctions cross-reference results into Neon Postgres.

Reads the opensanctions-results.json from the downloader and writes
sanctions flags, PEP status, and dataset matches to the persons table
and a new sanctions_matches table.
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
from rich.console import Console
from rich.table import Table

console = Console()


def _ensure_sanctions_table(conn: psycopg.Connection) -> None:
    """Create the sanctions_matches table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sanctions_matches (
            id SERIAL PRIMARY KEY,
            person_id TEXT NOT NULL,
            person_name TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            caption TEXT,
            schema_type TEXT,
            score FLOAT,
            datasets TEXT[],
            first_seen TEXT,
            last_seen TEXT,
            is_sanctioned BOOLEAN DEFAULT false,
            is_pep BOOLEAN DEFAULT false,
            checked_at TIMESTAMPTZ DEFAULT NOW(),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(person_id, entity_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sanctions_person
        ON sanctions_matches(person_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sanctions_sanctioned
        ON sanctions_matches(is_sanctioned) WHERE is_sanctioned = true
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sanctions_pep
        ON sanctions_matches(is_pep) WHERE is_pep = true
    """)
    # Add columns to persons table if they don't exist
    for col, dtype, default in [
        ("is_sanctioned", "BOOLEAN", "false"),
        ("is_pep", "BOOLEAN", "false"),
        ("sanctions_datasets", "TEXT[]", "'{}'"),
        ("sanctions_score", "FLOAT", "NULL"),
        ("sanctions_checked_at", "TIMESTAMPTZ", "NULL"),
    ]:
        try:
            conn.execute(
                f"ALTER TABLE persons ADD COLUMN IF NOT EXISTS {col} {dtype} DEFAULT {default}"
            )
        except Exception:
            pass  # Column may already exist with different default

    conn.commit()


def import_opensanctions(
    results_path: Path,
    database_url: str,
    *,
    min_score: float = 0.4,
) -> None:
    """Import OpenSanctions results into Neon Postgres.

    Args:
        results_path: Path to opensanctions-results.json
        database_url: Neon Postgres connection URL
        min_score: Minimum match score to import (0-1)
    """
    if not results_path.exists():
        console.print(f"[red]Results file not found: {results_path}[/red]")
        return

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    metadata = data.get("metadata", {})

    console.print("[bold]Importing OpenSanctions Results[/bold]")
    console.print(f"Results file: [cyan]{results_path}[/cyan]")
    console.print(
        f"Persons checked: [cyan]{metadata.get('total_persons_checked', len(results))}[/cyan]"
    )
    console.print(f"Min import score: [cyan]{min_score}[/cyan]")
    console.print()

    imported_matches = 0
    updated_persons = 0
    errors = 0

    with psycopg.connect(database_url) as conn:
        _ensure_sanctions_table(conn)

        for result in results:
            person_id = result.get("person_id", "")
            person_name = result.get("person_name", "")
            is_sanctioned = result.get("is_sanctioned", False)
            is_pep = result.get("is_pep", False)
            best_score = result.get("best_score", 0.0)
            datasets = result.get("datasets", [])
            checked_at = result.get("checked_at")
            matches = result.get("matches", [])

            if not person_id:
                continue

            # Import individual matches
            for match in matches:
                if match.get("score", 0) < min_score:
                    continue
                try:
                    conn.execute(
                        """
                        INSERT INTO sanctions_matches
                            (person_id, person_name, entity_id, caption, schema_type,
                             score, datasets, first_seen, last_seen, is_sanctioned, is_pep, checked_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (person_id, entity_id) DO UPDATE SET
                            score = EXCLUDED.score,
                            datasets = EXCLUDED.datasets,
                            is_sanctioned = EXCLUDED.is_sanctioned,
                            is_pep = EXCLUDED.is_pep,
                            checked_at = EXCLUDED.checked_at
                        """,
                        (
                            person_id,
                            person_name,
                            match.get("entity_id", ""),
                            match.get("caption", ""),
                            match.get("schema_type", ""),
                            match.get("score", 0.0),
                            match.get("datasets", []),
                            match.get("first_seen"),
                            match.get("last_seen"),
                            is_sanctioned,
                            is_pep,
                            checked_at,
                        ),
                    )
                    imported_matches += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        console.print(f"[yellow]Warning: {e}[/yellow]")

            # Update person record with sanctions flags
            if matches and best_score >= min_score:
                try:
                    conn.execute(
                        """
                        UPDATE persons SET
                            is_sanctioned = %s,
                            is_pep = %s,
                            sanctions_datasets = %s,
                            sanctions_score = %s,
                            sanctions_checked_at = %s
                        WHERE id = %s OR slug = %s
                        """,
                        (
                            is_sanctioned,
                            is_pep,
                            datasets,
                            best_score,
                            checked_at,
                            person_id,
                            person_id,
                        ),
                    )
                    updated_persons += 1
                except Exception:
                    pass  # Person may not exist in DB yet

        conn.commit()

    console.print()

    # Summary
    table = Table(title="OpenSanctions Import Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Match records imported", str(imported_matches))
    table.add_row("Person records updated", str(updated_persons))
    table.add_row("Errors", str(errors))
    console.print(table)

    console.print("\n[green]Import complete.[/green]")
