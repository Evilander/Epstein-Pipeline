"""Import rhowardstone/Epstein-research-data knowledge graph into Pipeline format.

Downloads entities and relationships from the GitHub repository
and maps them to Pipeline's Person model + KnowledgeGraphBuilder.

Source: https://github.com/rhowardstone/Epstein-research-data
- 606 entities (persons, organizations, shell companies, properties, aircraft, locations)
- 2,302 relationships with evidence metadata
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import httpx
from rich.console import Console

from epstein_pipeline.processors.knowledge_graph import GraphEdge, GraphNode, KnowledgeGraph

logger = logging.getLogger(__name__)
console = Console()

# Raw GitHub URLs — repo was reorganised from rhowardstone/Epstein to
# rhowardstone/Epstein-research-data in early 2026
_ENTITIES_URL = (
    "https://raw.githubusercontent.com/rhowardstone/"
    "Epstein-research-data/main/knowledge_graph_entities.json"
)
_RELATIONSHIPS_URL = (
    "https://raw.githubusercontent.com/rhowardstone/"
    "Epstein-research-data/main/knowledge_graph_relationships.json"
)
_PERSONS_REGISTRY_URL = (
    "https://raw.githubusercontent.com/rhowardstone/"
    "Epstein-research-data/main/persons_registry.json"
)

# Map entity_type to Pipeline node types
_TYPE_MAP = {
    "person": "person",
    "shell_company": "org",
    "organization": "org",
    "org": "org",
    "property": "location",
    "aircraft": "location",
    "location": "location",
}

# Map entity_type to Pipeline Person categories
_CATEGORY_MAP = {
    "person": "individual",
    "shell_company": "corporate",
    "organization": "organization",
    "org": "organization",
    "property": "property",
    "aircraft": "aircraft",
    "location": "location",
}


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


def _parse_metadata(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse metadata field — may be a JSON string or already a dict."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(raw)
        return cast(dict[str, Any], payload) if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


class GhostCrawlGraphImporter:
    """Import the rhowardstone/Epstein-research-data knowledge graph."""

    def import_graph(self, output_dir: Path | None = None) -> KnowledgeGraph:
        """Download and import the full knowledge graph.

        Returns a KnowledgeGraph with nodes and edges.
        Optionally saves to output_dir as JSON files.
        """
        console.print()
        console.rule("[bold cyan]GhostCrawl Knowledge Graph Import[/bold cyan]")

        # Fetch entities
        console.print("  Fetching entities from GitHub...")
        try:
            raw_entities = _fetch_json(_ENTITIES_URL)
        except Exception as exc:
            console.print(f"  [red]Failed to fetch entities: {exc}[/red]")
            return KnowledgeGraph()
        if not isinstance(raw_entities, list):
            console.print("  [red]Entity payload was not a list.[/red]")
            return KnowledgeGraph()
        raw_entities = cast(list[dict[str, Any]], raw_entities)

        console.print(f"  [green]Got {len(raw_entities)} entities[/green]")

        # Fetch relationships
        console.print("  Fetching relationships from GitHub...")
        try:
            raw_rels = _fetch_json(_RELATIONSHIPS_URL)
        except Exception as exc:
            console.print(f"  [red]Failed to fetch relationships: {exc}[/red]")
            raw_rels = []
        if not isinstance(raw_rels, list):
            console.print("  [yellow]Relationship payload was not a list.[/yellow]")
            raw_rels = []
        raw_rels = cast(list[dict[str, Any]], raw_rels)

        console.print(f"  [green]Got {len(raw_rels)} relationships[/green]")

        # Build entity ID → name lookup (relationships use numeric IDs)
        id_to_name: dict[str, str] = {}

        # Build graph
        nodes: list[GraphNode] = []
        persons: list[dict[str, Any]] = []

        for i, entity in enumerate(raw_entities):
            name = entity.get("name", "")
            # New schema uses 'entity_type', old used 'type'
            etype = entity.get("entity_type") or entity.get("type", "person")
            eid = str(entity.get("id", i))
            meta = _parse_metadata(entity.get("metadata"))

            id_to_name[eid] = name

            node = GraphNode(
                id=eid,
                label=name,
                type=_TYPE_MAP.get(etype, "person"),
                attributes={
                    "original_type": etype,
                    "aliases": entity.get("aliases") or [],
                    "description": meta.get("occupation", ""),
                    "person_type": meta.get("person_type", ""),
                    "legal_status": meta.get("legal_status", ""),
                    "mention_count": meta.get("ds10_mention_count", 0),
                    "source": "ghostcrawl-graph",
                },
            )
            nodes.append(node)

            slug = name.lower().replace(" ", "-").replace(".", "").replace(",", "")
            bio_parts = []
            if meta.get("occupation"):
                bio_parts.append(meta["occupation"])
            if meta.get("person_type"):
                bio_parts.append(meta["person_type"])
            if meta.get("legal_status"):
                bio_parts.append(meta["legal_status"])

            persons.append(
                {
                    "id": f"gc-{eid}",
                    "slug": slug,
                    "name": name,
                    "aliases": entity.get("aliases") or [],
                    "category": _CATEGORY_MAP.get(etype, "individual"),
                    "shortBio": " — ".join(bio_parts) if bio_parts else "",
                }
            )

        # Build edges — new schema uses source_entity_id/target_entity_id (numeric)
        edges: list[GraphEdge] = []
        for rel in raw_rels:
            source = str(rel.get("source_entity_id") or rel.get("source") or rel.get("from", ""))
            target = str(rel.get("target_entity_id") or rel.get("target") or rel.get("to", ""))
            rel_type = (
                rel.get("relationship_type")
                or rel.get("type")
                or rel.get("relationship", "associated_with")
            )
            weight = float(rel.get("weight", 1.0))
            rel_meta = _parse_metadata(rel.get("metadata"))

            edges.append(
                GraphEdge(
                    source=source,
                    target=target,
                    type=rel_type,
                    weight=weight,
                    attributes={
                        "evidence": rel_meta.get("notes", ""),
                        "evidence_type": rel_meta.get("evidence_type", ""),
                        "efta_ref": rel_meta.get("efta", ""),
                        "data_source": rel_meta.get("source", ""),
                        "date_first": rel.get("date_first"),
                        "date_last": rel.get("date_last"),
                        "source": "ghostcrawl-graph",
                    },
                )
            )

        graph = KnowledgeGraph(nodes=nodes, edges=edges)

        # Save outputs
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            persons_path = output_dir / "ghostcrawl-persons.json"
            persons_path.write_text(
                json.dumps(persons, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            console.print(f"  [green]Saved {len(persons)} persons to {persons_path}[/green]")

            graph_path = output_dir / "ghostcrawl-graph.json"
            graph_data = {
                "nodes": [
                    {"id": n.id, "label": n.label, "type": n.type, **n.attributes} for n in nodes
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "source_name": id_to_name.get(e.source, e.source),
                        "target_name": id_to_name.get(e.target, e.target),
                        "type": e.type,
                        "weight": e.weight,
                        **{k: v for k, v in e.attributes.items() if v},
                    }
                    for e in edges
                ],
            }
            graph_path.write_text(
                json.dumps(graph_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            console.print(f"  [green]Saved graph to {graph_path}[/green]")

        console.print()
        console.rule("[bold green]Graph Import Complete[/bold green]")
        console.print(f"  Nodes:         {len(nodes):,}")
        console.print(f"  Edges:         {len(edges):,}")
        console.print(f"  Persons:       {len(persons):,}")

        return graph
