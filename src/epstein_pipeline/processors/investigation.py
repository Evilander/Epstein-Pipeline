"""Multi-hop investigation engine for the Epstein knowledge graph.

Turns flat document search into threaded investigation — traverse connections,
detect communities, thread timelines, and discover cross-entity patterns.

Sits on top of KnowledgeGraph (from knowledge_graph.py) and adds:
- N-hop graph traversal from any person
- Louvain community detection (social circles)
- Temporal threading (document mentions over time)
- Cross-entity document discovery
- Path finding between entities
- Degree/betweenness centrality ranking
- Interactive investigation REPL
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from epstein_pipeline.models.document import Document, Email, Flight, Person
from epstein_pipeline.processors.knowledge_graph import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures for investigation results
# ---------------------------------------------------------------------------


@dataclass
class ConnectionPath:
    """A path between two entities through the graph."""

    source: str
    target: str
    hops: list[str]  # node IDs along the path
    edges: list[GraphEdge]  # edges traversed
    total_weight: float
    relationship_types: list[str]


@dataclass
class Community:
    """A detected community (social circle) in the graph."""

    id: int
    members: list[str]  # node IDs
    member_labels: dict[str, str]  # id -> label
    size: int
    internal_edges: int
    dominant_relationship: str  # most common edge type within
    label: str = ""  # auto-generated description


@dataclass
class TemporalThread:
    """A timeline of document mentions for an entity or entity pair."""

    entity_ids: list[str]
    events: list[dict[str, Any]]  # sorted by date
    date_range: tuple[str | None, str | None]
    total_mentions: int
    sources_breakdown: dict[str, int]  # source type -> count


@dataclass
class EntityProfile:
    """Comprehensive profile of an entity from the graph."""

    id: str
    label: str
    node_type: str
    degree: int  # total connections
    weighted_degree: float
    top_connections: list[tuple[str, str, float]]  # (id, label, weight)
    community_id: int | None
    centrality_rank: int | None
    edge_type_breakdown: dict[str, int]
    document_count: int
    flight_count: int
    email_count: int


@dataclass
class InvestigationResult:
    """Container for investigation query results."""

    query_type: str
    query_params: dict[str, Any]
    results: Any
    summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Core investigation engine
# ---------------------------------------------------------------------------


class InvestigationEngine:
    """Multi-hop investigation engine over the Epstein knowledge graph.

    Build the graph first with KnowledgeGraphBuilder, then pass it here
    for interactive querying and analysis.

    Usage:
        builder = KnowledgeGraphBuilder(settings)
        builder.add_documents(docs)
        builder.add_flights(flights)
        graph = builder.build()

        engine = InvestigationEngine(graph)
        engine.load_documents(docs)
        engine.load_persons(persons)

        # Find everyone within 2 hops of a person
        neighbors = engine.traverse("p-0001", max_hops=2)

        # Detect social circles
        communities = engine.detect_communities()

        # Timeline of co-appearances
        timeline = engine.temporal_thread(["p-0001", "p-0042"])
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

        # Build adjacency structures for fast traversal
        self._adj: dict[str, list[tuple[str, GraphEdge]]] = defaultdict(list)
        self._node_map: dict[str, GraphNode] = {}
        self._edge_index: dict[tuple[str, str], list[GraphEdge]] = defaultdict(list)

        # Document/flight/email indexes (loaded separately)
        self._docs_by_person: dict[str, list[Document]] = defaultdict(list)
        self._flights_by_person: dict[str, list[Flight]] = defaultdict(list)
        self._emails_by_person: dict[str, list[Email]] = defaultdict(list)
        self._persons: dict[str, Person] = {}
        self._all_docs: list[Document] = []

        # Cached analysis results
        self._communities: list[Community] | None = None
        self._centrality: dict[str, float] | None = None
        self._degree_cache: dict[str, int] = {}

        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build adjacency list and lookup structures from the graph."""
        for node in self.graph.nodes:
            self._node_map[node.id] = node

        for edge in self.graph.edges:
            self._adj[edge.source].append((edge.target, edge))
            self._adj[edge.target].append((edge.source, edge))
            # Normalize edge key (sorted pair)
            key = (min(edge.source, edge.target), max(edge.source, edge.target))
            self._edge_index[key].append(edge)

        for node_id in self._node_map:
            self._degree_cache[node_id] = len(self._adj[node_id])

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_documents(self, documents: list[Document]) -> None:
        """Index documents by person ID for cross-reference queries."""
        self._all_docs = documents
        self._docs_by_person.clear()
        for doc in documents:
            for pid in doc.personIds:
                self._docs_by_person[pid].append(doc)

    def load_flights(self, flights: list[Flight]) -> None:
        """Index flights by person ID."""
        self._flights_by_person.clear()
        for flight in flights:
            for pid in flight.passengerIds + flight.pilotIds:
                self._flights_by_person[pid].append(flight)

    def load_emails(self, emails: list[Email]) -> None:
        """Index emails by person ID."""
        self._emails_by_person.clear()
        for email in emails:
            for pid in email.personIds:
                self._emails_by_person[pid].append(email)

    def load_persons(self, persons: list[Person] | dict[str, Person]) -> None:
        """Load person registry for name lookups."""
        if isinstance(persons, dict):
            self._persons = persons
        else:
            self._persons = {p.id: p for p in persons}

    def _label(self, node_id: str) -> str:
        """Get human-readable label for a node."""
        if node_id in self._persons:
            return self._persons[node_id].name
        if node_id in self._node_map:
            return self._node_map[node_id].label
        return node_id

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def traverse(
        self,
        start: str,
        max_hops: int = 2,
        edge_types: list[str] | None = None,
        min_weight: float = 0.0,
    ) -> dict[str, dict[str, Any]]:
        """BFS traversal from a starting node, returning all reachable nodes.

        Returns dict of node_id -> {distance, path, weight, label, edges}.
        """
        if start not in self._node_map:
            # Try fuzzy match by label
            resolved_start = self._resolve_entity(start)
            if not resolved_start:
                return {}
            start = resolved_start

        visited: dict[str, dict[str, Any]] = {
            start: {
                "distance": 0,
                "path": [start],
                "weight": 0.0,
                "label": self._label(start),
                "edges": [],
            }
        }
        frontier = [start]

        for hop in range(max_hops):
            next_frontier = []
            for node_id in frontier:
                for neighbor_id, edge in self._adj[node_id]:
                    if edge_types and edge.type not in edge_types:
                        continue
                    if edge.weight < min_weight:
                        continue
                    if neighbor_id not in visited:
                        visited[neighbor_id] = {
                            "distance": hop + 1,
                            "path": visited[node_id]["path"] + [neighbor_id],
                            "weight": visited[node_id]["weight"] + edge.weight,
                            "label": self._label(neighbor_id),
                            "edges": visited[node_id]["edges"] + [edge.type],
                        }
                        next_frontier.append(neighbor_id)
            frontier = next_frontier

        return visited

    def find_path(
        self,
        source: str,
        target: str,
        max_hops: int = 6,
    ) -> ConnectionPath | None:
        """Find shortest path between two entities (BFS)."""
        source = self._resolve_entity(source) or source
        target = self._resolve_entity(target) or target

        if source not in self._node_map or target not in self._node_map:
            return None

        # BFS
        visited: dict[str, tuple[str | None, GraphEdge | None]] = {
            source: (None, None)
        }  # node -> (parent, edge)
        queue = [source]

        while queue:
            current = queue.pop(0)
            if current == target:
                break

            for neighbor_id, edge in self._adj[current]:
                if neighbor_id not in visited:
                    visited[neighbor_id] = (current, edge)
                    if len(self._reconstruct_path(visited, source, neighbor_id)) > max_hops + 1:
                        continue
                    queue.append(neighbor_id)
        else:
            return None

        if target not in visited:
            return None

        # Reconstruct path
        path = self._reconstruct_path(visited, source, target)
        edges = []
        rel_types = []
        total_weight = 0.0

        for i in range(len(path) - 1):
            key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            edge_list = self._edge_index.get(key, [])
            if edge_list:
                best = max(edge_list, key=lambda e: e.weight)
                edges.append(best)
                rel_types.append(best.type)
                total_weight += best.weight

        return ConnectionPath(
            source=source,
            target=target,
            hops=path,
            edges=edges,
            total_weight=total_weight,
            relationship_types=rel_types,
        )

    def _reconstruct_path(self, visited: dict, source: str, target: str) -> list[str]:
        """Reconstruct path from BFS visited dict."""
        path = [target]
        current = target
        while current != source:
            parent, _ = visited[current]
            if parent is None:
                break
            path.append(parent)
            current = parent
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Community detection (Louvain)
    # ------------------------------------------------------------------

    def detect_communities(self, resolution: float = 1.0) -> list[Community]:
        """Detect communities using Louvain method.

        Falls back to connected-components if python-louvain not installed.
        """
        if self._communities is not None:
            return self._communities

        node_ids = list(self._node_map.keys())
        if len(node_ids) < 2:
            return []

        try:
            partition = self._louvain_partition(resolution)
        except ImportError:
            logger.info("python-louvain not installed, using connected components")
            partition = self._connected_components_partition()

        # Group nodes by community
        communities_raw: dict[int, list[str]] = defaultdict(list)
        for node_id, comm_id in partition.items():
            communities_raw[comm_id].append(node_id)

        communities = []
        for comm_id, members in sorted(communities_raw.items(), key=lambda x: -len(x[1])):
            # Count internal edges and find dominant relationship
            internal_edge_types: Counter = Counter()
            internal_count = 0
            member_set = set(members)

            for m in members:
                for neighbor_id, edge in self._adj[m]:
                    if neighbor_id in member_set:
                        internal_edge_types[edge.type] += 1
                        internal_count += 1

            internal_count //= 2  # undirected, counted twice
            dominant = (
                internal_edge_types.most_common(1)[0][0] if internal_edge_types else "unknown"
            )

            member_labels = {m: self._label(m) for m in members}

            communities.append(
                Community(
                    id=comm_id,
                    members=members,
                    member_labels=member_labels,
                    size=len(members),
                    internal_edges=internal_count,
                    dominant_relationship=dominant,
                    label=self._auto_label_community(members, dominant),
                )
            )

        self._communities = communities
        return communities

    def _louvain_partition(self, resolution: float) -> dict[str, int]:
        """Run Louvain community detection via networkx + community module."""
        import networkx as nx

        try:
            from community import community_louvain
        except ImportError:
            # Try alternate import path
            import community as community_louvain

        G = nx.Graph()
        for node in self.graph.nodes:
            G.add_node(node.id, label=node.label, type=node.type)
        for edge in self.graph.edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight, type=edge.type)

        return cast(dict[str, int], community_louvain.best_partition(G, resolution=resolution))

    def _connected_components_partition(self) -> dict[str, int]:
        """Fallback: label connected components as communities."""
        visited: set[str] = set()
        partition: dict[str, int] = {}
        comm_id = 0

        for node_id in self._node_map:
            if node_id in visited:
                continue
            # BFS to find connected component
            queue = [node_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                partition[current] = comm_id
                for neighbor_id, _ in self._adj[current]:
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)
            comm_id += 1

        return partition

    def _auto_label_community(self, members: list[str], dominant_rel: str) -> str:
        """Generate a human-readable label for a community."""
        named = [self._label(m) for m in members[:5]]
        suffix = f" (+{len(members) - 5} more)" if len(members) > 5 else ""
        return f"{', '.join(named)}{suffix} [{dominant_rel}]"

    # ------------------------------------------------------------------
    # Centrality analysis
    # ------------------------------------------------------------------

    def compute_centrality(self) -> dict[str, float]:
        """Compute betweenness centrality for all nodes.

        Uses NetworkX if available, otherwise falls back to degree centrality.
        """
        if self._centrality is not None:
            return self._centrality

        try:
            import networkx as nx

            G = nx.Graph()
            for edge in self.graph.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)

            self._centrality = nx.betweenness_centrality(G, weight="weight")
        except ImportError:
            # Fallback: degree centrality (connection count / max possible)
            max_degree = max(self._degree_cache.values()) if self._degree_cache else 1
            self._centrality = {nid: deg / max_degree for nid, deg in self._degree_cache.items()}

        return self._centrality

    def top_entities(self, n: int = 20, by: str = "degree") -> list[EntityProfile]:
        """Rank entities by degree, weighted degree, or centrality."""
        profiles = []
        centrality = self.compute_centrality() if by == "centrality" else {}
        communities = {m: c.id for c in (self._communities or []) for m in c.members}

        for node in self.graph.nodes:
            if node.type != "person":
                continue

            neighbors = self._adj[node.id]
            edge_types: Counter = Counter()
            connections: list[tuple[str, str, float]] = []
            weighted_deg = 0.0

            for neighbor_id, edge in neighbors:
                edge_types[edge.type] += 1
                weighted_deg += edge.weight
                connections.append((neighbor_id, self._label(neighbor_id), edge.weight))

            connections.sort(key=lambda x: -x[2])

            profiles.append(
                EntityProfile(
                    id=node.id,
                    label=self._label(node.id),
                    node_type=node.type,
                    degree=len(neighbors),
                    weighted_degree=weighted_deg,
                    top_connections=connections[:10],
                    community_id=communities.get(node.id),
                    centrality_rank=None,  # filled below
                    edge_type_breakdown=dict(edge_types),
                    document_count=len(self._docs_by_person.get(node.id, [])),
                    flight_count=len(self._flights_by_person.get(node.id, [])),
                    email_count=len(self._emails_by_person.get(node.id, [])),
                )
            )

        # Sort
        if by == "centrality":
            profiles.sort(key=lambda p: centrality.get(p.id, 0), reverse=True)
        elif by == "weighted":
            profiles.sort(key=lambda p: p.weighted_degree, reverse=True)
        else:
            profiles.sort(key=lambda p: p.degree, reverse=True)

        # Assign ranks
        for i, p in enumerate(profiles):
            p.centrality_rank = i + 1

        return profiles[:n]

    # ------------------------------------------------------------------
    # Temporal threading
    # ------------------------------------------------------------------

    def temporal_thread(
        self,
        entity_ids: list[str],
        date_range: tuple[str | None, str | None] = (None, None),
    ) -> TemporalThread:
        """Build a chronological timeline of all mentions/appearances.

        For a single entity: all documents, flights, emails mentioning them.
        For multiple entities: only events where ALL specified entities co-appear.
        """
        entity_ids = [self._resolve_entity(eid) or eid for eid in entity_ids]
        entity_set = set(entity_ids)
        events: list[dict[str, Any]] = []
        sources: Counter = Counter()

        # Documents where all entities co-appear
        if entity_ids:
            # Get doc sets for each entity, intersect
            doc_sets = [set(id(d) for d in self._docs_by_person.get(eid, [])) for eid in entity_ids]
            if doc_sets:
                common_doc_ids = doc_sets[0]
                for ds in doc_sets[1:]:
                    common_doc_ids &= ds

                for doc in self._all_docs:
                    if id(doc) in common_doc_ids and doc.date:
                        if self._in_date_range(doc.date, date_range):
                            events.append(
                                {
                                    "date": doc.date,
                                    "type": "document",
                                    "id": doc.id,
                                    "title": doc.title,
                                    "source": doc.source,
                                    "category": doc.category,
                                    "persons": doc.personIds,
                                }
                            )
                            sources["document"] += 1

        # Flights where all entities co-appear
        if entity_ids:
            for eid in entity_ids:
                for flight in self._flights_by_person.get(eid, []):
                    all_pax = set(flight.passengerIds + flight.pilotIds)
                    if entity_set.issubset(all_pax) and flight.date:
                        if self._in_date_range(flight.date, date_range):
                            events.append(
                                {
                                    "date": flight.date,
                                    "type": "flight",
                                    "id": flight.id,
                                    "origin": flight.origin,
                                    "destination": flight.destination,
                                    "aircraft": flight.aircraft,
                                    "passengers": flight.passengerIds,
                                }
                            )
                            sources["flight"] += 1
                break  # only need to check one entity's flights

        # Emails where all entities co-appear
        if entity_ids:
            for eid in entity_ids:
                for email in self._emails_by_person.get(eid, []):
                    if entity_set.issubset(set(email.personIds)) and email.date:
                        if self._in_date_range(email.date, date_range):
                            events.append(
                                {
                                    "date": email.date,
                                    "type": "email",
                                    "id": email.id,
                                    "subject": email.subject,
                                    "persons": email.personIds,
                                }
                            )
                            sources["email"] += 1
                break

        # Deduplicate by (type, id)
        seen = set()
        unique_events = []
        for ev in events:
            key = (ev["type"], ev["id"])
            if key not in seen:
                seen.add(key)
                unique_events.append(ev)

        # Sort chronologically
        unique_events.sort(key=lambda e: e.get("date", "9999"))

        dates = [e["date"] for e in unique_events if e.get("date")]
        return TemporalThread(
            entity_ids=entity_ids,
            events=unique_events,
            date_range=(dates[0] if dates else None, dates[-1] if dates else None),
            total_mentions=len(unique_events),
            sources_breakdown=dict(sources),
        )

    @staticmethod
    def _in_date_range(date: str, date_range: tuple[str | None, str | None]) -> bool:
        """Check if a date falls within the given range."""
        start, end = date_range
        if start and date < start:
            return False
        if end and date > end:
            return False
        return True

    # ------------------------------------------------------------------
    # Cross-entity discovery
    # ------------------------------------------------------------------

    def shared_documents(
        self,
        person_a: str,
        person_b: str,
    ) -> list[Document]:
        """Find all documents where both persons are mentioned."""
        person_a = self._resolve_entity(person_a) or person_a
        person_b = self._resolve_entity(person_b) or person_b

        docs_a = set(id(d) for d in self._docs_by_person.get(person_a, []))
        docs_b = set(id(d) for d in self._docs_by_person.get(person_b, []))
        common = docs_a & docs_b

        return [d for d in self._all_docs if id(d) in common]

    def shared_flights(
        self,
        person_a: str,
        person_b: str,
    ) -> list[Flight]:
        """Find all flights where both persons were on board."""
        person_a = self._resolve_entity(person_a) or person_a
        person_b = self._resolve_entity(person_b) or person_b

        flights_a = self._flights_by_person.get(person_a, [])
        result = []
        for flight in flights_a:
            all_pax = set(flight.passengerIds + flight.pilotIds)
            if person_b in all_pax:
                result.append(flight)
        return result

    def who_connects(
        self,
        person_a: str,
        person_b: str,
    ) -> list[dict[str, Any]]:
        """Find intermediary people who connect two non-adjacent entities.

        Returns list of {connector_id, connector_label, weight_to_a, weight_to_b}.
        """
        person_a = self._resolve_entity(person_a) or person_a
        person_b = self._resolve_entity(person_b) or person_b

        neighbors_a = {nid: edge for nid, edge in self._adj[person_a]}
        neighbors_b = {nid: edge for nid, edge in self._adj[person_b]}

        # Find common neighbors
        common = set(neighbors_a.keys()) & set(neighbors_b.keys())
        common.discard(person_a)
        common.discard(person_b)

        connectors: list[dict[str, Any]] = []
        for cid in common:
            connectors.append(
                {
                    "id": cid,
                    "label": self._label(cid),
                    "weight_to_a": neighbors_a[cid].weight,
                    "weight_to_b": neighbors_b[cid].weight,
                    "combined_weight": neighbors_a[cid].weight + neighbors_b[cid].weight,
                    "rel_to_a": neighbors_a[cid].type,
                    "rel_to_b": neighbors_b[cid].type,
                }
            )

        connectors.sort(key=lambda c: -float(c["combined_weight"]))
        return connectors

    # ------------------------------------------------------------------
    # Entity resolution (fuzzy matching)
    # ------------------------------------------------------------------

    def _resolve_entity(self, query: str) -> str | None:
        """Resolve a name/partial-ID to a node ID.

        Tries: exact ID match, exact name match, substring match.
        """
        # Exact ID
        if query in self._node_map:
            return query

        # Person registry exact match
        for pid, person in self._persons.items():
            if person.name.lower() == query.lower():
                return pid
            if query.lower() in [a.lower() for a in person.aliases]:
                return pid

        # Node label match
        query_lower = query.lower()
        for node in self.graph.nodes:
            if node.label.lower() == query_lower:
                return node.id

        # Substring match (return best)
        candidates = []
        for node in self.graph.nodes:
            if query_lower in node.label.lower():
                candidates.append(node.id)
        for pid, person in self._persons.items():
            if query_lower in person.name.lower():
                if pid not in candidates:
                    candidates.append(pid)

        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            # Return the one with highest degree
            return max(candidates, key=lambda c: self._degree_cache.get(c, 0))

        return None

    def search_entities(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search for entities by name (fuzzy)."""
        query_lower = query.lower()
        results: list[dict[str, Any]] = []

        for node in self.graph.nodes:
            score = 0
            if query_lower == node.label.lower():
                score = 100
            elif node.label.lower().startswith(query_lower):
                score = 80
            elif query_lower in node.label.lower():
                score = 60

            if score > 0:
                results.append(
                    {
                        "id": node.id,
                        "label": node.label,
                        "type": node.type,
                        "degree": self._degree_cache.get(node.id, 0),
                        "score": score,
                    }
                )

        # Also search person registry
        for pid, person in self._persons.items():
            if pid in {r["id"] for r in results}:
                continue
            score = 0
            if query_lower == person.name.lower():
                score = 100
            elif person.name.lower().startswith(query_lower):
                score = 80
            elif query_lower in person.name.lower():
                score = 60
            for alias in person.aliases:
                if query_lower in alias.lower():
                    score = max(score, 50)

            if score > 0:
                results.append(
                    {
                        "id": pid,
                        "label": person.name,
                        "type": "person",
                        "degree": self._degree_cache.get(pid, 0),
                        "score": score,
                    }
                )

        results.sort(key=lambda r: (-int(r["score"]), -int(r["degree"])))
        return results[:limit]

    # ------------------------------------------------------------------
    # Export / serialization
    # ------------------------------------------------------------------

    def export_investigation(self, path: Path, results: list[InvestigationResult]) -> None:
        """Export investigation results to JSON."""
        data = {
            "investigation": {
                "generated": datetime.now().isoformat(),
                "graph_stats": {
                    "nodes": self.graph.node_count,
                    "edges": self.graph.edge_count,
                },
                "queries": [
                    {
                        "type": r.query_type,
                        "params": r.query_params,
                        "summary": r.summary,
                        "timestamp": r.timestamp,
                    }
                    for r in results
                ],
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def export_community_report(self, path: Path) -> None:
        """Export community detection results."""
        communities = self.detect_communities()
        data = {
            "communities": [
                {
                    "id": c.id,
                    "size": c.size,
                    "label": c.label,
                    "dominant_relationship": c.dominant_relationship,
                    "internal_edges": c.internal_edges,
                    "members": [{"id": m, "label": c.member_labels.get(m, m)} for m in c.members],
                }
                for c in communities
            ],
            "total_communities": len(communities),
            "generated": datetime.now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def export_entity_rankings(self, path: Path, n: int = 50) -> None:
        """Export top entities by multiple ranking criteria."""
        data = {
            "by_degree": [
                {
                    "rank": p.centrality_rank,
                    "id": p.id,
                    "label": p.label,
                    "degree": p.degree,
                    "docs": p.document_count,
                    "flights": p.flight_count,
                }
                for p in self.top_entities(n, by="degree")
            ],
            "by_centrality": [
                {
                    "rank": p.centrality_rank,
                    "id": p.id,
                    "label": p.label,
                    "degree": p.degree,
                    "docs": p.document_count,
                    "flights": p.flight_count,
                }
                for p in self.top_entities(n, by="centrality")
            ],
            "generated": datetime.now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


class InvestigationREPL:
    """Interactive investigation shell.

    Commands:
        search <name>             — Find entities by name
        profile <entity>          — Full entity profile
        traverse <entity> [hops]  — N-hop neighbor discovery
        path <entity1> <entity2>  — Shortest path between entities
        shared <entity1> <entity2> — Shared documents/flights
        connects <entity1> <entity2> — Intermediary connectors
        timeline <entity> [entity2] — Temporal thread
        communities               — Detect and list social circles
        top [n] [by]              — Top entities (degree/centrality/weighted)
        export <type> <path>      — Export results
        help                      — Show commands
        quit                      — Exit
    """

    def __init__(self, engine: InvestigationEngine) -> None:
        self.engine = engine
        self.history: list[InvestigationResult] = []

    def run(self) -> None:
        """Start the interactive REPL."""
        try:
            from rich.console import Console

            self._console = Console()
            self._rich = True
        except ImportError:
            self._rich = False

        self._print_banner()

        while True:
            try:
                line = input("\n[investigate] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting investigation.")
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()
            args = parts[1:]

            try:
                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "help":
                    self._print_help()
                elif cmd == "search":
                    self._cmd_search(args)
                elif cmd == "profile":
                    self._cmd_profile(args)
                elif cmd == "traverse":
                    self._cmd_traverse(args)
                elif cmd == "path":
                    self._cmd_path(args)
                elif cmd == "shared":
                    self._cmd_shared(args)
                elif cmd == "connects":
                    self._cmd_connects(args)
                elif cmd == "timeline":
                    self._cmd_timeline(args)
                elif cmd == "communities":
                    self._cmd_communities(args)
                elif cmd == "top":
                    self._cmd_top(args)
                elif cmd == "export":
                    self._cmd_export(args)
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for commands.")
            except Exception as exc:
                print(f"Error: {exc}")
                logger.exception("REPL command failed")

    def _print_banner(self) -> None:
        if self._rich:
            self._console.print(
                "\n[bold cyan]═══ Epstein Investigation Engine ═══[/bold cyan]\n"
                f"[dim]Graph: {self.engine.graph.node_count} entities, "
                f"{self.engine.graph.edge_count} connections[/dim]\n"
                "[dim]Type 'help' for commands[/dim]"
            )
        else:
            print("\n=== Epstein Investigation Engine ===")
            print(
                f"Graph: {self.engine.graph.node_count} entities, "
                f"{self.engine.graph.edge_count} connections"
            )
            print("Type 'help' for commands")

    def _print_help(self) -> None:
        cmds = [
            ("search <name>", "Find entities by name"),
            ("profile <entity>", "Full entity profile"),
            ("traverse <entity> [hops]", "N-hop neighbor discovery (default: 2)"),
            ("path <A> to <B>", "Shortest path between entities"),
            ("shared <A> and <B>", "Documents/flights in common"),
            ("connects <A> and <B>", "Who bridges two entities"),
            ("timeline <entity> [entity2]", "Chronological thread"),
            ("communities", "Detect social circles (Louvain)"),
            ("top [n] [degree|centrality]", "Top entities by metric"),
            ("export communities <path>", "Export community report"),
            ("export rankings <path>", "Export entity rankings"),
            ("quit", "Exit"),
        ]
        if self._rich:
            from rich.table import Table

            table = Table(title="Investigation Commands", show_header=True)
            table.add_column("Command", style="cyan")
            table.add_column("Description")
            for cmd, desc in cmds:
                table.add_row(cmd, desc)
            self._console.print(table)
        else:
            for cmd, desc in cmds:
                print(f"  {cmd:<35} {desc}")

    def _cmd_search(self, args: list[str]) -> None:
        query = " ".join(args)
        if not query:
            print("Usage: search <name>")
            return
        results = self.engine.search_entities(query)
        if not results:
            print(f"No entities found matching '{query}'")
            return
        if self._rich:
            from rich.table import Table

            table = Table(title=f"Search: {query}")
            table.add_column("ID", style="dim")
            table.add_column("Name", style="bold")
            table.add_column("Type")
            table.add_column("Connections", justify="right")
            for r in results:
                table.add_row(r["id"], r["label"], r["type"], str(r["degree"]))
            self._console.print(table)
        else:
            for r in results:
                print(f"  {r['id']:<10} {r['label']:<30} [{r['type']}] deg={r['degree']}")

    def _cmd_profile(self, args: list[str]) -> None:
        entity = " ".join(args)
        if not entity:
            print("Usage: profile <entity>")
            return

        resolved = self.engine._resolve_entity(entity)
        if not resolved:
            print(f"Entity not found: {entity}")
            return

        # Build profile manually
        node = self.engine._node_map.get(resolved)
        neighbors = self.engine._adj[resolved]
        edge_types: Counter = Counter()
        connections = []
        for nid, edge in neighbors:
            edge_types[edge.type] += 1
            connections.append((nid, self.engine._label(nid), edge.weight, edge.type))
        connections.sort(key=lambda x: -x[2])

        docs = self.engine._docs_by_person.get(resolved, [])
        flights = self.engine._flights_by_person.get(resolved, [])
        emails = self.engine._emails_by_person.get(resolved, [])

        if self._rich:
            from rich.panel import Panel
            from rich.table import Table

            lines = [
                f"[bold]{self.engine._label(resolved)}[/bold] ({resolved})",
                f"Type: {node.type if node else 'unknown'}",
                f"Connections: {len(neighbors)}",
                f"Documents: {len(docs)} | Flights: {len(flights)} | Emails: {len(emails)}",
                "",
                "[bold]Edge types:[/bold]",
            ]
            for etype, count in edge_types.most_common():
                lines.append(f"  {etype}: {count}")

            self._console.print(Panel("\n".join(lines), title="Entity Profile"))

            if connections:
                table = Table(title="Top Connections")
                table.add_column("Entity")
                table.add_column("Relationship")
                table.add_column("Weight", justify="right")
                for nid, label, weight, etype in connections[:15]:
                    table.add_row(label, etype, f"{weight:.1f}")
                self._console.print(table)
        else:
            print(f"\n  {self.engine._label(resolved)} ({resolved})")
            print(f"  Connections: {len(neighbors)}")
            print(f"  Docs: {len(docs)} | Flights: {len(flights)} | Emails: {len(emails)}")
            for nid, label, weight, etype in connections[:10]:
                print(f"    -> {label} [{etype}] w={weight:.1f}")

    def _cmd_traverse(self, args: list[str]) -> None:
        if not args:
            print("Usage: traverse <entity> [max_hops]")
            return
        hops = 2
        entity_parts = args
        if args[-1].isdigit():
            hops = int(args[-1])
            entity_parts = args[:-1]
        entity = " ".join(entity_parts)

        result = self.engine.traverse(entity, max_hops=hops)
        if not result:
            print(f"Entity not found: {entity}")
            return

        # Group by distance
        by_distance: dict[int, list] = defaultdict(list)
        for nid, info in result.items():
            by_distance[info["distance"]].append((nid, info))

        if self._rich:
            from rich.tree import Tree

            tree = Tree(f"[bold]{self.engine._label(entity)}[/bold] ({len(result)} reachable)")
            for dist in sorted(by_distance.keys()):
                if dist == 0:
                    continue
                branch = tree.add(f"[cyan]Hop {dist}[/cyan] ({len(by_distance[dist])} entities)")
                for nid, info in sorted(by_distance[dist], key=lambda x: -x[1]["weight"])[:20]:
                    branch.add(f"{info['label']} [dim]via {' → '.join(info['edges'])}[/dim]")
            self._console.print(tree)
        else:
            print(f"\n  {entity} — {len(result)} reachable within {hops} hops")
            for dist in sorted(by_distance.keys()):
                if dist == 0:
                    continue
                print(f"\n  Hop {dist} ({len(by_distance[dist])} entities):")
                for nid, info in sorted(by_distance[dist], key=lambda x: -x[1]["weight"])[:15]:
                    print(f"    {info['label']} via {' → '.join(info['edges'])}")

    def _cmd_path(self, args: list[str]) -> None:
        # Parse "A to B" or "A B"
        text = " ".join(args)
        parts = re.split(r"\s+to\s+|\s+and\s+", text, maxsplit=1)
        if len(parts) < 2:
            # Try splitting on last space-separated token
            if len(args) >= 2:
                parts = [" ".join(args[:-1]), args[-1]]
            else:
                print("Usage: path <entity1> to <entity2>")
                return

        result = self.engine.find_path(parts[0].strip(), parts[1].strip())
        if not result:
            print(f"No path found between '{parts[0].strip()}' and '{parts[1].strip()}'")
            return

        labels = [self.engine._label(h) for h in result.hops]
        if self._rich:
            chain = " → ".join(
                f"[bold]{label}[/bold]" if i == 0 or i == len(labels) - 1 else label
                for i, label in enumerate(labels)
            )
            self._console.print(
                f"\n  Path ({len(result.hops) - 1} hops, weight {result.total_weight:.1f}):"
            )
            self._console.print(f"  {chain}")
            for i, edge in enumerate(result.edges):
                self._console.print(f"    [dim]{labels[i]} —[{edge.type}]→ {labels[i + 1]}[/dim]")
        else:
            print(f"\n  Path ({len(result.hops) - 1} hops):")
            print(f"  {' → '.join(labels)}")

    def _cmd_shared(self, args: list[str]) -> None:
        text = " ".join(args)
        parts = re.split(r"\s+and\s+|\s+with\s+", text, maxsplit=1)
        if len(parts) < 2:
            print("Usage: shared <entity1> and <entity2>")
            return

        a, b = parts[0].strip(), parts[1].strip()
        docs = self.engine.shared_documents(a, b)
        flights = self.engine.shared_flights(a, b)

        print(f"\n  {a} & {b}:")
        print(f"  {len(docs)} shared documents, {len(flights)} shared flights")

        if docs:
            print("\n  Documents:")
            for doc in sorted(docs, key=lambda d: d.date or "")[:20]:
                print(f"    [{doc.date or '?'}] {doc.title[:60]} ({doc.source})")

        if flights:
            print("\n  Flights:")
            for f in sorted(flights, key=lambda f: f.date or ""):
                print(f"    [{f.date or '?'}] {f.origin} → {f.destination} ({f.aircraft or '?'})")

    def _cmd_connects(self, args: list[str]) -> None:
        text = " ".join(args)
        parts = re.split(r"\s+and\s+|\s+to\s+", text, maxsplit=1)
        if len(parts) < 2:
            print("Usage: connects <entity1> and <entity2>")
            return

        connectors = self.engine.who_connects(parts[0].strip(), parts[1].strip())
        if not connectors:
            print("No intermediary connectors found.")
            return

        print(f"\n  Connectors between {parts[0].strip()} and {parts[1].strip()}:")
        for c in connectors[:15]:
            print(
                f"    {c['label']:<30} "
                f"[{c['rel_to_a']}] w={c['weight_to_a']:.1f} | "
                f"[{c['rel_to_b']}] w={c['weight_to_b']:.1f}"
            )

    def _cmd_timeline(self, args: list[str]) -> None:
        text = " ".join(args)
        entities = [e.strip() for e in re.split(r"\s+and\s+|,", text) if e.strip()]
        if not entities:
            print("Usage: timeline <entity> [and <entity2>]")
            return

        thread = self.engine.temporal_thread(entities)
        print(f"\n  Timeline: {', '.join(self.engine._label(e) for e in thread.entity_ids)}")
        print(
            f"  {thread.total_mentions} events, "
            f"range: {thread.date_range[0] or '?'} to {thread.date_range[1] or '?'}"
        )
        print(f"  Sources: {thread.sources_breakdown}")

        for ev in thread.events[:30]:
            if ev["type"] == "document":
                print(f"    [{ev['date']}] DOC: {ev['title'][:55]} ({ev['source']})")
            elif ev["type"] == "flight":
                print(
                    f"    [{ev['date']}] FLT: {ev.get('origin', '?')} → {ev.get('destination', '?')}"
                )
            elif ev["type"] == "email":
                print(f"    [{ev['date']}] EMAIL: {ev['subject'][:55]}")

    def _cmd_communities(self, args: list[str]) -> None:
        communities = self.engine.detect_communities()
        print(f"\n  {len(communities)} communities detected:")

        for c in communities:
            if c.size < 2:
                continue
            print(f"\n  Community {c.id} ({c.size} members, {c.internal_edges} internal edges)")
            print(f"  Dominant relationship: {c.dominant_relationship}")
            print(f"  Members: {c.label}")

    def _cmd_top(self, args: list[str]) -> None:
        n = 20
        by = "degree"
        for a in args:
            if a.isdigit():
                n = int(a)
            elif a in ("degree", "centrality", "weighted"):
                by = a

        profiles = self.engine.top_entities(n, by=by)
        if self._rich:
            from rich.table import Table

            table = Table(title=f"Top {n} Entities (by {by})")
            table.add_column("#", justify="right", style="dim")
            table.add_column("Name", style="bold")
            table.add_column("Connections", justify="right")
            table.add_column("Docs", justify="right")
            table.add_column("Flights", justify="right")
            for p in profiles:
                table.add_row(
                    str(p.centrality_rank),
                    p.label,
                    str(p.degree),
                    str(p.document_count),
                    str(p.flight_count),
                )
            self._console.print(table)
        else:
            print(f"\n  Top {n} entities by {by}:")
            for p in profiles:
                print(
                    f"    {p.centrality_rank:>3}. {p.label:<30} "
                    f"deg={p.degree} docs={p.document_count} flights={p.flight_count}"
                )

    def _cmd_export(self, args: list[str]) -> None:
        if len(args) < 2:
            print("Usage: export <communities|rankings> <output_path>")
            return
        export_type = args[0]
        path = Path(args[1])

        if export_type == "communities":
            self.engine.export_community_report(path)
            print(f"  Exported community report to {path}")
        elif export_type == "rankings":
            self.engine.export_entity_rankings(path)
            print(f"  Exported entity rankings to {path}")
        else:
            print(f"Unknown export type: {export_type}")
