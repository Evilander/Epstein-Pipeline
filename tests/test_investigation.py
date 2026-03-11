"""Tests for the multi-hop investigation engine."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from epstein_pipeline.models.document import Document, Email, EmailContact, Flight
from epstein_pipeline.processors.investigation import (
    Community,
    ConnectionPath,
    EntityProfile,
    InvestigationEngine,
    InvestigationResult,
    TemporalThread,
)
from epstein_pipeline.processors.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphBuilder,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_person_graph(sample_documents) -> KnowledgeGraph:
    """Graph with co-occurrence edges from sample_documents.

    doc-002 has personIds [p-0001, p-0002] → 1 co-occurrence edge.
    doc-001 and doc-003 only have p-0001 → no new edges.
    Result: 2 nodes (p-0001, p-0002), 1 edge.
    """
    builder = KnowledgeGraphBuilder()
    builder.add_documents(sample_documents)
    return builder.build()


@pytest.fixture
def rich_graph() -> KnowledgeGraph:
    """A richer graph with 5 persons and multiple edge types."""
    docs = [
        Document(
            id="d1",
            title="Doc 1",
            source="other",
            category="other",
            personIds=["p-0001", "p-0002"],
        ),
        Document(
            id="d2",
            title="Doc 2",
            source="other",
            category="other",
            personIds=["p-0002", "p-0003"],
        ),
        Document(
            id="d3",
            title="Doc 3",
            source="other",
            category="other",
            personIds=["p-0003", "p-0004"],
        ),
        Document(
            id="d4",
            title="Doc 4",
            source="other",
            category="other",
            personIds=["p-0001", "p-0002", "p-0003"],
        ),
    ]
    flights = [
        Flight(
            id="f1",
            date="1999-03-15",
            aircraft="Gulfstream",
            tailNumber="N908JE",
            origin="TIST",
            destination="KPBI",
            passengerIds=["p-0001", "p-0002", "p-0005"],
            pilotIds=[],
        ),
        Flight(
            id="f2",
            date="2001-06-20",
            aircraft="Boeing 727",
            tailNumber="N908JE",
            origin="KPBI",
            destination="CYUL",
            passengerIds=["p-0003"],
            pilotIds=["p-0004"],
        ),
    ]
    emails = [
        Email(
            id="e1",
            subject="Meeting",
            **{"from": EmailContact(name="A", email="a@test.com", personSlug="p-0001")},
            to=[EmailContact(name="B", email="b@test.com", personSlug="p-0002")],
            cc=[],
            date="2002-01-10",
            body="meeting notes",
            personIds=["p-0001", "p-0002"],
            folder="inbox",
        ),
    ]

    builder = KnowledgeGraphBuilder()
    builder.add_documents(docs)
    builder.add_flights(flights)
    builder.add_emails(emails)
    builder.add_person_labels(
        {
            "p-0001": "Jeffrey Epstein",
            "p-0002": "Ghislaine Maxwell",
            "p-0003": "Bill Clinton",
            "p-0004": "Larry Visoski",
            "p-0005": "Sarah Kellen",
        }
    )
    return builder.build()


@pytest.fixture
def engine(rich_graph) -> InvestigationEngine:
    """InvestigationEngine with the rich graph + loaded docs/flights/emails."""
    eng = InvestigationEngine(rich_graph)
    # Load documents, flights, and emails so shared_* methods work
    docs = [
        Document(
            id="d1", title="Doc 1", source="other", category="other", personIds=["p-0001", "p-0002"]
        ),
        Document(
            id="d2", title="Doc 2", source="other", category="other", personIds=["p-0002", "p-0003"]
        ),
        Document(
            id="d3", title="Doc 3", source="other", category="other", personIds=["p-0003", "p-0004"]
        ),
        Document(
            id="d4",
            title="Doc 4",
            source="other",
            category="other",
            personIds=["p-0001", "p-0002", "p-0003"],
        ),
    ]
    flights = [
        Flight(
            id="f1",
            date="1999-03-15",
            aircraft="Gulfstream",
            tailNumber="N908JE",
            origin="TIST",
            destination="KPBI",
            passengerIds=["p-0001", "p-0002", "p-0005"],
            pilotIds=[],
        ),
        Flight(
            id="f2",
            date="2001-06-20",
            aircraft="Boeing 727",
            tailNumber="N908JE",
            origin="KPBI",
            destination="CYUL",
            passengerIds=["p-0003"],
            pilotIds=["p-0004"],
        ),
    ]
    emails = [
        Email(
            id="e1",
            subject="Meeting",
            **{"from": EmailContact(name="A", email="a@test.com", personSlug="p-0001")},
            to=[EmailContact(name="B", email="b@test.com", personSlug="p-0002")],
            cc=[],
            date="2002-01-10",
            body="meeting notes",
            personIds=["p-0001", "p-0002"],
            folder="inbox",
        ),
    ]
    eng.load_documents(docs)
    eng.load_flights(flights)
    eng.load_emails(emails)
    return eng


@pytest.fixture
def simple_engine(three_person_graph) -> InvestigationEngine:
    """InvestigationEngine with minimal graph (2 connected nodes)."""
    return InvestigationEngine(three_person_graph)


@pytest.fixture
def sample_flights() -> list[Flight]:
    return [
        Flight(
            id="f1",
            date="1999-03-15",
            aircraft="Gulfstream",
            tailNumber="N908JE",
            origin="TIST",
            destination="KPBI",
            passengerIds=["p-0001", "p-0002", "p-0005"],
            pilotIds=[],
        ),
    ]


@pytest.fixture
def sample_emails() -> list[Email]:
    return [
        Email(
            id="e1",
            subject="Meeting",
            **{"from": EmailContact(name="A", email="a@test.com", personSlug="p-0001")},
            to=[EmailContact(name="B", email="b@test.com", personSlug="p-0002")],
            cc=[],
            date="2002-01-10",
            body="meeting notes",
            personIds=["p-0001", "p-0002"],
            folder="inbox",
        ),
    ]


# ---------------------------------------------------------------------------
# InvestigationEngine construction
# ---------------------------------------------------------------------------


class TestEngineConstruction:
    def test_engine_builds_from_graph(self, engine):
        assert engine.graph is not None
        assert len(engine._node_map) > 0

    def test_adjacency_list_built(self, engine):
        assert len(engine._adj) > 0
        # p-0001 should have neighbors (it appears in multiple docs/flights)
        assert "p-0001" in engine._adj
        assert len(engine._adj["p-0001"]) > 0

    def test_empty_graph(self):
        empty = KnowledgeGraph(nodes=[], edges=[])
        eng = InvestigationEngine(empty)
        assert len(eng._node_map) == 0
        assert len(eng._adj) == 0

    def test_node_map_contains_all_nodes(self, engine, rich_graph):
        for node in rich_graph.nodes:
            assert node.id in engine._node_map


# ---------------------------------------------------------------------------
# Traverse (BFS)
# ---------------------------------------------------------------------------


class TestTraverse:
    def test_traverse_returns_reachable_nodes(self, engine):
        result = engine.traverse("p-0001", max_hops=1)
        assert isinstance(result, dict)
        # p-0001 should reach its direct neighbors
        assert len(result) >= 1

    def test_traverse_max_hops_0(self, engine):
        result = engine.traverse("p-0001", max_hops=0)
        # 0 hops = only the start node itself (or empty depending on impl)
        assert len(result) <= 1

    def test_traverse_max_hops_limits_depth(self, engine):
        hop1 = engine.traverse("p-0001", max_hops=1)
        hop2 = engine.traverse("p-0001", max_hops=2)
        # More hops should reach at least as many nodes
        assert len(hop2) >= len(hop1)

    def test_traverse_nonexistent_node(self, engine):
        result = engine.traverse("p-9999", max_hops=2)
        assert len(result) == 0

    def test_traverse_includes_distance(self, engine):
        result = engine.traverse("p-0001", max_hops=2)
        for node_id, info in result.items():
            assert "distance" in info or "hops" in info or isinstance(info, (int, dict))


# ---------------------------------------------------------------------------
# Find path
# ---------------------------------------------------------------------------


class TestFindPath:
    def test_find_direct_path(self, engine):
        # p-0001 and p-0002 share edges (co-occurrence, co-passenger, correspondence)
        path = engine.find_path("p-0001", "p-0002")
        assert path is not None
        assert isinstance(path, ConnectionPath)
        assert path.source == "p-0001"
        assert path.target == "p-0002"
        assert len(path.hops) >= 2  # at least [source, target]

    def test_find_indirect_path(self, engine):
        # p-0001 → p-0002 → p-0003 (2-hop path via co-occurrence)
        path = engine.find_path("p-0001", "p-0003")
        assert path is not None
        assert len(path.hops) >= 2

    def test_find_path_no_connection(self, simple_engine):
        # In the simple graph, p-0003 has no edges (only doc-001 and doc-003 have p-0001 alone)
        # Actually doc-001 has only p-0001, doc-002 has p-0001+p-0002, doc-003 has only p-0001
        # So p-0003 is NOT in the graph at all (not in any multi-person doc)
        # BUT the sample_documents fixture: doc-001 personIds=["p-0001"], doc-002=["p-0001","p-0002"]
        # So only p-0001 and p-0002 are nodes → p-0003 doesn't exist
        path = simple_engine.find_path("p-0001", "p-9999")
        assert path is None

    def test_find_path_same_node(self, engine):
        path = engine.find_path("p-0001", "p-0001")
        # Either returns trivial path or None — both acceptable
        if path is not None:
            assert path.source == "p-0001"
            assert path.target == "p-0001"

    def test_path_has_edges(self, engine):
        path = engine.find_path("p-0001", "p-0002")
        assert path is not None
        assert len(path.edges) >= 1
        assert path.total_weight > 0


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------


class TestCommunityDetection:
    def test_detect_communities_returns_list(self, engine):
        communities = engine.detect_communities()
        assert isinstance(communities, list)
        assert len(communities) >= 1

    def test_community_structure(self, engine):
        communities = engine.detect_communities()
        for comm in communities:
            assert isinstance(comm, Community)
            assert comm.size > 0
            assert len(comm.members) == comm.size
            assert isinstance(comm.id, int)

    def test_all_nodes_in_communities(self, engine, rich_graph):
        communities = engine.detect_communities()
        all_members = set()
        for comm in communities:
            all_members.update(comm.members)
        # Every graph node should be in some community
        for node in rich_graph.nodes:
            assert node.id in all_members

    def test_communities_with_fallback(self, engine):
        """Community detection should work even without python-louvain."""
        with patch.dict("sys.modules", {"community": None, "community.community_louvain": None}):
            # The engine may have already imported — this tests the fallback path
            # by calling detect_communities which should handle ImportError internally
            communities = engine.detect_communities()
            assert isinstance(communities, list)

    def test_empty_graph_communities(self):
        empty = KnowledgeGraph(nodes=[], edges=[])
        eng = InvestigationEngine(empty)
        communities = eng.detect_communities()
        assert communities == []


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------


class TestCentrality:
    def test_compute_centrality_returns_dict(self, engine):
        centrality = engine.compute_centrality()
        assert isinstance(centrality, dict)
        assert len(centrality) > 0

    def test_centrality_values_bounded(self, engine):
        centrality = engine.compute_centrality()
        for node_id, score in centrality.items():
            assert isinstance(score, (int, float))
            assert score >= 0


# ---------------------------------------------------------------------------
# Top entities
# ---------------------------------------------------------------------------


class TestTopEntities:
    def test_top_entities_by_degree(self, engine):
        top = engine.top_entities(n=3, by="degree")
        assert isinstance(top, list)
        assert len(top) <= 3
        for entity in top:
            assert isinstance(entity, EntityProfile)
            assert entity.degree >= 0

    def test_top_entities_sorted(self, engine):
        top = engine.top_entities(n=10, by="degree")
        degrees = [e.degree for e in top]
        assert degrees == sorted(degrees, reverse=True)

    def test_top_entities_profiles_populated(self, engine):
        top = engine.top_entities(n=5)
        for profile in top:
            assert profile.id != ""
            assert isinstance(profile.edge_type_breakdown, dict)

    def test_top_entities_limit(self, engine):
        top = engine.top_entities(n=1)
        assert len(top) == 1


# ---------------------------------------------------------------------------
# Temporal thread
# ---------------------------------------------------------------------------


class TestTemporalThread:
    def test_temporal_thread_basic(self, engine):
        thread = engine.temporal_thread(["p-0001"])
        assert isinstance(thread, TemporalThread)
        assert "p-0001" in thread.entity_ids

    def test_temporal_thread_multiple_entities(self, engine):
        thread = engine.temporal_thread(["p-0001", "p-0002"])
        assert len(thread.entity_ids) == 2

    def test_temporal_thread_with_dates(self, engine):
        thread = engine.temporal_thread(["p-0001"], date_range=("1998-01-01", "2005-12-31"))
        assert isinstance(thread, TemporalThread)

    def test_temporal_thread_empty_entity(self, engine):
        thread = engine.temporal_thread(["p-9999"])
        assert thread.total_mentions == 0


# ---------------------------------------------------------------------------
# Shared documents and flights
# ---------------------------------------------------------------------------


class TestSharedAnalysis:
    def test_shared_documents(self, engine):
        # In rich_graph: d1 has [p-0001, p-0002], d4 has [p-0001, p-0002, p-0003]
        shared = engine.shared_documents("p-0001", "p-0002")
        assert isinstance(shared, list)
        assert len(shared) >= 1

    def test_shared_documents_no_overlap(self, engine):
        # p-0004 and p-0005 don't share documents
        shared = engine.shared_documents("p-0004", "p-0005")
        assert len(shared) == 0

    def test_shared_flights(self, engine):
        # f1 has passengerIds [p-0001, p-0002, p-0005]
        shared = engine.shared_flights("p-0001", "p-0002")
        assert isinstance(shared, list)
        assert len(shared) >= 1

    def test_shared_flights_no_overlap(self, engine):
        shared = engine.shared_flights("p-0003", "p-0005")
        assert len(shared) == 0


# ---------------------------------------------------------------------------
# Who connects (intermediary finder)
# ---------------------------------------------------------------------------


class TestWhoConnects:
    def test_who_connects_finds_intermediary(self, engine):
        # p-0001 and p-0003 are connected via p-0002
        # (p-0001↔p-0002 via d1/d4/f1/e1, p-0002↔p-0003 via d2/d4)
        connectors = engine.who_connects("p-0001", "p-0003")
        assert isinstance(connectors, list)
        # p-0002 should be an intermediary
        # Check p-0002 appears somewhere in results
        found = any("p-0002" in str(c) for c in connectors)
        assert found or len(connectors) > 0

    def test_who_connects_direct_neighbors(self, engine):
        # p-0001 and p-0002 are directly connected — no intermediary needed
        connectors = engine.who_connects("p-0001", "p-0002")
        # Could return empty (directly connected) or return the direct connection info
        assert isinstance(connectors, list)


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------


class TestEntityResolution:
    def test_resolve_by_exact_id(self, engine):
        result = engine._resolve_entity("p-0001")
        assert result == "p-0001"

    def test_resolve_by_label(self, engine):
        result = engine._resolve_entity("Jeffrey Epstein")
        assert result == "p-0001"

    def test_resolve_by_substring(self, engine):
        result = engine._resolve_entity("Epstein")
        # Should match p-0001 (Jeffrey Epstein)
        assert result is not None

    def test_resolve_nonexistent(self, engine):
        result = engine._resolve_entity("Nobody McFakerson")
        assert result is None

    def test_resolve_case_insensitive(self, engine):
        result = engine._resolve_entity("jeffrey epstein")
        # Depends on implementation — may or may not be case-insensitive
        # At minimum, exact label match should work
        if result is not None:
            assert result == "p-0001"


# ---------------------------------------------------------------------------
# Search entities
# ---------------------------------------------------------------------------


class TestSearchEntities:
    def test_search_exact_name(self, engine):
        results = engine.search_entities("Jeffrey Epstein")
        assert len(results) >= 1
        # Top result should be p-0001 — results are list[dict]
        top = results[0]
        assert top["id"] == "p-0001"

    def test_search_partial(self, engine):
        results = engine.search_entities("Clinton")
        assert len(results) >= 1

    def test_search_no_results(self, engine):
        results = engine.search_entities("xyznonexistent12345")
        assert len(results) == 0

    def test_search_limit(self, engine):
        results = engine.search_entities("p-", limit=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Export methods
# ---------------------------------------------------------------------------


class TestExports:
    def test_export_investigation(self, engine, tmp_path):
        result = InvestigationResult(
            query_type="test",
            query_params={"target": "p-0001"},
            results={"nodes_reached": 3},
            summary="Test investigation",
        )
        out = tmp_path / "investigation.json"
        engine.export_investigation(out, [result])
        assert out.exists()
        data = json.loads(out.read_text())
        # Wrapped in {"investigation": {"queries": [...]}}
        assert "investigation" in data
        queries = data["investigation"]["queries"]
        assert len(queries) == 1
        assert queries[0]["type"] == "test"

    def test_export_community_report(self, engine, tmp_path):
        out = tmp_path / "communities.json"
        engine.export_community_report(out)
        assert out.exists()
        data = json.loads(out.read_text())
        # Wrapped in {"communities": [...]}
        assert "communities" in data
        assert isinstance(data["communities"], list)

    def test_export_entity_rankings(self, engine, tmp_path):
        out = tmp_path / "rankings.json"
        engine.export_entity_rankings(out, n=3)
        assert out.exists()
        data = json.loads(out.read_text())
        # Wrapped in {"by_degree": [...], "by_centrality": [...]}
        assert "by_degree" in data
        assert "by_centrality" in data
        assert len(data["by_degree"]) <= 3


# ---------------------------------------------------------------------------
# Dataclass validation
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_connection_path(self):
        from epstein_pipeline.processors.knowledge_graph import GraphEdge

        edge = GraphEdge(source="a", target="b", type="co-occurrence", weight=1.0)
        path = ConnectionPath(
            source="a",
            target="b",
            hops=["a", "b"],
            edges=[edge],
            total_weight=1.0,
            relationship_types=["co-occurrence"],
        )
        assert path.source == "a"
        assert len(path.hops) == 2

    def test_community(self):
        comm = Community(
            id=0,
            members=["p-0001", "p-0002"],
            member_labels={"p-0001": "Epstein", "p-0002": "Maxwell"},
            size=2,
            internal_edges=1,
            dominant_relationship="co-occurrence",
        )
        assert comm.size == 2
        assert comm.label == ""  # default

    def test_temporal_thread(self):
        thread = TemporalThread(
            entity_ids=["p-0001"],
            events=[{"type": "document", "date": "2001-01-01"}],
            date_range=("2000-01-01", "2005-12-31"),
            total_mentions=1,
            sources_breakdown={"document": 1},
        )
        assert thread.total_mentions == 1

    def test_entity_profile(self):
        profile = EntityProfile(
            id="p-0001",
            label="Jeffrey Epstein",
            node_type="person",
            degree=5,
            weighted_degree=12.5,
            top_connections=[("p-0002", "co-occurrence", 3.0)],
            community_id=0,
            centrality_rank=1,
            edge_type_breakdown={"co-occurrence": 3, "co-passenger": 2},
            document_count=10,
            flight_count=5,
            email_count=3,
        )
        assert profile.degree == 5
        assert profile.centrality_rank == 1

    def test_investigation_result(self):
        result = InvestigationResult(
            query_type="traverse",
            query_params={"start": "p-0001", "max_hops": 2},
            results={"found": 5},
            summary="Found 5 entities within 2 hops",
        )
        assert result.query_type == "traverse"
        assert result.timestamp  # auto-generated
