"""
Unit tests for network analysis functionality.

These tests validate graph construction and basic analysis outputs using small,
deterministic fixtures.
"""

import pytest

from cib.core import CIBMatrix, Scenario
from cib.network_analysis import ImpactPathwayAnalyzer, NetworkAnalyzer, NetworkGraphBuilder


def _toy_chain_matrix() -> CIBMatrix:
    descriptors = {
        "A": ["Low", "High"],
        "B": ["Low", "High"],
        "C": ["Low", "High"],
    }
    matrix = CIBMatrix(descriptors)

    for a_state in descriptors["A"]:
        for b_state in descriptors["B"]:
            matrix.set_impact("A", a_state, "B", b_state, 2.0)

    for b_state in descriptors["B"]:
        for c_state in descriptors["C"]:
            matrix.set_impact("B", b_state, "C", c_state, 1.0)

    return matrix


class TestNetworkGraphBuilder:
    """Test suite for NetworkGraphBuilder class."""

    def test_build_impact_network_has_expected_edges(self) -> None:
        """Build an aggregated impact network with expected edges."""
        matrix = _toy_chain_matrix()
        builder = NetworkGraphBuilder(matrix)
        graph = builder.build_impact_network()

        assert set(graph.nodes()) == {"A", "B", "C"}
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")
        assert graph.number_of_edges() == 2

        attrs = graph.edges["A", "B"]
        assert attrs["impact_direction"] == "promoting"
        assert float(attrs["weight"]) > 0.0
        assert float(attrs["abs_weight"]) > 0.0

    def test_build_state_specific_network_uses_scenario_states(self) -> None:
        """Build a state-specific network using scenario states for both endpoints."""
        descriptors = {
            "A": ["Low", "High"],
            "B": ["Low", "High"],
        }
        matrix = CIBMatrix(descriptors)
        matrix.set_impact("A", "Low", "B", "Low", 3.0)
        matrix.set_impact("A", "High", "B", "Low", -3.0)

        scenario = Scenario({"A": "Low", "B": "Low"}, matrix)
        builder = NetworkGraphBuilder(matrix)
        graph = builder.build_state_specific_network(scenario, min_abs_weight=0.1)

        assert graph.has_edge("A", "B")
        assert float(graph.edges["A", "B"]["weight"]) == 3.0


class TestNetworkAnalyzer:
    """Test suite for NetworkAnalyzer class."""

    def test_compute_centrality_measures_orders_chain_center_highest(self) -> None:
        """Centrality measures with highest degree for chain centre are computed."""
        matrix = _toy_chain_matrix()
        analyzer = NetworkAnalyzer(matrix)
        measures = analyzer.compute_centrality_measures()

        assert set(measures.keys()) == {"A", "B", "C"}
        assert measures["B"]["degree"] > measures["A"]["degree"]
        assert measures["B"]["degree"] > measures["C"]["degree"]

    def test_find_impact_pathways_returns_ranked_paths(self) -> None:
        """Find impact pathways and ensure path is returned."""
        matrix = _toy_chain_matrix()
        analyzer = NetworkAnalyzer(matrix)
        paths = analyzer.find_impact_pathways("A", "C", max_length=3)

        assert paths == [["A", "B", "C"]]

    def test_find_impact_pathways_respects_max_paths_metadata(self) -> None:
        """Bound path enumeration and expose truncation metadata."""
        descriptors = {"A": ["L", "H"], "B": ["L", "H"], "C": ["L", "H"], "D": ["L", "H"]}
        matrix = CIBMatrix(descriptors)
        for src in descriptors:
            for tgt in descriptors:
                if src == tgt:
                    continue
                for s_src in descriptors[src]:
                    for s_tgt in descriptors[tgt]:
                        matrix.set_impact(src, s_src, tgt, s_tgt, 1.0)
        analyzer = NetworkAnalyzer(matrix)
        paths, meta = analyzer.find_impact_pathways(
            "A",
            "D",
            max_length=3,
            max_paths=2,
            return_metadata=True,
        )

        assert len(paths) <= 2
        assert meta["truncated_by_max_paths"] is True
        assert meta["is_complete"] is False
        assert int(meta["enumerated_paths"]) >= len(paths)

    def test_find_impact_pathways_respects_time_limit_metadata(self) -> None:
        """Time-limited search should return truncation metadata."""
        descriptors = {"A": ["L", "H"], "B": ["L", "H"], "C": ["L", "H"], "D": ["L", "H"]}
        matrix = CIBMatrix(descriptors)
        for src in descriptors:
            for tgt in descriptors:
                if src == tgt:
                    continue
                for s_src in descriptors[src]:
                    for s_tgt in descriptors[tgt]:
                        matrix.set_impact(src, s_src, tgt, s_tgt, 1.0)
        analyzer = NetworkAnalyzer(matrix)
        _paths, meta = analyzer.find_impact_pathways(
            "A",
            "D",
            max_length=3,
            time_limit_s=0.0,
            return_metadata=True,
        )

        assert meta["truncated_by_time_limit"] is True
        assert meta["is_complete"] is False

    def test_find_strongest_pathways_respects_bounds_and_metadata(self) -> None:
        """Strongest-path API should expose bounded truncation metadata."""
        descriptors = {"A": ["L", "H"], "B": ["L", "H"], "C": ["L", "H"], "D": ["L", "H"]}
        matrix = CIBMatrix(descriptors)
        for src in descriptors:
            for tgt in descriptors:
                if src == tgt:
                    continue
                for s_src in descriptors[src]:
                    for s_tgt in descriptors[tgt]:
                        matrix.set_impact(src, s_src, tgt, s_tgt, 1.0)

        pathway_analyzer = ImpactPathwayAnalyzer(matrix)
        top_paths, meta = pathway_analyzer.find_strongest_pathways(
            n=3,
            max_length=3,
            max_paths=2,
            return_metadata=True,
        )

        assert len(top_paths) <= 2
        assert meta["truncated_by_max_paths"] is True
        assert int(meta["enumerated_paths"]) >= len(top_paths)

    def test_find_impact_pathways_invalid_descriptor_raises(self) -> None:
        """Raise ValueError for unknown source or target."""
        matrix = _toy_chain_matrix()
        analyzer = NetworkAnalyzer(matrix)

        with pytest.raises(ValueError, match="Source descriptor"):
            analyzer.find_impact_pathways("X", "C")
