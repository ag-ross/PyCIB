"""
Network analysis tools for CIB systems.

This module provides network graph construction and qualitative network analysis
capabilities for understanding descriptor interrelationships and system
structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cib.core import CIBMatrix, Scenario

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Network analysis requires networkx. Install with: pip install networkx"
    ) from exc


def _impact_direction(value: float) -> str:
    """
    Map a signed impact value to a direction label.

    Args:
        value: Signed impact value.

    Returns:
        "promoting" for positive values, otherwise "hindering".
    """
    return "promoting" if float(value) > 0.0 else "hindering"


@dataclass(frozen=True)
class AggregationResult:
    """
    Aggregate impact summary for an ordered descriptor pair.

    Attributes:
        signed: Signed representative impact.
        absolute: Absolute representative impact.
    """

    signed: float
    absolute: float


class NetworkGraphBuilder:
    """
    Builds networkx graphs from CIB matrices for analysis.

    Converts CIB impact relationships into directed graphs where nodes
    represent descriptors and edges represent impact relationships.
    """

    def __init__(self, matrix: CIBMatrix) -> None:
        """
        Initialize graph builder with a CIB matrix.

        Args:
            matrix: CIB matrix containing impact relationships.

        Raises:
            ValueError: If matrix has no descriptors.
        """
        if not matrix.descriptors:
            raise ValueError("Matrix must have at least one descriptor")
        self.matrix = matrix

    def build_impact_network(
        self,
        *,
        aggregation: str = "mean_absolute",
        min_abs_weight: float = 0.0,
        include_state_impacts: bool = False,
        state_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> nx.DiGraph:
        """
        Build a directed graph representing all impact relationships.

        Nodes are descriptors. Edges represent impacts between descriptors.
        Edges store both signed and absolute weights to preserve impact
        direction while supporting magnitude-based analysis.

        Args:
            aggregation: Aggregation method for impacts across all state pairs.
                Supported values are "mean_absolute", "mean_signed",
                "max_absolute", and "weighted_mean".
            min_abs_weight: Do not include edges with abs_weight below this
                threshold.
            include_state_impacts: If True, include a "state_impacts" mapping
                per edge containing all (src_state, tgt_state) impacts.
            state_probabilities: Optional probabilities for "weighted_mean".
                Format is {descriptor: {state_label: probability}}.

        Returns:
            Directed graph with descriptor nodes and impact edges.

        Raises:
            ValueError: If aggregation is unknown or weighted inputs are invalid.
        """
        graph = nx.DiGraph()
        for desc in self.matrix.descriptors.keys():
            graph.add_node(desc)

        for src_desc in self.matrix.descriptors.keys():
            for tgt_desc in self.matrix.descriptors.keys():
                if src_desc == tgt_desc:
                    continue

                agg = self._aggregate_pair(
                    src_desc=src_desc,
                    tgt_desc=tgt_desc,
                    aggregation=aggregation,
                    state_probabilities=state_probabilities,
                )
                if agg.absolute < float(min_abs_weight) or agg.absolute <= 0.0:
                    continue

                attrs: Dict[str, object] = {
                    "weight": float(agg.signed),
                    "abs_weight": float(agg.absolute),
                    "impact_strength": float(agg.absolute),
                    "impact_direction": _impact_direction(agg.signed),
                }

                if include_state_impacts:
                    attrs["state_impacts"] = self._collect_state_impacts(
                        src_desc=src_desc,
                        tgt_desc=tgt_desc,
                    )

                graph.add_edge(src_desc, tgt_desc, **attrs)

        return graph

    def build_state_specific_network(
        self,
        scenario: Optional[Scenario] = None,
        *,
        min_abs_weight: float = 0.0,
        aggregation: str = "mean_absolute",
        state_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> nx.DiGraph:
        """
        Build a network for a specific scenario state.

        When scenario is provided:
        - For each ordered descriptor pair (src, tgt), src != tgt:
          - src_state = scenario.get_state(src)
          - tgt_state = scenario.get_state(tgt)
          - weight = matrix.get_impact(src, src_state, tgt, tgt_state)
          - Add the edge only if abs(weight) exceeds the configured threshold.

        When scenario is None:
        - Returns the aggregated network (same as build_impact_network()).

        Args:
            scenario: Scenario to use for state-specific impacts. If None, uses
                aggregated impacts across all states.
            min_abs_weight: Do not include edges with abs_weight below this
                threshold.
            aggregation: Aggregation method when scenario is None.
            state_probabilities: Optional probabilities for "weighted_mean".

        Returns:
            Directed graph with impact edges.
        """
        if scenario is None:
            return self.build_impact_network(
                aggregation=aggregation,
                min_abs_weight=min_abs_weight,
                include_state_impacts=False,
                state_probabilities=state_probabilities,
            )

        graph = nx.DiGraph()
        for desc in self.matrix.descriptors.keys():
            graph.add_node(desc)

        for src_desc in self.matrix.descriptors.keys():
            src_state = scenario.get_state(src_desc)
            for tgt_desc in self.matrix.descriptors.keys():
                if src_desc == tgt_desc:
                    continue
                tgt_state = scenario.get_state(tgt_desc)
                w = float(self.matrix.get_impact(src_desc, src_state, tgt_desc, tgt_state))
                if abs(w) < float(min_abs_weight) or abs(w) <= 0.0:
                    continue
                graph.add_edge(
                    src_desc,
                    tgt_desc,
                    weight=w,
                    abs_weight=abs(w),
                    impact_strength=abs(w),
                    impact_direction=_impact_direction(w),
                )

        return graph

    def _aggregate_pair(
        self,
        *,
        src_desc: str,
        tgt_desc: str,
        aggregation: str,
        state_probabilities: Optional[Dict[str, Dict[str, float]]],
    ) -> AggregationResult:
        """
        Aggregate impacts for one ordered descriptor pair.

        Args:
            src_desc: Source descriptor name.
            tgt_desc: Target descriptor name.
            aggregation: Aggregation method.
            state_probabilities: Optional state probabilities for weighted mean.

        Returns:
            AggregationResult containing signed and absolute representative impacts.

        Raises:
            ValueError: If aggregation is unknown or weighted inputs are invalid.
        """
        src_states = self.matrix.descriptors[src_desc]
        tgt_states = self.matrix.descriptors[tgt_desc]

        impacts: List[float] = []
        for src_state in src_states:
            for tgt_state in tgt_states:
                impacts.append(
                    float(self.matrix.get_impact(src_desc, src_state, tgt_desc, tgt_state))
                )
        if not impacts:
            return AggregationResult(signed=0.0, absolute=0.0)

        method = str(aggregation).strip().lower()
        if method == "mean_absolute":
            signed = sum(impacts) / len(impacts)
            absolute = sum(abs(x) for x in impacts) / len(impacts)
            return AggregationResult(signed=signed, absolute=absolute)
        if method == "mean_signed":
            signed = sum(impacts) / len(impacts)
            return AggregationResult(signed=signed, absolute=abs(signed))
        if method == "max_absolute":
            signed = max(impacts, key=lambda x: abs(x))
            return AggregationResult(signed=signed, absolute=abs(signed))
        if method == "weighted_mean":
            if state_probabilities is None:
                raise ValueError(
                    "state_probabilities must be provided for aggregation='weighted_mean'"
                )
            p_src = state_probabilities.get(src_desc)
            p_tgt = state_probabilities.get(tgt_desc)
            if p_src is None or p_tgt is None:
                raise ValueError(
                    "state_probabilities must include entries for both src and tgt descriptors"
                )

            weights: List[float] = []
            weighted_impacts: List[float] = []
            weighted_abs: List[float] = []
            for src_state in src_states:
                for tgt_state in tgt_states:
                    w = float(p_src.get(src_state, 0.0)) * float(p_tgt.get(tgt_state, 0.0))
                    if w <= 0.0:
                        continue
                    impact = float(self.matrix.get_impact(src_desc, src_state, tgt_desc, tgt_state))
                    weights.append(w)
                    weighted_impacts.append(impact * w)
                    weighted_abs.append(abs(impact) * w)
            if not weights or sum(weights) <= 0.0:
                return AggregationResult(signed=0.0, absolute=0.0)

            denom = float(sum(weights))
            signed = float(sum(weighted_impacts)) / denom
            absolute = float(sum(weighted_abs)) / denom
            return AggregationResult(signed=signed, absolute=absolute)

        raise ValueError(f"Unknown aggregation method: {aggregation!r}")

    def _collect_state_impacts(
        self,
        *,
        src_desc: str,
        tgt_desc: str,
    ) -> Dict[Tuple[str, str], float]:
        """
        Collect all state-to-state impacts for one descriptor pair.

        Args:
            src_desc: Source descriptor name.
            tgt_desc: Target descriptor name.

        Returns:
            Mapping from (src_state, tgt_state) to signed impact.
        """
        result: Dict[Tuple[str, str], float] = {}
        for src_state in self.matrix.descriptors[src_desc]:
            for tgt_state in self.matrix.descriptors[tgt_desc]:
                result[(src_state, tgt_state)] = float(
                    self.matrix.get_impact(src_desc, src_state, tgt_desc, tgt_state)
                )
        return result


class NetworkAnalyzer:
    """
    Performs qualitative network analysis on CIB impact networks.
    """

    def __init__(self, matrix: CIBMatrix) -> None:
        """
        Initialize analyzer with a CIB matrix.

        Args:
            matrix: CIB matrix to analyze.
        """
        self.matrix = matrix
        self._builder = NetworkGraphBuilder(matrix)

    def compute_centrality_measures(
        self, scenario: Optional[Scenario] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute network centrality measures for descriptors.

        Args:
            scenario: Optional scenario for state-specific analysis.

        Returns:
            Dictionary mapping descriptor names to centrality metrics.
        """
        graph = self._builder.build_state_specific_network(scenario)

        degree = nx.degree_centrality(graph)
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        try:
            eigen = nx.eigenvector_centrality(graph, weight="abs_weight", max_iter=1000)
        except Exception:
            eigen = {n: 0.0 for n in graph.nodes()}

        results: Dict[str, Dict[str, float]] = {}
        for desc in graph.nodes():
            results[str(desc)] = {
                "degree": float(degree.get(desc, 0.0)),
                "betweenness": float(betweenness.get(desc, 0.0)),
                "eigenvector": float(eigen.get(desc, 0.0)),
            }
        return results

    def find_impact_pathways(
        self,
        source: str,
        target: str,
        max_length: int = 3,
        scenario: Optional[Scenario] = None,
    ) -> List[List[str]]:
        """
        Find all impact pathways between two descriptors.

        Uses nx.all_simple_paths() to enumerate all simple paths (no cycles).
        Paths are ranked by cumulative impact strength (sum of abs_weight).
        max_length refers to number of edges.

        Args:
            source: Source descriptor name.
            target: Target descriptor name.
            max_length: Maximum pathway length in edges (default: 3).
            scenario: Optional scenario for state-specific pathways.

        Returns:
            List of pathways, each as a list of descriptor names, sorted by
            cumulative impact strength (descending).

        Raises:
            ValueError: If source or target descriptor not found in matrix.
        """
        if source not in self.matrix.descriptors:
            raise ValueError(f"Source descriptor '{source}' not found")
        if target not in self.matrix.descriptors:
            raise ValueError(f"Target descriptor '{target}' not found")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if source == target:
            return []

        graph = self._builder.build_state_specific_network(scenario)
        if not nx.has_path(graph, source, target):
            return []

        paths = list(nx.all_simple_paths(graph, source=source, target=target, cutoff=max_length))

        def _path_strength(path: Sequence[str]) -> float:
            strength = 0.0
            for u, v in zip(path, path[1:]):
                strength += float(graph.edges[u, v].get("abs_weight", 0.0))
            return strength

        paths.sort(key=_path_strength, reverse=True)
        return [list(map(str, p)) for p in paths]

    def compute_network_metrics(
        self, scenario: Optional[Scenario] = None
    ) -> Dict[str, float]:
        """
        Compute overall network structure metrics.

        Args:
            scenario: Optional scenario for state-specific metrics.

        Returns:
            Dictionary of network metrics.
        """
        graph = self._builder.build_state_specific_network(scenario)
        undirected = graph.to_undirected()

        metrics: Dict[str, float] = {
            "n_nodes": float(graph.number_of_nodes()),
            "n_edges": float(graph.number_of_edges()),
            "density": float(nx.density(graph)),
        }

        components = list(nx.connected_components(undirected))
        metrics["n_components"] = float(len(components))

        try:
            metrics["avg_clustering"] = float(nx.average_clustering(undirected))
        except Exception:
            metrics["avg_clustering"] = 0.0

        largest: Optional[Iterable[str]] = None
        if components:
            largest = max(components, key=len)
        if largest is not None and len(list(largest)) >= 2:
            sub = undirected.subgraph(largest).copy()
            try:
                metrics["avg_shortest_path_length"] = float(nx.average_shortest_path_length(sub))
            except Exception:
                metrics["avg_shortest_path_length"] = 0.0
        else:
            metrics["avg_shortest_path_length"] = 0.0

        return metrics

    def identify_communities(
        self,
        scenario: Optional[Scenario] = None,
        resolution: float = 1.0,
    ) -> Dict[str, int]:
        """
        Identify descriptor communities using the Louvain clustering algorithm.

        Args:
            scenario: Optional scenario for state-specific communities.
            resolution: Louvain resolution parameter (default: 1.0).

        Returns:
            Dictionary mapping descriptor names to community IDs (integers).
        """
        graph = self._builder.build_state_specific_network(scenario)
        undirected = graph.to_undirected()
        communities = nx.community.louvain_communities(undirected, resolution=float(resolution))

        result: Dict[str, int] = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                result[str(node)] = int(idx)
        return result

    def get_most_influential_descriptors(
        self,
        n: int = 5,
        metric: str = "degree",
        scenario: Optional[Scenario] = None,
    ) -> List[Tuple[str, float]]:
        """
        Identify the most influential descriptors in the network.

        Args:
            n: Number of top descriptors to return.
            metric: Centrality metric to use ("degree", "betweenness", "eigenvector").
            scenario: Optional scenario for state-specific analysis.

        Returns:
            List of (descriptor, score) tuples, sorted by influence.

        Raises:
            ValueError: If metric is invalid or n is non-positive.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        metric_key = str(metric).strip().lower()
        if metric_key not in {"degree", "betweenness", "eigenvector"}:
            raise ValueError(f"Unknown centrality metric: {metric!r}")

        measures = self.compute_centrality_measures(scenario)
        items = [(desc, float(vals.get(metric_key, 0.0))) for desc, vals in measures.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[: int(n)]


class ImpactPathwayAnalyzer:
    """
    Analyzes impact pathways and influence flows in CIB networks.
    """

    def __init__(self, matrix: CIBMatrix) -> None:
        """
        Initialize pathway analyzer with a CIB matrix.

        Args:
            matrix: CIB matrix to analyze.
        """
        self.matrix = matrix

    def trace_impact_flow(
        self,
        source: str,
        source_state: str,
        max_depth: int = 3,
        scenario: Optional[Scenario] = None,
        decay_factor: float = 1.0,
        *,
        min_abs_weight: float = 0.0,
    ) -> Dict[str, float]:
        """
        Trace impact flow from a specific descriptor-state combination.

        Impacts are accumulated additively along each simple path, with optional
        depth-based attenuation. The returned score for a target is the sum of
        contributions from all simple paths from source to target up to
        max_depth edges.

        Args:
            source: Source descriptor name.
            source_state: Source state label to use.
            max_depth: Maximum depth in edges to trace (default: 3).
            scenario: Optional scenario for state-specific impacts. If provided,
                its state assignment is used for all descriptors except the
                source descriptor, which uses source_state.
            decay_factor: Attenuation factor per edge (default: 1.0).
            min_abs_weight: Do not traverse edges with abs(weight) below this threshold.

        Returns:
            Dictionary mapping target descriptors to cumulative impact scores.

        Raises:
            ValueError: If inputs are invalid.
        """
        if source not in self.matrix.descriptors:
            raise ValueError(f"Source descriptor '{source}' not found")
        if source_state not in self.matrix.descriptors[source]:
            raise ValueError(
                f"State '{source_state}' not found for descriptor '{source}'"
            )
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if float(decay_factor) <= 0.0:
            raise ValueError("decay_factor must be positive")

        state_map: Dict[str, str]
        if scenario is None:
            # Arbitrary baseline states are used for non-source descriptors.
            state_map = {d: self.matrix.descriptors[d][0] for d in self.matrix.descriptors.keys()}
        else:
            state_map = scenario.to_dict()
        state_map[source] = str(source_state)

        results: Dict[str, float] = {d: 0.0 for d in self.matrix.descriptors.keys() if d != source}

        def _neighbors(desc: str) -> Iterable[Tuple[str, float]]:
            src_state_local = state_map[desc]
            for tgt_desc in self.matrix.descriptors.keys():
                if tgt_desc == desc:
                    continue
                tgt_state_local = state_map[tgt_desc]
                w = float(self.matrix.get_impact(desc, src_state_local, tgt_desc, tgt_state_local))
                if abs(w) <= 0.0 or abs(w) < float(min_abs_weight):
                    continue
                yield tgt_desc, w

        def _dfs(
            current: str,
            *,
            depth: int,
            visited: set[str],
            cumulative: float,
        ) -> None:
            if depth >= int(max_depth):
                return
            for nxt, w in _neighbors(current):
                if nxt in visited:
                    continue
                next_depth = depth + 1
                next_cumulative = cumulative + w
                contribution = next_cumulative * (float(decay_factor) ** next_depth)
                if nxt in results:
                    results[nxt] += contribution
                visited.add(nxt)
                _dfs(nxt, depth=next_depth, visited=visited, cumulative=next_cumulative)
                visited.remove(nxt)

        _dfs(source, depth=0, visited={source}, cumulative=0.0)
        return results

    def find_strongest_pathways(
        self,
        n: int = 10,
        scenario: Optional[Scenario] = None,
        *,
        max_length: int = 3,
        aggregation: str = "mean_absolute",
        min_abs_weight: float = 0.0,
    ) -> List[Tuple[List[str], float]]:
        """
        Find the strongest impact pathways in the network.

        This method enumerates simple paths up to max_length edges and ranks them
        by cumulative impact strength (sum of abs_weight). For large graphs, the
        search space can grow quickly; callers should keep max_length small.

        Args:
            n: Number of top pathways to return.
            scenario: Optional scenario for state-specific pathways.
            max_length: Maximum pathway length in edges (default: 3).
            aggregation: Aggregation method when scenario is None.
            min_abs_weight: Do not include edges with abs_weight below this threshold.

        Returns:
            List of (pathway, strength) tuples, sorted by strength.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        builder = NetworkGraphBuilder(self.matrix)
        graph = builder.build_state_specific_network(
            scenario,
            min_abs_weight=min_abs_weight,
            aggregation=aggregation,
        )

        nodes = list(graph.nodes())
        if len(nodes) < 2 or graph.number_of_edges() == 0:
            return []

        candidates: List[Tuple[List[str], float]] = []

        def _path_strength(path: Sequence[str]) -> float:
            s = 0.0
            for u, v in zip(path, path[1:]):
                s += float(graph.edges[u, v].get("abs_weight", 0.0))
            return s

        for i, src in enumerate(nodes):
            for tgt in nodes[i + 1 :]:
                if not nx.has_path(graph, src, tgt):
                    continue
                for p in nx.all_simple_paths(graph, source=src, target=tgt, cutoff=int(max_length)):
                    candidates.append((list(map(str, p)), _path_strength(p)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[: int(n)]

