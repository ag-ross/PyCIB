"""
Visualization tools for CIB analysis.

This module provides plotting capabilities for cross-impact matrices,
scenarios, and their relationships.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from cib.core import CIBMatrix, Scenario

if TYPE_CHECKING:
    from cib.succession import SuccessionOperator
    from cib.transformation_matrix import TransformationMatrix


class MatrixVisualizer:
    """
    Visualizes cross-impact matrices.

    Provides methods to create heatmaps and other visualizations of
    impact relationships in CIB matrices.
    """

    @staticmethod
    def plot_matrix(
        matrix: CIBMatrix,
        descriptor_pair: Optional[Tuple[str, str]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot a heatmap of impact matrix for a descriptor pair.

        Args:
            matrix: CIB matrix to visualize.
            descriptor_pair: Tuple of (source, target) descriptor names.
                If None, plots the first available pair.
            ax: Matplotlib axes to plot on. If None, creates new figure.

        Returns:
            Matplotlib axes object.

        Raises:
            ValueError: If descriptor pair is invalid.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        if descriptor_pair is None:
            descriptors = list(matrix.descriptors.keys())
            if len(descriptors) < 2:
                raise ValueError("Need at least 2 descriptors to plot")
            descriptor_pair = (descriptors[0], descriptors[1])

        src_desc, tgt_desc = descriptor_pair

        if src_desc not in matrix.descriptors:
            raise ValueError(f"Source descriptor '{src_desc}' not found")
        if tgt_desc not in matrix.descriptors:
            raise ValueError(f"Target descriptor '{tgt_desc}' not found")

        src_states = matrix.descriptors[src_desc]
        tgt_states = matrix.descriptors[tgt_desc]

        impact_data = np.zeros((len(src_states), len(tgt_states)))
        for i, src_state in enumerate(src_states):
            for j, tgt_state in enumerate(tgt_states):
                impact_data[i, j] = matrix.get_impact(
                    src_desc, src_state, tgt_desc, tgt_state
                )

        im = ax.imshow(impact_data, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(np.arange(len(tgt_states)))
        ax.set_yticks(np.arange(len(src_states)))
        ax.set_xticklabels(tgt_states)
        ax.set_yticklabels(src_states)
        ax.set_xlabel(f"Target: {tgt_desc}")
        ax.set_ylabel(f"Source: {src_desc}")
        ax.set_title(f"Impact Matrix: {src_desc} → {tgt_desc}")

        plt.colorbar(im, ax=ax, label="Impact Value")

        return ax

    @staticmethod
    def heatmap(
        matrix: CIBMatrix, ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Create a heatmap visualization of the matrix.

        For matrices with multiple descriptor pairs, this plots the first
        available pair. Use plot_matrix for specific pairs.

        Args:
            matrix: CIB matrix to visualize.
            ax: Matplotlib axes to plot on. If None, creates new figure.

        Returns:
            Matplotlib axes object.
        """
        return MatrixVisualizer.plot_matrix(matrix, ax=ax)


class ScenarioVisualizer:
    """
    Visualizes scenarios and their relationships.

    Provides methods to plot scenarios, create network graphs, and
    visualize scenario evolution.
    """

    @staticmethod
    def plot_scenarios(
        scenarios: List[Scenario],
        matrix: CIBMatrix,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Visualize scenarios as points in descriptor space.

        For systems with 2-3 descriptors, creates a scatter or 3D plot.
        For larger systems, creates a summary visualization.

        Args:
            scenarios: List of scenarios to plot.
            matrix: CIB matrix providing descriptor definitions.
            ax: Matplotlib axes to plot on. If None, creates new figure.

        Returns:
            Matplotlib axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if len(matrix.descriptors) == 2:
            desc_names = list(matrix.descriptors.keys())
            x_values: List[int] = []
            y_values: List[int] = []

            for scenario in scenarios:
                x_idx = scenario.get_state_index(desc_names[0])
                y_idx = scenario.get_state_index(desc_names[1])
                x_values.append(x_idx)
                y_values.append(y_idx)

            ax.scatter(x_values, y_values, alpha=0.6)
            ax.set_xlabel(desc_names[0])
            ax.set_ylabel(desc_names[1])
            ax.set_title(f"Scenarios in {desc_names[0]}-{desc_names[1]} Space")
        else:
            ax.text(
                0.5,
                0.5,
                f"{len(scenarios)} scenarios\n"
                f"{len(matrix.descriptors)} descriptors",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Scenario Summary")
            ax.axis("off")

        return ax

    @staticmethod
    def scenario_network(
        scenarios: List[Scenario],
        matrix: CIBMatrix,
        ax: Optional[plt.Axes] = None,
        *,
        layout: str = "spring",
        show_transitions: bool = True,
        succession_operator: Optional["SuccessionOperator"] = None,
        edge_metric: str = "hamming",
        max_edges_per_node: int = 5,
        node_weights: Optional[Mapping[Scenario, float]] = None,
        label_counts: bool = False,
    ) -> plt.Axes:
        """
        Create a network graph of scenario relationships using NetworkX.

        Edge creation rules:
        - If show_transitions is True and a succession_operator is provided, add
          directed edges for succession transitions (scenario -> successor).
        - If edge_metric is "hamming", connect scenarios with Hamming distance
          ≤ 2, and weight = 1 / (1 + distance).
        - If edge_metric is "impact_similarity", connect scenarios by cosine
          similarity of their per-descriptor chosen-state impact scores.
        - At most max_edges_per_node similarity edges are kept per node.

        Args:
            scenarios: List of scenarios to visualize.
            matrix: CIB matrix providing descriptor definitions.
            ax: Matplotlib axes to plot on. If None, creates new figure.
            layout: Graph layout algorithm ("spring", "circular", "hierarchical").
            show_transitions: Whether to show succession transition edges.
            succession_operator: Optional succession operator for transition edges.
            edge_metric: Similarity metric ("hamming", "impact_similarity").
            max_edges_per_node: Maximum similarity edges per node.
            node_weights: Optional mapping from Scenario to a weight (for example,
                Monte Carlo counts). If provided, node sizes are scaled by this
                weight.
            label_counts: Whether to append node weight (as an integer) to node
                labels when node_weights is provided.

        Returns:
            Matplotlib axes object.

        Raises:
            ValueError: If inputs are invalid.
            ImportError: If networkx is not installed.
        """
        if not scenarios:
            raise ValueError("scenarios cannot be empty")
        if max_edges_per_node <= 0:
            raise ValueError("max_edges_per_node must be positive")

        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Scenario network visualization requires networkx. "
                "Install with: pip install networkx"
            ) from exc

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        if len(scenarios) > 250:
            ax.text(
                0.5,
                0.5,
                f"{len(scenarios)} scenarios\n"
                "Network visualization is limited to ≤250 scenarios by default.",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Scenario Network")
            ax.axis("off")
            return ax

        scenario_to_idx: Dict[Scenario, int] = {s: i for i, s in enumerate(scenarios)}
        graph = nx.DiGraph()
        for idx, scenario in enumerate(scenarios):
            graph.add_node(idx, scenario=scenario, label=f"S{idx}")

        metric = str(edge_metric).strip().lower()
        if metric not in {"hamming", "impact_similarity"}:
            raise ValueError(f"Invalid edge_metric: {edge_metric!r}")

        def _hamming(a: Scenario, b: Scenario) -> int:
            return int(sum(x != y for x, y in zip(a.to_indices(), b.to_indices())))

        def _impact_vector(s: Scenario) -> np.ndarray:
            from cib.core import ImpactBalance

            balance = ImpactBalance(s, matrix).balance
            vals: List[float] = []
            for desc in matrix.descriptors.keys():
                chosen = s.get_state(desc)
                vals.append(float(balance[desc][chosen]))
            return np.array(vals, dtype=float)

        if metric == "hamming":
            for i, a in enumerate(scenarios):
                distances: List[Tuple[int, int]] = []
                for j, b in enumerate(scenarios):
                    if i == j:
                        continue
                    d = _hamming(a, b)
                    if 0 < d <= 2:
                        distances.append((j, d))
                distances.sort(key=lambda x: x[1])
                for j, d in distances[: int(max_edges_per_node)]:
                    w = 1.0 / (1.0 + float(d))
                    graph.add_edge(i, j, weight=w, edge_type="similarity")
                    graph.add_edge(j, i, weight=w, edge_type="similarity")
        else:
            vectors = [_impact_vector(s) for s in scenarios]
            norms = [float(np.linalg.norm(v)) for v in vectors]
            for i, v in enumerate(vectors):
                sims: List[Tuple[int, float]] = []
                for j, u in enumerate(vectors):
                    if i == j:
                        continue
                    denom = norms[i] * norms[j]
                    if denom <= 0.0:
                        continue
                    sim = float(np.dot(v, u) / denom)
                    sims.append((j, sim))
                sims.sort(key=lambda x: x[1], reverse=True)
                for j, sim in sims[: int(max_edges_per_node)]:
                    if sim <= 0.0:
                        continue
                    graph.add_edge(i, j, weight=sim, edge_type="similarity")
                    graph.add_edge(j, i, weight=sim, edge_type="similarity")

        if show_transitions and succession_operator is not None:
            for i, scen in enumerate(scenarios):
                successor = succession_operator.find_successor(scen, matrix)
                j = scenario_to_idx.get(successor)
                if j is None:
                    continue
                graph.add_edge(i, j, weight=1.0, edge_type="succession")

        layout_name = str(layout).strip().lower()
        if layout_name == "circular":
            pos = nx.circular_layout(graph)
        elif layout_name == "hierarchical":
            try:
                from networkx.drawing.nx_agraph import graphviz_layout

                pos = graphviz_layout(graph, prog="dot")
            except Exception:
                pos = nx.spring_layout(graph, seed=42, k=1.0, iterations=50)
        else:
            pos = nx.spring_layout(graph, seed=42, k=1.0, iterations=50)

        weights = [float(d.get("weight", 0.0)) for _u, _v, d in graph.edges(data=True)]
        max_w = max(weights) if weights else 1.0
        widths = [0.6 + 3.4 * (w / max_w) ** 2 for w in weights]
        colors = [
            ("black" if d.get("edge_type") == "succession" else "#95a5a6")
            for _u, _v, d in graph.edges(data=True)
        ]

        node_size_by_idx: Dict[int, float] = {}
        if node_weights is None:
            node_size_by_idx = {i: 350.0 for i in range(len(scenarios))}
        else:
            weights_by_idx: Dict[int, float] = {}
            for i, scen in enumerate(scenarios):
                weights_by_idx[i] = float(node_weights.get(scen, 0.0))
            max_node_w = max(weights_by_idx.values()) if weights_by_idx else 1.0
            max_node_w = max(max_node_w, 1.0)
            for i, w in weights_by_idx.items():
                ratio = float(w) / float(max_node_w)
                node_size_by_idx[i] = 140.0 + 2600.0 * (ratio**1.25)

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=[node_size_by_idx[i] for i in range(len(scenarios))],
            node_color="#3498db",
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            width=widths,
            edge_color=colors,
            arrows=True,
            arrowsize=10,
            alpha=0.8,
            ax=ax,
        )
        labels: Dict[int, str] = {}
        for i in range(len(scenarios)):
            if node_weights is not None and label_counts:
                count = int(round(float(node_weights.get(scenarios[i], 0.0))))
                labels[i] = f"S{i}\n{count}"
            else:
                labels[i] = f"S{i}"
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

        ax.set_title(f"Scenario Network ({len(scenarios)} scenarios)")
        ax.axis("off")
        return ax

    @staticmethod
    def transformation_graph(
        transformation_matrix: "TransformationMatrix",
        matrix: CIBMatrix,
        ax: Optional[plt.Axes] = None,
        *,
        layout: str = "spring",
        min_success_rate: float = 0.0,
    ) -> plt.Axes:
        """
        Visualize transformation matrix as a directed graph.

        Scenarios are shown as nodes, and transformations are shown as
        directed edges. Edge thickness represents success rate.

        Args:
            transformation_matrix: TransformationMatrix to visualize.
            matrix: CIB matrix providing descriptor definitions.
            ax: Matplotlib axes to plot on. If None, creates new figure.
            layout: Graph layout algorithm ("spring", "circular", "hierarchical").
            min_success_rate: Minimum success rate to show an edge.

        Returns:
            Matplotlib axes object.
        """
        import networkx as nx

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        scenarios = transformation_matrix.scenarios
        if len(scenarios) > 250:
            ax.text(
                0.5,
                0.5,
                f"{len(scenarios)} scenarios\n"
                "Transformation graph is limited to ≤250 scenarios by default.",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Transformation Graph")
            ax.axis("off")
            return ax

        scenario_to_idx: Dict[Scenario, int] = {
            s: i for i, s in enumerate(scenarios)
        }
        graph = nx.DiGraph()
        for idx, scenario in enumerate(scenarios):
            graph.add_node(idx, scenario=scenario, label=f"S{idx}")

        for (source, target), perturbations in transformation_matrix.transformations.items():
            source_idx = scenario_to_idx.get(source)
            target_idx = scenario_to_idx.get(target)
            if source_idx is None or target_idx is None:
                continue

            best_pert = max(perturbations, key=lambda p: p.success_rate)
            if best_pert.success_rate < min_success_rate:
                continue

            if graph.has_edge(source_idx, target_idx):
                existing_weight = graph[source_idx][target_idx].get("weight", 0.0)
                if best_pert.success_rate > existing_weight:
                    graph[source_idx][target_idx]["weight"] = best_pert.success_rate
                    graph[source_idx][target_idx]["pert_type"] = best_pert.perturbation_type
                    graph[source_idx][target_idx]["magnitude"] = best_pert.magnitude
            else:
                graph.add_edge(
                    source_idx,
                    target_idx,
                    weight=best_pert.success_rate,
                    pert_type=best_pert.perturbation_type,
                    magnitude=best_pert.magnitude,
                )

        layout_name = str(layout).strip().lower()
        if layout_name == "circular":
            pos = nx.circular_layout(graph)
        elif layout_name == "hierarchical":
            try:
                from networkx.drawing.nx_agraph import graphviz_layout

                pos = graphviz_layout(graph, prog="dot")
            except Exception:
                pos = nx.spring_layout(graph, seed=42, k=1.0, iterations=50)
        else:
            pos = nx.spring_layout(graph, seed=42, k=1.0, iterations=50)

        weights = [
            float(d.get("weight", 0.0)) for _u, _v, d in graph.edges(data=True)
        ]
        max_w = max(weights) if weights else 1.0
        if max_w > 0:
            # Minimum width ensures all edges are visible.
            widths = [max(0.5, 0.5 + 3.5 * (w / max_w) ** 0.8) for w in weights]
        else:
            widths = [1.0] * len(weights)

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=500,
            node_color="#3498db",
            ax=ax,
        )
        # Edge colors are based on success rate thresholds.
        edge_colors = []
        for w in weights:
            if w >= 0.1:
                edge_colors.append("#2c3e50")  # High success rate
            elif w >= 0.05:
                edge_colors.append("#7f8c8d")  # Medium success rate
            else:
                edge_colors.append("#95a5a6")  # Low success rate

        nx.draw_networkx_edges(
            graph,
            pos,
            width=widths,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=15,
            alpha=0.7,
            ax=ax,
        )
        labels: Dict[int, str] = {
            i: f"S{i}" for i in range(len(scenarios))
        }
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

        total_pairs = len(scenarios) * (len(scenarios) - 1)
        transformations_found = len(transformation_matrix.transformations)
        edges_shown = len(graph.edges())

        title = (
            f"Transformation Graph ({len(scenarios)} scenarios, "
            f"{edges_shown} transformations shown"
        )
        if transformations_found > edges_shown:
            title += f", {transformations_found - edges_shown} filtered"
        title += ")"

        ax.set_title(title)
        ax.axis("off")
        return ax


class ShockVisualizer:
    """
    Visualizations for robustness and shock-related metrics.
    """

    @staticmethod
    def plot_robustness_scores(
        scores: Mapping[Scenario, float],
        ax: Optional[plt.Axes] = None,
        title: str = "Scenario Robustness",
    ) -> plt.Axes:
        """
        Plot robustness scores as a bar chart.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        scenarios = list(scores.keys())
        values = [float(scores[s]) for s in scenarios]
        labels = [f"S{i}" for i in range(len(scenarios))]

        ax.bar(labels, values, alpha=0.8)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Robustness (fraction consistent)")
        ax.set_title(title)
        return ax


class UncertaintyVisualizer:
    """
    Visualizations for uncertainty-aware outputs.
    """

    @staticmethod
    def plot_probability_intervals(
        probabilities: Mapping[Scenario, float],
        intervals: Mapping[Scenario, Tuple[float, float]],
        ax: Optional[plt.Axes] = None,
        title: str = "Scenario consistency probability (with CI)",
    ) -> plt.Axes:
        """
        Plot scenario probabilities with error bars (confidence intervals).
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        scenarios = list(probabilities.keys())
        p = np.array([float(probabilities[s]) for s in scenarios], dtype=float)
        ci = [intervals[s] for s in scenarios]
        lower = np.array([float(x[0]) for x in ci], dtype=float)
        upper = np.array([float(x[1]) for x in ci], dtype=float)

        x = np.arange(len(scenarios))
        yerr = np.vstack([p - lower, upper - p])
        ax.errorbar(x, p, yerr=yerr, fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i}" for i in range(len(scenarios))])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("P(consistent)")
        ax.set_title(title)
        return ax


class DynamicVisualizer:
    """
    Visualizations for dynamic (multi-period) outputs.
    """

    @staticmethod
    def plot_state_probability_bands(
        timelines: Dict[int, Dict[str, Dict[str, float]]],
        descriptor: str,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        state_order: Optional[Sequence[str]] = None,
        order_by_probability: bool = True,
    ) -> plt.Axes:
        """
        Plot stacked probability bands P(z_i(t)=k) over time for one descriptor.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        periods = sorted(int(t) for t in timelines.keys())
        # All states that appear for this descriptor are collected.
        all_states: List[str] = []
        for t in periods:
            if descriptor not in timelines[t]:
                continue
            for state in timelines[t][descriptor].keys():
                if state not in all_states:
                    all_states.append(state)

        if not all_states:
            raise ValueError(f"Descriptor {descriptor!r} not present in timelines")

        if state_order is not None:
            order = [str(s) for s in state_order]
            ordered = [s for s in order if s in all_states]
            extras = [s for s in all_states if s not in set(ordered)]
            all_states = ordered + extras
        elif order_by_probability:
            totals: Dict[str, float] = {s: 0.0 for s in all_states}
            for t in periods:
                probs = timelines[t].get(descriptor, {})
                for s in all_states:
                    totals[s] += float(probs.get(s, 0.0))
            all_states = sorted(all_states, key=lambda s: totals.get(s, 0.0), reverse=True)

        data = np.zeros((len(all_states), len(periods)), dtype=float)
        for j, t in enumerate(periods):
            probs = timelines[t].get(descriptor, {})
            for i, state in enumerate(all_states):
                data[i, j] = float(probs.get(state, 0.0))

        ax.stackplot(periods, data, labels=all_states, alpha=0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(periods)
        ax.set_xlabel("Period")
        ax.set_ylabel("Probability")
        ax.set_title(title or f"State probabilities over time: {descriptor}")
        ax.legend(loc="upper right", ncols=min(3, len(all_states)))
        return ax

    @staticmethod
    def plot_numeric_fan_chart(
        quantiles_by_period: Dict[int, Tuple[float, float, float]],
        ax: Optional[plt.Axes] = None,
        title: str = "Fan chart (median + confidence band)",
        y_label: str = "Value",
    ) -> plt.Axes:
        """
        Plot a fan chart with a median line and a single confidence band.

        Args:
            quantiles_by_period: period -> (q_low, q_mid, q_high)
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        periods = sorted(int(t) for t in quantiles_by_period.keys())
        q_low = np.array([quantiles_by_period[t][0] for t in periods], dtype=float)
        q_mid = np.array([quantiles_by_period[t][1] for t in periods], dtype=float)
        q_high = np.array([quantiles_by_period[t][2] for t in periods], dtype=float)

        ax.plot(periods, q_mid, linewidth=2.0, label="Median")
        ax.fill_between(periods, q_low, q_high, alpha=0.25, label="Band")
        ax.set_xlabel("Period")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(loc="best")
        return ax

    @staticmethod
    def plot_descriptor_stochastic_summary(
        *,
        timelines: Dict[int, Dict[str, Dict[str, float]]],
        quantiles_by_period: Dict[int, Tuple[float, float, float]],
        numeric_expectation_by_period: Dict[int, float],
        descriptor: str,
        y_label: str = "Numeric mapping",
        title: Optional[str] = None,
        ax_probs: Optional[plt.Axes] = None,
        ax_numeric: Optional[plt.Axes] = None,
        # Optionally, a third subplot with per-run step traces is added.
        spaghetti_paths: Optional[Sequence[object]] = None,
        spaghetti_numeric_mapping: Optional[Mapping[str, float]] = None,
        spaghetti_max_runs: int = 250,
        spaghetti_alpha: float = 0.06,
        ax_spaghetti: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Axes, plt.Axes, Optional[plt.Axes]]:
        """
        Plot categorical + numeric summaries, optionally with a spaghetti subplot.

        Top: stacked categorical probability bands.
        Bottom: expected value main line with quantile confidence band.
        Optional third: step-style per-run traces for the descriptor.
        """
        want_spaghetti = spaghetti_paths is not None
        if ax_probs is None or ax_numeric is None or (want_spaghetti and ax_spaghetti is None):
            if want_spaghetti:
                fig, (ax_probs, ax_numeric, ax_spaghetti) = plt.subplots(
                    3, 1, figsize=(11, 9), sharex=True
                )
            else:
                fig, (ax_probs, ax_numeric) = plt.subplots(
                    2, 1, figsize=(11, 7), sharex=True
                )
            if title:
                fig.suptitle(title                )

        # Categorical probability bands are plotted.
        state_order: Optional[List[str]] = None
        if spaghetti_numeric_mapping is not None:
            # If the descriptor has an ordered numeric mapping, it is used to keep
            # intuitive ordering (e.g., Very Low ... Medium ... Very High).
            mapping = {str(k): float(v) for k, v in spaghetti_numeric_mapping.items()}
            try:
                state_order = [
                    s for s, _v in sorted(mapping.items(), key=lambda kv: kv[1])
                ]
            except Exception:
                state_order = None
        DynamicVisualizer.plot_state_probability_bands(
            timelines=timelines,
            descriptor=descriptor,
            ax=ax_probs,
            title=f"State probability bands: {descriptor}",
            state_order=state_order,
        )

        # Numeric expectation and quantile band are plotted.
        periods = sorted(int(t) for t in quantiles_by_period.keys())
        exp = np.array([float(numeric_expectation_by_period[t]) for t in periods], dtype=float)
        q_low = np.array([float(quantiles_by_period[t][0]) for t in periods], dtype=float)
        q_mid = np.array([float(quantiles_by_period[t][1]) for t in periods], dtype=float)
        q_high = np.array([float(quantiles_by_period[t][2]) for t in periods], dtype=float)

        ax_numeric.plot(periods, exp, linewidth=2.0, label="Expected value")
        ax_numeric.plot(periods, q_mid, linewidth=1.5, linestyle="--", label="Median")
        ax_numeric.fill_between(periods, q_low, q_high, alpha=0.25, label="Quantile band")
        ax_numeric.set_ylabel(y_label)
        ax_numeric.set_xlabel("Period")
        ax_numeric.set_title("Numeric summary (expected value + band)")
        ax_numeric.legend(loc="best")
        ax_numeric.set_xticks(periods)
        ax_numeric.set_xlim(min(periods), max(periods))
        ax_probs.set_xticks(periods)

        # An optional spaghetti subplot is added.
        if want_spaghetti:
            if ax_spaghetti is None:
                raise ValueError("ax_spaghetti must be provided when spaghetti_paths is provided")
            if spaghetti_numeric_mapping is None:
                raise ValueError(
                    "spaghetti_numeric_mapping must be provided when spaghetti_paths is provided"
                )
            mapping = {k: float(v) for k, v in spaghetti_numeric_mapping.items()}
            shown = 0
            for p in spaghetti_paths:
                if shown >= int(spaghetti_max_runs):
                    break
                # Duck-typed interface is used: p.periods, p.scenarios, Scenario.to_dict().
                scen = list(getattr(p, "scenarios"))
                x = list(getattr(p, "periods"))
                y = [mapping[s.to_dict()[descriptor]] for s in scen]
                ax_spaghetti.step(x, y, where="post", color="C0", alpha=float(spaghetti_alpha), linewidth=1.0)
                shown += 1
            ax_spaghetti.set_ylabel(y_label)
            ax_spaghetti.set_title(f"Per-run step traces (first {shown} runs)")
            ax_spaghetti.set_xticks(periods)
            ax_spaghetti.set_xlim(min(periods), max(periods))

        return ax_probs, ax_numeric, ax_spaghetti if want_spaghetti else None

    @staticmethod
    def plot_descriptor_branching_summary(
        *,
        branching: object,
        descriptor: str,
        numeric_mapping: Mapping[str, float],
        y_label: str = "Numeric mapping",
        quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95),
        title: Optional[str] = None,
        max_paths: int = 10,
        ax_probs: Optional[plt.Axes] = None,
        ax_numeric: Optional[plt.Axes] = None,
        ax_paths: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
        """
        Branching-result analogue of `plot_descriptor_stochastic_summary`.

        This uses the branching graph's node/edge weights to derive:
          - P(descriptor=state) bands
          - weighted numeric quantiles + expectation
          - optional top-path step traces (from branching.top_paths)
        """
        from cib.pathway import (
            branching_numeric_expectation_by_period,
            branching_numeric_quantile_timelines,
            branching_state_probability_timelines,
        )

        timelines = branching_state_probability_timelines(branching)
        q = branching_numeric_quantile_timelines(
            branching, descriptor=descriptor, numeric_mapping=numeric_mapping, quantiles=quantiles
        )
        exp = branching_numeric_expectation_by_period(
            branching, descriptor=descriptor, numeric_mapping=numeric_mapping
        )

        periods = sorted(int(t) for t in q.keys())
        want_paths = True
        if ax_probs is None or ax_numeric is None or ax_paths is None:
            fig, (ax_probs, ax_numeric, ax_paths) = plt.subplots(
                3, 1, figsize=(11, 9), sharex=True
            )
            if title:
                fig.suptitle(title)

        DynamicVisualizer.plot_state_probability_bands(
            timelines=timelines,
            descriptor=descriptor,
            ax=ax_probs,
            title=f"State probability bands: {descriptor}",
            state_order=[s for s, _v in sorted({str(k): float(v) for k, v in numeric_mapping.items()}.items(), key=lambda kv: kv[1])],
        )

        exp_arr = np.array([float(exp[t]) for t in periods], dtype=float)
        q_low = np.array([float(q[t][0]) for t in periods], dtype=float)
        q_mid = np.array([float(q[t][1]) for t in periods], dtype=float)
        q_high = np.array([float(q[t][2]) for t in periods], dtype=float)

        ax_numeric.plot(periods, exp_arr, linewidth=2.0, label="Expected value")
        ax_numeric.plot(periods, q_mid, linewidth=1.5, linestyle="--", label="Median")
        ax_numeric.fill_between(periods, q_low, q_high, alpha=0.25, label="Quantile band")
        ax_numeric.set_ylabel(y_label)
        ax_numeric.set_xlabel("Period")
        ax_numeric.set_title("Numeric summary (expected value + band)")
        ax_numeric.legend(loc="best")
        ax_numeric.set_xticks(periods)
        ax_numeric.set_xlim(min(periods), max(periods))
        ax_probs.set_xticks(periods)

        # Top-path traces are plotted (deterministic step lines, alpha by weight).
        top_paths = list(getattr(branching, "top_paths", ()))
        scenarios_by_period = list(getattr(branching, "scenarios_by_period"))
        if not top_paths:
            ax_paths.text(0.5, 0.5, "No top_paths available", ha="center", va="center")
        else:
            top_paths = top_paths[: max(1, int(max_paths))]
            max_w = max(float(w) for _idxs, w in top_paths) if top_paths else 1.0
            for idxs, w in top_paths:
                idxs = list(map(int, idxs))
                y = [
                    float(numeric_mapping[scenarios_by_period[p_idx][idx].to_dict()[descriptor]])
                    for p_idx, idx in enumerate(idxs)
                ]
                alpha = 0.15 + 0.75 * (float(w) / max_w) ** 0.5
                ax_paths.step(periods, y, where="post", color="C0", alpha=alpha, linewidth=1.5)
        ax_paths.set_ylabel(y_label)
        ax_paths.set_title(f"Top-path step traces (first {min(len(top_paths), int(max_paths))} paths)")
        ax_paths.set_xticks(periods)
        ax_paths.set_xlim(min(periods), max(periods))

        return ax_probs, ax_numeric, ax_paths

    @staticmethod
    def plot_pathway_tree(
        *,
        periods: Sequence[int],
        scenarios_by_period: Sequence[Sequence[Scenario]],
        edges: Mapping[Tuple[int, int], Mapping[int, float]],
        top_paths: Optional[Sequence[Tuple[Sequence[int], float]]] = None,
        ax: Optional[plt.Axes] = None,
        title: str = "Branching pathway tree",
        min_edge_weight: float = 0.02,
        max_linewidth: float = 6.0,
        node_size: float = 60.0,
        label_max_chars: int = 38,
        key_descriptors: Optional[Sequence[str]] = None,
    ) -> plt.Axes:
        """
        Plot a simple branching pathway tree with edge thickness by weight.

        Args:
            periods: Period labels in chronological order.
            scenarios_by_period: A layered list of Scenario nodes per period.
            edges: Edge weights between consecutive layers as
                edges[(period_idx, src_idx)][tgt_idx] -> weight.
            min_edge_weight: Do not draw edges below this weight.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 4))

        if not periods:
            raise ValueError("periods cannot be empty")
        if len(periods) != len(scenarios_by_period):
            raise ValueError("periods and scenarios_by_period must have same length")

        # Optional pruning is performed: only nodes/edges that appear on top paths are shown.
        allowed_nodes: Optional[List[set[int]]] = None
        allowed_edges: Optional[set[Tuple[int, int, int]]] = None  # (p_idx, src_idx, tgt_idx)
        if top_paths is not None:
            allowed_nodes = [set() for _ in range(len(periods))]
            allowed_edges = set()
            for path_idxs, _w in top_paths:
                if len(path_idxs) != len(periods):
                    continue
                for p_idx, node_idx in enumerate(path_idxs):
                    allowed_nodes[p_idx].add(int(node_idx))
                for p_idx in range(len(periods) - 1):
                    allowed_edges.add((p_idx, int(path_idxs[p_idx]), int(path_idxs[p_idx + 1])))

        # Node coordinates are set: x by period index; y spread evenly per layer.
        xs = list(range(len(periods)))
        y_positions: List[List[float]] = []
        for layer in scenarios_by_period:
            n = max(1, len(layer))
            if n == 1:
                y_positions.append([0.0])
            else:
                y_positions.append(list(np.linspace(-1.0, 1.0, n)))

        def _label(s: Scenario) -> str:
            items = list(s.to_dict().items())
            if key_descriptors is not None:
                items = [(k, v) for k, v in items if k in set(key_descriptors)]
            # Keys are abbreviated to reduce clutter.
            parts = [f"{k[:3]}={v}" for k, v in items]
            txt = ",".join(parts)
            if len(txt) > int(label_max_chars):
                return txt[: int(label_max_chars) - 1] + "…"
            return txt

        # Edges are drawn.
        for p_idx in range(len(periods) - 1):
            layer = scenarios_by_period[p_idx]
            next_layer = scenarios_by_period[p_idx + 1]
            for src_idx in range(len(layer)):
                if allowed_nodes is not None and src_idx not in allowed_nodes[p_idx]:
                    continue
                out = edges.get((p_idx, src_idx), {})
                for tgt_idx, w in out.items():
                    w = float(w)
                    if w < float(min_edge_weight):
                        continue
                    if int(tgt_idx) < 0 or int(tgt_idx) >= len(next_layer):
                        continue
                    if allowed_edges is not None and (p_idx, src_idx, int(tgt_idx)) not in allowed_edges:
                        continue
                    lw = max(0.5, float(max_linewidth) * w)
                    ax.plot(
                        [xs[p_idx], xs[p_idx + 1]],
                        [y_positions[p_idx][src_idx], y_positions[p_idx + 1][int(tgt_idx)]],
                        color="C0",
                        alpha=min(0.9, 0.2 + 0.8 * w),
                        linewidth=lw,
                    )

        # Nodes and labels are drawn.
        for p_idx, layer in enumerate(scenarios_by_period):
            for i, s in enumerate(layer):
                if allowed_nodes is not None and i not in allowed_nodes[p_idx]:
                    continue
                ax.scatter(
                    [xs[p_idx]],
                    [y_positions[p_idx][i]],
                    s=float(node_size),
                    color="black",
                    alpha=0.8,
                )
                ax.text(
                    xs[p_idx] + 0.02,
                    y_positions[p_idx][i],
                    _label(s),
                    fontsize=8,
                    va="center",
                )

        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(t)) for t in periods])
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_xlabel("Period")
        return ax

    @staticmethod
    def scenario_network(
        scenarios: List[Scenario],
        matrix: CIBMatrix,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Create a network graph of scenario relationships.

        For small systems, creates a graph showing scenario transitions.
        For larger systems, creates a simplified visualization.

        Args:
            scenarios: List of scenarios to visualize.
            matrix: CIB matrix providing descriptor definitions.
            ax: Matplotlib axes to plot on. If None, creates new figure.

        Returns:
            Matplotlib axes object.

        Note:
            Full network graph implementation would require networkx.
            This is a simplified version for basic visualization.
        """
        try:
            return ScenarioVisualizer.scenario_network(
                scenarios=scenarios,
                matrix=matrix,
                ax=ax,
                layout="spring",
                show_transitions=False,
                succession_operator=None,
                edge_metric="hamming",
                max_edges_per_node=5,
            )
        except Exception:
            # The previous simplified visualization is preserved as a fallback.
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 8))

            positions: Dict[Scenario, Tuple[float, float]] = {}
            for idx, scenario in enumerate(scenarios):
                angle = 2 * np.pi * idx / max(1, len(scenarios))
                positions[scenario] = (float(np.cos(angle)), float(np.sin(angle)))

            for idx, scenario in enumerate(scenarios):
                x, y = positions[scenario]
                ax.scatter(x, y, s=100, alpha=0.6)
                ax.text(x, y + 0.1, f"S{idx}", ha="center", fontsize=8)

            ax.set_title(f"Scenario Network ({len(scenarios)} scenarios)")
            ax.set_aspect("equal")
            ax.axis("off")
            return ax
