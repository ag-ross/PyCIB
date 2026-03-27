"""
Transformation pathway representation and lightweight analysis utilities.

This module provides dataclasses for pathway output (including extended
disequilibrium-aware pathways), event and memory state types, and utilities
for state-probability timelines, quantile timelines, and pathway summaries.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix, Scenario


@dataclass(frozen=True)
class TransformationPathway:
    """
    A discrete scenario pathway across periods.
    """

    periods: Tuple[int, ...]
    scenarios: Tuple[Scenario, ...]
    equilibrium_scenarios: Optional[Tuple[Scenario, ...]] = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if len(self.periods) != len(self.scenarios):
            raise ValueError("periods and scenarios must have the same length")
        if self.equilibrium_scenarios is not None and len(self.periods) != len(
            self.equilibrium_scenarios
        ):
            raise ValueError(
                "periods and equilibrium_scenarios must have the same length"
            )

    def to_dicts(self) -> List[Dict[str, str]]:
        return [s.to_dict() for s in self.scenarios]

    def scenarios_for_mode(
        self, mode: Literal["realized", "equilibrium"]
    ) -> Tuple[Scenario, ...]:
        """
        The pathway scenarios under the requested mode are returned.

        Args:
            mode: "realized" returns `scenarios`. "equilibrium" returns
                `equilibrium_scenarios` if present.

        Returns:
            Tuple of scenarios aligned to `periods`.

        Raises:
            ValueError: If equilibrium scenarios are requested but not present.
        """
        if mode == "realized":
            return self.scenarios
        if self.equilibrium_scenarios is None:
            raise ValueError("equilibrium_scenarios is not present on this pathway")
        return self.equilibrium_scenarios


@dataclass(frozen=True)
class PerPeriodDisequilibriumMetrics:
    """
    Per-period disequilibrium diagnostics aligned to one realised scenario.

    When ``distance_to_consistent_set`` or ``distance_to_attractor`` is ``None``,
    inspect the matching ``*_error`` field for the exception summary if present.
    """

    period: int
    is_consistent: bool
    consistency_margin: float
    descriptor_margins: Dict[str, float]
    brink_descriptors: Tuple[str, ...]
    distance_to_equilibrium: Optional[float]
    time_to_equilibrium: Optional[int]
    entered_consistent_set: bool
    distance_to_consistent_set: Optional[float] = None
    distance_to_attractor: Optional[float] = None
    nearest_attractor_kind: Optional[str] = None
    attractor_size: Optional[int] = None
    is_on_attractor: Optional[bool] = None
    consistent_set_distance_error: Optional[str] = None
    attractor_distance_error: Optional[str] = None


@dataclass(frozen=True)
class ActiveMatrixState:
    """
    Provenance summary for the active matrix used in one period.
    """

    period: int
    regime_name: str
    base_matrix_id: str
    active_matrix_id: str
    applied_threshold_rules: Tuple[str, ...] = ()
    applied_structural_shocks: Tuple[str, ...] = ()
    applied_judgment_sampling: Tuple[str, ...] = ()
    adaptive_updates: Tuple[str, ...] = ()
    threshold_regime_reaffirmations: Tuple[str, ...] = ()
    entered_regime: bool = False
    regime_entry_period: Optional[int] = None
    regime_spell_index: int = 0
    diff_summary: Dict[str, float] = field(default_factory=dict)
    provenance_labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TransitionEvent:
    """
    One logged transition or provenance event.
    """

    period: int
    event_type: str
    label: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryState:
    """
    Serialisable path-dependent memory state for one period.
    """

    period: int
    values: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    export_label: str = "memory"


@dataclass(frozen=True)
class StructuralConsistencyState:
    """
    Structural consistency diagnostics for one period.
    """

    period: int
    is_structurally_consistent: bool
    violations: Tuple[str, ...] = ()
    summary: str = ""


@dataclass(frozen=True)
class PathDependentState:
    """
    Shared runtime state for path-dependent replay and explanation.
    """

    period: int
    scenario: Scenario
    regime_name: str
    active_matrix: CIBMatrix
    memory_state: Optional[MemoryState] = None
    history_signature: Tuple[Tuple[int, ...], ...] = ()
    transition_events: Tuple[TransitionEvent, ...] = ()


@dataclass(frozen=True)
class ExtendedTransformationPathway:
    """
    Rich pathway output for disequilibrium-aware dynamic workflows.
    """

    periods: Tuple[int, ...]
    realised_scenarios: Tuple[Scenario, ...]
    equilibrium_scenarios: Optional[Tuple[Scenario, ...]] = None
    extension_mode: str = "transient"
    disequilibrium_metrics: Tuple[PerPeriodDisequilibriumMetrics, ...] = ()
    active_regimes: Tuple[str, ...] = ()
    active_matrices: Tuple[ActiveMatrixState, ...] = ()
    transition_events: Tuple[TransitionEvent, ...] = ()
    memory_states: Tuple[MemoryState, ...] = ()
    structural_consistency: Tuple[StructuralConsistencyState, ...] = ()
    diagnostics: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_periods = len(self.periods)
        if n_periods != len(self.realised_scenarios):
            raise ValueError("periods and realised_scenarios must have the same length")
        if self.equilibrium_scenarios is not None and len(self.equilibrium_scenarios) != n_periods:
            raise ValueError("equilibrium_scenarios must align to periods")
        if self.disequilibrium_metrics and len(self.disequilibrium_metrics) != n_periods:
            raise ValueError("disequilibrium_metrics must align to periods")
        if self.active_regimes and len(self.active_regimes) != n_periods:
            raise ValueError("active_regimes must align to periods")
        if self.active_matrices and len(self.active_matrices) != n_periods:
            raise ValueError("active_matrices must align to periods")
        if self.memory_states and len(self.memory_states) != n_periods:
            raise ValueError("memory_states must align to periods")
        if self.structural_consistency and len(self.structural_consistency) != n_periods:
            raise ValueError("structural_consistency must align to periods")

    @property
    def scenarios(self) -> Tuple[Scenario, ...]:
        """
        Backward-compatible alias for realised scenarios.
        """

        return self.realised_scenarios

    def to_dicts(self) -> List[Dict[str, str]]:
        return [s.to_dict() for s in self.realised_scenarios]

    def scenarios_for_mode(
        self, mode: Literal["realized", "equilibrium"]
    ) -> Tuple[Scenario, ...]:
        if mode == "realized":
            return self.realised_scenarios
        if self.equilibrium_scenarios is None:
            raise ValueError("equilibrium_scenarios is not present on this pathway")
        return self.equilibrium_scenarios

    def to_serializable_dict(self) -> Dict[str, Any]:
        """
        A JSON-serialisable representation of the pathway is returned.
        """

        return {
            "periods": [int(t) for t in self.periods],
            "realised_scenarios": [s.to_dict() for s in self.realised_scenarios],
            "equilibrium_scenarios": (
                [s.to_dict() for s in self.equilibrium_scenarios]
                if self.equilibrium_scenarios is not None
                else None
            ),
            "extension_mode": str(self.extension_mode),
            "disequilibrium_metrics": [asdict(m) for m in self.disequilibrium_metrics],
            "active_regimes": list(self.active_regimes),
            "active_matrices": [asdict(m) for m in self.active_matrices],
            "transition_events": [asdict(e) for e in self.transition_events],
            "memory_states": [asdict(m) for m in self.memory_states],
            "structural_consistency": [asdict(s) for s in self.structural_consistency],
            "diagnostics": {
                str(k): list(v) if isinstance(v, tuple) else v for k, v in self.diagnostics.items()
            },
        }


def pathway_frequencies(
    pathways: Sequence[TransformationPathway],
) -> Dict[Tuple[Tuple[int, Tuple[Tuple[str, str], ...]], ...], int]:
    """
    Identical pathways (by period and scenario assignments) are counted.

    Returns:
        A dict mapping a hashable pathway signature to its count.
    """
    freqs: Dict[Tuple[Tuple[int, Tuple[Tuple[str, str], ...]], ...], int] = {}
    for p in pathways:
        sig: List[Tuple[int, Tuple[Tuple[str, str], ...]]] = []
        for t, s in zip(p.periods, p.scenarios):
            sig.append((int(t), tuple(s.to_dict().items())))
        key = tuple(sig)
        freqs[key] = freqs.get(key, 0) + 1
    return freqs


def state_probability_timelines(
    pathways: Sequence[TransformationPathway],
    *,
    scenario_mode: Literal["realized", "equilibrium"] = "realized",
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Per-period categorical probabilities P(descriptor=state) across pathways are computed.
    """
    if not pathways:
        raise ValueError("pathways cannot be empty")

    periods = pathways[0].periods
    for p in pathways:
        if p.periods != periods:
            raise ValueError("All pathways must share the same periods for aggregation")

    n = float(len(pathways))
    out: Dict[int, Dict[str, Dict[str, float]]] = {}

    for idx, t in enumerate(periods):
        counts: Dict[str, Dict[str, int]] = {}
        for p in pathways:
            sd = p.scenarios_for_mode(scenario_mode)[idx].to_dict()
            for d, state in sd.items():
                counts.setdefault(d, {})
                counts[d][state] = counts[d].get(state, 0) + 1

        out[int(t)] = {d: {s: c / n for s, c in row.items()} for d, row in counts.items()}

    return out


def summarize_disequilibrium_path(
    pathway: ExtendedTransformationPathway,
) -> Dict[str, Any]:
    """
    One disequilibrium-aware pathway is summarised without mutating it.
    """

    if not pathway.disequilibrium_metrics:
        raise ValueError("pathway does not contain disequilibrium_metrics")

    from cib.scoring import cumulative_disequilibrium_burden

    metrics = tuple(pathway.disequilibrium_metrics)
    margins = np.asarray([float(m.consistency_margin) for m in metrics], dtype=float)
    inconsistent = [m for m in metrics if not bool(m.is_consistent)]
    consistent_set_distances = [
        float(
            m.distance_to_consistent_set
            if m.distance_to_consistent_set is not None
            else m.distance_to_equilibrium
        )
        for m in metrics
        if (
            m.distance_to_consistent_set is not None
            or m.distance_to_equilibrium is not None
        )
    ]
    attractor_distances = [
        float(m.distance_to_attractor)
        for m in metrics
        if m.distance_to_attractor is not None
    ]
    first_consistent_period = next(
        (int(m.period) for m in metrics if bool(m.is_consistent)),
        None,
    )
    time_to_equilibrium = next(
        (int(m.time_to_equilibrium) for m in metrics if m.time_to_equilibrium is not None),
        None,
    )
    brink_counter: Counter[str] = Counter()
    attractor_kind_counter: Counter[str] = Counter()
    for metric in metrics:
        brink_counter.update(metric.brink_descriptors)
        if metric.nearest_attractor_kind is not None:
            attractor_kind_counter.update((str(metric.nearest_attractor_kind),))

    return {
        "extension_mode": str(pathway.extension_mode),
        "fraction_periods_inconsistent": float(len(inconsistent)) / float(len(metrics)),
        "min_consistency_margin": float(np.min(margins)),
        "mean_consistency_margin": float(np.mean(margins)),
        "first_consistent_period": first_consistent_period,
        "time_to_equilibrium": time_to_equilibrium,
        "mean_distance_to_consistent_set": (
            float(np.mean(np.asarray(consistent_set_distances, dtype=float)))
            if consistent_set_distances
            else None
        ),
        "mean_distance_to_attractor": (
            float(np.mean(np.asarray(attractor_distances, dtype=float)))
            if attractor_distances
            else None
        ),
        "attractor_kinds_observed": tuple(
            kind for kind, _count in attractor_kind_counter.most_common()
        ),
        "cumulative_disequilibrium_burden": float(cumulative_disequilibrium_burden(metrics)),
        "descriptors_most_frequently_contributing": tuple(
            descriptor for descriptor, _count in brink_counter.most_common()
        ),
    }


def summarize_disequilibrium_ensemble(
    pathways: Sequence[ExtendedTransformationPathway],
) -> Dict[str, Any]:
    """
    A disequilibrium-aware ensemble is summarised without mutating input pathways.
    """

    if not pathways:
        raise ValueError("pathways cannot be empty")

    summaries = [summarize_disequilibrium_path(pathway) for pathway in pathways]
    burden = np.asarray(
        [float(summary["cumulative_disequilibrium_burden"]) for summary in summaries],
        dtype=float,
    )
    inconsistent_fraction = np.asarray(
        [float(summary["fraction_periods_inconsistent"]) for summary in summaries],
        dtype=float,
    )
    margin_means = np.asarray(
        [float(summary["mean_consistency_margin"]) for summary in summaries],
        dtype=float,
    )
    margin_mins = np.asarray(
        [float(summary["min_consistency_margin"]) for summary in summaries],
        dtype=float,
    )
    first_consistent = [
        int(summary["first_consistent_period"])
        for summary in summaries
        if summary["first_consistent_period"] is not None
    ]
    time_to_equilibrium = [
        int(summary["time_to_equilibrium"])
        for summary in summaries
        if summary["time_to_equilibrium"] is not None
    ]
    descriptor_counter: Counter[str] = Counter()
    for summary in summaries:
        descriptor_counter.update(summary["descriptors_most_frequently_contributing"])

    return {
        "n_pathways": int(len(pathways)),
        "mean_fraction_periods_inconsistent": float(np.mean(inconsistent_fraction)),
        "mean_consistency_margin": float(np.mean(margin_means)),
        "minimum_consistency_margin": float(np.min(margin_mins)),
        "mean_cumulative_disequilibrium_burden": float(np.mean(burden)),
        "first_consistent_period_distribution": tuple(sorted(first_consistent)),
        "time_to_equilibrium_distribution": tuple(sorted(time_to_equilibrium)),
        "descriptors_most_frequently_contributing": tuple(
            descriptor for descriptor, _count in descriptor_counter.most_common()
        ),
    }


def numeric_quantile_timelines(
    pathways: Sequence[TransformationPathway],
    *,
    descriptor: str,
    numeric_mapping: Mapping[str, float],
    quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95),
    scenario_mode: Literal["realized", "equilibrium"] = "realized",
) -> Dict[int, Tuple[float, float, float]]:
    """
    Numeric quantiles across time for a discrete descriptor are computed.

    Fan-chart style reporting is supported when (and only when) a descriptor's
    states have an explicit numeric mapping.

    Args:
        pathways: Ensemble of pathways (must share identical periods).
        descriptor: Descriptor name to summarise.
        numeric_mapping: Mapping from state label -> numeric value.
        quantiles: (low, mid, high) quantiles; defaults to (0.05, 0.5, 0.95).

    Returns:
        Dict mapping period -> (q_low, q_mid, q_high).
    """
    if not pathways:
        raise ValueError("pathways cannot be empty")

    q_low, q_mid, q_high = (float(quantiles[0]), float(quantiles[1]), float(quantiles[2]))
    if not (0.0 <= q_low <= q_mid <= q_high <= 1.0):
        raise ValueError("quantiles must satisfy 0 <= q_low <= q_mid <= q_high <= 1")

    periods = pathways[0].periods
    for p in pathways:
        if p.periods != periods:
            raise ValueError("All pathways must share the same periods for aggregation")

    out: Dict[int, Tuple[float, float, float]] = {}
    for idx, t in enumerate(periods):
        vals: List[float] = []
        for p in pathways:
            state = p.scenarios_for_mode(scenario_mode)[idx].to_dict().get(descriptor)
            if state is None:
                raise ValueError(f"Descriptor {descriptor!r} missing from scenario")
            if state not in numeric_mapping:
                raise ValueError(
                    f"State {state!r} missing from numeric_mapping for {descriptor!r}"
                )
            vals.append(float(numeric_mapping[state]))

        arr = np.asarray(vals, dtype=float)
        q = np.quantile(arr, [q_low, q_mid, q_high])
        out[int(t)] = (float(q[0]), float(q[1]), float(q[2]))

    return out


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, qs: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Weighted quantiles for 1D data are computed.
    """
    if values.ndim != 1 or weights.ndim != 1 or values.shape[0] != weights.shape[0]:
        raise ValueError("values and weights must be 1D arrays of the same length")
    if values.shape[0] == 0:
        raise ValueError("values cannot be empty")

    q_low, q_mid, q_high = (float(qs[0]), float(qs[1]), float(qs[2]))
    if not (0.0 <= q_low <= q_mid <= q_high <= 1.0):
        raise ValueError("quantiles must satisfy 0 <= q_low <= q_mid <= q_high <= 1")

    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("weights must sum to a positive value")
    w = w / w_sum

    cw = np.cumsum(w)
    # Linear interpolation is used on the weighted CDF.
    def _q(q: float) -> float:
        idx = int(np.searchsorted(cw, q, side="left"))
        idx = max(0, min(idx, v.shape[0] - 1))
        return float(v[idx])

    return (_q(q_low), _q(q_mid), _q(q_high))


def branching_node_probabilities(
    branching: "object",
) -> Dict[int, Dict[int, float]]:
    """
    Per-period node probabilities are computed from a BranchingResult.

    Returns:
        Dict mapping period_idx -> {node_idx -> probability}.
    """
    periods = list(getattr(branching, "periods"))
    edges = getattr(branching, "edges")
    scenarios_by_period = getattr(branching, "scenarios_by_period")
    if not periods:
        raise ValueError("branching.periods cannot be empty")
    if len(periods) != len(scenarios_by_period):
        raise ValueError("branching.periods and branching.scenarios_by_period must align")

    dist: Dict[int, Dict[int, float]] = {0: {0: 1.0}}
    for p_idx in range(len(periods) - 1):
        cur = dist.get(p_idx, {})
        nxt: Dict[int, float] = {}
        for src_idx, p_src in cur.items():
            out = edges.get((p_idx, int(src_idx)), {})
            for tgt_idx, w in out.items():
                nxt[int(tgt_idx)] = nxt.get(int(tgt_idx), 0.0) + float(p_src) * float(w)
        dist[p_idx + 1] = nxt
    return dist


def branching_state_probability_timelines(
    branching: "object",
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Per-period categorical probabilities are computed from a BranchingResult.
    """
    periods = list(getattr(branching, "periods"))
    scenarios_by_period = list(getattr(branching, "scenarios_by_period"))
    node_p = branching_node_probabilities(branching)

    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for p_idx, t in enumerate(periods):
        counts: Dict[str, Dict[str, float]] = {}
        for node_idx, prob in node_p.get(p_idx, {}).items():
            s = scenarios_by_period[p_idx][int(node_idx)]
            sd = s.to_dict()
            for d, state in sd.items():
                counts.setdefault(d, {})
                counts[d][state] = counts[d].get(state, 0.0) + float(prob)
        out[int(t)] = counts
    return out


def branching_regime_residence_timelines(
    branching: "object",
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Weighted regime-residence probabilities are computed from a BranchingResult.
    """
    periods = list(getattr(branching, "periods"))
    regime_states_by_period = list(getattr(branching, "regime_states_by_period", ()))
    active_regimes = list(getattr(branching, "active_regimes", ()))
    node_p = branching_node_probabilities(branching)

    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for p_idx, t in enumerate(periods):
        regime_counts: Dict[str, float] = {}
        entry_counts: Dict[str, float] = {}
        reaffirmation_counts: Dict[str, float] = {}
        node_probabilities = node_p.get(p_idx, {})
        for node_idx, prob in node_probabilities.items():
            if regime_states_by_period:
                regime_state = regime_states_by_period[p_idx][int(node_idx)]
                regime_name = str(getattr(regime_state, "regime_name"))
                if bool(getattr(regime_state, "entered_regime")):
                    entry_counts[regime_name] = (
                        entry_counts.get(regime_name, 0.0) + float(prob)
                    )
                for label in getattr(
                    regime_state, "threshold_regime_reaffirmations", ()
                ):
                    reaffirmation_counts[str(label)] = (
                        reaffirmation_counts.get(str(label), 0.0) + float(prob)
                    )
            else:
                regime_name = str(active_regimes[p_idx][int(node_idx)])
            regime_counts[regime_name] = regime_counts.get(regime_name, 0.0) + float(prob)
        out[int(t)] = {
            "active_regimes": regime_counts,
            "entered_regime": entry_counts,
            "threshold_regime_reaffirmations": reaffirmation_counts,
        }
    return out


def branching_numeric_quantile_timelines(
    branching: "object",
    *,
    descriptor: str,
    numeric_mapping: Mapping[str, float],
    quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95),
) -> Dict[int, Tuple[float, float, float]]:
    """
    Weighted numeric quantiles per period are computed from a BranchingResult.
    """
    periods = list(getattr(branching, "periods"))
    scenarios_by_period = list(getattr(branching, "scenarios_by_period"))
    node_p = branching_node_probabilities(branching)

    out: Dict[int, Tuple[float, float, float]] = {}
    for p_idx, t in enumerate(periods):
        vals: List[float] = []
        ws: List[float] = []
        for node_idx, prob in node_p.get(p_idx, {}).items():
            s = scenarios_by_period[p_idx][int(node_idx)]
            state = s.to_dict().get(descriptor)
            if state is None:
                raise ValueError(f"Descriptor {descriptor!r} missing from scenario")
            if state not in numeric_mapping:
                raise ValueError(
                    f"State {state!r} missing from numeric_mapping for {descriptor!r}"
                )
            vals.append(float(numeric_mapping[state]))
            ws.append(float(prob))

        out[int(t)] = _weighted_quantiles(
            np.asarray(vals, dtype=float), np.asarray(ws, dtype=float), quantiles
        )
    return out


def branching_numeric_expectation_by_period(
    branching: "object",
    *,
    descriptor: str,
    numeric_mapping: Mapping[str, float],
) -> Dict[int, float]:
    """
    Weighted numeric expectation per period is computed from a BranchingResult.
    """
    periods = list(getattr(branching, "periods"))
    scenarios_by_period = list(getattr(branching, "scenarios_by_period"))
    node_p = branching_node_probabilities(branching)

    out: Dict[int, float] = {}
    for p_idx, t in enumerate(periods):
        total = 0.0
        for node_idx, prob in node_p.get(p_idx, {}).items():
            s = scenarios_by_period[p_idx][int(node_idx)]
            state = s.to_dict().get(descriptor)
            if state is None:
                raise ValueError(f"Descriptor {descriptor!r} missing from scenario")
            if state not in numeric_mapping:
                raise ValueError(
                    f"State {state!r} missing from numeric_mapping for {descriptor!r}"
                )
            total += float(prob) * float(numeric_mapping[state])
        out[int(t)] = float(total)
    return out

