"""
Transformation pathway representation and lightweight analysis utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.core import Scenario


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
        Return the pathway scenarios under the requested mode.

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


def pathway_frequencies(
    pathways: Sequence[TransformationPathway],
) -> Dict[Tuple[Tuple[int, Tuple[Tuple[str, str], ...]], ...], int]:
    """
    Count identical pathways (by period and scenario assignments).

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
    Compute per-period categorical probabilities: P(descriptor=state) across pathways.
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


def numeric_quantile_timelines(
    pathways: Sequence[TransformationPathway],
    *,
    descriptor: str,
    numeric_mapping: Mapping[str, float],
    quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95),
    scenario_mode: Literal["realized", "equilibrium"] = "realized",
) -> Dict[int, Tuple[float, float, float]]:
    """
    Compute numeric quantiles across time for a discrete descriptor.

    This supports fan-chart style reporting when (and only when) a descriptor's
    states have an explicit numeric mapping.

    Args:
        pathways: Ensemble of pathways (must share identical periods).
        descriptor: Descriptor name to summarize.
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
    Compute weighted quantiles for 1D data.
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
    Compute per-period node probabilities from a BranchingResult.

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
    Compute per-period categorical probabilities from a BranchingResult.
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


def branching_numeric_quantile_timelines(
    branching: "object",
    *,
    descriptor: str,
    numeric_mapping: Mapping[str, float],
    quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95),
) -> Dict[int, Tuple[float, float, float]]:
    """
    Compute weighted numeric quantiles per period from a BranchingResult.
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
    Compute weighted numeric expectation per period from a BranchingResult.
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

