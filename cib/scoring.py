"""
Scenario diagnostics and scoring utilities for CIB.

This module provides a small, explicit “scenario quality” surface:
  - consistency (binary)
  - detailed inconsistency diagnostics
  - margin-to-inconsistency (brink detector)
  - total impact score (sum of chosen-state impact scores)
  - qualitative labels for numeric impacts (hindering/promoting)
"""

from __future__ import annotations

from itertools import product
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cib.core import CIBMatrix, ConsistencyChecker, ImpactBalance, Scenario
from cib.succession import GlobalSuccession, SuccessionOperator


@dataclass(frozen=True)
class ScenarioDiagnostics:
    is_consistent: bool
    chosen_states: Dict[str, str]
    balances: Dict[str, Dict[str, float]]
    inconsistencies: List[Dict[str, object]]
    # Minimum over descriptors of (chosen_score - best_alternative_score).
    # Negative values indicate inconsistency; 0 indicates a tie at the maximum.
    consistency_margin: float
    # Sum of impact scores of chosen states across descriptors.
    total_impact_score: float

    def brink_descriptors(self, threshold: float = 0.0) -> List[str]:
        """
        Descriptors at or below the margin threshold are returned.

        This is useful for identifying descriptors that are on the brink of switching.
        """
        threshold = float(threshold)
        brink: List[str] = []
        for d, bal in self.balances.items():
            chosen_state = self.chosen_states.get(d)
            if chosen_state is None:
                continue
            chosen = bal.get(chosen_state)
            if chosen is None:
                continue
            # The best alternative is used (chosen state is excluded).
            best_alt = float("-inf")
            for s, v in bal.items():
                if str(s) == str(chosen_state):
                    continue
                best_alt = max(best_alt, float(v))
            margin = 0.0 if best_alt == float("-inf") else float(chosen) - float(best_alt)
            if margin <= threshold:
                brink.append(d)
        return brink


@dataclass(frozen=True)
class AttractorDiagnostics:
    """
    Relationship between a scenario and its nearest local attractor.
    """

    distance_to_attractor: float
    attractor_kind: str
    attractor_size: int
    is_on_attractor: bool


def scenario_diagnostics(
    scenario: Scenario,
    matrix: CIBMatrix,
    *,
    float_atol: float = 1e-08,
    float_rtol: float = 1e-05,
) -> ScenarioDiagnostics:
    """
    Standard CIB diagnostics for a scenario are computed.

    Args:
        scenario: Scenario to analyse.
        matrix: CIB matrix containing impact relationships.

    Returns:
        ScenarioDiagnostics object containing consistency status, impact
        balances, margins, and total impact score.

    Raises:
        ValueError: If scenario or matrix contain invalid descriptor or
            state references.
    """
    detail = ConsistencyChecker.check_consistency_detailed(
        scenario,
        matrix,
        float_atol=float(float_atol),
        float_rtol=float(float_rtol),
    )
    balances = detail["balances"]
    inconsistencies = detail["inconsistencies"]
    is_consistent = bool(detail["is_consistent"])

    ib = ImpactBalance(scenario, matrix)
    total = 0.0
    margins: List[float] = []
    chosen_states = {}
    for d, states in matrix.descriptors.items():
        chosen_state = scenario.get_state(d)
        chosen_states[d] = chosen_state
        chosen_score = float(ib.get_score(d, chosen_state))
        total += chosen_score
        # Margin to switching is defined against the best alternative state.
        # This makes the margin strictly positive for strongly-consistent descriptors,
        # 0.0 for ties at the maximum, and negative for inconsistent descriptors.
        best_alt = float("-inf")
        for s in states:
            if s == chosen_state:
                continue
            best_alt = max(best_alt, float(ib.get_score(d, s)))
        if best_alt == float("-inf"):
            # Degenerate descriptor (single state): no alternative exists.
            margins.append(0.0)
        else:
            margins.append(chosen_score - best_alt)

    margin = float(min(margins)) if margins else 0.0
    diag = ScenarioDiagnostics(
        is_consistent=is_consistent,
        chosen_states=chosen_states,
        balances={k: {s: float(v) for s, v in row.items()} for k, row in balances.items()},
        inconsistencies=list(inconsistencies),
        consistency_margin=margin,
        total_impact_score=float(total),
    )
    return diag


def descriptor_disequilibrium_contributions(
    scenario: Scenario,
    matrix: CIBMatrix,
    *,
    top_k_sources: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Descriptor-level disequilibrium contributions for one scenario are computed.

    The returned structure is intentionally lightweight and serialisable so it
    can be embedded directly in pathway summaries or diagnostics payloads.
    """

    from cib.attribution import attribute_scenario

    attribution = attribute_scenario(
        scenario, matrix, top_k_sources=top_k_sources
    )
    out: Dict[str, Dict[str, Any]] = {}
    for item in attribution.per_descriptor:
        out[item.target_descriptor] = {
            "chosen_state": str(item.chosen_state),
            "alternative_state": str(item.alternative_state),
            "margin_to_switch": float(item.margin_to_switch),
            "source_deltas": {
                str(contribution.src_descriptor): float(contribution.delta)
                for contribution in item.contributions
            },
        }
    return out


def _hamming_distance(a: Scenario, b: Scenario) -> float:
    return float(sum(x != y for x, y in zip(a.to_indices(), b.to_indices())))


def _scenario_space_size(matrix: CIBMatrix) -> int:
    space_size = 1
    for states in matrix.descriptors.values():
        space_size *= int(len(states))
    return int(space_size)


def attractor_distance(
    scenario: Scenario,
    matrix: CIBMatrix,
    *,
    max_exact_states: int = 4096,
    fallback: str = "trace_attractor",
    succession_operator: Optional[SuccessionOperator] = None,
    max_iterations: int = 1000,
) -> AttractorDiagnostics:
    """
    The nearest-attractor relationship for a scenario is computed.

    The returned diagnostics are intentionally distinct from distance to the
    consistent set. A scenario may be on a cycle attractor and still be
    inconsistent under the local CIB criterion.
    """

    if succession_operator is None:
        succession_operator = GlobalSuccession()

    if _scenario_space_size(matrix) <= int(max_exact_states):
        best_distance = float("inf")
        best_kind = "fixed_point"
        best_size = 1
        desc_names = list(matrix.descriptors.keys())
        state_lists = [matrix.descriptors[name] for name in desc_names]
        visited_attractors: set[Tuple[Tuple[int, ...], ...]] = set()
        for values in product(*state_lists):
            candidate = Scenario(dict(zip(desc_names, values)), matrix)
            result = succession_operator.find_attractor(
                candidate, matrix, max_iterations=max_iterations
            )
            if result.is_cycle:
                cycle = result.attractor
                if not isinstance(cycle, list):
                    raise TypeError("cycle attractor must be a list of scenarios")
                signature = tuple(sorted(tuple(member.to_indices()) for member in cycle))
                if signature in visited_attractors:
                    continue
                visited_attractors.add(signature)
                distance = min(
                    _hamming_distance(scenario, member) for member in cycle
                )
                kind = "cycle"
                size = len(cycle)
            else:
                attractor = result.attractor
                if not isinstance(attractor, Scenario):
                    raise TypeError("fixed-point attractor must be a Scenario")
                signature = (tuple(attractor.to_indices()),)
                if signature in visited_attractors:
                    continue
                visited_attractors.add(signature)
                distance = _hamming_distance(scenario, attractor)
                kind = "fixed_point"
                size = 1
            if (
                distance < best_distance
                or (
                    distance == best_distance
                    and (best_kind != "fixed_point" and kind == "fixed_point")
                )
                or (
                    distance == best_distance
                    and kind == best_kind
                    and size < best_size
                )
            ):
                best_distance = float(distance)
                best_kind = str(kind)
                best_size = int(size)
        if best_distance < float("inf"):
            return AttractorDiagnostics(
                distance_to_attractor=float(best_distance),
                attractor_kind=str(best_kind),
                attractor_size=int(best_size),
                is_on_attractor=bool(best_distance == 0.0),
            )

    if fallback == "none":
        raise ValueError("Exact attractor search is infeasible under current settings")
    if fallback != "trace_attractor":
        raise ValueError("fallback must be 'trace_attractor' or 'none'")

    result = succession_operator.find_attractor(
        scenario, matrix, max_iterations=max_iterations
    )
    if result.is_cycle:
        cycle = result.attractor
        if not isinstance(cycle, list):
            raise TypeError("cycle attractor must be a list of scenarios")
        distance = min(_hamming_distance(scenario, member) for member in cycle)
        return AttractorDiagnostics(
            distance_to_attractor=float(distance),
            attractor_kind="cycle",
            attractor_size=len(cycle),
            is_on_attractor=bool(distance == 0.0),
        )
    attractor = result.attractor
    if not isinstance(attractor, Scenario):
        raise TypeError("fixed-point attractor must be a Scenario")
    distance = _hamming_distance(scenario, attractor)
    return AttractorDiagnostics(
        distance_to_attractor=float(distance),
        attractor_kind="fixed_point",
        attractor_size=1,
        is_on_attractor=bool(distance == 0.0),
    )


def equilibrium_distance(
    scenario: Scenario,
    matrix: CIBMatrix,
    *,
    max_exact_states: int = 4096,
    fallback: str = "trace_attractor",
    succession_operator: Optional[SuccessionOperator] = None,
    max_iterations: int = 1000,
) -> float:
    """
    A distance from a scenario to a local attractor under `matrix` is computed.

    Args:
        scenario: Scenario to analyse.
        matrix: Active CIB matrix.
        max_exact_states: Maximum number of scenarios allowed for exact search.
        fallback: Fallback strategy used when exact search is infeasible.
            `"trace_attractor"`: the Hamming distance to the attractor
            reached from the supplied scenario is computed. `"none"`:
            an error is raised.
        succession_operator: Optional succession operator for fallback tracing.
        max_iterations: Maximum iterations used by the fallback trace.

    Returns:
        Hamming distance to the nearest exact or fallback attractor.

    Raises:
        ValueError: If the fallback mode is unsupported.
    """
    return float(
        attractor_distance(
            scenario,
            matrix,
            max_exact_states=max_exact_states,
            fallback=fallback,
            succession_operator=succession_operator,
            max_iterations=max_iterations,
        ).distance_to_attractor
    )


def consistent_set_distance(
    scenario: Scenario,
    matrix: CIBMatrix,
    *,
    max_exact_states: int = 4096,
    fallback: str = "trace_consistent_entry",
    succession_operator: Optional[SuccessionOperator] = None,
    max_iterations: int = 1000,
) -> float:
    """
    A distance from a scenario to the local consistent set under `matrix` is computed.

    Exact search is performed over the full scenario space when feasible. When
    the search space is too large, a configurable fallback is used.
    """
    if succession_operator is None:
        succession_operator = GlobalSuccession()

    if _scenario_space_size(matrix) <= int(max_exact_states):
        best = float("inf")
        desc_names = list(matrix.descriptors.keys())
        state_lists = [matrix.descriptors[name] for name in desc_names]
        for values in product(*state_lists):
            candidate = Scenario(dict(zip(desc_names, values)), matrix)
            if not ConsistencyChecker.check_consistency(candidate, matrix):
                continue
            best = min(best, _hamming_distance(scenario, candidate))
        if best < float("inf"):
            return float(best)

    if fallback == "none":
        raise ValueError("Exact equilibrium search is infeasible under current settings")
    if fallback != "trace_consistent_entry":
        raise ValueError(
            "fallback must be 'trace_consistent_entry' or 'none'"
        )

    current = scenario
    if ConsistencyChecker.check_consistency(current, matrix):
        return 0.0
    for _step in range(1, int(max_iterations) + 1):
        current = succession_operator.find_successor(current, matrix)
        if ConsistencyChecker.check_consistency(current, matrix):
            return float(_hamming_distance(scenario, current))
    raise ValueError(
        "A consistent state was not reached under the supplied fallback trace"
    )


def cumulative_disequilibrium_burden(
    items: Sequence[object],
) -> float:
    """
    Aggregate negative margins across a path as a non-negative burden score.

    The function accepts either a sequence of pathway metric objects exposing a
    `consistency_margin` attribute or a sequence of raw margin values.
    """

    burden = 0.0
    for item in items:
        if hasattr(item, "consistency_margin"):
            margin = float(getattr(item, "consistency_margin"))
        else:
            margin = float(item)  # type: ignore[arg-type]
        burden += max(0.0, -margin)
    return float(burden)


def impact_label(
    value: float,
    *,
    weak_threshold: float = 0.5,
    strong_threshold: float = 1.5,
) -> str:
    """
    A numeric impact value is mapped to a qualitative label.

    Args:
        value: Numeric impact value to label.
        weak_threshold: Threshold for weak impact classification.
        strong_threshold: Threshold for strong impact classification.

    Returns:
        Qualitative label: "strongly_hindering", "hindering", "neutral",
        "promoting", or "strongly_promoting".

    Raises:
        ValueError: If weak_threshold <= 0 or strong_threshold <= weak_threshold.
    """
    v = float(value)
    weak = float(weak_threshold)
    strong = float(strong_threshold)
    if strong <= weak or weak <= 0:
        raise ValueError("Require 0 < weak_threshold < strong_threshold")

    if v <= -strong:
        return "strongly_hindering"
    if v <= -weak:
        return "hindering"
    if v < weak:
        return "neutral"
    if v < strong:
        return "promoting"
    return "strongly_promoting"


def judgment_section_labels(
    matrix: CIBMatrix,
    *,
    src_desc: str,
    tgt_desc: str,
    weak_threshold: float = 0.5,
    strong_threshold: float = 1.5,
) -> Dict[Tuple[str, str], str]:
    """
    A judgement section (src_desc -> tgt_desc) is labelled with qualitative labels.

    Args:
        matrix: CIB matrix containing impact relationships.
        src_desc: Source descriptor name.
        tgt_desc: Target descriptor name.
        weak_threshold: Threshold for weak impact classification.
        strong_threshold: Threshold for strong impact classification.

    Returns:
        Dictionary mapping (src_state, tgt_state) tuples to qualitative
        labels.

    Raises:
        ValueError: If src_desc == tgt_desc (diagonal sections omitted), or
            if threshold parameters are invalid.
    """
    if src_desc == tgt_desc:
        raise ValueError("Diagonal sections are omitted by convention")
    out: Dict[Tuple[str, str], str] = {}
    for src_state in matrix.descriptors[src_desc]:
        for tgt_state in matrix.descriptors[tgt_desc]:
            v = matrix.get_impact(src_desc, src_state, tgt_desc, tgt_state)
            out[(src_state, tgt_state)] = impact_label(
                v, weak_threshold=weak_threshold, strong_threshold=strong_threshold
            )
    return out

