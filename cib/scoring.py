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

from dataclasses import dataclass
from typing import Dict, List, Tuple

from cib.core import CIBMatrix, ConsistencyChecker, ImpactBalance, Scenario


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
        Return descriptors at or below the margin threshold.

        This is useful for identifying descriptors “on the brink” of switching.
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
            best_alt = max(float(v) for v in bal.values())
            margin = float(chosen) - float(best_alt)
            if margin <= threshold:
                brink.append(d)
        return brink


def scenario_diagnostics(
    scenario: Scenario, matrix: CIBMatrix
) -> ScenarioDiagnostics:
    """
    Compute standard CIB diagnostics for a scenario.

    Args:
        scenario: Scenario to analyze.
        matrix: CIB matrix containing impact relationships.

    Returns:
        ScenarioDiagnostics object containing consistency status, impact
        balances, margins, and total impact score.

    Raises:
        ValueError: If scenario or matrix contain invalid descriptor or
            state references.
    """
    detail = ConsistencyChecker.check_consistency_detailed(scenario, matrix)
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
        best = max(float(ib.get_score(d, s)) for s in states)
        margins.append(chosen_score - best)

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


def impact_label(
    value: float,
    *,
    weak_threshold: float = 0.5,
    strong_threshold: float = 1.5,
) -> str:
    """
    Map a numeric impact value to a qualitative label.

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
    Label a judgment section (src_desc -> tgt_desc) with qualitative labels.

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

