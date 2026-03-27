"""
Unit tests for disequilibrium metrics and extended pathway summaries.

These tests validate PerPeriodDisequilibriumMetrics, path/ensemble summaries,
and descriptor-level disequilibrium contributions and burden.
"""

from __future__ import annotations

from cib.core import CIBMatrix, Scenario
from cib.pathway import (
    ActiveMatrixState,
    ExtendedTransformationPathway,
    PerPeriodDisequilibriumMetrics,
    summarize_disequilibrium_ensemble,
    summarize_disequilibrium_path,
)
from cib.scoring import (
    attractor_distance,
    consistent_set_distance,
    cumulative_disequilibrium_burden,
    descriptor_disequilibrium_contributions,
    equilibrium_distance,
)


def _simple_matrix() -> CIBMatrix:
    desc = {"A": ["Low", "High"], "B": ["Low", "High"]}
    m = CIBMatrix(desc)
    m.set_impact("A", "Low", "B", "Low", 2.0)
    m.set_impact("A", "Low", "B", "High", -2.0)
    m.set_impact("A", "High", "B", "Low", -2.0)
    m.set_impact("A", "High", "B", "High", 2.0)
    m.set_impact("B", "Low", "A", "Low", 2.0)
    m.set_impact("B", "Low", "A", "High", -2.0)
    m.set_impact("B", "High", "A", "Low", -2.0)
    m.set_impact("B", "High", "A", "High", 2.0)
    return m


def _cycle_and_fixed_point_matrix() -> CIBMatrix:
    desc = {"A": ["Low", "High"], "B": ["Low", "High"], "C": ["Low", "High"]}
    m = CIBMatrix(desc)

    impacts = {
        ("A", "Low", "B", "Low"): -2.0,
        ("A", "Low", "B", "High"): 2.0,
        ("A", "High", "B", "Low"): 1.0,
        ("A", "High", "B", "High"): 2.0,
        ("A", "Low", "C", "High"): 2.0,
        ("A", "High", "C", "Low"): -1.0,
        ("A", "High", "C", "High"): -1.0,
        ("B", "Low", "A", "Low"): 2.0,
        ("B", "Low", "A", "High"): 1.0,
        ("B", "High", "A", "Low"): 2.0,
        ("B", "Low", "C", "Low"): 1.0,
        ("B", "Low", "C", "High"): 1.0,
        ("B", "High", "C", "High"): -2.0,
        ("C", "Low", "A", "High"): 2.0,
        ("C", "High", "A", "Low"): -2.0,
        ("C", "High", "A", "High"): 1.0,
        ("C", "Low", "B", "Low"): 2.0,
        ("C", "High", "B", "Low"): -1.0,
        ("C", "High", "B", "High"): -1.0,
    }
    m.set_impacts(impacts)
    return m


def test_equilibrium_distance_measures_distance_to_nearest_attractor() -> None:
    m = _simple_matrix()
    consistent = Scenario({"A": "Low", "B": "Low"}, m)
    inconsistent = Scenario({"A": "Low", "B": "High"}, m)

    assert equilibrium_distance(consistent, m) == 0.0
    assert equilibrium_distance(inconsistent, m) == 0.0
    assert consistent_set_distance(inconsistent, m) == 1.0


def test_equilibrium_distance_distinguishes_attractor_proximity_from_consistent_set_distance() -> None:
    m = _cycle_and_fixed_point_matrix()
    scenario = Scenario({"A": "Low", "B": "Low", "C": "Low"}, m)

    assert equilibrium_distance(scenario, m) == 0.0
    assert consistent_set_distance(scenario, m) == 1.0

    attractor = attractor_distance(scenario, m)
    assert attractor.distance_to_attractor == 0.0
    assert attractor.attractor_kind == "cycle"


def test_descriptor_disequilibrium_contributions_returns_per_descriptor_payload() -> None:
    m = _simple_matrix()
    scenario = Scenario({"A": "Low", "B": "High"}, m)

    contributions = descriptor_disequilibrium_contributions(scenario, m)

    assert "A" in contributions
    assert "B" in contributions
    assert contributions["A"]["alternative_state"] == "High"


def test_cumulative_disequilibrium_burden_aggregates_negative_margins() -> None:
    metrics = [
        PerPeriodDisequilibriumMetrics(
            period=0,
            is_consistent=False,
            consistency_margin=-2.0,
            descriptor_margins={"A": -2.0},
            brink_descriptors=("A",),
            distance_to_equilibrium=1.0,
            time_to_equilibrium=1,
            entered_consistent_set=False,
        ),
        PerPeriodDisequilibriumMetrics(
            period=1,
            is_consistent=True,
            consistency_margin=1.0,
            descriptor_margins={"A": 1.0},
            brink_descriptors=(),
            distance_to_equilibrium=0.0,
            time_to_equilibrium=0,
            entered_consistent_set=True,
        ),
    ]

    assert cumulative_disequilibrium_burden(metrics) == 2.0


def test_extended_pathway_summaries_and_serialization() -> None:
    m = _simple_matrix()
    s0 = Scenario({"A": "Low", "B": "High"}, m)
    s1 = Scenario({"A": "High", "B": "High"}, m)
    metrics = (
        PerPeriodDisequilibriumMetrics(
            period=0,
            is_consistent=False,
            consistency_margin=-4.0,
            descriptor_margins={"A": -4.0, "B": -4.0},
            brink_descriptors=("A", "B"),
            distance_to_equilibrium=1.0,
            time_to_equilibrium=1,
            entered_consistent_set=False,
        ),
        PerPeriodDisequilibriumMetrics(
            period=1,
            is_consistent=True,
            consistency_margin=4.0,
            descriptor_margins={"A": 4.0, "B": 4.0},
            brink_descriptors=(),
            distance_to_equilibrium=0.0,
            time_to_equilibrium=0,
            entered_consistent_set=True,
        ),
    )
    path = ExtendedTransformationPathway(
        periods=(0, 1),
        realised_scenarios=(s0, s1),
        equilibrium_scenarios=None,
        extension_mode="transient",
        disequilibrium_metrics=metrics,
        active_regimes=("baseline", "baseline"),
        active_matrices=(
            ActiveMatrixState(
                period=0,
                regime_name="baseline",
                base_matrix_id="base",
                active_matrix_id="base",
            ),
            ActiveMatrixState(
                period=1,
                regime_name="baseline",
                base_matrix_id="base",
                active_matrix_id="base",
            ),
        ),
        transition_events=(),
        memory_states=(),
        structural_consistency=(),
        diagnostics={},
    )

    summary = summarize_disequilibrium_path(path)
    ensemble = summarize_disequilibrium_ensemble([path])
    payload = path.to_serializable_dict()

    assert summary["first_consistent_period"] == 1
    assert summary["cumulative_disequilibrium_burden"] == 4.0
    assert ensemble["n_pathways"] == 1
    assert payload["extension_mode"] == "transient"
    assert payload["realised_scenarios"][0]["B"] == "High"
