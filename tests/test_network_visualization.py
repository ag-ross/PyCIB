"""
Unit tests for network visualization functionality.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from cib.core import CIBMatrix, Scenario
from cib.visualization import ScenarioVisualizer


def test_scenario_network_runs_and_returns_axes() -> None:
    """Create a scenario network plot without raising errors."""
    descriptors = {
        "A": ["Low", "High"],
        "B": ["Low", "High"],
    }
    matrix = CIBMatrix(descriptors)
    scenarios = [
        Scenario({"A": "Low", "B": "Low"}, matrix),
        Scenario({"A": "Low", "B": "High"}, matrix),
        Scenario({"A": "High", "B": "Low"}, matrix),
        Scenario({"A": "High", "B": "High"}, matrix),
    ]

    ax = ScenarioVisualizer.scenario_network(scenarios, matrix, edge_metric="hamming")
    assert ax is not None


def test_scenario_network_rejects_incompatible_scenarios_for_hamming() -> None:
    matrix_a = CIBMatrix({"A": ["Low", "High"]})
    matrix_b = CIBMatrix({"A": ["Low", "High"], "B": ["Low", "High"]})
    scenarios = [
        Scenario({"A": "Low"}, matrix_a),
        Scenario({"A": "Low", "B": "Low"}, matrix_b),
    ]

    with pytest.raises(ValueError, match="descriptor schema mismatch"):
        _ = ScenarioVisualizer.scenario_network(
            scenarios, matrix_b, edge_metric="hamming"
        )
