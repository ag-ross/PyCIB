"""
Unit tests for network visualization functionality.
"""

import matplotlib

matplotlib.use("Agg")

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
