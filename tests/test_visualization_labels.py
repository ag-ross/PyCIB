"""
Unit tests for visualization label semantics.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from cib.visualization import DynamicVisualizer, UncertaintyVisualizer


def test_state_probability_bands_y_label_is_ensemble_share() -> None:
    timelines = {
        2025: {"X": {"Low": 0.6, "High": 0.4}},
        2030: {"X": {"Low": 0.4, "High": 0.6}},
    }
    ax = DynamicVisualizer.plot_state_probability_bands(timelines, descriptor="X")
    assert ax.get_ylabel() == "Ensemble share"


def test_probability_intervals_y_label_remains_probability() -> None:
    probabilities = {}
    intervals = {}
    ax = UncertaintyVisualizer.plot_probability_intervals(probabilities, intervals)
    assert ax.get_ylabel() == "P(consistent)"
