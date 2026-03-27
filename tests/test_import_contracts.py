"""
Unit tests for public import contract behavior.
"""

from __future__ import annotations

import cib


def test_optional_exports_are_not_in___all__() -> None:
    """Star-import surface should remain usable without optional extras."""
    optional_names = {
        "MatrixVisualizer",
        "ScenarioVisualizer",
        "UncertaintyVisualizer",
        "ShockVisualizer",
        "DynamicVisualizer",
        "NetworkGraphBuilder",
        "NetworkAnalyzer",
        "ImpactPathwayAnalyzer",
    }
    assert optional_names.isdisjoint(set(cib.__all__))
