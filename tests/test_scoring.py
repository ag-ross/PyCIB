"""
Unit tests for scenario diagnostics and qualitative impact labels.
"""

from __future__ import annotations

import pytest

from cib.core import CIBMatrix, Scenario
from cib.scoring import (
    _hamming_distance,
    impact_label,
    judgment_section_labels,
    scenario_diagnostics,
)


def _simple_matrix() -> CIBMatrix:
    # Two-descriptor coordination system with fixed points at (Low,Low) and (High,High).
    desc = {"A": ["Low", "High"], "B": ["Low", "High"]}
    m = CIBMatrix(desc)
    # A influences B.
    m.set_impact("A", "Low", "B", "Low", 2.0)
    m.set_impact("A", "Low", "B", "High", -2.0)
    m.set_impact("A", "High", "B", "Low", -2.0)
    m.set_impact("A", "High", "B", "High", 2.0)
    # B influences A.
    m.set_impact("B", "Low", "A", "Low", 2.0)
    m.set_impact("B", "Low", "A", "High", -2.0)
    m.set_impact("B", "High", "A", "Low", -2.0)
    m.set_impact("B", "High", "A", "High", 2.0)
    return m


class TestImpactLabel:
    def test_bucketing(self) -> None:
        assert impact_label(-3.0) == "strongly_hindering"
        assert impact_label(-1.0) == "hindering"
        assert impact_label(0.0) == "neutral"
        assert impact_label(1.0) == "promoting"
        assert impact_label(3.0) == "strongly_promoting"

    def test_invalid_thresholds(self) -> None:
        with pytest.raises(ValueError):
            impact_label(0.0, weak_threshold=1.0, strong_threshold=1.0)


class TestScenarioDiagnostics:
    def test_consistent_scenario_has_non_negative_margin(self) -> None:
        m = _simple_matrix()
        s = Scenario({"A": "Low", "B": "Low"}, m)
        d = scenario_diagnostics(s, m)
        assert d.is_consistent is True
        assert d.consistency_margin >= 0.0
        assert d.consistency_margin == 4.0

    def test_inconsistent_scenario_has_negative_margin(self) -> None:
        m = _simple_matrix()
        s = Scenario({"A": "Low", "B": "High"}, m)
        d = scenario_diagnostics(s, m)
        assert d.is_consistent is False
        assert d.consistency_margin < 0.0
        assert len(d.inconsistencies) >= 1

    def test_total_impact_score_present(self) -> None:
        m = _simple_matrix()
        s = Scenario({"A": "High", "B": "High"}, m)
        d = scenario_diagnostics(s, m)
        assert isinstance(d.total_impact_score, float)

    def test_scenario_diagnostics_respects_tolerance_passthrough(self) -> None:
        m = CIBMatrix({"A": ["a0", "a1"], "B": ["b0", "b1"]})
        m.set_impact("A", "a0", "B", "b0", 1.0)
        m.set_impact("A", "a0", "B", "b1", 1.0 + 5e-9)
        s = Scenario({"A": "a0", "B": "b0"}, m)

        strict = scenario_diagnostics(s, m, float_atol=0.0, float_rtol=0.0)
        tolerant = scenario_diagnostics(s, m, float_atol=1e-8, float_rtol=0.0)

        assert strict.is_consistent is False
        assert tolerant.is_consistent is True

    def test_hamming_distance_rejects_incompatible_scenarios(self) -> None:
        matrix_a = CIBMatrix({"A": ["Low", "High"]})
        matrix_b = CIBMatrix({"A": ["Low", "High"], "B": ["Low", "High"]})
        scenario_a = Scenario({"A": "Low"}, matrix_a)
        scenario_b = Scenario({"A": "Low", "B": "Low"}, matrix_b)

        with pytest.raises(ValueError, match="descriptor schema mismatch"):
            _ = _hamming_distance(scenario_a, scenario_b)


class TestJudgmentSectionLabels:
    def test_labels_cover_full_section(self) -> None:
        m = _simple_matrix()
        labels = judgment_section_labels(m, src_desc="A", tgt_desc="B")
        assert len(labels) == 4
        assert labels[("Low", "Low")] == "strongly_promoting"
        assert labels[("Low", "High")] == "strongly_hindering"

