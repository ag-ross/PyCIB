"""
Contract tests for ScenarioAnalyzer.find_all_consistent mode semantics.
"""

from __future__ import annotations

import pytest

from cib.analysis import FindAllConsistentResult, ScenarioAnalyzer
from cib.constraints import ForbiddenPair
from cib.core import CIBMatrix


def _large_matrix() -> CIBMatrix:
    descriptors = {f"D{i}": ["L", "M", "H"] for i in range(11)}
    return CIBMatrix(descriptors)


def test_find_all_consistent_exhaustive_mode_raises_when_infeasible() -> None:
    analyzer = ScenarioAnalyzer(_large_matrix())
    with pytest.raises(ValueError, match="mode='exhaustive'"):
        analyzer.find_all_consistent(mode="exhaustive")


def test_find_all_consistent_exhaustive_mode_respects_max_scenarios_contract() -> None:
    analyzer = ScenarioAnalyzer(_large_matrix())
    with pytest.raises(ValueError, match="max_scenarios=5"):
        analyzer.find_all_consistent(mode="exhaustive", max_scenarios=5)


def test_find_all_consistent_shortlist_mode_returns_metadata() -> None:
    analyzer = ScenarioAnalyzer(_large_matrix())
    res = analyzer.find_all_consistent(
        mode="shortlist",
        n_restarts=20,
        seed=123,
        return_metadata=True,
    )
    assert isinstance(res, FindAllConsistentResult)
    assert res.enumeration_mode == "random_restarts"
    assert res.is_complete is False
    assert res.requested_mode == "shortlist"
    assert res.effective_mode == "shortlist"
    assert res.switch_reason == "explicit_shortlist_mode"


def test_find_all_consistent_auto_mode_warns_and_marks_shortlist() -> None:
    analyzer = ScenarioAnalyzer(_large_matrix())
    with pytest.warns(UserWarning, match="random-restart shortlist"):
        res = analyzer.find_all_consistent(
            mode="auto",
            n_restarts=20,
            seed=123,
            return_metadata=True,
        )
    assert isinstance(res, FindAllConsistentResult)
    assert res.is_complete is False
    assert res.requested_mode == "auto"
    assert res.effective_mode == "shortlist"
    assert res.switch_reason == "auto_threshold_exceeded"


def test_find_all_consistent_auto_mode_reports_threshold_and_cap_exceeded() -> None:
    analyzer = ScenarioAnalyzer(_large_matrix())
    with pytest.warns(UserWarning, match="random-restart shortlist"):
        res = analyzer.find_all_consistent(
            mode="auto",
            n_restarts=20,
            seed=123,
            max_scenarios=5,
            return_metadata=True,
        )
    assert isinstance(res, FindAllConsistentResult)
    assert res.switch_reason == "auto_threshold_and_max_scenarios_exceeded"


def test_find_all_consistent_exhaustive_metadata_is_complete() -> None:
    descriptors = {"A": ["L", "H"], "B": ["L", "H"]}
    matrix = CIBMatrix(descriptors)
    analyzer = ScenarioAnalyzer(matrix)
    res = analyzer.find_all_consistent(return_metadata=True)
    assert isinstance(res, FindAllConsistentResult)
    assert res.enumeration_mode == "exhaustive"
    assert res.is_complete is True
    assert res.requested_mode == "exhaustive"
    assert res.effective_mode == "exhaustive"
    assert res.switch_reason == "none"


def test_random_restart_shortlist_uses_constraint_aware_succession() -> None:
    descriptors = {"A": ["L", "H"], "B": ["L", "H"]}
    matrix = CIBMatrix(descriptors)
    analyzer = ScenarioAnalyzer(matrix)
    captured: dict[str, object] = {}

    def fake_find_attractors_via_random_restarts(**kwargs):
        captured["operator"] = kwargs["succession_operator"]
        return []

    analyzer.find_attractors_via_random_restarts = fake_find_attractors_via_random_restarts  # type: ignore[method-assign]
    _ = analyzer.find_consistent_via_random_restarts(
        n_restarts=5,
        seed=1,
        constraints=[ForbiddenPair("A", "H", "B", "H")],
    )
    assert captured["operator"].__class__.__name__ == "ConstrainedGlobalSuccession"


def test_random_restart_shortlist_passes_constrained_policy_parameters() -> None:
    descriptors = {"A": ["L", "H"], "B": ["L", "H"]}
    matrix = CIBMatrix(descriptors)
    analyzer = ScenarioAnalyzer(matrix)
    captured: dict[str, object] = {}

    def fake_find_attractors_via_random_restarts(**kwargs):
        captured["operator"] = kwargs["succession_operator"]
        return []

    analyzer.find_attractors_via_random_restarts = fake_find_attractors_via_random_restarts  # type: ignore[method-assign]
    _ = analyzer.find_consistent_via_random_restarts(
        n_restarts=5,
        seed=1,
        constraints=[ForbiddenPair("A", "H", "B", "H")],
        constrained_mode="strict",
        constrained_top_k=3,
        constrained_backtracking_depth=1,
    )
    op = captured["operator"]
    assert op is not None
    assert getattr(op, "constraint_mode") == "strict"
    assert getattr(op, "constrained_top_k") == 3
    assert getattr(op, "constrained_backtracking_depth") == 1
