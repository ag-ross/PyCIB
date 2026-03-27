"""
Unit tests for the exact pruned solver.
"""

from __future__ import annotations

import pytest

from cib.analysis import ScenarioAnalyzer
from cib.benchmark_data import benchmark_matrix_b1
from cib.constraints import AllowedStates
from cib.solvers.config import ExactSolverConfig


def test_exact_pruned_matches_bruteforce_on_b1() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    brute = analyzer.find_all_consistent(max_scenarios=50_000)
    brute_set = {tuple(s.to_indices()) for s in brute}

    cfg = ExactSolverConfig(ordering="given", bound="safe_upper_bound_v1")
    res = analyzer.find_all_consistent_exact(config=cfg)
    got_set = {tuple(s.to_indices()) for s in res.scenarios}

    assert res.is_complete is True
    assert got_set == brute_set


def test_exact_pruned_without_bounds_matches_bruteforce_on_b1() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    brute = analyzer.find_all_consistent(max_scenarios=50_000)
    brute_set = {tuple(s.to_indices()) for s in brute}

    cfg = ExactSolverConfig(ordering="given", bound="none")
    res = analyzer.find_all_consistent_exact(config=cfg)
    got_set = {tuple(s.to_indices()) for s in res.scenarios}

    assert res.is_complete is True
    assert got_set == brute_set


def test_exact_pruned_time_limit_can_stop_early() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg = ExactSolverConfig(ordering="random", bound="safe_upper_bound_v1", time_limit_s=1e-9)
    res = analyzer.find_all_consistent_exact(config=cfg)

    assert res.is_complete is False
    assert res.status in {"timeout", "max_solutions"}


def test_exact_bruteforce_fallback_respects_constraints() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)
    constrained = [AllowedStates(desc="D00", allowed={"S0"})]

    fast_cfg = ExactSolverConfig(
        ordering="given",
        bound="safe_upper_bound_v1",
        constraints=constrained,
    )
    fallback_cfg = ExactSolverConfig(
        ordering="given",
        bound="safe_upper_bound_v1",
        constraints=constrained,
        use_fast_scoring=False,
        allow_bruteforce=True,
    )

    fast_res = analyzer.find_all_consistent_exact(config=fast_cfg)
    fallback_res = analyzer.find_all_consistent_exact(config=fallback_cfg)

    assert fast_res.is_complete is True
    assert fallback_res.is_complete is True
    assert fallback_res.diagnostics["fallback"] == "bruteforce"
    assert {tuple(s.to_indices()) for s in fast_res.scenarios} == {
        tuple(s.to_indices()) for s in fallback_res.scenarios
    }


def test_exact_bruteforce_fallback_respects_max_solutions() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg = ExactSolverConfig(
        ordering="given",
        bound="safe_upper_bound_v1",
        use_fast_scoring=False,
        allow_bruteforce=True,
        max_solutions=1,
    )
    res = analyzer.find_all_consistent_exact(config=cfg)

    assert res.status == "max_solutions"
    assert res.is_complete is False
    assert len(res.scenarios) == 1
    assert res.diagnostics["fallback"] == "bruteforce"


def test_exact_bruteforce_fallback_respects_time_limit() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg = ExactSolverConfig(
        ordering="given",
        bound="safe_upper_bound_v1",
        use_fast_scoring=False,
        allow_bruteforce=True,
        time_limit_s=1e-12,
    )
    res = analyzer.find_all_consistent_exact(config=cfg)

    assert res.status == "timeout"
    assert res.is_complete is False
    assert res.diagnostics["fallback"] == "bruteforce"


def test_exact_strict_fast_surfaces_fast_scorer_failure(monkeypatch) -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    def _boom(*args, **kwargs):
        raise ValueError("forced fast scorer failure")

    monkeypatch.setattr("cib.fast_scoring.FastCIBScorer.from_matrix", _boom)
    cfg = ExactSolverConfig(strict_fast=True)
    with pytest.raises(ValueError, match="forced fast scorer failure"):
        analyzer.find_all_consistent_exact(config=cfg)


def test_exact_non_strict_fast_fallback_warns_and_sets_diagnostics(monkeypatch) -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    def _boom(*args, **kwargs):
        raise ValueError("forced fallback")

    monkeypatch.setattr("cib.fast_scoring.FastCIBScorer.from_matrix", _boom)
    cfg = ExactSolverConfig(
        strict_fast=False,
        use_fast_scoring=True,
        allow_bruteforce=True,
    )
    with pytest.warns(UserWarning, match="falling back to slow path"):
        res = analyzer.find_all_consistent_exact(config=cfg)
    assert res.diagnostics.get("fast_scorer_fallback") is True
    assert res.diagnostics.get("fallback_stage") == "fast_scorer_initialization"
    assert res.diagnostics.get("fallback_from") == "fast_scorer"

