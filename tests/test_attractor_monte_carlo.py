"""
Unit tests for Monte Carlo attractor discovery.
"""

from __future__ import annotations

import pytest

from cib.analysis import ScenarioAnalyzer
from cib.benchmark_data import benchmark_matrix_b1
from cib.core import ConsistencyChecker, Scenario
from cib.solvers.config import MonteCarloAttractorConfig


def test_monte_carlo_diagnostics_weights_normalization() -> None:
    m = benchmark_matrix_b1()
    r = ScenarioAnalyzer(m).find_attractors_monte_carlo(
        config=MonteCarloAttractorConfig(
            runs=20,
            seed=1,
            succession="global",
            min_completion_fraction=None,
        )
    )
    assert r.diagnostics.get("weights_normalization") == "completed_runs_only"
    assert r.diagnostics.get("requested_runs") == 20
    assert r.diagnostics.get("requested_runs_normalization") == "requested_runs"
    assert isinstance(r.diagnostics.get("weights_requested_runs"), dict)
    assert isinstance(r.diagnostics.get("weights_requested_runs_serialized"), dict)
    assert "completion_fraction" in r.diagnostics


def test_monte_carlo_min_completion_fraction_invalid() -> None:
    with pytest.raises(ValueError, match="min_completion_fraction"):
        MonteCarloAttractorConfig(min_completion_fraction=1.5).validate()


def test_monte_carlo_completion_status_target_fraction_invalid() -> None:
    with pytest.raises(ValueError, match="completion_status_target_fraction"):
        MonteCarloAttractorConfig(completion_status_target_fraction=0.0).validate()


def test_monte_carlo_fail_on_timeout_raises_when_timeouts_occur() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)
    cfg = MonteCarloAttractorConfig(
        runs=200,
        seed=11_001,
        succession="global",
        max_iterations=1,
        fail_on_timeout=True,
    )
    with pytest.raises(RuntimeError, match="fail_on_timeout"):
        analyzer.find_attractors_monte_carlo(config=cfg)


def test_monte_carlo_default_min_completion_fraction_raises_on_timeouts() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)
    cfg = MonteCarloAttractorConfig(
        runs=200,
        seed=11_001,
        succession="global",
        max_iterations=1,
    )
    with pytest.raises(RuntimeError, match="min_completion_fraction"):
        analyzer.find_attractors_monte_carlo(config=cfg)


def test_monte_carlo_default_threshold_can_be_overridden_to_permissive() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)
    cfg = MonteCarloAttractorConfig(
        runs=200,
        seed=11_001,
        succession="global",
        max_iterations=1,
        min_completion_fraction=None,
    )
    res = analyzer.find_attractors_monte_carlo(config=cfg)
    assert int(res.diagnostics.get("n_timeouts", 0)) > 0


def test_monte_carlo_permissive_mode_surfaces_incompleteness_status() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)
    cfg = MonteCarloAttractorConfig(
        runs=200,
        seed=11_001,
        succession="global",
        max_iterations=1,
        min_completion_fraction=None,
    )
    res = analyzer.find_attractors_monte_carlo(config=cfg)
    assert int(res.diagnostics.get("n_timeouts", 0)) > 0
    assert float(res.diagnostics.get("completion_fraction", 1.0)) < float(
        res.diagnostics.get("completion_status_target_fraction", 1.0)
    )
    assert res.status in {"partial_timeout", "incomplete"}
    assert res.status != "no_attractors"


def test_monte_carlo_status_is_deterministic_from_diagnostics() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)
    cfg = MonteCarloAttractorConfig(
        runs=200,
        seed=11_001,
        succession="global",
        max_iterations=1,
        min_completion_fraction=None,
    )
    res = analyzer.find_attractors_monte_carlo(config=cfg)
    n_completed = int(res.diagnostics["n_completed_runs"])
    completion_fraction = float(res.diagnostics["completion_fraction"])
    completion_target = float(res.diagnostics["completion_status_target_fraction"])
    if n_completed <= 0:
        expected = "incomplete"
    elif completion_fraction + 1e-15 < completion_target:
        expected = "partial_timeout" if res.counts else "incomplete"
    else:
        expected = "ok" if res.counts else "no_attractors"
    assert res.status == expected


def test_monte_carlo_attractor_reproducibility() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg = MonteCarloAttractorConfig(runs=200, seed=123, succession="global")
    r1 = analyzer.find_attractors_monte_carlo(config=cfg)
    r2 = analyzer.find_attractors_monte_carlo(config=cfg)

    assert r1.counts == r2.counts
    assert r1.attractor_keys_ranked == r2.attractor_keys_ranked

    cfg_sparse = MonteCarloAttractorConfig(
        runs=200, seed=123, succession="global", fast_backend="sparse"
    )
    s1 = analyzer.find_attractors_monte_carlo(config=cfg_sparse)
    s2 = analyzer.find_attractors_monte_carlo(config=cfg_sparse)
    assert s1.counts == s2.counts
    assert s1.attractor_keys_ranked == s2.attractor_keys_ranked


def test_monte_carlo_attractor_multiprocess_matches_single_process() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg1 = MonteCarloAttractorConfig(runs=200, seed=999, succession="global", n_jobs=1)
    cfg2 = MonteCarloAttractorConfig(runs=200, seed=999, succession="global", n_jobs=2)

    r1 = analyzer.find_attractors_monte_carlo(config=cfg1)
    r2 = analyzer.find_attractors_monte_carlo(config=cfg2)

    assert r1.counts == r2.counts
    assert r1.attractor_keys_ranked == r2.attractor_keys_ranked

    cfg1s = MonteCarloAttractorConfig(
        runs=200, seed=999, succession="global", n_jobs=1, fast_backend="sparse"
    )
    cfg2s = MonteCarloAttractorConfig(
        runs=200, seed=999, succession="global", n_jobs=2, fast_backend="sparse"
    )
    s1 = analyzer.find_attractors_monte_carlo(config=cfg1s)
    s2 = analyzer.find_attractors_monte_carlo(config=cfg2s)
    assert s1.counts == s2.counts
    assert s1.attractor_keys_ranked == s2.attractor_keys_ranked


def test_monte_carlo_attractor_keys_are_valid_fixed_points() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg = MonteCarloAttractorConfig(runs=200, seed=456, succession="global")
    res = analyzer.find_attractors_monte_carlo(config=cfg)

    # At least one attractor should be discovered for this benchmark.
    assert len(res.counts) >= 1

    # Fixed-point keys should correspond to consistent scenarios.
    for key in res.attractor_keys_ranked[:5]:
        if key.kind != "fixed":
            continue
        v = key.value
        assert isinstance(v, tuple)
        s = Scenario(list(v), m)
        assert ConsistencyChecker.check_consistency(s, m) is True


def test_monte_carlo_cycle_key_policy_rotate_min_has_cycle_structure() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg = MonteCarloAttractorConfig(
        runs=200,
        seed=321,
        succession="global",
        cycle_mode="keep_cycle",
        cycle_key_policy="rotate_min",
    )
    res = analyzer.find_attractors_monte_carlo(config=cfg)
    assert len(res.counts) >= 1

    for key in res.attractor_keys_ranked:
        if key.kind != "cycle":
            continue
        v = key.value
        assert isinstance(v, tuple)
        assert len(v) >= 1
        assert isinstance(v[0], tuple)
        break


def test_monte_carlo_cycle_mode_representative_first_is_distinct_from_keep_cycle() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg_keep = MonteCarloAttractorConfig(
        runs=200, seed=222, succession="global", cycle_mode="keep_cycle"
    )
    cfg_rep = MonteCarloAttractorConfig(
        runs=200, seed=222, succession="global", cycle_mode="representative_first"
    )
    r_keep = analyzer.find_attractors_monte_carlo(config=cfg_keep)
    r_rep = analyzer.find_attractors_monte_carlo(config=cfg_rep)

    assert r_keep.diagnostics["n_completed_runs"] == r_rep.diagnostics["n_completed_runs"]
    assert r_keep.cycles is not None
    assert r_rep.cycles is None


def test_monte_carlo_cycle_mode_representative_random_is_reproducible_across_jobs() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg1 = MonteCarloAttractorConfig(
        runs=300,
        seed=777,
        succession="global",
        cycle_mode="representative_random",
        n_jobs=1,
    )
    cfg2 = MonteCarloAttractorConfig(
        runs=300,
        seed=777,
        succession="global",
        cycle_mode="representative_random",
        n_jobs=2,
    )
    r1 = analyzer.find_attractors_monte_carlo(config=cfg1)
    r2 = analyzer.find_attractors_monte_carlo(config=cfg2)
    assert r1.counts == r2.counts


def _near_tie_matrix() -> "CIBMatrix":
    from cib.core import CIBMatrix

    m = CIBMatrix({"A": ["Low", "High"], "B": ["Low", "High"]})
    eps = 1e-9
    for a_state in ("Low", "High"):
        m.set_impact("A", a_state, "B", "Low", 0.0)
        m.set_impact("A", a_state, "B", "High", eps)
    for b_state in ("Low", "High"):
        m.set_impact("B", b_state, "A", "Low", 0.0)
        m.set_impact("B", b_state, "A", "High", eps)
    return m


def test_monte_carlo_tolerances_affect_near_tie_behavior() -> None:
    m = _near_tie_matrix()
    analyzer = ScenarioAnalyzer(m)

    strict = analyzer.find_attractors_monte_carlo(
        config=MonteCarloAttractorConfig(
            runs=40,
            seed=2026,
            succession="global",
            use_fast_scoring=False,
            float_atol=0.0,
            float_rtol=0.0,
        )
    )
    tolerant = analyzer.find_attractors_monte_carlo(
        config=MonteCarloAttractorConfig(
            runs=40,
            seed=2026,
            succession="global",
            use_fast_scoring=False,
            float_atol=1e-6,
            float_rtol=0.0,
        )
    )

    assert strict.counts != tolerant.counts
    assert strict.diagnostics["float_atol"] == pytest.approx(0.0)
    assert tolerant.diagnostics["float_atol"] == pytest.approx(1e-6)


def test_monte_carlo_fast_and_slow_match_with_tolerances() -> None:
    m = _near_tie_matrix()
    analyzer = ScenarioAnalyzer(m)

    slow = analyzer.find_attractors_monte_carlo(
        config=MonteCarloAttractorConfig(
            runs=60,
            seed=707,
            succession="local",
            use_fast_scoring=False,
            float_atol=1e-6,
            float_rtol=0.0,
        )
    )
    fast = analyzer.find_attractors_monte_carlo(
        config=MonteCarloAttractorConfig(
            runs=60,
            seed=707,
            succession="local",
            use_fast_scoring=True,
            float_atol=1e-6,
            float_rtol=0.0,
        )
    )

    assert slow.counts == fast.counts


def test_monte_carlo_parallel_preserves_tolerance_diagnostics() -> None:
    m = _near_tie_matrix()
    analyzer = ScenarioAnalyzer(m)

    res = analyzer.find_attractors_monte_carlo(
        config=MonteCarloAttractorConfig(
            runs=24,
            seed=909,
            succession="global",
            use_fast_scoring=False,
            n_jobs=2,
            float_atol=1e-6,
            float_rtol=0.0,
        )
    )

    assert res.diagnostics["float_atol"] == pytest.approx(1e-6)
    assert res.diagnostics["float_rtol"] == pytest.approx(0.0)


def test_monte_carlo_strict_fast_surfaces_fast_scorer_failure(monkeypatch) -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    def _boom(*args, **kwargs):
        raise ValueError("forced fast scorer failure")

    monkeypatch.setattr("cib.fast_scoring.FastCIBScorer.from_matrix", _boom)
    cfg = MonteCarloAttractorConfig(
        runs=20,
        seed=123,
        strict_fast=True,
        min_completion_fraction=None,
    )
    with pytest.raises(ValueError, match="forced fast scorer failure"):
        analyzer.find_attractors_monte_carlo(config=cfg)


def test_monte_carlo_non_strict_fast_fallback_warns_and_sets_diagnostics(monkeypatch) -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    def _boom(*args, **kwargs):
        raise ValueError("forced fallback")

    monkeypatch.setattr("cib.fast_scoring.FastCIBScorer.from_matrix", _boom)
    cfg = MonteCarloAttractorConfig(
        runs=20,
        seed=123,
        strict_fast=False,
        min_completion_fraction=None,
    )
    with pytest.warns(UserWarning, match="falling back to slow path"):
        res = analyzer.find_attractors_monte_carlo(config=cfg)
    assert res.diagnostics.get("fast_scorer_fallback") is True
    assert res.diagnostics.get("fallback_stage") == "fast_scorer_initialization"
    assert res.diagnostics.get("fallback_from") == "fast_scorer"
