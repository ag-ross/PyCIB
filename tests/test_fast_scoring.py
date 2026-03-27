"""
Unit tests for the fast scoring backend.
"""

from __future__ import annotations

import pytest
import numpy as np

from cib.benchmark_data import benchmark_matrix_b1, benchmark_matrix_b2
from cib.core import ConsistencyChecker, Scenario
from cib.fast_scoring import FastCIBScorer
from cib.sparse_scoring import SparseCIBScorer


def _random_scenario(matrix, seed: int) -> Scenario:
    rng = np.random.default_rng(int(seed))
    sdict = {}
    for d, states in matrix.descriptors.items():
        sdict[d] = states[int(rng.integers(0, len(states)))]
    return Scenario(sdict, matrix)


def test_fast_scorer_workspace_guard() -> None:
    m = benchmark_matrix_b1()
    with pytest.raises(MemoryError, match="max_workspace_bytes"):
        FastCIBScorer.from_matrix(m, max_workspace_bytes=1)


def test_fast_scores_match_slow_balances_b1() -> None:
    m = benchmark_matrix_b1()
    scorer = FastCIBScorer.from_matrix(m)

    s = _random_scenario(m, seed=501)
    z = scorer.scenario_to_indices(s)
    scores = scorer.scores_for_scenario(z)

    for d in m.descriptors.keys():
        j = scorer.descriptor_index[d]
        slow = m.calculate_impact_balance(s, d)
        for state_label, v in slow.items():
            l = scorer.state_index[j][state_label]
            assert float(scores[j, l]) == float(v)


def test_sparse_scores_match_slow_balances_b1() -> None:
    m = benchmark_matrix_b1()
    scorer = SparseCIBScorer.from_matrix(m)

    s = _random_scenario(m, seed=502)
    z = scorer.scenario_to_indices(s)
    scores = scorer.scores_for_scenario(z)

    for d in m.descriptors.keys():
        j = scorer.descriptor_index[d]
        slow = m.calculate_impact_balance(s, d)
        for state_label, v in slow.items():
            l = scorer.state_index[j][state_label]
            assert float(scores[j, l]) == float(v)


def test_fast_consistency_matches_reference_b2() -> None:
    m = benchmark_matrix_b2()
    scorer = FastCIBScorer.from_matrix(m)

    for seed in range(10):
        s = _random_scenario(m, seed=1000 + seed)
        slow = ConsistencyChecker.check_consistency(s, m)
        fast = scorer.is_consistent(scorer.scenario_to_indices(s))
        assert bool(fast) == bool(slow)


def test_sparse_consistency_matches_reference_b2() -> None:
    m = benchmark_matrix_b2()
    scorer = SparseCIBScorer.from_matrix(m)

    for seed in range(10):
        s = _random_scenario(m, seed=1100 + seed)
        slow = ConsistencyChecker.check_consistency(s, m)
        fast = scorer.is_consistent(scorer.scenario_to_indices(s))
        assert bool(fast) == bool(slow)


def test_consistency_checker_fast_path_runs() -> None:
    m = benchmark_matrix_b1()
    s = _random_scenario(m, seed=707)
    slow = ConsistencyChecker.check_consistency(s, m)
    fast = ConsistencyChecker.check_consistency(s, m, use_fast=True)
    assert bool(slow) == bool(fast)


def test_consistency_checker_fast_path_workspace_cap_falls_back() -> None:
    m = benchmark_matrix_b1()
    s = _random_scenario(m, seed=808)
    out: dict[str, object] = {}
    fast = ConsistencyChecker.check_consistency(
        s,
        m,
        use_fast=True,
        max_fast_workspace_bytes=1,
        out_diagnostics=out,
    )
    slow = ConsistencyChecker.check_consistency(s, m, use_fast=False)
    assert bool(fast) == bool(slow)
    assert out.get("fast_path_failed") is True


def test_consistency_checker_fast_path_workspace_cap_override_none() -> None:
    m = benchmark_matrix_b1()
    s = _random_scenario(m, seed=809)
    out: dict[str, object] = {}
    fast = ConsistencyChecker.check_consistency(
        s,
        m,
        use_fast=True,
        max_fast_workspace_bytes=None,
        out_diagnostics=out,
    )
    slow = ConsistencyChecker.check_consistency(s, m, use_fast=False)
    assert bool(fast) == bool(slow)
    assert out.get("fast_path_failed") is None


def test_consistency_checker_out_diagnostics_positional_is_compatible() -> None:
    m = benchmark_matrix_b1()
    s = _random_scenario(m, seed=810)
    out: dict[str, object] = {}
    _ = ConsistencyChecker.check_consistency(s, m, True, out)
    assert isinstance(out, dict)


def test_fast_consistency_matches_reference_with_explicit_tolerances() -> None:
    """Fast and reference consistency should agree under matched tolerances."""
    m = benchmark_matrix_b2()
    scorer = FastCIBScorer.from_matrix(m)

    for seed in range(10):
        s = _random_scenario(m, seed=1200 + seed)
        slow = ConsistencyChecker.check_consistency(
            s,
            m,
            float_atol=1e-7,
            float_rtol=1e-6,
        )
        fast = scorer.is_consistent(
            scorer.scenario_to_indices(s),
            float_atol=1e-7,
            float_rtol=1e-6,
        )
        assert bool(fast) == bool(slow)

