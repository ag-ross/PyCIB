"""
Unit tests for small direct probabilistic fits.
"""

import numpy as np
import pytest

from cib.prob import fit_direct as fit_direct_mod
from cib.prob.fit_direct import fit_joint_direct
from cib.prob.model import ProbabilisticCIAModel
from cib.prob.types import FactorSpec, ScenarioIndex


def test_fit_direct_two_by_two_recovers_conditionals() -> None:
    """Recover marginals and selected conditionals in 2x2 case."""
    factors = [
        FactorSpec("A", ["a0", "a1"]),
        FactorSpec("B", ["b0", "b1"]),
    ]

    # Fixed marginals.
    marginals = {
        "A": {"a0": 0.6, "a1": 0.4},
        "B": {"b0": 0.7, "b1": 0.3},
    }

    # Conditionals are chosen consistent with P(A=a1)=0.4:
    # 0.7 * P(a1|b0) + 0.3 * P(a1|b1) = 0.4.
    p_a1_b1 = 0.6
    p_a1_b0 = (0.4 - 0.3 * p_a1_b1) / 0.7  # 0.314285714...

    # Convert to multipliers m = P(A=a|B=b)/P(A=a) for both outcomes a0,a1.
    multipliers = {
        (("A", "a1"), ("B", "b1")): p_a1_b1 / marginals["A"]["a1"],  # 1.5
        (("A", "a0"), ("B", "b1")): (1.0 - p_a1_b1) / marginals["A"]["a0"],
        (("A", "a1"), ("B", "b0")): p_a1_b0 / marginals["A"]["a1"],
        (("A", "a0"), ("B", "b0")): (1.0 - p_a1_b0) / marginals["A"]["a0"],
    }

    model = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers=multipliers)
    dist = model.fit_joint(method="direct", kl_weight=1e-8, solver_maxiter=3000)

    assert np.isclose(float(np.sum(dist.p)), 1.0, atol=1e-9)
    assert float(np.min(dist.p)) >= -1e-12

    implied_A = dist.marginal("A")
    implied_B = dist.marginal("B")
    assert abs(implied_A["a0"] - 0.6) < 1e-6
    assert abs(implied_A["a1"] - 0.4) < 1e-6
    assert abs(implied_B["b0"] - 0.7) < 1e-6
    assert abs(implied_B["b1"] - 0.3) < 1e-6

    # Check conditional recovery.
    implied = dist.conditional(("A", "a1"), ("B", "b1"))
    assert abs(float(implied) - float(p_a1_b1)) < 5e-4


def test_fit_direct_no_multipliers_honours_kl_baseline() -> None:
    """KL baseline should affect no-multiplier direct fits when enabled."""
    factors = [
        FactorSpec("A", ["a0", "a1"]),
        FactorSpec("B", ["b0", "b1"]),
    ]
    marginals = {
        "A": {"a0": 0.5, "a1": 0.5},
        "B": {"b0": 0.5, "b1": 0.5},
    }
    index = ScenarioIndex(factors)

    p = fit_joint_direct(
        index=index,
        marginals=marginals,
        multipliers={},
        kl_weight=50.0,
        kl_baseline=np.array([0.7, 0.1, 0.1, 0.1], dtype=float),
        solver_maxiter=4000,
    )

    baseline = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    assert np.max(np.abs(p - baseline)) > 1e-3
    assert np.isclose(float(np.sum(p)), 1.0, atol=1e-9)


def test_fit_direct_seed_none_forwards_nondeterministic_rng_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    original_default_rng = np.random.default_rng

    def _spy_default_rng(seed=None):  # type: ignore[no-untyped-def]
        calls.append(seed)
        return original_default_rng(0)

    class _DummyResult:
        def __init__(self, x: np.ndarray) -> None:
            self.success = True
            self.message = "dummy"
            self.x = x
            self.nit = 1
            self.status = 0

    def _dummy_minimize(fun, x0, **kwargs):  # type: ignore[no-untyped-def]
        return _DummyResult(np.asarray(x0, dtype=float))

    monkeypatch.setattr(fit_direct_mod.np.random, "default_rng", _spy_default_rng)
    monkeypatch.setattr(fit_direct_mod, "minimize", _dummy_minimize)

    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}
    index = ScenarioIndex(factors)
    multipliers = {(("A", "a1"), ("B", "b1")): 1.2}

    _ = fit_joint_direct(
        index=index,
        marginals=marginals,
        multipliers=multipliers,
        kl_weight=0.0,
        random_seed=None,
        solver_maxiter=10,
    )

    assert calls
    assert calls[0] is None


def test_fit_direct_explicit_seed_remains_reproducible() -> None:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}
    index = ScenarioIndex(factors)
    multipliers = {(("A", "a1"), ("B", "b1")): 1.2}

    p1 = fit_joint_direct(
        index=index,
        marginals=marginals,
        multipliers=multipliers,
        kl_weight=0.0,
        random_seed=123,
        solver_maxiter=4000,
    )
    p2 = fit_joint_direct(
        index=index,
        marginals=marginals,
        multipliers=multipliers,
        kl_weight=0.0,
        random_seed=123,
        solver_maxiter=4000,
    )
    assert np.allclose(p1, p2, atol=1e-12)


def test_fit_direct_seed_none_produces_variability_across_runs_with_stochastic_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyResult:
        def __init__(self, x: np.ndarray) -> None:
            self.success = True
            self.message = "dummy"
            self.x = x
            self.nit = 1
            self.status = 0

    def _dummy_minimize(fun, x0, **kwargs):  # type: ignore[no-untyped-def]
        return _DummyResult(np.asarray(x0, dtype=float))

    monkeypatch.setattr(fit_direct_mod, "minimize", _dummy_minimize)

    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}
    index = ScenarioIndex(factors)
    multipliers = {(("A", "a1"), ("B", "b1")): 1.2}

    signatures = []
    for _ in range(6):
        p = fit_joint_direct(
            index=index,
            marginals=marginals,
            multipliers=multipliers,
            kl_weight=0.0,
            random_seed=None,
            solver_maxiter=10,
        )
        signatures.append(tuple(float(x) for x in np.round(p, 10)))
    assert len(set(signatures)) > 1


def test_fit_direct_seed_none_variability_also_applies_with_kl_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyResult:
        def __init__(self, x: np.ndarray) -> None:
            self.success = True
            self.message = "dummy"
            self.x = x
            self.nit = 1
            self.status = 0

    def _dummy_minimize(fun, x0, **kwargs):  # type: ignore[no-untyped-def]
        return _DummyResult(np.asarray(x0, dtype=float))

    monkeypatch.setattr(fit_direct_mod, "minimize", _dummy_minimize)

    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.5, "a1": 0.5}, "B": {"b0": 0.5, "b1": 0.5}}
    index = ScenarioIndex(factors)
    multipliers = {(("A", "a1"), ("B", "b1")): 1.2}
    baseline = np.array([0.6, 0.2, 0.1, 0.1], dtype=float)

    signatures = []
    for _ in range(6):
        p = fit_joint_direct(
            index=index,
            marginals=marginals,
            multipliers=multipliers,
            kl_weight=1.0,
            kl_baseline=baseline,
            random_seed=None,
            solver_maxiter=10,
        )
        signatures.append(tuple(float(x) for x in np.round(p, 10)))
    assert len(set(signatures)) > 1


def test_fit_direct_rejects_negative_relevance_weights() -> None:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}
    index = ScenarioIndex(factors)
    multipliers = {(("A", "a1"), ("B", "b1")): 1.2}

    with pytest.raises(ValueError, match="non-negative"):
        _ = fit_joint_direct(
            index=index,
            marginals=marginals,
            multipliers=multipliers,
            relevance_weights={("A", "B"): -1.0},
        )

