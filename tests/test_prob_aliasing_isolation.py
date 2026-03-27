"""
Regression tests for probabilistic model boundary mutation isolation.
"""

from __future__ import annotations

import numpy as np

from cib.prob.model import ProbabilisticCIAModel
from cib.prob.types import FactorSpec


def _base_inputs():
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}
    multipliers = {
        (("A", "a1"), ("B", "b1")): 1.4,
        (("A", "a0"), ("B", "b1")): 0.7,
        (("A", "a1"), ("B", "b0")): 0.8,
        (("A", "a0"), ("B", "b0")): 1.1,
    }
    return factors, marginals, multipliers


def test_prob_model_copies_caller_owned_inputs() -> None:
    """Post-init mutations of caller mappings should not alter model internals."""
    factors, marginals, multipliers = _base_inputs()
    model = ProbabilisticCIAModel(
        factors=factors,
        marginals=marginals,
        multipliers=multipliers,
    )

    marginals["A"]["a0"] = 0.1
    marginals["A"]["a1"] = 0.9
    multipliers[(("A", "a1"), ("B", "b1"))] = 9.0
    multipliers[(("A", "a0"), ("B", "b0"))] = 9.0

    assert model.marginals["A"]["a0"] == 0.6
    assert model.marginals["A"]["a1"] == 0.4
    assert model.multipliers[(("A", "a1"), ("B", "b1"))] == 1.4
    assert model.multipliers[(("A", "a0"), ("B", "b0"))] == 1.1


def test_prob_model_fit_is_stable_after_external_input_mutation() -> None:
    """Dense fit output should be unaffected by external post-init mutations."""
    factors, marginals, multipliers = _base_inputs()
    model = ProbabilisticCIAModel(
        factors=factors,
        marginals=marginals,
        multipliers=multipliers,
    )
    before = model.fit_joint(
        method="direct",
        random_seed=123,
        solver_maxiter=3000,
    )

    marginals["B"]["b0"] = 0.2
    marginals["B"]["b1"] = 0.8
    multipliers[(("A", "a1"), ("B", "b0"))] = 3.5

    after = model.fit_joint(
        method="direct",
        random_seed=123,
        solver_maxiter=3000,
    )
    np.testing.assert_allclose(
        np.asarray(before.p), np.asarray(after.p), atol=1e-12, rtol=0.0
    )


def test_prob_model_iterative_fit_is_stable_after_external_input_mutation() -> None:
    """Iterative fit output should be unaffected by external post-init mutations."""
    factors, marginals, multipliers = _base_inputs()
    model = ProbabilisticCIAModel(
        factors=factors,
        marginals=marginals,
        multipliers=multipliers,
    )
    before = model.fit_joint(
        method="iterative",
        random_seed=321,
        iterative_burn_in_sweeps=200,
        iterative_n_samples=1200,
        iterative_thinning=1,
    )

    marginals["A"]["a0"] = 0.2
    marginals["A"]["a1"] = 0.8
    multipliers[(("A", "a0"), ("B", "b1"))] = 2.0

    after = model.fit_joint(
        method="iterative",
        random_seed=321,
        iterative_burn_in_sweeps=200,
        iterative_n_samples=1200,
        iterative_thinning=1,
    )
    np.testing.assert_array_equal(
        np.asarray(before.support_indices), np.asarray(after.support_indices)
    )
    np.testing.assert_allclose(
        np.asarray(before.support_probabilities),
        np.asarray(after.support_probabilities),
        atol=1e-12,
        rtol=0.0,
    )
