"""
Input-contract tests for probabilistic fit methods.
"""

from __future__ import annotations

import pytest

from cib.prob.graph import RelevanceSpec
from cib.prob.model import ProbabilisticCIAModel
from cib.prob.types import FactorSpec


def _base_model(multipliers):
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}
    return ProbabilisticCIAModel(
        factors=factors,
        marginals=marginals,
        multipliers=multipliers,
    )


def test_model_rejects_unknown_multiplier_factor() -> None:
    with pytest.raises(ValueError, match="Unknown multiplier factor"):
        _ = _base_model({(("Z", "a1"), ("B", "b1")): 1.1})


def test_model_rejects_unknown_multiplier_outcome() -> None:
    with pytest.raises(ValueError, match="Unknown multiplier outcome"):
        _ = _base_model({(("A", "a2"), ("B", "b1")): 1.1})


def test_model_rejects_non_positive_multiplier() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _ = _base_model({(("A", "a1"), ("B", "b1")): 0.0})


def test_model_rejects_same_factor_multiplier_key() -> None:
    with pytest.raises(ValueError, match="must differ"):
        _ = _base_model({(("A", "a0"), ("A", "a1")): 1.1})


def test_fit_joint_iterative_rejects_invalid_sampling_parameters() -> None:
    model = _base_model({})
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        model.fit_joint(method="iterative", iterative_n_samples=0)
    with pytest.raises(ValueError, match="thinning must be >= 1"):
        model.fit_joint(method="iterative", iterative_thinning=0)
    with pytest.raises(ValueError, match="burn_in_sweeps must be >= 0"):
        model.fit_joint(method="iterative", iterative_burn_in_sweeps=-1)
    with pytest.raises(ValueError, match="eps must be > 0"):
        model.fit_joint(method="iterative", iterative_eps=0.0)


def test_fit_joint_direct_rejects_negative_relevance_weights() -> None:
    model = _base_model({(("A", "a1"), ("B", "b1")): 1.1})
    with pytest.raises(ValueError, match="non-negative"):
        model.fit_joint(
            method="direct",
            relevance=RelevanceSpec(parents={"A": {"B"}}, weights={("A", "B"): -0.5}),
            relevance_default_weight=0.0,
        )
