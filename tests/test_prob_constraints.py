"""
Unit tests for probabilistic marginal validation.
"""

from __future__ import annotations

import pytest

from cib.prob.constraints import validate_marginals
from cib.prob.types import FactorSpec


def _factors():
    return [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]


def test_validate_marginals_rejects_nan_values() -> None:
    marginals = {"A": {"a0": 0.5, "a1": float("nan")}, "B": {"b0": 0.4, "b1": 0.6}}
    with pytest.raises(ValueError, match="Non-finite marginal probability"):
        validate_marginals(_factors(), marginals)


def test_validate_marginals_rejects_inf_values() -> None:
    marginals = {"A": {"a0": 0.5, "a1": 0.5}, "B": {"b0": float("inf"), "b1": 0.0}}
    with pytest.raises(ValueError, match="Non-finite marginal probability"):
        validate_marginals(_factors(), marginals)


def test_validate_marginals_rejects_unknown_factor_keys() -> None:
    marginals = {
        "A": {"a0": 0.5, "a1": 0.5},
        "B": {"b0": 0.5, "b1": 0.5},
        "C": {"c0": 1.0},
    }
    with pytest.raises(ValueError, match="Unknown marginal factors"):
        validate_marginals(_factors(), marginals)


def test_validate_marginals_rejects_unknown_outcome_keys() -> None:
    marginals = {
        "A": {"a0": 0.5, "a1": 0.5, "a2": 0.0},
        "B": {"b0": 0.5, "b1": 0.5},
    }
    with pytest.raises(ValueError, match="Unknown marginal outcomes"):
        validate_marginals(_factors(), marginals)
