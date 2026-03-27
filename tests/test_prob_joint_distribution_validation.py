"""
Unit tests for dense joint-distribution simplex validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cib.prob.model import JointDistribution
from cib.prob.types import FactorSpec, ScenarioIndex


def _tiny_index() -> ScenarioIndex:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    return ScenarioIndex(factors)


def test_joint_distribution_accepts_valid_probability_vector() -> None:
    """Accept finite, non-negative vectors summing to one."""
    index = _tiny_index()
    p = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    dist = JointDistribution(index=index, p=p)
    assert float(np.sum(dist.p)) == pytest.approx(1.0)


def test_joint_distribution_rejects_negative_probability_mass() -> None:
    """Reject vectors with negative probability entries."""
    index = _tiny_index()
    with pytest.raises(ValueError, match="non-negative"):
        JointDistribution(index=index, p=np.array([0.5, 0.5, -1e-3, 0.0], dtype=float))


def test_joint_distribution_rejects_nan_or_inf_values() -> None:
    """Reject vectors containing non-finite values."""
    index = _tiny_index()
    with pytest.raises(ValueError, match="finite"):
        JointDistribution(index=index, p=np.array([0.5, np.nan, 0.5, 0.0], dtype=float))
    with pytest.raises(ValueError, match="finite"):
        JointDistribution(index=index, p=np.array([0.5, np.inf, 0.5, 0.0], dtype=float))


def test_joint_distribution_rejects_non_normalized_mass() -> None:
    """Reject vectors whose total mass is not one within tolerance."""
    index = _tiny_index()
    with pytest.raises(ValueError, match="sum to 1"):
        JointDistribution(index=index, p=np.array([0.4, 0.3, 0.2, 0.0], dtype=float))
