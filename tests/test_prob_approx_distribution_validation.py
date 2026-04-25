"""
Unit tests for approximate joint-distribution support validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cib.prob.approx import ApproxJointDistribution
from cib.prob.types import FactorSpec, ScenarioIndex


def _tiny_index() -> ScenarioIndex:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    return ScenarioIndex(factors)


def test_approx_joint_distribution_rejects_duplicate_support_indices() -> None:
    index = _tiny_index()
    with pytest.raises(ValueError, match="support_indices must be unique"):
        ApproxJointDistribution(
            index=index,
            support_indices=np.array([0, 0, 3], dtype=int),
            support_probabilities=np.array([0.2, 0.3, 0.5], dtype=float),
            n_samples=100,
        )
