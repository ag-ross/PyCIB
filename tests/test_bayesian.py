"""
Unit tests for expert aggregation and Gaussian uncertainty matrices.
"""

from __future__ import annotations

import math

from cib.analysis import MonteCarloAnalyzer
from cib.bayesian import ExpertAggregator, GaussianCIBMatrix
from cib.core import Scenario


class TestExpertAggregator:
    """Test suite for ExpertAggregator."""

    def test_weighted_mean_and_variance(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        key = ("A", "Low", "B", "Weak")

        expert1_impacts = {key: 2.0}
        expert2_impacts = {key: 0.0}

        agg = ExpertAggregator(descriptors)
        agg.add_expert(expert1_impacts, weight=0.75, confidence={key: 5})
        agg.add_expert(expert2_impacts, weight=0.25, confidence={key: 3})

        matrix = agg.aggregate()
        assert isinstance(matrix, GaussianCIBMatrix)

        mu = matrix.get_impact(*key)
        sigma = matrix.get_sigma(key)

        assert math.isclose(mu, 1.5, rel_tol=0.0, abs_tol=1e-12)

        # Expected:
        #   σ1=0.2 (c=5), σ2=0.8 (c=3)
        #   var_within = (0.75^2)*0.2^2 + (0.25^2)*0.8^2 = 0.0625
        #   var_between = 0.75*(2-1.5)^2 + 0.25*(0-1.5)^2 = 0.75
        #   total = 0.8125
        expected_sigma = math.sqrt(0.8125)
        assert math.isclose(sigma, expected_sigma, rel_tol=0.0, abs_tol=1e-9)

    def test_partial_coverage_renormalizes_weights(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        key1 = ("A", "Low", "B", "Weak")
        key2 = ("A", "High", "B", "Strong")

        expert1_impacts = {key1: 2.0, key2: -1.0}
        expert2_impacts = {key1: 0.0}  # missing key2

        agg = ExpertAggregator(descriptors)
        agg.add_expert(expert1_impacts, weight=0.6, confidence={key1: 4, key2: 4})
        agg.add_expert(expert2_impacts, weight=0.4, confidence={key1: 4})

        matrix = agg.aggregate()

        # key2 is expected to match expert1 only (weights renormalize to 1.0 on the single contributor).
        assert matrix.get_impact(*key2) == -1.0
        assert math.isclose(matrix.get_sigma(key2), 0.5, abs_tol=1e-12)  # c=4 => 0.5

    def test_monte_carlo_accepts_gaussian_matrix(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = GaussianCIBMatrix(descriptors, default_sigma=0.5)

        # A influences B.
        matrix.set_impact("A", "Low", "B", "Weak", 2.0, sigma=0.5)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0, sigma=0.5)
        matrix.set_impact("A", "High", "B", "Weak", -2.0, sigma=0.5)
        matrix.set_impact("A", "High", "B", "Strong", 2.0, sigma=0.5)

        # B influences A.
        matrix.set_impact("B", "Weak", "A", "Low", 1.0, sigma=0.5)
        matrix.set_impact("B", "Weak", "A", "High", -1.0, sigma=0.5)
        matrix.set_impact("B", "Strong", "A", "Low", -1.0, sigma=0.5)
        matrix.set_impact("B", "Strong", "A", "High", 1.0, sigma=0.5)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        mc = MonteCarloAnalyzer(matrix, n_samples=50, seed=123)
        p = mc.estimate_consistency_probability(scenario)
        assert 0.0 <= p <= 1.0

