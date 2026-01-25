"""
Unit tests for Monte Carlo consistency estimation.

Tests MonteCarloAnalyzer for probabilistic consistency probability
estimation and confidence interval calculation.
"""

import pytest

from cib.analysis import MonteCarloAnalyzer
from cib.core import Scenario
from cib.uncertainty import UncertainCIBMatrix


def _toy_descriptors() -> dict[str, list[str]]:
    return {
        "A": ["Low", "High"],
        "B": ["Low", "High"],
        "C": ["Low", "High"],
    }


def _toy_impacts() -> dict[tuple[str, str, str, str], float]:
    return {
        ("A", "Low", "B", "Low"): 2.0,
        ("A", "Low", "B", "High"): -2.0,
        ("A", "High", "B", "Low"): -2.0,
        ("A", "High", "B", "High"): 2.0,
        ("B", "Low", "C", "Low"): 2.0,
        ("B", "Low", "C", "High"): -2.0,
        ("B", "High", "C", "Low"): -2.0,
        ("B", "High", "C", "High"): 2.0,
        ("C", "Low", "A", "Low"): 2.0,
        ("C", "Low", "A", "High"): -2.0,
        ("C", "High", "A", "Low"): -2.0,
        ("C", "High", "A", "High"): 2.0,
    }


def _toy_confidence(impacts: dict[tuple[str, str, str, str], float], c: int = 3) -> dict[tuple[str, str, str, str], int]:
    return {k: int(c) for k in impacts}


class TestMonteCarloAnalyzer:
    """Test suite for MonteCarloAnalyzer class."""

    def test_initialization(self) -> None:
        """Test MonteCarloAnalyzer initialization."""
        desc = _toy_descriptors()
        impacts = _toy_impacts()
        matrix = UncertainCIBMatrix(desc)
        matrix.set_impacts(impacts, confidence=_toy_confidence(impacts, c=3))

        analyzer = MonteCarloAnalyzer(matrix, n_samples=1000, seed=123)

        assert analyzer.n_samples == 1000
        assert analyzer.base_seed == 123

    def test_estimate_consistency_probability(self) -> None:
        """Test consistency probability estimation."""
        desc = _toy_descriptors()
        impacts = _toy_impacts()
        matrix = UncertainCIBMatrix(desc)
        matrix.set_impacts(impacts, confidence=_toy_confidence(impacts, c=3))

        scenario = Scenario(
            {
                "A": "Low",
                "B": "Low",
                "C": "Low",
            },
            matrix,
        )

        analyzer = MonteCarloAnalyzer(matrix, n_samples=500, seed=123)
        prob = analyzer.estimate_consistency_probability(scenario)

        assert 0.0 <= prob <= 1.0

    def test_estimate_consistency_probability_reproducibility(self) -> None:
        """Test that estimation is reproducible with same seed."""
        desc = _toy_descriptors()
        impacts = _toy_impacts()
        matrix = UncertainCIBMatrix(desc)
        matrix.set_impacts(impacts, confidence=_toy_confidence(impacts, c=3))

        scenario = Scenario(
            {
                "A": "Low",
                "B": "Low",
                "C": "Low",
            },
            matrix,
        )

        analyzer1 = MonteCarloAnalyzer(matrix, n_samples=100, seed=123)
        analyzer2 = MonteCarloAnalyzer(matrix, n_samples=100, seed=123)

        prob1 = analyzer1.estimate_consistency_probability(scenario)
        prob2 = analyzer2.estimate_consistency_probability(scenario)

        assert prob1 == prob2

    def test_score_candidates(self) -> None:
        """Test scoring multiple candidate scenarios."""
        desc = _toy_descriptors()
        impacts = _toy_impacts()
        matrix = UncertainCIBMatrix(desc)
        matrix.set_impacts(impacts, confidence=_toy_confidence(impacts, c=3))

        scenarios = [
            Scenario(
                {
                    "A": "Low",
                    "B": "Low",
                    "C": "Low",
                },
                matrix,
            ),
            Scenario(
                {
                    "A": "High",
                    "B": "High",
                    "C": "High",
                },
                matrix,
            ),
        ]

        analyzer = MonteCarloAnalyzer(matrix, n_samples=200, seed=123)
        results = analyzer.score_candidates(scenarios)

        assert len(results.scenario_probabilities) == 2
        assert len(results.confidence_intervals) == 2
        assert results.n_samples == 200

        for scenario in scenarios:
            assert scenario in results.scenario_probabilities
            prob = results.scenario_probabilities[scenario]
            assert 0.0 <= prob <= 1.0

    def test_get_confidence_intervals(self) -> None:
        """Test confidence interval calculation."""
        desc = _toy_descriptors()
        impacts = _toy_impacts()
        matrix = UncertainCIBMatrix(desc)
        matrix.set_impacts(impacts, confidence=_toy_confidence(impacts, c=3))

        scenario = Scenario(
            {
                "A": "Low",
                "B": "Low",
                "C": "Low",
            },
            matrix,
        )

        analyzer = MonteCarloAnalyzer(matrix, n_samples=500, seed=123)
        ci = analyzer.get_confidence_intervals(scenario, level=0.95)

        assert len(ci) == 2
        lower, upper = ci
        assert 0.0 <= lower <= upper <= 1.0
