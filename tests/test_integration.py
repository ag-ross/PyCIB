"""
Integration tests for Phase 2 components.

Tests that deterministic analysis, Monte Carlo estimation, and robustness
testing work together correctly.
"""

from cib.analysis import MonteCarloAnalyzer, ScenarioAnalyzer
from cib.core import Scenario
from cib.shocks import RobustnessTester, ShockModel
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


class TestPhase2Integration:
    """Test suite for Phase 2 component integration."""

    def test_full_workflow(self) -> None:
        """Test complete Phase 2 workflow with a small synthetic fixture."""
        desc = _toy_descriptors()
        impacts = _toy_impacts()
        confidence = {k: 3 for k in impacts}
        matrix = UncertainCIBMatrix(desc)
        matrix.set_impacts(impacts, confidence=confidence)

        analyzer = ScenarioAnalyzer(matrix)
        candidates = analyzer.find_all_consistent()

        assert len(candidates) >= 1

        mc_analyzer = MonteCarloAnalyzer(matrix, n_samples=200, seed=123)
        mc_results = mc_analyzer.score_candidates(candidates)

        assert len(mc_results.scenario_probabilities) == len(candidates)

        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25)

        robustness_tester = RobustnessTester(matrix, shock_model)
        robustness_scores = robustness_tester.test_scenarios(
            candidates, n_simulations=100, seed=123
        )

        assert len(robustness_scores) == len(candidates)

    def test_high_confidence_low_uncertainty(self) -> None:
        """Test behavior with high confidence (low uncertainty)."""
        desc = _toy_descriptors()
        impacts = _toy_impacts()
        matrix = UncertainCIBMatrix(desc, default_confidence=5)

        high_confidence_impacts = {}
        high_confidence_conf = {}
        for key, value in impacts.items():
            high_confidence_impacts[key] = value
            high_confidence_conf[key] = 5

        matrix.set_impacts(high_confidence_impacts, confidence=high_confidence_conf)

        scenario = Scenario(
            {
                "A": "Low",
                "B": "Low",
                "C": "Low",
            },
            matrix,
        )

        mc_analyzer = MonteCarloAnalyzer(matrix, n_samples=200, seed=123)
        prob = mc_analyzer.estimate_consistency_probability(scenario)

        assert 0.0 <= prob <= 1.0

    def test_low_confidence_high_uncertainty(self) -> None:
        """Test behavior with low confidence (high uncertainty)."""
        desc = _toy_descriptors()
        impacts = _toy_impacts()
        matrix = UncertainCIBMatrix(desc, default_confidence=1)

        low_confidence_impacts = {}
        low_confidence_conf = {}
        for key, value in impacts.items():
            low_confidence_impacts[key] = value
            low_confidence_conf[key] = 1

        matrix.set_impacts(low_confidence_impacts, confidence=low_confidence_conf)

        scenario = Scenario(
            {
                "A": "Low",
                "B": "Low",
                "C": "Low",
            },
            matrix,
        )

        mc_analyzer = MonteCarloAnalyzer(matrix, n_samples=200, seed=123)
        prob = mc_analyzer.estimate_consistency_probability(scenario)

        assert 0.0 <= prob <= 1.0
