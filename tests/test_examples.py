"""
Unit tests for validation examples.

These tests use small synthetic fixtures to validate behavior without relying on
bundled example datasets (the repository keeps a single canonical 5-state demo).
"""

from cib.analysis import ScenarioAnalyzer
from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.succession import AttractorFinder


def _toy_descriptors() -> dict[str, list[str]]:
    return {
        "A": ["Low", "High"],
        "B": ["Low", "High"],
        "C": ["Low", "High"],
    }

def _toy_impacts() -> dict[tuple[str, str, str, str], float]:
    """
    Simple coordination structure with at least one fixed point.
    """
    # Each descriptor prefers to match the previous one in a cycle.
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

class TestSyntheticExamples:
    """Validation tests using synthetic fixtures."""

    def test_find_consistent_scenarios(self) -> None:
        """Find at least one consistent scenario."""
        matrix = CIBMatrix(_toy_descriptors())
        matrix.set_impacts(_toy_impacts())

        analyzer = ScenarioAnalyzer(matrix)
        consistent = analyzer.find_all_consistent()

        assert len(consistent) >= 1

        for scenario in consistent:
            is_consistent = ConsistencyChecker.check_consistency(
                scenario, matrix
            )
            assert is_consistent is True

    def test_succession_convergence(self) -> None:
        """Succession converges to an attractor."""
        matrix = CIBMatrix(_toy_descriptors())
        matrix.set_impacts(_toy_impacts())

        initial = Scenario(
            {
                "A": "Low",
                "B": "Low",
                "C": "Low",
            },
            matrix,
        )

        finder = AttractorFinder(matrix)
        attractors = finder.find_attractors(initial_scenarios=[initial])

        assert len(attractors) >= 1
