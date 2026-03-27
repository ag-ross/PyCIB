"""
Unit tests for succession operators.

Tests GlobalSuccession, LocalSuccession, and AttractorFinder for
correct convergence behaviour and attractor identification.
"""

import pytest

from cib.core import CIBMatrix, Scenario
from cib.dynamic import DynamicCIB
from cib.succession import (
    AttractorFinder,
    AttractorResult,
    GlobalSuccession,
    LocalSuccession,
    SuccessionOperator,
)


class TestGlobalSuccession:
    """Test suite for GlobalSuccession operator."""

    def test_find_successor(self) -> None:
        """Test finding successor scenario."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)

        scenario = Scenario({"A": "Low", "B": "Strong"}, matrix)
        succession = GlobalSuccession()
        successor = succession.find_successor(scenario, matrix)

        assert successor.get_state("B") == "Weak"

    def test_find_attractor_fixed_point(self) -> None:
        """Test finding fixed point attractor."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)
        matrix.set_impact("A", "High", "B", "Weak", -2.0)
        matrix.set_impact("A", "High", "B", "Strong", 2.0)

        matrix.set_impact("B", "Weak", "A", "Low", 1.0)
        matrix.set_impact("B", "Weak", "A", "High", -1.0)
        matrix.set_impact("B", "Strong", "A", "Low", -1.0)
        matrix.set_impact("B", "Strong", "A", "High", 1.0)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        succession = GlobalSuccession()
        result = succession.find_attractor(scenario, matrix)

        assert isinstance(result, AttractorResult)
        assert result.is_cycle is False
        assert isinstance(result.attractor, Scenario)

    def test_find_attractor_max_iterations(self) -> None:
        """Test that max_iterations parameter is accepted."""
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)

        scenario = Scenario({"A": "Low"}, matrix)
        succession = GlobalSuccession()
        result = succession.find_attractor(scenario, matrix, max_iterations=5)

        assert result.is_cycle is False
        assert isinstance(result.attractor, Scenario)

    def test_find_attractor_allow_partial_returns_last_state(self) -> None:
        """When allow_partial is True and cap is hit, last state is returned with converged=False."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)
        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)
        matrix.set_impact("A", "High", "B", "Weak", -2.0)
        matrix.set_impact("A", "High", "B", "Strong", 2.0)
        matrix.set_impact("B", "Weak", "A", "Low", 1.0)
        matrix.set_impact("B", "Weak", "A", "High", -1.0)
        matrix.set_impact("B", "Strong", "A", "Low", -1.0)
        matrix.set_impact("B", "Strong", "A", "High", 1.0)

        # From (Low, Strong) one GlobalSuccession step updates both: A->High, B->Weak; with max_iterations=1 we do not converge.
        scenario = Scenario({"A": "Low", "B": "Strong"}, matrix)
        succession = GlobalSuccession()
        result = succession.find_attractor(
            scenario, matrix, max_iterations=1, allow_partial=True
        )
        assert result.converged is False
        assert result.iterations == 1
        assert isinstance(result.attractor, Scenario)
        assert result.attractor.get_state("A") == "High"
        assert result.attractor.get_state("B") == "Weak"
        assert len(result.path) == 2

    def test_find_attractor_allow_partial_false_raises(self) -> None:
        """When allow_partial is False and cap is hit, RuntimeError is raised."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)
        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)
        matrix.set_impact("A", "High", "B", "Weak", -2.0)
        matrix.set_impact("A", "High", "B", "Strong", 2.0)
        matrix.set_impact("B", "Weak", "A", "Low", 1.0)
        matrix.set_impact("B", "Weak", "A", "High", -1.0)
        matrix.set_impact("B", "Strong", "A", "Low", -1.0)
        matrix.set_impact("B", "Strong", "A", "High", 1.0)

        scenario = Scenario({"A": "Low", "B": "Strong"}, matrix)
        succession = GlobalSuccession()
        with pytest.raises(RuntimeError, match="did not converge"):
            succession.find_attractor(
                scenario, matrix, max_iterations=1, allow_partial=False
            )


class TestLocalSuccession:
    """Test suite for LocalSuccession operator."""

    def test_find_successor(self) -> None:
        """Test finding successor with local update."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)

        scenario = Scenario({"A": "Low", "B": "Strong"}, matrix)
        succession = LocalSuccession()
        successor = succession.find_successor(scenario, matrix)

        assert successor.get_state("B") == "Weak"


class TestAttractorFinder:
    """Test suite for AttractorFinder class."""

    def test_find_attractors_exhaustive(self) -> None:
        """Test exhaustive attractor finding."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)
        matrix.set_impact("A", "High", "B", "Weak", -2.0)
        matrix.set_impact("A", "High", "B", "Strong", 2.0)

        matrix.set_impact("B", "Weak", "A", "Low", 1.0)
        matrix.set_impact("B", "Weak", "A", "High", -1.0)
        matrix.set_impact("B", "Strong", "A", "Low", -1.0)
        matrix.set_impact("B", "Strong", "A", "High", 1.0)

        finder = AttractorFinder(matrix)
        attractors = finder.find_attractors_exhaustive(matrix)

        assert len(attractors) >= 1

    def test_get_basin(self) -> None:
        """Test finding basin of attraction."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)
        matrix.set_impact("A", "High", "B", "Weak", -2.0)
        matrix.set_impact("A", "High", "B", "Strong", 2.0)

        matrix.set_impact("B", "Weak", "A", "Low", 1.0)
        matrix.set_impact("B", "Weak", "A", "High", -1.0)
        matrix.set_impact("B", "Strong", "A", "Low", -1.0)
        matrix.set_impact("B", "Strong", "A", "High", 1.0)

        finder = AttractorFinder(matrix)
        attractors = finder.find_attractors_exhaustive(matrix)

        if attractors:
            basin = finder.get_basin(attractors[0], matrix)
            assert len(basin) >= 1


class TestTimeToEquilibrium:
    def test_trace_to_equilibrium_reports_non_negative_time(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        matrix.set_impact("B", "Low", "A", "Low", 2.0)
        matrix.set_impact("B", "Low", "A", "High", -2.0)
        matrix.set_impact("B", "High", "A", "Low", 2.0)
        matrix.set_impact("B", "High", "A", "High", -2.0)

        dyn = DynamicCIB(matrix, periods=[1])
        path = dyn.trace_to_equilibrium(initial={"A": "High", "B": "High"})

        assert path.disequilibrium_metrics[-1].time_to_equilibrium == 0

    def test_trace_to_equilibrium_returns_none_when_consistent_set_is_never_entered(self) -> None:
        descriptors = {
            "A": ["Low", "High"],
            "B": ["Low", "High"],
            "C": ["Low", "High"],
        }
        matrix = CIBMatrix(descriptors)

        matrix.set_impacts(
            {
                ("A", "Low", "B", "Low"): -2.0,
                ("A", "Low", "B", "High"): 2.0,
                ("A", "High", "B", "Low"): 1.0,
                ("A", "High", "B", "High"): 2.0,
                ("A", "Low", "C", "High"): 2.0,
                ("A", "High", "C", "Low"): -1.0,
                ("A", "High", "C", "High"): -1.0,
                ("B", "Low", "A", "Low"): 2.0,
                ("B", "Low", "A", "High"): 1.0,
                ("B", "High", "A", "Low"): 2.0,
                ("B", "Low", "C", "Low"): 1.0,
                ("B", "Low", "C", "High"): 1.0,
                ("B", "High", "C", "High"): -2.0,
                ("C", "Low", "A", "High"): 2.0,
                ("C", "High", "A", "Low"): -2.0,
                ("C", "High", "A", "High"): 1.0,
                ("C", "Low", "B", "Low"): 2.0,
                ("C", "High", "B", "Low"): -1.0,
                ("C", "High", "B", "High"): -1.0,
            }
        )

        dyn = DynamicCIB(matrix, periods=[1])
        path = dyn.trace_to_equilibrium(
            initial={"A": "Low", "B": "Low", "C": "Low"},
            max_iterations=3,
        )

        assert all(
            metric.entered_consistent_set is False
            for metric in path.disequilibrium_metrics
        )
        assert path.disequilibrium_metrics[0].time_to_equilibrium is None
