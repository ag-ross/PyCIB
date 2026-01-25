"""
Unit tests for succession operators.

Tests GlobalSuccession, LocalSuccession, and AttractorFinder for
correct convergence behavior and attractor identification.
"""

import pytest

from cib.core import CIBMatrix, Scenario
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
