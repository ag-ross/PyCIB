"""
Unit tests for consistency checking.

Tests ConsistencyChecker for correct validation of scenarios against
the CIB consistency principle.
"""

import pytest

from cib.core import CIBMatrix, ConsistencyChecker, Scenario


class TestConsistencyChecker:
    """Test suite for ConsistencyChecker class."""

    def test_consistent_scenario(self) -> None:
        """Test checking a consistent scenario."""
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
        is_consistent = ConsistencyChecker.check_consistency(scenario, matrix)

        assert is_consistent is True

    def test_inconsistent_scenario(self) -> None:
        """Test checking an inconsistent scenario."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", -2.0)
        matrix.set_impact("A", "Low", "B", "Strong", 2.0)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        is_consistent = ConsistencyChecker.check_consistency(scenario, matrix)

        assert is_consistent is False

    def test_check_consistency_detailed(self) -> None:
        """Test detailed consistency checking."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", -2.0)
        matrix.set_impact("A", "Low", "B", "Strong", 2.0)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        result = ConsistencyChecker.check_consistency_detailed(
            scenario, matrix
        )

        assert result["is_consistent"] is False
        assert "balances" in result
        assert "inconsistencies" in result
        assert len(result["inconsistencies"]) > 0

    def test_find_inconsistent_descriptors(self) -> None:
        """Test finding inconsistent descriptors."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", -2.0)
        matrix.set_impact("A", "Low", "B", "Strong", 2.0)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        inconsistent = ConsistencyChecker.find_inconsistent_descriptors(
            scenario, matrix
        )

        assert "B" in inconsistent

    def test_all_zero_impacts(self) -> None:
        """Test consistency with all zero impacts."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        is_consistent = ConsistencyChecker.check_consistency(scenario, matrix)

        assert is_consistent is True

    def test_single_descriptor(self) -> None:
        """Test consistency with single descriptor."""
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)

        scenario = Scenario({"A": "Low"}, matrix)
        is_consistent = ConsistencyChecker.check_consistency(scenario, matrix)

        assert is_consistent is True

    def test_detailed_consistency_respects_tolerance_near_tie(self) -> None:
        """Detailed helper should respect caller-provided floating tolerances."""
        descriptors = {"A": ["a0", "a1"], "B": ["b0", "b1"]}
        matrix = CIBMatrix(descriptors)
        matrix.set_impact("A", "a0", "B", "b0", 1.0)
        matrix.set_impact("A", "a0", "B", "b1", 1.0 + 5e-9)

        scenario = Scenario({"A": "a0", "B": "b0"}, matrix)
        strict = ConsistencyChecker.check_consistency_detailed(
            scenario,
            matrix,
            float_atol=0.0,
            float_rtol=0.0,
        )
        tolerant = ConsistencyChecker.check_consistency_detailed(
            scenario,
            matrix,
            float_atol=1e-8,
            float_rtol=0.0,
        )

        assert strict["is_consistent"] is False
        assert tolerant["is_consistent"] is True

    def test_find_inconsistent_descriptors_respects_tolerance_near_tie(self) -> None:
        """Inconsistent descriptor helper should use provided tolerances."""
        descriptors = {"A": ["a0", "a1"], "B": ["b0", "b1"]}
        matrix = CIBMatrix(descriptors)
        matrix.set_impact("A", "a0", "B", "b0", 1.0)
        matrix.set_impact("A", "a0", "B", "b1", 1.0 + 5e-9)

        scenario = Scenario({"A": "a0", "B": "b0"}, matrix)
        strict = ConsistencyChecker.find_inconsistent_descriptors(
            scenario,
            matrix,
            float_atol=0.0,
            float_rtol=0.0,
        )
        tolerant = ConsistencyChecker.find_inconsistent_descriptors(
            scenario,
            matrix,
            float_atol=1e-8,
            float_rtol=0.0,
        )

        assert "B" in strict
        assert "B" not in tolerant
