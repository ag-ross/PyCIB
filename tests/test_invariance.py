"""
Unit tests for invariance operations.

Tests that IO-1 and IO-2 operations preserve consistency relationships
in CIB matrices.
"""

import pytest

from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.succession import AttractorFinder


class TestInvarianceOperations:
    """Test suite for invariance operations."""

    def test_io1_preserves_consistency(self) -> None:
        """Test that IO-1 (addition) preserves consistency."""
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

        original_attractors = AttractorFinder(matrix).find_attractors_exhaustive(
            matrix
        )

        matrix.apply_io1("A", "Low", 5.0)

        modified_attractors = AttractorFinder(matrix).find_attractors_exhaustive(
            matrix
        )

        assert len(original_attractors) == len(modified_attractors)

        original_states = {
            tuple(attr.to_dict().items()) for attr in original_attractors
        }
        modified_states = {
            tuple(attr.to_dict().items()) for attr in modified_attractors
        }

        assert original_states == modified_states

    def test_io2_preserves_consistency(self) -> None:
        """Test that IO-2 (multiplication) preserves consistency."""
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

        original_attractors = AttractorFinder(matrix).find_attractors_exhaustive(
            matrix
        )

        matrix.apply_io2("B", 2.0)

        modified_attractors = AttractorFinder(matrix).find_attractors_exhaustive(
            matrix
        )

        assert len(original_attractors) == len(modified_attractors)

        original_states = {
            tuple(attr.to_dict().items()) for attr in original_attractors
        }
        modified_states = {
            tuple(attr.to_dict().items()) for attr in modified_attractors
        }

        assert original_states == modified_states

    def test_io2_positive_multiplier(self) -> None:
        """Test that IO-2 requires positive multiplier."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        with pytest.raises(ValueError, match="must be positive"):
            matrix.apply_io2("B", -1.0)

        with pytest.raises(ValueError, match="must be positive"):
            matrix.apply_io2("B", 0.0)

    def test_io3_preserves_consistency(self) -> None:
        """Test that IO-3 (global multiplication) preserves consistency."""
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

        original_attractors = AttractorFinder(matrix).find_attractors_exhaustive(
            matrix
        )

        matrix.apply_io3(3.0)

        modified_attractors = AttractorFinder(matrix).find_attractors_exhaustive(
            matrix
        )

        original_states = {
            tuple(attr.to_dict().items()) for attr in original_attractors
        }
        modified_states = {
            tuple(attr.to_dict().items()) for attr in modified_attractors
        }
        assert original_states == modified_states

    def test_io4_preserves_consistency(self) -> None:
        """Test that IO-4 (transfer) preserves consistency."""
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

        original_attractors = AttractorFinder(matrix).find_attractors_exhaustive(
            matrix
        )

        # Transfer a constant between A's judgment groups in section A -> B.
        matrix.apply_io4(
            src_desc="A",
            from_state="Low",
            to_state="High",
            tgt_desc="B",
            value=5.0,
        )

        modified_attractors = AttractorFinder(matrix).find_attractors_exhaustive(
            matrix
        )

        original_states = {
            tuple(attr.to_dict().items()) for attr in original_attractors
        }
        modified_states = {
            tuple(attr.to_dict().items()) for attr in modified_attractors
        }
        assert original_states == modified_states
