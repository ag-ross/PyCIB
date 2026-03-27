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

    def test_standardize_is_invariant_to_sparse_vs_explicit_zero(self) -> None:
        """Test standardise output is independent of sparse representation."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        sparse = CIBMatrix(descriptors)
        dense = CIBMatrix(descriptors)

        sparse.set_impact("A", "Low", "B", "Weak", 2.0)
        sparse.set_impact("B", "Strong", "A", "High", -1.0)

        dense.set_impact("A", "Low", "B", "Weak", 2.0)
        dense.set_impact("A", "Low", "B", "Strong", 0.0)
        dense.set_impact("A", "High", "B", "Weak", 0.0)
        dense.set_impact("A", "High", "B", "Strong", 0.0)
        dense.set_impact("B", "Weak", "A", "Low", 0.0)
        dense.set_impact("B", "Weak", "A", "High", 0.0)
        dense.set_impact("B", "Strong", "A", "Low", 0.0)
        dense.set_impact("B", "Strong", "A", "High", -1.0)

        sparse.standardize()
        dense.standardize()

        for src_desc in descriptors:
            for src_state in descriptors[src_desc]:
                for tgt_desc in descriptors:
                    if src_desc == tgt_desc:
                        continue
                    vals_sparse = []
                    vals_dense = []
                    for tgt_state in descriptors[tgt_desc]:
                        vals_sparse.append(
                            sparse.get_impact(src_desc, src_state, tgt_desc, tgt_state)
                        )
                        vals_dense.append(
                            dense.get_impact(src_desc, src_state, tgt_desc, tgt_state)
                        )
                    assert sum(vals_sparse) == pytest.approx(0.0)
                    assert vals_sparse == pytest.approx(vals_dense)
