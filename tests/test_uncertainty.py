"""
Unit tests for uncertainty modeling.

Tests UncertainCIBMatrix and confidence code handling for Monte Carlo
sampling of uncertain impact matrices.
"""

import pytest

from cib.uncertainty import ConfidenceMapper, UncertainCIBMatrix


class TestConfidenceMapper:
    """Test suite for ConfidenceMapper class."""

    def test_confidence_to_sigma(self) -> None:
        """Test confidence to sigma mapping."""
        assert ConfidenceMapper.confidence_to_sigma(5) == 0.2
        assert ConfidenceMapper.confidence_to_sigma(4) == 0.5
        assert ConfidenceMapper.confidence_to_sigma(3) == 0.8
        assert ConfidenceMapper.confidence_to_sigma(2) == 1.2
        assert ConfidenceMapper.confidence_to_sigma(1) == 1.5

    def test_invalid_confidence(self) -> None:
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError):
            ConfidenceMapper.confidence_to_sigma(0)

        with pytest.raises(ValueError):
            ConfidenceMapper.confidence_to_sigma(6)


class TestUncertainCIBMatrix:
    """Test suite for UncertainCIBMatrix class."""

    def test_initialization(self) -> None:
        """Test UncertainCIBMatrix initialization."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = UncertainCIBMatrix(descriptors)

        assert matrix.n_descriptors == 2
        assert matrix.default_confidence == 3

    def test_set_impact_with_confidence(self) -> None:
        """Test setting impact with confidence code."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = UncertainCIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0, confidence=4)
        assert matrix.get_impact("A", "Low", "B", "Weak") == 2.0
        assert matrix.get_confidence("A", "Low", "B", "Weak") == 4

    def test_get_confidence_default(self) -> None:
        """Test that unset confidence returns default."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = UncertainCIBMatrix(descriptors, default_confidence=5)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        assert matrix.get_confidence("A", "Low", "B", "Weak") == 5

    def test_set_impacts_with_confidence(self) -> None:
        """Test bulk setting impacts with confidence."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = UncertainCIBMatrix(descriptors)

        impacts = {
            ("A", "Low", "B", "Weak"): 2.0,
            ("A", "High", "B", "Strong"): -1.0,
        }
        confidence = {
            ("A", "Low", "B", "Weak"): 4,
            ("A", "High", "B", "Strong"): 3,
        }

        matrix.set_impacts(impacts, confidence=confidence)

        assert matrix.get_confidence("A", "Low", "B", "Weak") == 4
        assert matrix.get_confidence("A", "High", "B", "Strong") == 3

    def test_sample_matrix_reproducibility(self) -> None:
        """Test that sampling is reproducible with same seed."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = UncertainCIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0, confidence=3)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0, confidence=3)

        sampled1 = matrix.sample_matrix(seed=123)
        sampled2 = matrix.sample_matrix(seed=123)

        assert sampled1.get_impact("A", "Low", "B", "Weak") == sampled2.get_impact(
            "A", "Low", "B", "Weak"
        )

    def test_sample_matrix_structure(self) -> None:
        """Test that sampled matrix has correct structure."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = UncertainCIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0, confidence=3)

        sampled = matrix.sample_matrix(seed=123)

        assert sampled.n_descriptors == matrix.n_descriptors
        assert sampled.descriptors == matrix.descriptors
