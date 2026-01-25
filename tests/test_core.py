"""
Unit tests for core CIB data structures.

Tests CIBMatrix, Scenario, ConsistencyChecker, and ImpactBalance classes
for correct initialization, data access, and basic operations.
"""

import pytest

from cib.core import CIBMatrix, ConsistencyChecker, ImpactBalance, Scenario


class TestCIBMatrix:
    """Test suite for CIBMatrix class."""

    def test_initialization(self) -> None:
        """Test CIBMatrix initialization with valid descriptors."""
        descriptors = {
            "A": ["Low", "High"],
            "B": ["Weak", "Strong"],
        }
        matrix = CIBMatrix(descriptors)

        assert matrix.n_descriptors == 2
        assert matrix.descriptors == descriptors
        assert matrix.state_counts == [2, 2]

    def test_initialization_empty(self) -> None:
        """Test that empty descriptors raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            CIBMatrix({})

    def test_initialization_duplicate_states(self) -> None:
        """Test that duplicate states raise ValueError."""
        with pytest.raises(ValueError, match="duplicate states"):
            CIBMatrix({"A": ["Low", "Low"]})

    def test_set_and_get_impact(self) -> None:
        """Test setting and retrieving impact values."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        assert matrix.get_impact("A", "Low", "B", "Weak") == 2.0

        matrix.set_impact("A", "High", "B", "Strong", -1.5)
        assert matrix.get_impact("A", "High", "B", "Strong") == -1.5

    def test_set_impact_self_impact_disallowed(self) -> None:
        """Test that diagonal/self-impacts are disallowed by convention."""
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)

        with pytest.raises(ValueError, match="Self-impacts"):
            matrix.set_impact("A", "Low", "A", "High", 1.0)

    def test_get_impact_default_zero(self) -> None:
        """Test that unset impacts default to zero."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        assert matrix.get_impact("A", "Low", "B", "Weak") == 0.0

    def test_set_impacts_bulk(self) -> None:
        """Test bulk setting of impacts."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        impacts = {
            ("A", "Low", "B", "Weak"): 2.0,
            ("A", "High", "B", "Strong"): -1.0,
        }
        matrix.set_impacts(impacts)

        assert matrix.get_impact("A", "Low", "B", "Weak") == 2.0
        assert matrix.get_impact("A", "High", "B", "Strong") == -1.0

    def test_calculate_impact_score(self) -> None:
        """Test impact score calculation."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        score = matrix.calculate_impact_score(scenario, "B", "Weak")

        assert score == 2.0

    def test_calculate_impact_balance(self) -> None:
        """Test impact balance calculation."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        balance = matrix.calculate_impact_balance(scenario, "B")

        assert "Weak" in balance
        assert "Strong" in balance
        assert balance["Weak"] == 2.0
        assert balance["Strong"] == -2.0


class TestScenario:
    """Test suite for Scenario class."""

    def test_initialization_from_dict(self) -> None:
        """Test Scenario creation from state dictionary."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)

        assert scenario.get_state("A") == "Low"
        assert scenario.get_state("B") == "Weak"

    def test_initialization_from_indices(self) -> None:
        """Test Scenario creation from index list."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        scenario = Scenario([0, 1], matrix)

        assert scenario.get_state("A") == "Low"
        assert scenario.get_state("B") == "Strong"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        state_dict = scenario.to_dict()

        assert state_dict == {"A": "Low", "B": "Weak"}

    def test_to_indices(self) -> None:
        """Test conversion to index list."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        scenario = Scenario({"A": "Low", "B": "Strong"}, matrix)
        indices = scenario.to_indices()

        assert indices == [0, 1]

    def test_equality(self) -> None:
        """Test scenario equality comparison."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        scenario1 = Scenario({"A": "Low", "B": "Weak"}, matrix)
        scenario2 = Scenario({"A": "Low", "B": "Weak"}, matrix)
        scenario3 = Scenario({"A": "High", "B": "Weak"}, matrix)

        assert scenario1 == scenario2
        assert scenario1 != scenario3

    def test_hash(self) -> None:
        """Test scenario hashing for use in sets."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        scenario1 = Scenario({"A": "Low", "B": "Weak"}, matrix)
        scenario2 = Scenario({"A": "Low", "B": "Weak"}, matrix)

        scenario_set = {scenario1, scenario2}
        assert len(scenario_set) == 1


class TestImpactBalance:
    """Test suite for ImpactBalance class."""

    def test_initialization(self) -> None:
        """Test ImpactBalance computation."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        balance = ImpactBalance(scenario, matrix)

        assert "B" in balance.balance
        assert balance.get_score("B", "Weak") == 2.0
        assert balance.get_score("B", "Strong") == -2.0

    def test_get_max_state(self) -> None:
        """Test finding maximum-impact state."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)

        scenario = Scenario({"A": "Low", "B": "Weak"}, matrix)
        balance = ImpactBalance(scenario, matrix)

        max_state = balance.get_max_state("B")
        assert max_state == "Weak"
