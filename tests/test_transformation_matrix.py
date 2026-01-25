"""
Unit tests for transformation matrix computation.

Tests TransformationMatrixBuilder, PerturbationInfo, and TransformationMatrix.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from cib.core import CIBMatrix, Scenario
from cib.example_data import (
    DATASET_B5_CONFIDENCE,
    DATASET_B5_DESCRIPTORS,
    DATASET_B5_IMPACTS,
)
from cib.transformation_matrix import (
    PerturbationInfo,
    TransformationMatrix,
    TransformationMatrixBuilder,
    _hamming_distance,
    _scenarios_match,
)
from cib.uncertainty import UncertainCIBMatrix
from cib.visualization import ScenarioVisualizer


class TestPerturbationInfo:
    """Test suite for PerturbationInfo dataclass."""

    def test_creation(self) -> None:
        """Test creating PerturbationInfo objects."""
        info = PerturbationInfo(
            perturbation_type="structural",
            magnitude=0.2,
            success_rate=0.75,
            details={"sigma": 0.2},
        )

        assert info.perturbation_type == "structural"
        assert info.magnitude == 0.2
        assert info.success_rate == 0.75
        assert info.details == {"sigma": 0.2}


class TestTransformationMatrix:
    """Test suite for TransformationMatrix dataclass."""

    def test_creation(self) -> None:
        """Test creating TransformationMatrix objects."""
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        scenarios = [
            Scenario({"A": "Low"}, matrix),
            Scenario({"A": "High"}, matrix),
        ]

        transformations = {}
        summary_stats = {"total_scenarios": 2}

        tm = TransformationMatrix(
            scenarios=scenarios,
            transformations=transformations,
            summary_stats=summary_stats,
        )

        assert len(tm.scenarios) == 2
        assert len(tm.transformations) == 0
        assert tm.summary_stats["total_scenarios"] == 2


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_hamming_distance(self) -> None:
        """Test Hamming distance calculation."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        scenario1 = Scenario({"A": "Low", "B": "Weak"}, matrix)
        scenario2 = Scenario({"A": "Low", "B": "Strong"}, matrix)
        scenario3 = Scenario({"A": "High", "B": "Strong"}, matrix)

        assert _hamming_distance(scenario1, scenario1) == 0
        assert _hamming_distance(scenario1, scenario2) == 1
        assert _hamming_distance(scenario1, scenario3) == 2

    def test_scenarios_match(self) -> None:
        """Test scenario matching logic."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        scenario1 = Scenario({"A": "Low", "B": "Weak"}, matrix)
        scenario2 = Scenario({"A": "Low", "B": "Strong"}, matrix)
        scenario3 = Scenario({"A": "High", "B": "Strong"}, matrix)

        assert _scenarios_match(scenario1, scenario1) is True
        assert _scenarios_match(scenario1, scenario2, max_hamming=1) is True
        assert _scenarios_match(scenario1, scenario3, max_hamming=1) is False
        assert _scenarios_match(scenario1, scenario3, max_hamming=2) is True


class TestTransformationMatrixBuilder:
    """Test suite for TransformationMatrixBuilder class."""

    def test_initialization(self) -> None:
        """Test builder initialization."""
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)

        builder = TransformationMatrixBuilder(base_matrix=matrix)
        assert builder.base_matrix == matrix

    def test_build_matrix_empty_scenarios(self) -> None:
        """Test that empty scenarios list raises ValueError."""
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)

        builder = TransformationMatrixBuilder(base_matrix=matrix)
        with pytest.raises(ValueError, match="scenarios list cannot be empty"):
            builder.build_matrix(scenarios=[])

    def test_build_matrix_single_scenario(self) -> None:
        """Test building matrix with single scenario."""
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)

        scenario = Scenario({"A": "Low"}, matrix)
        builder = TransformationMatrixBuilder(base_matrix=matrix)

        result = builder.build_matrix(
            scenarios=[scenario],
            n_trials_per_pair=10,
            seed=123,
        )

        assert len(result.scenarios) == 1
        assert len(result.transformations) == 0

    def test_build_matrix_structural_shocks(self) -> None:
        """Test building matrix with structural shock perturbations."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)
        matrix.set_impact("A", "High", "B", "Weak", -2.0)
        matrix.set_impact("A", "High", "B", "Strong", 2.0)

        scenario1 = Scenario({"A": "Low", "B": "Weak"}, matrix)
        scenario2 = Scenario({"A": "High", "B": "Strong"}, matrix)

        builder = TransformationMatrixBuilder(base_matrix=matrix)

        result = builder.build_matrix(
            scenarios=[scenario1, scenario2],
            perturbation_types=["structural"],
            n_trials_per_pair=20,
            structural_sigma_values=[0.1, 0.2],
            seed=456,
        )

        assert len(result.scenarios) == 2
        assert result.summary_stats["total_scenarios"] == 2

    def test_build_matrix_dynamic_shocks(self) -> None:
        """Test building matrix with dynamic shock perturbations."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        matrix = CIBMatrix(descriptors)

        matrix.set_impact("A", "Low", "B", "Weak", 2.0)
        matrix.set_impact("A", "Low", "B", "Strong", -2.0)

        scenario1 = Scenario({"A": "Low", "B": "Weak"}, matrix)
        scenario2 = Scenario({"A": "High", "B": "Strong"}, matrix)

        builder = TransformationMatrixBuilder(base_matrix=matrix)

        result = builder.build_matrix(
            scenarios=[scenario1, scenario2],
            perturbation_types=["dynamic"],
            n_trials_per_pair=20,
            dynamic_tau_values=[0.2, 0.3],
            seed=789,
        )

        assert len(result.scenarios) == 2

    def test_build_matrix_judgment_uncertainty(self) -> None:
        """Test building matrix with judgment uncertainty perturbations."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}

        uncertain_matrix = UncertainCIBMatrix(descriptors, default_confidence=3)
        uncertain_matrix.set_impact("A", "Low", "B", "Weak", 2.0, confidence=3)
        uncertain_matrix.set_impact("A", "Low", "B", "Strong", -2.0, confidence=3)

        scenario1 = Scenario({"A": "Low", "B": "Weak"}, uncertain_matrix)
        scenario2 = Scenario({"A": "High", "B": "Strong"}, uncertain_matrix)

        builder = TransformationMatrixBuilder(base_matrix=uncertain_matrix)

        result = builder.build_matrix(
            scenarios=[scenario1, scenario2],
            perturbation_types=["judgment"],
            n_trials_per_pair=20,
            judgment_sigma_scale_values=[0.5, 1.0],
            seed=101112,
        )

        assert len(result.scenarios) == 2

    def test_build_matrix_with_dataset_b5(self) -> None:
        """Test building transformation matrix with dataset B5."""
        matrix = UncertainCIBMatrix(DATASET_B5_DESCRIPTORS)
        matrix.set_impacts(DATASET_B5_IMPACTS, confidence=DATASET_B5_CONFIDENCE)
        from cib.analysis import ScenarioAnalyzer

        analyzer = ScenarioAnalyzer(matrix)
        consistent_scenarios = analyzer.find_all_consistent(max_scenarios=5)

        if len(consistent_scenarios) >= 2:
            builder = TransformationMatrixBuilder(base_matrix=matrix)
            result = builder.build_matrix(
                scenarios=consistent_scenarios[:3],
                perturbation_types=["structural"],
                n_trials_per_pair=30,
                structural_sigma_values=[0.1, 0.15],
                seed=12345,
            )

            assert len(result.scenarios) <= 3
            assert result.summary_stats["total_scenarios"] <= 3

    def test_transformation_matrix_visualization(self) -> None:
        """Test that transformation matrix can be visualized."""
        matrix = UncertainCIBMatrix(DATASET_B5_DESCRIPTORS)
        matrix.set_impacts(DATASET_B5_IMPACTS, confidence=DATASET_B5_CONFIDENCE)
        from cib.analysis import ScenarioAnalyzer

        analyzer = ScenarioAnalyzer(matrix)
        consistent_scenarios = analyzer.find_all_consistent(max_scenarios=5)

        if len(consistent_scenarios) >= 2:
            builder = TransformationMatrixBuilder(base_matrix=matrix)
            result = builder.build_matrix(
                scenarios=consistent_scenarios[:3],
                perturbation_types=["structural"],
                n_trials_per_pair=20,
                structural_sigma_values=[0.1],
                seed=99999,
            )

            # Transformation graph can be visualized.
            ax = ScenarioVisualizer.transformation_graph(result, matrix)
            assert ax is not None
