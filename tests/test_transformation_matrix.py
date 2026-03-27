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
    explain_regime_transformation,
    summarize_path_to_path_transformations,
)
from cib.pathway import MemoryState
from cib.transition_kernel import DefaultTransitionKernel
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
        """Builder initialisation is tested."""
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
        consistent_scenarios = analyzer.find_all_consistent(
            max_scenarios=5,
            mode="shortlist",
        )

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
        """That the transformation matrix can be visualised is tested."""
        matrix = UncertainCIBMatrix(DATASET_B5_DESCRIPTORS)
        matrix.set_impacts(DATASET_B5_IMPACTS, confidence=DATASET_B5_CONFIDENCE)
        from cib.analysis import ScenarioAnalyzer

        analyzer = ScenarioAnalyzer(matrix)
        consistent_scenarios = analyzer.find_all_consistent(
            max_scenarios=5,
            mode="shortlist",
        )

        if len(consistent_scenarios) >= 2:
            builder = TransformationMatrixBuilder(base_matrix=matrix)
            result = builder.build_matrix(
                scenarios=consistent_scenarios[:3],
                perturbation_types=["structural"],
                n_trials_per_pair=20,
                structural_sigma_values=[0.1],
                seed=99999,
            )

            # Transformation graph can be visualised.
            ax = ScenarioVisualizer.transformation_graph(result, matrix)
            assert ax is not None

    def test_build_matrix_records_extension_context(self) -> None:
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        scenario = Scenario({"A": "Low"}, matrix)
        builder = TransformationMatrixBuilder(base_matrix=matrix)

        result = builder.build_matrix(
            scenarios=[scenario],
            extension_mode="regime",
            regime_name="baseline",
            n_trials_per_pair=5,
            seed=123,
        )

        assert result.summary_stats["extension_mode"] == "regime"
        assert result.summary_stats["regime_name"] == "baseline"

    def test_build_matrix_pair_count_excludes_self_pairs(self) -> None:
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        scenarios = [
            Scenario({"A": "Low"}, matrix),
            Scenario({"A": "High"}, matrix),
        ]
        builder = TransformationMatrixBuilder(base_matrix=matrix)

        result = builder.build_matrix(
            scenarios=scenarios,
            perturbation_types=["dynamic"],
            dynamic_tau_values=[0.0],
            n_trials_per_pair=1,
            seed=123,
        )

        assert result.summary_stats["total_pairs_tested"] == 2

    def test_build_matrix_pair_count_handles_duplicate_equal_scenarios(self) -> None:
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        low = Scenario({"A": "Low"}, matrix)
        scenarios = [low, Scenario({"A": "Low"}, matrix), Scenario({"A": "High"}, matrix)]
        builder = TransformationMatrixBuilder(base_matrix=matrix)

        result = builder.build_matrix(
            scenarios=scenarios,
            perturbation_types=["dynamic"],
            dynamic_tau_values=[0.0],
            n_trials_per_pair=1,
            seed=123,
        )

        # Index-based self-pairs are skipped, so 3 * (3 - 1) = 6 pairs are evaluated.
        assert result.summary_stats["total_pairs_tested"] == 6

    def test_build_matrix_can_analyze_active_regime_matrix(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)

        boosted = CIBMatrix(descriptors)
        boosted.set_impacts(dict(baseline.iter_impacts()))
        boosted.set_impact("A", "Low", "B", "Low", -2.0)
        boosted.set_impact("A", "Low", "B", "High", 2.0)
        boosted.set_impact("B", "Low", "A", "Low", -2.0)
        boosted.set_impact("B", "Low", "A", "High", 2.0)

        source = Scenario({"A": "Low", "B": "Low"}, baseline)
        target = Scenario({"A": "High", "B": "High"}, baseline)
        builder = TransformationMatrixBuilder(base_matrix=baseline)

        baseline_result = builder.build_matrix(
            scenarios=[source, target],
            perturbation_types=["dynamic"],
            dynamic_tau_values=[0.0],
            n_trials_per_pair=1,
            seed=123,
        )
        boosted_result = builder.build_matrix(
            scenarios=[source, target],
            active_matrix=boosted,
            extension_mode="regime",
            regime_name="boosted",
            perturbation_types=["dynamic"],
            dynamic_tau_values=[0.0],
            n_trials_per_pair=1,
            seed=123,
        )

        assert (source, target) not in baseline_result.transformations
        assert (source, target) in boosted_result.transformations
        assert boosted_result.summary_stats["uses_active_matrix"] is True
        assert boosted_result.summary_stats["regime_name"] == "boosted"

    def test_analyze_path_to_path_transformations_uses_period_context(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)

        boosted = CIBMatrix(descriptors)
        boosted.set_impacts(dict(baseline.iter_impacts()))
        boosted.set_impact("A", "Low", "B", "Low", -2.0)
        boosted.set_impact("A", "Low", "B", "High", 2.0)
        boosted.set_impact("B", "Low", "A", "Low", -2.0)
        boosted.set_impact("B", "Low", "A", "High", 2.0)

        source_path = [
            Scenario({"A": "Low", "B": "Low"}, baseline),
            Scenario({"A": "Low", "B": "Low"}, baseline),
        ]
        target_path = [
            Scenario({"A": "Low", "B": "Low"}, baseline),
            Scenario({"A": "High", "B": "High"}, baseline),
        ]
        source_memory = [
            MemoryState(period=0, values={"phase": 0}, flags={}, export_label="m"),
            MemoryState(period=1, values={"phase": 0}, flags={}, export_label="m"),
        ]
        target_memory = [
            MemoryState(period=0, values={"phase": 0}, flags={}, export_label="m"),
            MemoryState(
                period=1,
                values={"phase": 1},
                flags={"locked_in": True},
                export_label="m",
            ),
        ]
        builder = TransformationMatrixBuilder(base_matrix=baseline)

        analysis = builder.analyze_path_to_path_transformations(
            source_path,
            target_path,
            periods=[2025, 2030],
            active_matrices=[baseline, boosted],
            active_regimes=["baseline", "boosted"],
            source_memory_states=source_memory,
            target_memory_states=target_memory,
            perturbation_types=["dynamic"],
            dynamic_tau_values=[0.0],
            n_trials_per_pair=1,
            seed=123,
            initial_scenario=Scenario({"A": "Low", "B": "Low"}, baseline),
            initial_memory_state=MemoryState(
                period=0, values={"phase": 0}, flags={}, export_label="m"
            ),
            transition_kernel=DefaultTransitionKernel(),
        )

        assert analysis["analysis_mode"] == "path_to_path"
        assert analysis["periods"] == (2025, 2030)
        assert analysis["changed_periods"] == (1,)
        assert analysis["memory_changed_periods"] == (1,)
        assert analysis["supported_changed_periods"] == ()
        assert analysis["all_changed_periods_supported"] is False
        assert analysis["path_replay_available"] is True
        assert analysis["path_replay_matches_target"] is True
        assert analysis["path_replay_matches_target_memory"] is False
        assert analysis["path_replay_matches_full_segment"] is False
        assert len(analysis["period_analyses"]) == 1
        assert analysis["period_analyses"][0]["period_index"] == 1
        assert analysis["period_analyses"][0]["regime_name"] == "boosted"
        assert analysis["period_analyses"][0]["memory_changed"] is True
        assert analysis["period_analyses"][0]["supported_by_perturbations"] is True
        assert analysis["period_analyses"][0]["supported_by_replay"] is False
        assert analysis["period_analyses"][0]["supported_by_source_state_replay"] is False
        assert analysis["period_analyses"][0]["supported_by_transition_law"] is False

    def test_analyze_path_to_path_transformations_validates_period_alignment(self) -> None:
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        path = [Scenario({"A": "Low"}, matrix)]
        builder = TransformationMatrixBuilder(base_matrix=matrix)

        with pytest.raises(ValueError, match="active_regimes must match the path length"):
            builder.analyze_path_to_path_transformations(
                path,
                path,
                active_regimes=["baseline", "boosted"],
                perturbation_types=["dynamic"],
                dynamic_tau_values=[0.0],
                n_trials_per_pair=1,
            )

    def test_analyze_path_to_path_transformations_analyzes_memory_only_divergence(self) -> None:
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        path = [Scenario({"A": "Low"}, matrix)]
        builder = TransformationMatrixBuilder(base_matrix=matrix)

        analysis = builder.analyze_path_to_path_transformations(
            path,
            path,
            periods=[2025],
            source_memory_states=[
                MemoryState(period=2025, values={"phase": 0}, flags={}, export_label="m")
            ],
            target_memory_states=[
                MemoryState(period=2025, values={"phase": 1}, flags={}, export_label="m")
            ],
        )

        assert analysis["changed_periods"] == ()
        assert analysis["memory_changed_periods"] == (0,)
        assert analysis["analyzed_periods"] == (0,)

    def test_analyze_path_replay_mapping_is_index_based_with_duplicate_period_labels(self) -> None:
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        low = Scenario({"A": "Low"}, matrix)
        high = Scenario({"A": "High"}, matrix)
        source_path = [low, low]
        target_path = [high, high]
        builder = TransformationMatrixBuilder(base_matrix=matrix)

        def two_step_kernel(**kwargs):
            current = kwargs["current_scenario"]
            active = kwargs["active_matrix"]
            memory = kwargs["memory_state"]
            previous = kwargs["previous_path"]
            if len(previous) == 0:
                return Scenario({"A": "High"}, active), memory, {"step_marker": 0}
            return Scenario({"A": "Low"}, active), memory, {"step_marker": 1}

        analysis = builder.analyze_path_to_path_transformations(
            source_path,
            target_path,
            periods=[2030, 2030],
            perturbation_types=["dynamic"],
            dynamic_tau_values=[0.0],
            n_trials_per_pair=1,
            seed=123,
            initial_scenario=low,
            transition_kernel=two_step_kernel,
        )

        assert analysis["changed_periods"] == (0, 1)
        assert len(analysis["period_analyses"]) == 2
        first = analysis["period_analyses"][0]
        second = analysis["period_analyses"][1]
        assert first["period_index"] == 0
        assert second["period_index"] == 1
        assert first["supported_by_replay"] is True
        assert second["supported_by_replay"] is False
        assert first["replay_metadata"].get("step_marker") == 0
        assert second["replay_metadata"].get("step_marker") == 1


def test_summarize_path_to_path_transformations() -> None:
    descriptors = {"A": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)
    source = [Scenario({"A": "Low"}, matrix), Scenario({"A": "Low"}, matrix)]
    target = [Scenario({"A": "High"}, matrix), Scenario({"A": "Low"}, matrix)]

    summary = summarize_path_to_path_transformations(source, target)

    assert summary["total_hamming"] == 1


def test_summarize_path_to_path_transformations_with_memory_context() -> None:
    descriptors = {"A": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)
    source = [Scenario({"A": "Low"}, matrix), Scenario({"A": "Low"}, matrix)]
    target = [Scenario({"A": "Low"}, matrix), Scenario({"A": "High"}, matrix)]
    source_memory = [
        MemoryState(period=0, values={"phase": 0}, flags={}, export_label="m"),
        MemoryState(period=1, values={"phase": 0}, flags={}, export_label="m"),
    ]
    target_memory = [
        MemoryState(period=0, values={"phase": 0}, flags={}, export_label="m"),
        MemoryState(period=1, values={"phase": 1}, flags={"locked_in": True}, export_label="m"),
    ]

    summary = summarize_path_to_path_transformations(
        source,
        target,
        source_memory_states=source_memory,
        target_memory_states=target_memory,
        active_regimes=("baseline", "boosted"),
    )

    assert summary["memory_active"] is True
    assert summary["changed_periods"] == (1,)
    assert summary["memory_changed_periods"] == (1,)
    assert summary["active_regimes"] == ("baseline", "boosted")


def test_summarize_path_memory_change_detects_period_drift() -> None:
    descriptors = {"A": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)
    source = [Scenario({"A": "Low"}, matrix)]
    target = [Scenario({"A": "Low"}, matrix)]
    source_memory = [
        MemoryState(period=2025, values={"phase": 0}, flags={}, export_label="m"),
    ]
    target_memory = [
        MemoryState(period=2030, values={"phase": 0}, flags={}, export_label="m"),
    ]

    summary = summarize_path_to_path_transformations(
        source,
        target,
        source_memory_states=source_memory,
        target_memory_states=target_memory,
    )

    assert summary["changed_periods"] == ()
    assert summary["memory_changed_periods"] == (0,)


def test_explain_regime_transformation() -> None:
    descriptors = {"A": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)
    source = Scenario({"A": "Low"}, matrix)
    target = Scenario({"A": "High"}, matrix)

    explanation = explain_regime_transformation(
        source=source,
        target=target,
        regime_name="baseline",
        active_matrix_id="m0",
    )

    assert explanation["regime_name"] == "baseline"
    assert explanation["hamming_distance"] == 1


class TestTransformationMatrixBuilderMemoryOnlyDivergence:
    def test_memory_only_divergence_not_marked_as_perturbation_supported(self) -> None:
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        path = [Scenario({"A": "Low"}, matrix)]
        builder = TransformationMatrixBuilder(base_matrix=matrix)

        analysis = builder.analyze_path_to_path_transformations(
            path,
            path,
            periods=[2025],
            source_memory_states=[
                MemoryState(period=2025, values={"phase": 0}, flags={}, export_label="m")
            ],
            target_memory_states=[
                MemoryState(period=2025, values={"phase": 1}, flags={}, export_label="m")
            ],
            perturbation_types=["dynamic"],
            dynamic_tau_values=[0.0],
            n_trials_per_pair=1,
            seed=123,
        )

        assert analysis["analyzed_periods"] == (0,)
        assert len(analysis["period_analyses"]) == 1
        record = analysis["period_analyses"][0]
        assert record["memory_changed"] is True
        assert record["supported_by_perturbations"] is False
        assert record["perturbations"] == ()
        assert record["summary_stats"]["total_pairs_tested"] == 0
