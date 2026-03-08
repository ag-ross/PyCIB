"""
Unit tests for shock modeling and robustness testing.

Tests ShockModel and RobustnessTester for structural shock application
and scenario robustness evaluation.
"""

import pytest
import numpy as np

from cib.core import CIBMatrix, Scenario
from cib.shocks import (
    RobustnessTester,
    ShockModel,
    calibrate_structural_sigma_from_confidence,
    suggest_dynamic_tau_bounds,
)
from cib.dynamic import DynamicCIB


def _toy_matrix() -> CIBMatrix:
    descriptors = {
        "A": ["Low", "High"],
        "B": ["Low", "High"],
        "C": ["Low", "High"],
    }
    m = CIBMatrix(descriptors)
    impacts = {
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
    m.set_impacts(impacts)
    return m


class TestShockModel:
    """Test suite for ShockModel class."""

    def test_initialization(self) -> None:
        """Test ShockModel initialization."""
        matrix = _toy_matrix()

        shock_model = ShockModel(matrix)

        assert shock_model.base_matrix == matrix
        assert shock_model.structural_sigma is None

    def test_add_structural_shocks(self) -> None:
        """Test configuring structural shocks."""
        matrix = _toy_matrix()

        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25)

        assert shock_model.structural_sigma == 0.25
        assert shock_model.structural_scaling_mode == "additive"
        assert shock_model.structural_scaling_alpha == 0.0

    def test_add_structural_shocks_invalid_sigma(self) -> None:
        """Test that non-positive sigma raises ValueError."""
        matrix = _toy_matrix()
        shock_model = ShockModel(matrix)

        with pytest.raises(ValueError, match="must be positive"):
            shock_model.add_structural_shocks(sigma=-1.0)

        with pytest.raises(ValueError, match="must be positive"):
            shock_model.add_structural_shocks(sigma=0.0)

    def test_add_structural_shocks_invalid_scaling_config(self) -> None:
        """Test scaling mode and alpha validation."""
        matrix = _toy_matrix()
        shock_model = ShockModel(matrix)

        with pytest.raises(ValueError, match="scaling_mode must be"):
            shock_model.add_structural_shocks(
                sigma=0.25,
                scaling_mode="unsupported_mode",
            )

        with pytest.raises(ValueError, match="scaling_alpha must be non-negative"):
            shock_model.add_structural_shocks(
                sigma=0.25,
                scaling_mode="multiplicative_magnitude",
                scaling_alpha=-0.1,
            )

        with pytest.raises(ValueError, match="keys must be"):
            shock_model.add_structural_shocks(
                sigma=0.25,
                scale_by_state={("A", "Low", "extra"): 1.2},  # type: ignore[dict-item]
            )

    def test_structural_shock_additive_explicit_matches_default(self) -> None:
        """Explicit additive scaling should match default behavior."""
        matrix = _toy_matrix()

        default_model = ShockModel(matrix)
        default_model.add_structural_shocks(sigma=0.25)
        explicit_model = ShockModel(matrix)
        explicit_model.add_structural_shocks(
            sigma=0.25, scaling_mode="additive", scaling_alpha=0.0
        )

        sampled_default = default_model.sample_shocked_matrix(seed=123)
        sampled_explicit = explicit_model.sample_shocked_matrix(seed=123)

        key = ("A", "Low", "B", "Low")
        assert sampled_default.get_impact(*key) == sampled_explicit.get_impact(*key)

    def test_structural_multiplicative_scaling_amplifies_nonzero_base(self) -> None:
        """Multiplicative scaling should enlarge shock size for non-zero impacts."""
        descriptors = {"A": ["S"], "B": ["T"]}
        matrix = CIBMatrix(descriptors)
        matrix.set_impact("A", "S", "B", "T", 1.5)  # non-zero base
        matrix.set_impact("B", "T", "A", "S", 0.0)  # zero base

        additive = ShockModel(matrix)
        additive.add_structural_shocks(sigma=0.01, scaling_mode="additive")
        mult = ShockModel(matrix)
        mult.add_structural_shocks(
            sigma=0.01,
            scaling_mode="multiplicative_magnitude",
            scaling_alpha=1.0,
        )

        sampled_add = additive.sample_shocked_matrix(seed=7)
        sampled_mul = mult.sample_shocked_matrix(seed=7)

        key_nonzero = ("A", "S", "B", "T")
        key_zero = ("B", "T", "A", "S")

        base_nonzero = matrix.get_impact(*key_nonzero)
        base_zero = matrix.get_impact(*key_zero)
        delta_add_nonzero = abs(sampled_add.get_impact(*key_nonzero) - base_nonzero)
        delta_mul_nonzero = abs(sampled_mul.get_impact(*key_nonzero) - base_nonzero)
        delta_add_zero = abs(sampled_add.get_impact(*key_zero) - base_zero)
        delta_mul_zero = abs(sampled_mul.get_impact(*key_zero) - base_zero)

        assert delta_mul_nonzero > delta_add_nonzero
        assert delta_mul_zero == pytest.approx(delta_add_zero)

    def test_structural_descriptor_and_state_scaling(self) -> None:
        """Descriptor/state multipliers should change structural shock magnitude."""
        descriptors = {"A": ["S"], "B": ["T"]}
        matrix = CIBMatrix(descriptors)
        matrix.set_impact("A", "S", "B", "T", 1.0)
        matrix.set_impact("B", "T", "A", "S", 1.0)

        baseline = ShockModel(matrix)
        baseline.add_structural_shocks(sigma=0.05, scaling_mode="additive")
        scaled = ShockModel(matrix)
        scaled.add_structural_shocks(
            sigma=0.05,
            scaling_mode="additive",
            scale_by_descriptor={"A": 2.0},
            scale_by_state={("A", "S"): 1.5},
        )

        b = baseline.sample_shocked_matrix(seed=21)
        s = scaled.sample_shocked_matrix(seed=21)
        key_scaled = ("A", "S", "B", "T")
        key_unscaled = ("B", "T", "A", "S")

        base = matrix.get_impact(*key_scaled)
        assert abs(s.get_impact(*key_scaled) - base) > abs(b.get_impact(*key_scaled) - base)
        assert s.get_impact(*key_unscaled) == pytest.approx(b.get_impact(*key_unscaled))

    def test_sample_shocked_matrix_reproducibility(self) -> None:
        """Test that shock sampling is reproducible."""
        matrix = _toy_matrix()

        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25)

        sampled1 = shock_model.sample_shocked_matrix(seed=123)
        sampled2 = shock_model.sample_shocked_matrix(seed=123)

        key = ("A", "Low", "B", "Low")
        val1 = sampled1.get_impact(key[0], key[1], key[2], key[3])
        val2 = sampled2.get_impact(key[0], key[1], key[2], key[3])

        assert val1 == val2

    def test_sample_shocked_matrix_reproducibility_multiplicative_scaling(self) -> None:
        """Multiplicative scaling mode should remain reproducible with fixed seed."""
        matrix = _toy_matrix()
        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(
            sigma=0.25,
            scaling_mode="multiplicative_magnitude",
            scaling_alpha=0.8,
        )

        sampled1 = shock_model.sample_shocked_matrix(seed=456)
        sampled2 = shock_model.sample_shocked_matrix(seed=456)

        key = ("A", "High", "B", "Low")
        assert sampled1.get_impact(*key) == sampled2.get_impact(*key)

    def test_sample_shocked_matrix_structure(self) -> None:
        """Test that shocked matrix has correct structure."""
        matrix = _toy_matrix()

        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25)

        sampled = shock_model.sample_shocked_matrix(seed=123)

        assert sampled.n_descriptors == matrix.n_descriptors
        assert sampled.descriptors == matrix.descriptors


class TestRobustnessTester:
    """Test suite for RobustnessTester class."""

    def test_initialization(self) -> None:
        """Test RobustnessTester initialization."""
        matrix = _toy_matrix()

        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25)

        tester = RobustnessTester(matrix, shock_model)

        assert tester.matrix == matrix
        assert tester.shock_model == shock_model

    def test_test_scenario(self) -> None:
        """Test robustness scoring for a single scenario."""
        matrix = _toy_matrix()

        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25)

        scenario = Scenario(
            {
                "A": "Low",
                "B": "Low",
                "C": "Low",
            },
            matrix,
        )

        tester = RobustnessTester(matrix, shock_model)
        robustness = tester.test_scenario(scenario, n_simulations=100, seed=123)

        assert 0.0 <= robustness <= 1.0

    def test_test_scenario_invalid_n_simulations(self) -> None:
        matrix = _toy_matrix()
        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25)
        scenario = Scenario({"A": "Low", "B": "Low", "C": "Low"}, matrix)
        tester = RobustnessTester(matrix, shock_model)

        with pytest.raises(ValueError, match="n_simulations must be positive"):
            tester.test_scenario(scenario, n_simulations=0, seed=123)

    def test_test_scenarios(self) -> None:
        """Test robustness scoring for multiple scenarios."""
        matrix = _toy_matrix()

        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25)

        scenarios = [
            Scenario(
                {
                    "A": "Low",
                    "B": "Low",
                    "C": "Low",
                },
                matrix,
            ),
            Scenario(
                {
                    "A": "High",
                    "B": "High",
                    "C": "High",
                },
                matrix,
            ),
        ]

        tester = RobustnessTester(matrix, shock_model)
        scores = tester.test_scenarios(scenarios, n_simulations=100, seed=123)

        assert len(scores) == 2
        for scenario in scenarios:
            assert scenario in scores
            assert 0.0 <= scores[scenario] <= 1.0

    def test_test_scenarios_invalid_n_simulations(self) -> None:
        matrix = _toy_matrix()
        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25)
        scenarios = [Scenario({"A": "Low", "B": "Low", "C": "Low"}, matrix)]
        tester = RobustnessTester(matrix, shock_model)

        with pytest.raises(ValueError, match="n_simulations must be positive"):
            tester.test_scenarios(scenarios, n_simulations=0, seed=123)

    def test_rank_by_robustness(self) -> None:
        """Test ranking scenarios by robustness."""
        matrix = _toy_matrix()
        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25)

        scenario1 = Scenario(
            {
                "A": "Low",
                "B": "Low",
                "C": "Low",
            },
            matrix,
        )
        scenario2 = Scenario(
            {
                "A": "High",
                "B": "High",
                "C": "High",
            },
            matrix,
        )

        scores = {scenario1: 0.8, scenario2: 0.6}

        tester = RobustnessTester(matrix, shock_model)
        ranked = tester.rank_by_robustness(scores)

        assert len(ranked) == 2
        assert ranked[0][0] == scenario1
        assert ranked[0][1] == 0.8
        assert ranked[1][0] == scenario2
        assert ranked[1][1] == 0.6


class TestAdvancedShockModeling:
    """Tests for correlated structural shocks and dynamic shocks."""

    def test_correlated_structural_shocks_shape_validation(self) -> None:
        matrix = _toy_matrix()

        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25, correlation_matrix=np.eye(3))

        with pytest.raises(ValueError, match="correlation_matrix must have shape"):
            shock_model.sample_shocked_matrix(seed=123)

    def test_correlated_structural_shocks_reproducible(self) -> None:
        matrix = _toy_matrix()

        # Identity correlation should behave like independent shocks for covariance structure,
        # but is sampled via multivariate_normal.
        impact_keys = []
        for src_desc in matrix.descriptors:
            for src_state in matrix.descriptors[src_desc]:
                for tgt_desc in matrix.descriptors:
                    if src_desc == tgt_desc:
                        continue
                    for tgt_state in matrix.descriptors[tgt_desc]:
                        impact_keys.append((src_desc, src_state, tgt_desc, tgt_state))

        corr = np.eye(len(impact_keys))
        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.25, correlation_matrix=corr)

        s1 = shock_model.sample_shocked_matrix(seed=123)
        s2 = shock_model.sample_shocked_matrix(seed=123)

        key = ("A", "Low", "B", "Low")
        assert s1.get_impact(*key) == s2.get_impact(*key)

    def test_dynamic_shocks_reproducible(self) -> None:
        matrix = _toy_matrix()
        shock_model = ShockModel(matrix)
        shock_model.add_dynamic_shocks(periods=[1, 2, 3], tau=0.10, rho=0.5)

        d1 = shock_model.sample_dynamic_shocks(seed=123)
        d2 = shock_model.sample_dynamic_shocks(seed=123)
        assert d1 == d2
        assert set(d1.keys()) == {1, 2, 3}

    def test_dynamic_shocks_influence_dynamic_simulation(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)

        dyn = DynamicCIB(matrix, periods=[1])
        # With a fully neutral matrix, global succession would select the first state
        # for each descriptor (Low). A positive dynamic shock is expected to tilt A to High.
        dynamic_shocks = {1: {("A", "High"): 1.0}}
        path = dyn.simulate_path(
            initial={"A": "Low", "B": "Low"},
            seed=123,
            dynamic_shocks_by_period=dynamic_shocks,
        )
        assert path.scenarios[0].to_dict()["A"] == "High"

    def test_dynamic_shock_descriptor_scaling(self) -> None:
        matrix = _toy_matrix()
        base = ShockModel(matrix)
        base.add_dynamic_shocks(periods=[1, 2], tau=0.2, rho=0.5)
        scaled = ShockModel(matrix)
        scaled.add_dynamic_shocks(
            periods=[1, 2],
            tau=0.2,
            rho=0.5,
            scale_by_descriptor={"A": 2.0},
            scale_by_state={("A", "High"): 1.5},
        )

        d_base = base.sample_dynamic_shocks(seed=42)
        d_scaled = scaled.sample_dynamic_shocks(seed=42)
        key = ("A", "High")
        assert abs(d_scaled[1][key]) > abs(d_base[1][key])


class TestRobustnessExtensions:
    def test_evaluate_scenario_returns_extended_metrics(self) -> None:
        matrix = _toy_matrix()
        shock_model = ShockModel(matrix)
        shock_model.add_structural_shocks(sigma=0.2)
        tester = RobustnessTester(matrix, shock_model)
        scenario = Scenario({"A": "Low", "B": "Low", "C": "Low"}, matrix)

        metrics = tester.evaluate_scenario(
            scenario,
            n_simulations=50,
            seed=123,
            max_iterations=100,
        )
        assert metrics.n_simulations == 50
        assert 0.0 <= metrics.consistency_rate <= 1.0
        assert 0.0 <= metrics.attractor_retention_rate <= 1.0
        assert 0.0 <= metrics.switch_rate <= 1.0
        assert metrics.mean_hamming_to_base_attractor >= 0.0


class TestShockCalibrationHelpers:
    def test_calibrate_structural_sigma_from_confidence(self) -> None:
        sigma_mean = calibrate_structural_sigma_from_confidence([5, 4, 3, 2, 1], method="mean")
        sigma_med = calibrate_structural_sigma_from_confidence([5, 4, 3, 2, 1], method="median")
        assert sigma_mean > 0
        assert sigma_med > 0

    def test_calibrate_structural_sigma_from_confidence_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            calibrate_structural_sigma_from_confidence([5, 4, 3], method="bad")

    def test_suggest_dynamic_tau_bounds(self) -> None:
        lo, hi = suggest_dynamic_tau_bounds(0.8, low_ratio=0.5, high_ratio=1.25)
        assert lo == pytest.approx(0.4)
        assert hi == pytest.approx(1.0)

    def test_suggest_dynamic_tau_bounds_invalid_ratios(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            suggest_dynamic_tau_bounds(0.8, low_ratio=0.0, high_ratio=1.0)
        with pytest.raises(ValueError, match="must be <="):
            suggest_dynamic_tau_bounds(0.8, low_ratio=1.1, high_ratio=1.0)
