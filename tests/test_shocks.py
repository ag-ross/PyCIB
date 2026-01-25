"""
Unit tests for shock modeling and robustness testing.

Tests ShockModel and RobustnessTester for structural shock application
and scenario robustness evaluation.
"""

import pytest
import numpy as np

from cib.core import CIBMatrix, Scenario
from cib.shocks import RobustnessTester, ShockModel
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

    def test_add_structural_shocks_invalid_sigma(self) -> None:
        """Test that non-positive sigma raises ValueError."""
        matrix = _toy_matrix()
        shock_model = ShockModel(matrix)

        with pytest.raises(ValueError, match="must be positive"):
            shock_model.add_structural_shocks(sigma=-1.0)

        with pytest.raises(ValueError, match="must be positive"):
            shock_model.add_structural_shocks(sigma=0.0)

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
