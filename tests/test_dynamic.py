"""
Unit tests for dynamic (multi-period) CIB simulation.
"""

from __future__ import annotations

import pytest

from cib.constraints import ConstraintIndex, ForbiddenPair
from cib.core import CIBMatrix, ConsistencyChecker
from cib.cyclic import CyclicDescriptor
from cib.dynamic import DynamicCIB
from cib.succession import GlobalSuccession, LocalSuccession
from cib.threshold import ThresholdRule
from cib.example_data import (
    DATASET_B5_CONFIDENCE,
    DATASET_B5_DESCRIPTORS,
    DATASET_B5_IMPACTS,
    DATASET_B5_INITIAL_SCENARIO,
    dataset_b5_cyclic_descriptors,
    dataset_b5_threshold_rule_fast_permitting,
)
from cib.uncertainty import UncertainCIBMatrix


class TestDynamicCIB:
    """Test suite for DynamicCIB."""

    def test_cyclic_descriptor_drives_path(self) -> None:
        descriptors = {"Cycle": ["Low", "High"], "Y": ["Low", "High"]}
        m = CIBMatrix(descriptors)

        # Cycle <-> Y coordination (two fixed points).
        m.set_impact("Cycle", "Low", "Y", "Low", 2.0)
        m.set_impact("Cycle", "Low", "Y", "High", -2.0)
        m.set_impact("Cycle", "High", "Y", "Low", -2.0)
        m.set_impact("Cycle", "High", "Y", "High", 2.0)

        m.set_impact("Y", "Low", "Cycle", "Low", 2.0)
        m.set_impact("Y", "Low", "Cycle", "High", -2.0)
        m.set_impact("Y", "High", "Cycle", "Low", -2.0)
        m.set_impact("Y", "High", "Cycle", "High", 2.0)

        dyn = DynamicCIB(m, periods=[1, 2, 3])
        dyn.add_cyclic_descriptor(
            CyclicDescriptor(
                name="Cycle",
                transition={
                    "Low": {"High": 1.0},
                    "High": {"Low": 1.0},
                },
            )
        )

        path = dyn.simulate_path(initial={"Cycle": "Low", "Y": "Low"}, seed=123)
        states = [s.to_dict() for s in path.scenarios]

        assert states[0] == {"Cycle": "Low", "Y": "Low"}
        # Cyclic descriptors are evolved between periods and are held fixed during
        # within-period succession. Coordination is therefore induced between Y and
        # the exogenous Cycle state each period.
        assert states[1] == {"Cycle": "High", "Y": "High"}
        assert states[2] == {"Cycle": "Low", "Y": "Low"}

    def test_threshold_rule_modifies_matrix(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)

        # Neutral ties are used unless a threshold modifier is applied.
        def modifier(base: CIBMatrix) -> CIBMatrix:
            out = CIBMatrix(base.descriptors)
            out.set_impacts(dict(base._impacts))  # type: ignore[attr-defined]
            out.set_impact("A", "High", "B", "Low", -3.0)
            out.set_impact("A", "High", "B", "High", 3.0)
            out.set_impact("A", "Low", "B", "Low", -1.0)
            out.set_impact("A", "Low", "B", "High", 1.0)
            return out

        dyn = DynamicCIB(m, periods=[1])
        dyn.add_threshold_rule(
            ThresholdRule(
                name="IfAHighBoostBHigh",
                condition=lambda s: s.get_state("A") == "High",
                modifier=modifier,
            )
        )

        path = dyn.simulate_path(initial={"A": "High", "B": "Low"}, seed=123)
        final_state = path.scenarios[0].to_dict()

        assert final_state["B"] == "High"

    def test_threshold_match_policy_controls_multiple_rule_application(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)

        # A is stabilised to High regardless of B.
        m.set_impact("B", "Low", "A", "Low", 0.0)
        m.set_impact("B", "Low", "A", "High", 1.0)
        m.set_impact("B", "High", "A", "Low", 0.0)
        m.set_impact("B", "High", "A", "High", 1.0)

        def modifier_b_high(base: CIBMatrix) -> CIBMatrix:
            out = CIBMatrix(base.descriptors)
            out.set_impacts(dict(base._impacts))  # type: ignore[attr-defined]
            out.set_impact("A", "High", "B", "Low", 0.0)
            out.set_impact("A", "High", "B", "High", 1.0)
            return out

        def modifier_b_low(base: CIBMatrix) -> CIBMatrix:
            out = CIBMatrix(base.descriptors)
            out.set_impacts(dict(base._impacts))  # type: ignore[attr-defined]
            out.set_impact("A", "High", "B", "Low", 1.0)
            out.set_impact("A", "High", "B", "High", 0.0)
            return out

        rule1 = ThresholdRule(
            name="Rule1_IfAHighThenBHigh",
            condition=lambda s: s.get_state("A") == "High",
            modifier=modifier_b_high,
        )
        rule2 = ThresholdRule(
            name="Rule2_IfAHighThenBLow",
            condition=lambda s: s.get_state("A") == "High",
            modifier=modifier_b_low,
        )

        dyn_first = DynamicCIB(m, periods=[1], threshold_match_policy="first_match")
        dyn_first.add_threshold_rule(rule1)
        dyn_first.add_threshold_rule(rule2)
        p_first = dyn_first.simulate_path(initial={"A": "High", "B": "Low"}, seed=123)
        assert p_first.scenarios[0].to_dict()["B"] == "High"

        dyn_all = DynamicCIB(m, periods=[1], threshold_match_policy="all_matches")
        dyn_all.add_threshold_rule(rule1)
        dyn_all.add_threshold_rule(rule2)
        p_all = dyn_all.simulate_path(initial={"A": "High", "B": "Low"}, seed=123)
        assert p_all.scenarios[0].to_dict()["B"] == "Low"

    def test_ensemble_reproducible(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Low", 2.0)
        m.set_impact("A", "Low", "B", "High", -2.0)
        m.set_impact("A", "High", "B", "Low", -2.0)
        m.set_impact("A", "High", "B", "High", 2.0)
        m.set_impact("B", "Low", "A", "Low", 2.0)
        m.set_impact("B", "Low", "A", "High", -2.0)
        m.set_impact("B", "High", "A", "Low", -2.0)
        m.set_impact("B", "High", "A", "High", 2.0)

        dyn = DynamicCIB(m, periods=[1, 2])
        paths1 = dyn.simulate_ensemble(initial={"A": "Low", "B": "Low"}, n_runs=10, base_seed=123)
        paths2 = dyn.simulate_ensemble(initial={"A": "Low", "B": "Low"}, n_runs=10, base_seed=123)

        assert [p.to_dicts() for p in paths1] == [p.to_dicts() for p in paths2]

    def test_equilibrium_mode_relaxes_to_unshocked_attractor(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)

        # A strict preference for Low is induced regardless of B, with B remaining neutral.
        m.set_impact("B", "Low", "A", "Low", 2.0)
        m.set_impact("B", "Low", "A", "High", -2.0)
        m.set_impact("B", "High", "A", "Low", 2.0)
        m.set_impact("B", "High", "A", "High", -2.0)

        dyn = DynamicCIB(m, periods=[1])
        dynamic_shocks = {1: {("A", "High"): 10.0}}
        path = dyn.simulate_path(
            initial={"A": "Low", "B": "Low"},
            seed=123,
            dynamic_shocks_by_period=dynamic_shocks,
            equilibrium_mode="relax_unshocked",
        )

        assert path.equilibrium_scenarios is not None
        realised = path.scenarios[0]
        equilibrium = path.equilibrium_scenarios[0]

        assert ConsistencyChecker.check_consistency(equilibrium, m) is True
        assert ConsistencyChecker.check_consistency(realised, m) is False

    def test_simulate_path_can_collect_diagnostics(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Low", 2.0)
        m.set_impact("A", "Low", "B", "High", -2.0)
        m.set_impact("B", "Low", "A", "Low", 2.0)
        m.set_impact("B", "Low", "A", "High", -2.0)

        dyn = DynamicCIB(m, periods=[1, 2, 3])
        diag = {}
        path = dyn.simulate_path(initial={"A": "Low", "B": "Low"}, seed=123, diagnostics=diag)
        assert len(path.scenarios) == 3
        assert len(diag.get("iterations", [])) == 3
        assert len(diag.get("is_cycle", [])) == 3
        assert "threshold_rules_applied" not in diag

    def test_simulate_path_records_threshold_applications(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)

        def modifier(base: CIBMatrix) -> CIBMatrix:
            out = CIBMatrix(base.descriptors)
            out.set_impacts(dict(base.iter_impacts()))
            out.set_impact("A", "High", "B", "Low", -3.0)
            out.set_impact("A", "High", "B", "High", 3.0)
            return out

        dyn = DynamicCIB(m, periods=[1, 2])
        dyn.add_threshold_rule(
            ThresholdRule(
                name="IfAHighBoostBHigh",
                condition=lambda s: s.get_state("A") == "High",
                modifier=modifier,
            )
        )

        diag = {}
        _ = dyn.simulate_path(initial={"A": "High", "B": "Low"}, seed=123, diagnostics=diag)
        applied = diag.get("threshold_rules_applied")
        assert isinstance(applied, list)
        assert len(applied) == 2
        assert applied[0] == ["IfAHighBoostBHigh"]

    def test_dataset_b5_demo_path_shapes(self) -> None:
        """
        Smoke-check the canonical 5-state demo wiring.

        This intentionally keeps runtime small (few runs, few periods) while
        ensuring the demo dataset and helpers remain coherent.
        """
        periods = [2025, 2030, 2035]
        matrix = UncertainCIBMatrix(DATASET_B5_DESCRIPTORS)
        matrix.set_impacts(DATASET_B5_IMPACTS, confidence=DATASET_B5_CONFIDENCE)

        dyn = DynamicCIB(matrix, periods=periods)
        for cd in dataset_b5_cyclic_descriptors():
            dyn.add_cyclic_descriptor(cd)
        dyn.add_threshold_rule(dataset_b5_threshold_rule_fast_permitting())

        paths = dyn.simulate_ensemble(initial=DATASET_B5_INITIAL_SCENARIO, n_runs=20, base_seed=123)
        assert len(paths) == 20
        for p in paths:
            assert list(p.periods) == periods
            assert len(p.scenarios) == len(periods)

    def test_structural_shock_scaling_validation(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])

        with pytest.raises(ValueError, match="Unsupported structural_shock_scaling_mode"):
            dyn.simulate_path(
                initial={"A": "Low", "B": "Low"},
                structural_sigma=0.1,
                structural_shock_scaling_mode="bad_mode",  # type: ignore[arg-type]
            )

        with pytest.raises(
            ValueError, match="structural_shock_scaling_alpha must be non-negative"
        ):
            dyn.simulate_ensemble(
                initial={"A": "Low", "B": "Low"},
                n_runs=2,
                base_seed=123,
                structural_sigma=0.1,
                structural_shock_scaling_alpha=-0.1,
            )

    def test_structural_shock_additive_pass_through_matches_default(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Low", 2.0)
        m.set_impact("A", "Low", "B", "High", -2.0)
        m.set_impact("A", "High", "B", "Low", -2.0)
        m.set_impact("A", "High", "B", "High", 2.0)
        m.set_impact("B", "Low", "A", "Low", 2.0)
        m.set_impact("B", "Low", "A", "High", -2.0)
        m.set_impact("B", "High", "A", "Low", -2.0)
        m.set_impact("B", "High", "A", "High", 2.0)

        dyn = DynamicCIB(m, periods=[1, 2])
        paths_default = dyn.simulate_ensemble(
            initial={"A": "Low", "B": "Low"},
            n_runs=10,
            base_seed=321,
            structural_sigma=0.15,
        )
        paths_additive = dyn.simulate_ensemble(
            initial={"A": "Low", "B": "Low"},
            n_runs=10,
            base_seed=321,
            structural_sigma=0.15,
            structural_shock_scaling_mode="additive",
            structural_shock_scaling_alpha=0.0,
        )

        assert [p.to_dicts() for p in paths_default] == [p.to_dicts() for p in paths_additive]

    def test_structural_shock_scale_map_validation_in_dynamic_api(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])

        with pytest.raises(ValueError, match="unknown descriptor"):
            dyn.simulate_path(
                initial={"A": "Low", "B": "Low"},
                structural_sigma=0.1,
                structural_shock_scale_by_descriptor={"Unknown": 2.0},
            )

    def test_dynamic_shock_scale_map_validation_in_dynamic_api(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])

        with pytest.raises(ValueError, match="unknown descriptor"):
            dyn.simulate_ensemble(
                initial={"A": "Low", "B": "Low"},
                n_runs=2,
                base_seed=123,
                dynamic_tau=0.2,
                dynamic_shock_scale_by_descriptor={"Unknown": 1.5},
            )

    def test_dynamic_constraints_strict_rejects_initial_infeasibility(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])
        constraints = [ForbiddenPair("A", "High", "B", "High")]
        with pytest.raises(ValueError, match="Initial scenario is infeasible"):
            dyn.simulate_path(
                initial={"A": "High", "B": "High"},
                constraints=constraints,
                constraint_mode="strict",
            )

    def test_dynamic_constraints_repair_repairs_initial_infeasibility(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])
        constraints = [ForbiddenPair("A", "High", "B", "High")]
        p = dyn.simulate_path(
            initial={"A": "High", "B": "High"},
            constraints=constraints,
            constraint_mode="repair",
            constrained_top_k=2,
            constrained_backtracking_depth=2,
        )
        out = p.scenarios[0].to_dict()
        assert not (out["A"] == "High" and out["B"] == "High")

    def test_dynamic_constraints_cyclic_retry_exhaustion_strict(self) -> None:
        descriptors = {"Cycle": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("Cycle", "Low", "B", "Low", 0.0)
        m.set_impact("Cycle", "Low", "B", "High", 1.0)
        m.set_impact("Cycle", "High", "B", "Low", 0.0)
        m.set_impact("Cycle", "High", "B", "High", 1.0)
        dyn = DynamicCIB(m, periods=[1, 2])
        dyn.add_cyclic_descriptor(
            CyclicDescriptor(
                name="Cycle",
                transition={
                    "Low": {"High": 1.0},
                    "High": {"High": 1.0},
                },
            )
        )
        constraints = [ForbiddenPair("Cycle", "High", "B", "High")]
        with pytest.raises(ValueError, match="Cyclic transition produced infeasible state"):
            dyn.simulate_path(
                initial={"Cycle": "Low", "B": "High"},
                constraints=constraints,
                constraint_mode="strict",
                cyclic_infeasible_retries=1,
            )

    def test_dynamic_constraints_repair_preserves_cyclic_lock(self) -> None:
        descriptors = {"Cycle": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        for cycle_state in descriptors["Cycle"]:
            m.set_impact("Cycle", cycle_state, "B", "Low", 0.0)
            m.set_impact("Cycle", cycle_state, "B", "High", 1.0)
        for b_state in descriptors["B"]:
            m.set_impact("B", b_state, "Cycle", "Low", 0.0)
            m.set_impact("B", b_state, "Cycle", "High", 1.0)

        dyn = DynamicCIB(m, periods=[1, 2])
        dyn.add_cyclic_descriptor(
            CyclicDescriptor(
                name="Cycle",
                transition={
                    "Low": {"High": 1.0},
                    "High": {"High": 1.0},
                },
            )
        )
        constraints = [ForbiddenPair("Cycle", "High", "B", "High")]
        diag = {}
        p = dyn.simulate_path(
            initial={"Cycle": "Low", "B": "Low"},
            constraints=constraints,
            constraint_mode="repair",
            constrained_top_k=2,
            constrained_backtracking_depth=2,
            diagnostics=diag,
        )
        states = [s.to_dict() for s in p.scenarios]
        assert states[0] == {"Cycle": "Low", "B": "High"}
        assert states[1] == {"Cycle": "High", "B": "Low"}
        assert diag.get("constraint_repairs_applied") == ["post_cyclic_transition"]

    def test_dynamic_constraints_repair_preserves_cyclic_lock_in_equilibrium(self) -> None:
        descriptors = {"Cycle": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        for cycle_state in descriptors["Cycle"]:
            m.set_impact("Cycle", cycle_state, "B", "Low", 0.0)
            m.set_impact("Cycle", cycle_state, "B", "High", 1.0)
        for b_state in descriptors["B"]:
            m.set_impact("B", b_state, "Cycle", "Low", 0.0)
            m.set_impact("B", b_state, "Cycle", "High", 1.0)

        dyn = DynamicCIB(m, periods=[1, 2])
        dyn.add_cyclic_descriptor(
            CyclicDescriptor(
                name="Cycle",
                transition={
                    "Low": {"High": 1.0},
                    "High": {"High": 1.0},
                },
            )
        )
        constraints = [ForbiddenPair("Cycle", "High", "B", "High")]
        diag = {}
        p = dyn.simulate_path(
            initial={"Cycle": "Low", "B": "Low"},
            constraints=constraints,
            constraint_mode="repair",
            constrained_top_k=2,
            constrained_backtracking_depth=2,
            equilibrium_mode="relax_unshocked",
            diagnostics=diag,
        )
        assert p.equilibrium_scenarios is not None
        eq_states = [s.to_dict() for s in p.equilibrium_scenarios]
        assert eq_states[0] == {"Cycle": "Low", "B": "High"}
        assert eq_states[1] == {"Cycle": "High", "B": "Low"}
        assert diag.get("constraint_repairs_applied") == [
            "post_cyclic_transition",
            "post_equilibrium_selection",
        ]

    def test_dynamic_constraints_repair_rejects_non_global_succession(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])
        constraints = [ForbiddenPair("A", "High", "B", "High")]
        with pytest.raises(
            ValueError,
            match="constraint_mode='repair' currently supports only the built-in GlobalSuccession operator",
        ):
            dyn.simulate_path(
                initial={"A": "Low", "B": "Low"},
                constraints=constraints,
                constraint_mode="repair",
                succession_operator=LocalSuccession(),
            )

    def test_dynamic_constraints_repair_rejects_global_subclass(self) -> None:
        class DerivedGlobalSuccession(GlobalSuccession):
            pass

        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])
        constraints = [ForbiddenPair("A", "High", "B", "High")]
        with pytest.raises(
            ValueError,
            match="constraint_mode='repair' currently supports only the built-in GlobalSuccession operator",
        ):
            dyn.simulate_path(
                initial={"A": "Low", "B": "Low"},
                constraints=constraints,
                constraint_mode="repair",
                succession_operator=DerivedGlobalSuccession(),
            )

    def test_dynamic_shocks_reject_non_global_succession_in_simulate_path(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])
        dynamic_shocks = {1: {("A", "High"): 1.0}}
        with pytest.raises(
            ValueError,
            match="dynamic_shocks_by_period currently supports only the built-in GlobalSuccession operator",
        ):
            dyn.simulate_path(
                initial={"A": "Low", "B": "Low"},
                dynamic_shocks_by_period=dynamic_shocks,
                succession_operator=LocalSuccession(),
            )

    def test_dynamic_tau_rejects_non_global_succession_in_simulate_ensemble(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1, 2])
        with pytest.raises(
            ValueError,
            match="dynamic_tau currently supports only the built-in GlobalSuccession operator",
        ):
            dyn.simulate_ensemble(
                initial={"A": "Low", "B": "Low"},
                n_runs=2,
                base_seed=123,
                dynamic_tau=0.2,
                succession_operator=LocalSuccession(),
            )

    def test_dynamic_constraints_repair_returns_only_feasible_scenarios(self) -> None:
        descriptors = {"Cycle": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        for cycle_state in descriptors["Cycle"]:
            m.set_impact("Cycle", cycle_state, "B", "Low", 0.0)
            m.set_impact("Cycle", cycle_state, "B", "High", 1.0)
        for b_state in descriptors["B"]:
            m.set_impact("B", b_state, "Cycle", "Low", 0.0)
            m.set_impact("B", b_state, "Cycle", "High", 1.0)

        dyn = DynamicCIB(m, periods=[1, 2, 3])
        dyn.add_cyclic_descriptor(
            CyclicDescriptor(
                name="Cycle",
                transition={
                    "Low": {"High": 1.0},
                    "High": {"High": 1.0},
                },
            )
        )
        constraints = [ForbiddenPair("Cycle", "High", "B", "High")]
        cidx = ConstraintIndex.from_specs(m, constraints)
        assert cidx is not None
        p = dyn.simulate_path(
            initial={"Cycle": "Low", "B": "Low"},
            constraints=constraints,
            constraint_mode="repair",
            constrained_top_k=2,
            constrained_backtracking_depth=2,
        )
        assert all(cidx.is_full_valid(s.to_indices()) for s in p.scenarios)

    def test_dynamic_constraints_first_period_output_mode_initial(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("B", "Low", "A", "Low", 0.0)
        m.set_impact("B", "Low", "A", "High", 1.0)
        m.set_impact("B", "High", "A", "Low", 0.0)
        m.set_impact("B", "High", "A", "High", 1.0)

        dyn = DynamicCIB(m, periods=[1, 2])
        constraints = [ForbiddenPair("A", "High", "B", "High")]
        p = dyn.simulate_path(
            initial={"A": "Low", "B": "High"},
            constraints=constraints,
            constraint_mode="repair",
            first_period_output_mode="initial",
            constrained_top_k=2,
            constrained_backtracking_depth=2,
        )
        assert p.scenarios[0].to_dict() == {"A": "Low", "B": "High"}
        assert p.scenarios[1].to_dict() != {"A": "Low", "B": "High"}

    def test_dynamic_constraints_ensemble_reproducible(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Low", 2.0)
        m.set_impact("A", "Low", "B", "High", -2.0)
        m.set_impact("A", "High", "B", "Low", -2.0)
        m.set_impact("A", "High", "B", "High", 2.0)
        m.set_impact("B", "Low", "A", "Low", 2.0)
        m.set_impact("B", "Low", "A", "High", -2.0)
        m.set_impact("B", "High", "A", "Low", -2.0)
        m.set_impact("B", "High", "A", "High", 2.0)
        dyn = DynamicCIB(m, periods=[1, 2, 3])
        constraints = [ForbiddenPair("A", "High", "B", "High")]
        paths1 = dyn.simulate_ensemble(
            initial={"A": "Low", "B": "Low"},
            n_runs=8,
            base_seed=42,
            constraint_mode="repair",
            constraints=constraints,
            constrained_top_k=2,
            constrained_backtracking_depth=2,
        )
        paths2 = dyn.simulate_ensemble(
            initial={"A": "Low", "B": "Low"},
            n_runs=8,
            base_seed=42,
            constraint_mode="repair",
            constraints=constraints,
            constrained_top_k=2,
            constrained_backtracking_depth=2,
        )
        assert [p.to_dicts() for p in paths1] == [p.to_dicts() for p in paths2]

