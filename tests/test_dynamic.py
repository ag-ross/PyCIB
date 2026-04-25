"""
Unit tests for dynamic (multi-period) CIB simulation.

These tests validate DynamicCIB path simulation, cyclic descriptors, threshold
rules, extended pathway modes (transient, regime, path_dependent), and related
behaviour.
"""

from __future__ import annotations

import pytest

from cib.constraints import ConstraintIndex, ForbiddenPair
from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.cyclic import CyclicDescriptor
from cib.dynamic import DynamicCIB
from cib.succession import GlobalSuccession, LocalSuccession, SuccessionOperator
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
from cib.pathway import MemoryState
from cib.regimes import RegimeSpec
from cib.transition_kernel import DefaultTransitionKernel


class _SamplingSpyMatrix(CIBMatrix):
    def __init__(self, descriptors):
        super().__init__(descriptors)
        self.sampled_seeds = []

    def sample_matrix(self, seed: int, sigma_scale: float = 1.0):  # type: ignore[override]
        self.sampled_seeds.append(int(seed))
        return self


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

    def test_threshold_modifier_inplace_mutation_is_isolated_from_base_matrix(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Low", 0.0)
        m.set_impact("A", "Low", "B", "High", 0.0)
        m.set_impact("A", "High", "B", "Low", 0.0)
        m.set_impact("A", "High", "B", "High", 0.0)

        def mutating_modifier(base: CIBMatrix) -> CIBMatrix:
            base.set_impact("A", "High", "B", "Low", -3.0)
            base.set_impact("A", "High", "B", "High", 3.0)
            return base

        dyn = DynamicCIB(m, periods=[1])
        dyn.add_threshold_rule(
            ThresholdRule(
                name="MutatingThreshold",
                condition=lambda s: s.get_state("A") == "High",
                modifier=mutating_modifier,
            )
        )

        _ = dyn.simulate_path(initial={"A": "High", "B": "Low"}, seed=123)
        low_path = dyn.simulate_path(initial={"A": "Low", "B": "Low"}, seed=123)

        assert low_path.scenarios[0].to_dict()["B"] == "Low"
        assert m.get_impact("A", "High", "B", "Low") == pytest.approx(0.0)
        assert m.get_impact("A", "High", "B", "High") == pytest.approx(0.0)

    def test_threshold_modifier_inplace_mutation_ensemble_is_reproducible(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Low", 0.0)
        m.set_impact("A", "Low", "B", "High", 0.0)
        m.set_impact("A", "High", "B", "Low", 0.0)
        m.set_impact("A", "High", "B", "High", 0.0)

        def mutating_modifier(base: CIBMatrix) -> CIBMatrix:
            base.set_impact("A", "High", "B", "Low", -3.0)
            base.set_impact("A", "High", "B", "High", 3.0)
            return base

        dyn = DynamicCIB(m, periods=[1, 2])
        dyn.add_threshold_rule(
            ThresholdRule(
                name="MutatingThreshold",
                condition=lambda s: s.get_state("A") == "High",
                modifier=mutating_modifier,
            )
        )
        paths1 = dyn.simulate_ensemble(
            initial={"A": "Low", "B": "Low"},
            n_runs=10,
            base_seed=123,
        )
        paths2 = dyn.simulate_ensemble(
            initial={"A": "Low", "B": "Low"},
            n_runs=10,
            base_seed=123,
        )

        assert [p.to_dicts() for p in paths1] == [p.to_dicts() for p in paths2]

    def test_simulate_path_warns_when_threshold_rule_is_regime_only(self) -> None:
        descriptors = {"A": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])
        dyn.add_threshold_rule(
            ThresholdRule(
                name="RegimeOnlyThreshold",
                condition=lambda s: s.get_state("A") == "Low",
                target_regime="baseline",
            )
        )

        with pytest.warns(UserWarning, match="regime-transition threshold rules are ignored"):
            _ = dyn.simulate_path(initial={"A": "Low"}, seed=123)

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

    def test_simulate_path_seed_none_uses_nondeterministic_judgment_sampling(self) -> None:
        m = _SamplingSpyMatrix({"A": ["Low", "High"]})
        dyn = DynamicCIB(m, periods=[1, 2, 3])

        _ = dyn.simulate_path(
            initial={"A": "Low"},
            seed=None,
            judgment_sigma_scale_by_period={1: 1.0, 2: 1.0, 3: 1.0},
        )
        seeds_first = tuple(m.sampled_seeds)
        m.sampled_seeds.clear()
        _ = dyn.simulate_path(
            initial={"A": "Low"},
            seed=None,
            judgment_sigma_scale_by_period={1: 1.0, 2: 1.0, 3: 1.0},
        )
        seeds_second = tuple(m.sampled_seeds)

        assert len(seeds_first) == 3
        assert len(seeds_second) == 3
        assert seeds_first != seeds_second

    def test_simulate_path_seed_none_uses_nondeterministic_structural_seed_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = CIBMatrix({"A": ["Low", "High"]})
        dyn = DynamicCIB(m, periods=[1, 2, 3])

        import cib.shocks as shocks_mod

        captured = {"seeds": []}
        original_sample = shocks_mod.ShockModel.sample_shocked_matrix

        def _spy_sample(self, random_seed):  # type: ignore[no-untyped-def]
            captured["seeds"].append(int(random_seed))
            return original_sample(self, random_seed)

        monkeypatch.setattr(shocks_mod.ShockModel, "sample_shocked_matrix", _spy_sample)

        _ = dyn.simulate_path(
            initial={"A": "Low"},
            seed=None,
            structural_sigma=0.01,
        )
        seeds_first = tuple(captured["seeds"])
        captured["seeds"].clear()
        _ = dyn.simulate_path(
            initial={"A": "Low"},
            seed=None,
            structural_sigma=0.01,
        )
        seeds_second = tuple(captured["seeds"])

        assert len(seeds_first) == 3
        assert len(seeds_second) == 3
        assert seeds_first != seeds_second

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

    def test_simulate_path_extended_transient_returns_disequilibrium_metrics(self) -> None:
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

        dyn = DynamicCIB(m, periods=[1])
        path = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "High"},
            extension_mode="transient",
        )

        assert path.extension_mode == "transient"
        assert len(path.disequilibrium_metrics) == 1
        assert path.disequilibrium_metrics[0].is_consistent is False
        assert path.disequilibrium_metrics[0].distance_to_consistent_set == 1.0
        assert path.disequilibrium_metrics[0].distance_to_equilibrium == 0.0

    def test_simulate_path_extended_regime_tracks_regime_history(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1, 2])
        boosted = CIBMatrix(descriptors)
        boosted.set_impacts(dict(m.iter_impacts()))
        boosted.set_impact("A", "High", "B", "Low", -2.0)
        boosted.set_impact("A", "High", "B", "High", 2.0)
        dyn.add_regime(RegimeSpec(name="boosted", base_matrix=boosted))
        dyn.set_regime_transition_rule(
            lambda **kwargs: (
                "boosted" if kwargs["realized_scenario"].get_state("A") == "High" else kwargs["current_regime"]
            )
        )

        path = dyn.simulate_path_extended(
            initial={"A": "High", "B": "Low"},
            extension_mode="regime",
            initial_regime="baseline",
        )

        assert len(path.active_regimes) == 2
        assert path.active_regimes[0] in {"baseline", "boosted"}
        assert len(path.active_matrices) == 2

    def test_simulate_path_extended_regime_enforces_outputs_and_stable_provenance(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        boosted = CIBMatrix(descriptors)
        boosted.set_impacts(dict(m.iter_impacts()))
        boosted.set_impact("A", "High", "B", "Low", -2.0)
        boosted.set_impact("A", "High", "B", "High", 2.0)

        def modifier(base: CIBMatrix) -> CIBMatrix:
            out = CIBMatrix(base.descriptors)
            out.set_impacts(dict(base.iter_impacts()))
            out.set_impact("A", "Low", "B", "Low", -1.0)
            out.set_impact("A", "Low", "B", "High", 1.0)
            return out

        dyn = DynamicCIB(m, periods=[1])
        dyn.add_regime(RegimeSpec(name="boosted", base_matrix=boosted))
        dyn.set_regime_transition_rule(lambda **kwargs: "boosted")
        dyn.add_threshold_rule(
            ThresholdRule(
                name="WithinRegimeModifier",
                condition=lambda s: s.get_state("A") == "Low",
                modifier=modifier,
            )
        )

        path1 = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Low"},
            extension_mode="regime",
            return_disequilibrium=False,
            return_active_matrices=False,
            return_transition_events=False,
            return_regime_history=False,
        )
        path2 = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Low"},
            extension_mode="regime",
        )

        assert len(path1.disequilibrium_metrics) == 1
        assert path1.active_regimes == ("boosted",)
        assert len(path1.active_matrices) == 1
        assert len(path1.transition_events) >= 2
        assert path1.active_matrices[0].base_matrix_id == path2.active_matrices[0].base_matrix_id
        assert path1.active_matrices[0].active_matrix_id == path2.active_matrices[0].active_matrix_id
        assert path1.active_matrices[0].diff_summary["n_changed_cells"] > 0.0
        assert "regime:boosted" in path1.active_matrices[0].provenance_labels
        assert (
            "threshold_modifier:WithinRegimeModifier"
            in path1.active_matrices[0].provenance_labels
        )
        assert any(
            event.event_type == "threshold_activation"
            and event.metadata.get("activation_kind") == "modifier"
            for event in path1.transition_events
        )

    def test_threshold_modifier_event_reports_object_identity_semantics(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)

        def mutating_modifier(base: CIBMatrix) -> CIBMatrix:
            base.set_impact("A", "Low", "B", "Low", -1.0)
            base.set_impact("A", "Low", "B", "High", 1.0)
            return base

        dyn = DynamicCIB(m, periods=[1])
        dyn.add_threshold_rule(
            ThresholdRule(
                name="MutatingThreshold",
                condition=lambda s: s.get_state("A") == "Low",
                modifier=mutating_modifier,
            )
        )

        path = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Low"},
            extension_mode="regime",
        )

        modifier_events = [
            event
            for event in path.transition_events
            if event.event_type == "threshold_activation"
            and event.metadata.get("activation_kind") == "modifier"
            and event.metadata.get("threshold_rule") == "MutatingThreshold"
        ]
        assert modifier_events
        assert modifier_events[0].metadata.get(
            "modifier_returned_distinct_object"
        ) is False

    def test_simulate_path_extended_threshold_can_trigger_regime_transition(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)
        boosted = CIBMatrix(descriptors)
        boosted.set_impact("A", "High", "B", "Low", -2.0)
        boosted.set_impact("A", "High", "B", "High", 2.0)

        dyn = DynamicCIB(baseline, periods=[1])
        dyn.add_regime(RegimeSpec(name="boosted", base_matrix=boosted))
        dyn.add_threshold_rule(
            ThresholdRule(
                name="HighAEntersBoostedRegime",
                condition=lambda s: s.get_state("A") == "High",
                target_regime="boosted",
            )
        )

        path = dyn.simulate_path_extended(
            initial={"A": "High", "B": "Low"},
            extension_mode="regime",
            initial_regime="baseline",
        )

        assert path.active_regimes == ("boosted",)
        assert "threshold_regime_transition:HighAEntersBoostedRegime" in (
            path.active_matrices[0].provenance_labels
        )
        assert "threshold_modifier:HighAEntersBoostedRegime" not in (
            path.active_matrices[0].provenance_labels
        )
        assert any(
            event.event_type == "regime_transition"
            and event.source == "threshold_rule"
            and event.metadata.get("activation_kind") == "regime_transition"
            and event.metadata.get("threshold_rule") == "HighAEntersBoostedRegime"
            for event in path.transition_events
        )

    def test_simulate_path_extended_threshold_reaffirmation_does_not_log_regime_transition(
        self,
    ) -> None:
        descriptors = {"A": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)

        dyn = DynamicCIB(baseline, periods=[1, 2])
        dyn.add_threshold_rule(
            ThresholdRule(
                name="StayBaseline",
                condition=lambda s: s.get_state("A") == "Low",
                target_regime="baseline",
            )
        )

        path = dyn.simulate_path_extended(
            initial={"A": "Low"},
            extension_mode="regime",
            initial_regime="baseline",
        )

        assert all(
            "threshold_regime_transition:StayBaseline"
            not in state.provenance_labels
            for state in path.active_matrices
        )
        assert all(
            "threshold_regime_reaffirmation:StayBaseline" in state.provenance_labels
            for state in path.active_matrices
        )
        assert all(
            event.event_type != "regime_transition"
            for event in path.transition_events
            if event.metadata.get("threshold_rule") == "StayBaseline"
        )
        assert any(
            event.event_type == "threshold_activation"
            and event.metadata.get("activation_kind") == "regime_reaffirmation"
            and event.metadata.get("threshold_rule") == "StayBaseline"
            for event in path.transition_events
        )
        assert path.active_matrices[0].entered_regime is True
        assert path.active_matrices[1].entered_regime is False
        assert path.active_matrices[0].regime_entry_period == 1
        assert path.active_matrices[1].regime_entry_period == 1

    def test_simulate_path_extended_reports_active_matrix_equilibrium(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("B", "Low", "A", "Low", 2.0)
        m.set_impact("B", "Low", "A", "High", -2.0)
        m.set_impact("B", "High", "A", "Low", 2.0)
        m.set_impact("B", "High", "A", "High", -2.0)

        dyn = DynamicCIB(m, periods=[1])
        path = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Low"},
            extension_mode="transient",
            dynamic_shocks_by_period={1: {("A", "High"): 10.0}},
            equilibrium_mode="relax_unshocked",
        )

        assert path.equilibrium_scenarios is not None
        realised = path.realised_scenarios[0]
        equilibrium = path.equilibrium_scenarios[0]
        assert realised.to_dict() != equilibrium.to_dict()
        assert equilibrium.to_dict() == {"A": "Low", "B": "Low"}
        assert path.scenarios_for_mode("equilibrium")[0] == equilibrium

    def test_simulate_path_extended_allow_partial_records_converged(self) -> None:
        """With allow_partial=True and a low cap, diagnostics record converged per period."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Weak", 2.0)
        m.set_impact("A", "Low", "B", "Strong", -2.0)
        m.set_impact("A", "High", "B", "Weak", -2.0)
        m.set_impact("A", "High", "B", "Strong", 2.0)
        m.set_impact("B", "Weak", "A", "Low", 1.0)
        m.set_impact("B", "Weak", "A", "High", -1.0)
        m.set_impact("B", "Strong", "A", "Low", -1.0)
        m.set_impact("B", "Strong", "A", "High", 1.0)

        dyn = DynamicCIB(m, periods=[1, 2])
        diag = {}
        path = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Strong"},
            extension_mode="transient",
            succession_operator=LocalSuccession(),
            max_iterations=1,
            allow_partial=True,
            diagnostics=diag,
        )
        assert len(path.realised_scenarios) == 2
        assert "converged" in diag
        assert len(diag["converged"]) == 2
        assert any(c is False for c in diag["converged"])

    def test_simulate_path_extended_allow_partial_equilibrium_uses_higher_cap(self) -> None:
        """With allow_partial and equilibrium requested, equilibrium is relaxed with separate cap."""
        descriptors = {"A": ["Low", "High"], "B": ["Weak", "Strong"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Weak", 2.0)
        m.set_impact("A", "Low", "B", "Strong", -2.0)
        m.set_impact("A", "High", "B", "Weak", -2.0)
        m.set_impact("A", "High", "B", "Strong", 2.0)
        m.set_impact("B", "Weak", "A", "Low", 1.0)
        m.set_impact("B", "Weak", "A", "High", -1.0)
        m.set_impact("B", "Strong", "A", "Low", -1.0)
        m.set_impact("B", "Strong", "A", "High", 1.0)

        dyn = DynamicCIB(m, periods=[1])
        path = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Strong"},
            extension_mode="transient",
            succession_operator=LocalSuccession(),
            max_iterations=1,
            allow_partial=True,
            equilibrium_mode="relax_unshocked",
            equilibrium_max_iterations=100,
        )
        assert path.equilibrium_scenarios is not None
        assert len(path.equilibrium_scenarios) == 1
        # Equilibrium run uses the higher cap and should converge to a consistent scenario.
        eq = path.equilibrium_scenarios[0]
        assert ConsistencyChecker.check_consistency(eq, m) is True

    def test_simulate_path_extended_path_dependent_tracks_memory_and_consistency(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1, 2])
        path = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Low"},
            extension_mode="path_dependent",
            memory_state=MemoryState(
                period=0,
                values={"required_regime": "baseline"},
                flags={},
                export_label="memory",
            ),
            transition_kernel=DefaultTransitionKernel(),
        )

        assert len(path.memory_states) == 2
        assert len(path.structural_consistency) == 2
        assert path.structural_consistency[0].is_structurally_consistent is True

    def test_simulate_path_extended_path_dependent_memory_changes_realized_transition(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("B", "Low", "A", "Low", 2.0)
        m.set_impact("B", "Low", "A", "High", -2.0)
        m.set_impact("B", "High", "A", "Low", 2.0)
        m.set_impact("B", "High", "A", "High", -2.0)

        dyn = DynamicCIB(m, periods=[1])
        path = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Low"},
            extension_mode="path_dependent",
            memory_state=MemoryState(
                period=0,
                values={"locked_descriptors": {"B": "High"}},
                flags={"locked_in": True},
                export_label="memory",
            ),
            transition_kernel=DefaultTransitionKernel(),
        )

        assert path.realised_scenarios[0].to_dict()["B"] == "High"
        assert any(
            event.event_type == "irreversible_transition"
            for event in path.transition_events
        )
        assert path.structural_consistency[0].is_structurally_consistent is True

    def test_simulate_path_extended_path_dependent_does_not_log_plain_kernel_metadata_as_memory_update(
        self,
    ) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])

        path = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Low"},
            extension_mode="path_dependent",
            transition_kernel=DefaultTransitionKernel(),
        )

        assert path.transition_events == ()

    def test_simulate_path_extended_path_dependent_logs_memory_update_when_kernel_changes_memory(
        self,
    ) -> None:
        descriptors = {"A": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])

        path = dyn.simulate_path_extended(
            initial={"A": "Low"},
            extension_mode="path_dependent",
            transition_kernel=lambda **kwargs: (
                kwargs["current_scenario"],
                MemoryState(
                    period=0,
                    values={"phase": 1},
                    flags={},
                    export_label="memory",
                ),
                {"used_history": False},
            ),
        )

        assert any(
            event.event_type == "memory_update"
            for event in path.transition_events
        )

    def test_simulate_path_extended_does_not_alias_caller_memory_state(self) -> None:
        descriptors = {"A": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        dyn = DynamicCIB(matrix, periods=[1])
        initial_memory = MemoryState(
            period=0,
            values={"phase": {"value": 0}},
            flags={},
            export_label="memory",
        )

        def mutating_kernel(**kwargs):
            memory_state = kwargs["memory_state"]
            if memory_state is not None:
                memory_state.values["phase"]["value"] = 99
            return kwargs["current_scenario"], memory_state, {}

        _ = dyn.simulate_path_extended(
            initial={"A": "Low"},
            extension_mode="path_dependent",
            memory_state=initial_memory,
            transition_kernel=mutating_kernel,
        )

        assert initial_memory.values["phase"]["value"] == 0

    def test_simulate_path_extended_initial_output_keeps_period_zero_state_coherent(
        self,
    ) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[2025, 2030])

        path = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Low"},
            extension_mode="path_dependent",
            first_period_output_mode="initial",
            memory_state=MemoryState(
                period=0,
                values={"required_regime": "baseline"},
                flags={},
                export_label="memory",
            ),
            transition_kernel=DefaultTransitionKernel(),
        )

        assert path.realised_scenarios[0].to_dict() == {"A": "Low", "B": "Low"}
        assert path.memory_states[0].values == {"required_regime": "baseline"}
        assert path.memory_states[0].period == 2025
        assert path.structural_consistency[0].is_structurally_consistent is True

    def test_first_period_initial_output_keeps_internal_history_for_path_callbacks(
        self,
    ) -> None:
        descriptors = {"A": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1, 2])
        call_count = {"n": 0}

        def kernel(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return Scenario({"A": "High"}, m), kwargs["memory_state"], {}
            previous_path = kwargs["previous_path"]
            if not previous_path:
                raise ValueError("missing previous path for second period")
            if previous_path[-1].to_dict()["A"] != "High":
                raise ValueError("internal previous path did not preserve realised attractor")
            return kwargs["current_scenario"], kwargs["memory_state"], {}

        path = dyn.simulate_path_extended(
            initial={"A": "Low"},
            extension_mode="path_dependent",
            first_period_output_mode="initial",
            transition_kernel=kernel,
        )

        assert path.realised_scenarios[0].to_dict()["A"] == "Low"
        assert path.realised_scenarios[1].to_dict()["A"] == "High"

    def test_simulate_path_extended_metrics_use_locked_period_operator(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Low", 2.0)
        m.set_impact("A", "Low", "B", "High", -2.0)
        m.set_impact("A", "High", "B", "Low", -2.0)
        m.set_impact("A", "High", "B", "High", 2.0)

        dyn = DynamicCIB(m, periods=[1, 2])
        dyn.add_cyclic_descriptor(
            CyclicDescriptor(
                name="B",
                transition={
                    "Low": {"Low": 0.0, "High": 1.0},
                    "High": {"Low": 0.0, "High": 1.0},
                },
            )
        )

        path = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Low"},
            extension_mode="transient",
            seed=123,
        )

        assert path.realised_scenarios[1].to_dict() == {"A": "Low", "B": "High"}
        assert path.disequilibrium_metrics[1].is_consistent is False
        assert path.disequilibrium_metrics[1].time_to_equilibrium is None

    def test_simulate_path_extended_equilibrium_respects_constraints_and_locks(self) -> None:
        descriptors = {"Cycle": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        for cycle_state in descriptors["Cycle"]:
            m.set_impact("Cycle", cycle_state, "B", "Low", 0.0)
            m.set_impact("Cycle", cycle_state, "B", "High", 1.0)

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

        path = dyn.simulate_path_extended(
            initial={"Cycle": "Low", "B": "Low"},
            extension_mode="transient",
            constraints=constraints,
            constraint_mode="repair",
            constrained_top_k=2,
            constrained_backtracking_depth=2,
            equilibrium_mode="relax_unshocked",
        )

        assert path.equilibrium_scenarios is not None
        eq_states = [s.to_dict() for s in path.equilibrium_scenarios]
        assert eq_states == [
            {"Cycle": "Low", "B": "High"},
            {"Cycle": "High", "B": "Low"},
        ]

    def test_trace_to_equilibrium_stops_on_first_consistent_state(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("B", "Low", "A", "Low", 2.0)
        m.set_impact("B", "Low", "A", "High", -2.0)
        m.set_impact("B", "High", "A", "Low", 2.0)
        m.set_impact("B", "High", "A", "High", -2.0)

        dyn = DynamicCIB(m, periods=[1])
        path = dyn.trace_to_equilibrium(initial={"A": "High", "B": "High"})

        assert path.disequilibrium_metrics[0].time_to_equilibrium == 1
        assert path.disequilibrium_metrics[0].entered_consistent_set is False
        assert path.disequilibrium_metrics[-1].is_consistent is True
        assert path.disequilibrium_metrics[-1].time_to_equilibrium == 0
        assert path.disequilibrium_metrics[-1].entered_consistent_set is True

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

    def test_dynamic_shocks_reject_custom_succession_in_simulate_path(self) -> None:
        """Custom operators (not GlobalSuccession or LocalSuccession) are rejected with dynamic shocks."""

        class CustomSuccession(SuccessionOperator):
            def find_successor(self, scenario, matrix):
                return scenario

        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])
        dynamic_shocks = {1: {("A", "High"): 1.0}}
        with pytest.raises(
            ValueError,
            match="GlobalSuccession or LocalSuccession operator",
        ):
            dyn.simulate_path(
                initial={"A": "Low", "B": "Low"},
                dynamic_shocks_by_period=dynamic_shocks,
                succession_operator=CustomSuccession(),
            )

    def test_dynamic_shocks_accept_local_succession_in_simulate_path(self) -> None:
        """simulate_path with dynamic_shocks_by_period accepts LocalSuccession."""
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1])
        dynamic_shocks = {1: {("A", "High"): 1.0}}
        path = dyn.simulate_path(
            initial={"A": "Low", "B": "Low"},
            dynamic_shocks_by_period=dynamic_shocks,
            succession_operator=LocalSuccession(),
            max_iterations=50,
        )
        assert len(path.scenarios) >= 1
        assert path.scenarios[0].to_dict()["A"] in ("Low", "High")
        assert path.scenarios[0].to_dict()["B"] in ("Low", "High")

    def test_simulate_path_extended_with_dynamic_shocks_and_local_succession(
        self,
    ) -> None:
        """simulate_path_extended with dynamic_shocks_by_period and LocalSuccession runs."""
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Low", 2.0)
        m.set_impact("A", "Low", "B", "High", -2.0)
        m.set_impact("A", "High", "B", "Low", -2.0)
        m.set_impact("A", "High", "B", "High", 2.0)
        m.set_impact("B", "Low", "A", "Low", 1.0)
        m.set_impact("B", "Low", "A", "High", -1.0)
        m.set_impact("B", "High", "A", "Low", -1.0)
        m.set_impact("B", "High", "A", "High", 1.0)
        dyn = DynamicCIB(m, periods=[1, 2])
        dynamic_shocks = {1: {("A", "High"): 0.5}, 2: {("B", "High"): 0.5}}
        pathway = dyn.simulate_path_extended(
            initial={"A": "Low", "B": "Low"},
            extension_mode="transient",
            first_period_output_mode="initial",
            dynamic_shocks_by_period=dynamic_shocks,
            succession_operator=LocalSuccession(),
            max_iterations=50,
            seed=42,
        )
        assert pathway.periods is not None
        assert len(pathway.periods) == 2
        scenarios = pathway.scenarios_for_mode("realized")
        assert len(scenarios) == 2
        for s in scenarios:
            assert s.to_dict()["A"] in ("Low", "High")
            assert s.to_dict()["B"] in ("Low", "High")

    def test_dynamic_tau_rejects_custom_succession_in_simulate_ensemble(self) -> None:
        """Custom operators are rejected when dynamic_tau is used."""

        class CustomSuccession(SuccessionOperator):
            def find_successor(self, scenario, matrix):
                return scenario

        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        dyn = DynamicCIB(m, periods=[1, 2])
        with pytest.raises(
            ValueError,
            match="GlobalSuccession or LocalSuccession operator",
        ):
            dyn.simulate_ensemble(
                initial={"A": "Low", "B": "Low"},
                n_runs=2,
                base_seed=123,
                dynamic_tau=0.2,
                succession_operator=CustomSuccession(),
            )

    def test_dynamic_tau_accepts_local_succession_in_simulate_ensemble(self) -> None:
        """simulate_ensemble with dynamic_tau accepts LocalSuccession."""
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("A", "Low", "B", "Low", 2.0)
        m.set_impact("A", "Low", "B", "High", -2.0)
        m.set_impact("A", "High", "B", "Low", -2.0)
        m.set_impact("A", "High", "B", "High", 2.0)
        m.set_impact("B", "Low", "A", "Low", 1.0)
        m.set_impact("B", "Low", "A", "High", -1.0)
        m.set_impact("B", "High", "A", "Low", -1.0)
        m.set_impact("B", "High", "A", "High", 1.0)
        dyn = DynamicCIB(m, periods=[1, 2])
        pathways = dyn.simulate_ensemble(
            initial={"A": "Low", "B": "Low"},
            n_runs=2,
            base_seed=123,
            dynamic_tau=0.1,
            succession_operator=LocalSuccession(),
            max_iterations=50,
        )
        assert len(pathways) == 2
        for pathway in pathways:
            assert pathway.periods is not None
            assert len(pathway.periods) >= 1
            scenarios = pathway.scenarios_for_mode("realized")
            assert len(scenarios) >= 1

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

