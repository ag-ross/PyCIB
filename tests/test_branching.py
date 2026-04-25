"""
Unit tests for hybrid branching pathway construction.

These tests validate BranchingPathwayBuilder for multi-branch pathways with
regimes, transition kernels, and structural consistency.
"""

from __future__ import annotations

import pytest

from cib.branching import BranchingPathwayBuilder
from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.cyclic import CyclicDescriptor
from cib.pathway import MemoryState, branching_regime_residence_timelines
from cib.regimes import RegimeSpec
from cib.succession import GlobalSuccession
from cib.threshold import ThresholdRule
from cib.transition_kernel import DefaultTransitionKernel


def _coordination_matrix() -> CIBMatrix:
    """
    Two-descriptor coordination system with two fixed points:
      (Low,Low) and (High,High).
    """
    desc = {"A": ["Low", "High"], "B": ["Low", "High"]}
    m = CIBMatrix(desc)
    # A influences B.
    m.set_impact("A", "Low", "B", "Low", 2.0)
    m.set_impact("A", "Low", "B", "High", -2.0)
    m.set_impact("A", "High", "B", "Low", -2.0)
    m.set_impact("A", "High", "B", "High", 2.0)
    # B influences A.
    m.set_impact("B", "Low", "A", "Low", 2.0)
    m.set_impact("B", "Low", "A", "High", -2.0)
    m.set_impact("B", "High", "A", "Low", -2.0)
    m.set_impact("B", "High", "A", "High", 2.0)
    return m


def _asymmetric_basin_matrix() -> CIBMatrix:
    """
    Three-descriptor system with two fixed points and asymmetric basin sizes.
    """
    desc = {"A": ["Low", "High"], "B": ["Low", "High"], "C": ["Low", "High"]}
    m = CIBMatrix(desc)
    # A follows a low-biased majority of B and C.
    for b in desc["B"]:
        for c in desc["C"]:
            low_score = (2.0 if b == "Low" else 0.0) + (2.0 if c == "Low" else 0.0)
            high_score = (1.0 if b == "High" else 0.0) + (1.0 if c == "High" else 0.0)
            m.set_impact("B", b, "A", "Low", 2.0 if b == "Low" else 0.0)
            m.set_impact("B", b, "A", "High", 1.0 if b == "High" else 0.0)
            m.set_impact("C", c, "A", "Low", 2.0 if c == "Low" else 0.0)
            m.set_impact("C", c, "A", "High", 1.0 if c == "High" else 0.0)
            _ = low_score + high_score
    # B and C each follow A.
    m.set_impact("A", "Low", "B", "Low", 1.0)
    m.set_impact("A", "Low", "B", "High", 0.0)
    m.set_impact("A", "High", "B", "Low", 0.0)
    m.set_impact("A", "High", "B", "High", 1.0)
    m.set_impact("A", "Low", "C", "Low", 1.0)
    m.set_impact("A", "Low", "C", "High", 0.0)
    m.set_impact("A", "High", "C", "Low", 0.0)
    m.set_impact("A", "High", "C", "High", 1.0)
    return m


class TestBranchingPathwayBuilder:
    def test_enumeration_branch_builds_two_nodes(self) -> None:
        m = _coordination_matrix()
        b = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "Low", "B": "High"},
            max_states_to_enumerate=10_000,
            n_transition_samples=50,
            base_seed=123,
        )
        r = b.build(top_k=5)

        assert r.periods == (1, 2)
        # A deterministic attractor is selected from the initial scenario.
        assert len(r.scenarios_by_period[0]) == 1
        # Both consistent futures are enumerated in the next period.
        assert len(r.scenarios_by_period[1]) == 2
        assert r.transition_method[1] == "enumerate"

        # Outgoing edge weights from the root are normalised to 1.
        out = r.edges[(0, 0)]
        assert abs(sum(out.values()) - 1.0) < 1e-9

    def test_enumeration_branch_uses_basin_weighted_edges(self) -> None:
        m = _asymmetric_basin_matrix()
        b = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "Low", "B": "High", "C": "High"},
            max_states_to_enumerate=10_000,
            n_transition_samples=20,
            base_seed=123,
        )
        r = b.build(top_k=5)
        assert r.transition_method[1] == "enumerate"
        out = r.edges[(0, 0)]
        assert len(out) >= 2

        # Compute exact basin shares under the same deterministic succession law.
        succ = GlobalSuccession()
        basin_counts: dict[tuple[int, ...], int] = {}
        for a in m.descriptors["A"]:
            for b_state in m.descriptors["B"]:
                for c in m.descriptors["C"]:
                    s0 = Scenario({"A": a, "B": b_state, "C": c}, m)
                    res = succ.find_attractor(s0, m, max_iterations=1000)
                    if res.is_cycle:
                        continue
                    attr = res.attractor
                    assert isinstance(attr, Scenario)
                    key = tuple(int(v) for v in attr.to_indices())
                    basin_counts[key] = basin_counts.get(key, 0) + 1

        node_by_idx = {
            idx: tuple(int(v) for v in scenario.to_indices())
            for idx, scenario in enumerate(r.scenarios_by_period[1])
        }
        denom = float(sum(basin_counts.values()))
        expected = {
            idx: float(basin_counts[node_key]) / denom
            for idx, node_key in node_by_idx.items()
            if node_key in basin_counts
        }
        for idx, w in expected.items():
            assert out[idx] == pytest.approx(w, abs=1e-9)

    def test_sampling_fallback_is_reproducible(self) -> None:
        # Sampling is forced by setting the enumeration threshold below the full space size.
        m = _coordination_matrix()
        b1 = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "Low", "B": "High"},
            max_states_to_enumerate=1,
            n_transition_samples=200,
            base_seed=123,
            dynamic_tau=0.25,
            dynamic_rho=0.5,
            dynamic_innovation_dist="student_t",
            dynamic_innovation_df=5.0,
            dynamic_jump_prob=0.05,
            dynamic_jump_scale=0.8,
        )
        b2 = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "Low", "B": "High"},
            max_states_to_enumerate=1,
            n_transition_samples=200,
            base_seed=123,
            dynamic_tau=0.25,
            dynamic_rho=0.5,
            dynamic_innovation_dist="student_t",
            dynamic_innovation_df=5.0,
            dynamic_jump_prob=0.05,
            dynamic_jump_scale=0.8,
        )

        r1 = b1.build(top_k=5)
        r2 = b2.build(top_k=5)

        assert r1.transition_method[1] == "sample"
        assert r1.edges == r2.edges
        assert r1.top_paths == r2.top_paths

    def test_non_memory_sampling_marks_approximation_contract(self) -> None:
        m = _coordination_matrix()
        builder = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "Low", "B": "High"},
            max_states_to_enumerate=1,
            n_transition_samples=20,
            base_seed=123,
        )

        result = builder.build(top_k=5)

        assert result.transition_method[1] == "sample"
        assert result.approximation_contract == "approximate_scenario_regime_branching"

    def test_sampling_nodes_can_be_equilibrium_consistent(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        m = CIBMatrix(descriptors)
        m.set_impact("B", "Low", "A", "Low", 2.0)
        m.set_impact("B", "Low", "A", "High", -2.0)
        m.set_impact("B", "High", "A", "Low", 2.0)
        m.set_impact("B", "High", "A", "High", -2.0)

        b = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "Low", "B": "Low"},
            max_states_to_enumerate=1,
            n_transition_samples=50,
            base_seed=123,
            dynamic_tau=0.5,
            dynamic_rho=0.5,
        )
        r = b.build(top_k=5)

        for s in r.scenarios_by_period[0]:
            assert ConsistencyChecker.check_consistency(s, m) is True
        for s in r.scenarios_by_period[1]:
            assert ConsistencyChecker.check_consistency(s, m) is True

    def test_per_parent_topk_pruning_reduces_branching(self) -> None:
        m = _coordination_matrix()
        b = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "Low", "B": "High"},
            max_states_to_enumerate=10_000,
            n_transition_samples=50,
            prune_policy="per_parent_topk",
            per_parent_top_k=1,
            base_seed=123,
        )
        r = b.build(top_k=5)
        out = r.edges[(0, 0)]
        assert len(out) == 1
        assert abs(sum(out.values()) - 1.0) < 1e-9

    def test_min_edge_weight_pruning_returns_dead_end_when_all_removed(self) -> None:
        m = _coordination_matrix()
        b = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "Low", "B": "High"},
            max_states_to_enumerate=10_000,
            n_transition_samples=50,
            prune_policy="min_edge_weight",
            min_edge_weight=0.9,
            base_seed=123,
        )
        r = b.build(top_k=5)
        out = r.edges[(0, 0)]
        assert out == {}

    def test_threshold_evaluated_on_post_cyclic_scenario(self) -> None:
        """
        Threshold rules and cyclic descriptors are used together. The active CIM
        for period t+1 is determined by the scenario at the start of period t+1
        (after cyclic transitions), so that behaviour matches DynamicCIB.
        """
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

        def modifier(base: CIBMatrix) -> CIBMatrix:
            out = CIBMatrix(base.descriptors)
            out.set_impacts(dict(base._impacts))  # type: ignore[attr-defined]
            out.set_impact("A", "High", "B", "Low", -3.0)
            out.set_impact("A", "High", "B", "High", 3.0)
            return out

        rule = ThresholdRule(
            name="IfAHighBoostBHigh",
            condition=lambda s: s.get_state("A") == "High",
            modifier=modifier,
        )
        cyclic_a = CyclicDescriptor(
            name="A",
            transition={"Low": {"High": 1.0}, "High": {"High": 1.0}},
        )

        b = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "Low", "B": "Low"},
            cyclic_descriptors=[cyclic_a],
            threshold_rules=[rule],
            max_states_to_enumerate=10_000,
            n_transition_samples=50,
            base_seed=123,
        )
        r = b.build(top_k=5)

        assert r.periods == (1, 2)
        assert len(r.scenarios_by_period[0]) >= 1
        assert len(r.scenarios_by_period[1]) >= 1
        # After cyclic transition, A becomes High; threshold is evaluated on
        # that post-cyclic scenario, so the modifier is applied and B is drawn to High.
        period1_states = [s.to_dict() for s in r.scenarios_by_period[1]]
        assert any(d.get("A") == "High" and d.get("B") == "High" for d in period1_states), (
            "Expected at least one period-1 scenario with A=High, B=High when "
            "threshold is evaluated on post-cyclic scenario."
        )

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

        b_first = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "High", "B": "Low"},
            threshold_rules=[rule1, rule2],
            threshold_match_policy="first_match",
            max_states_to_enumerate=10_000,
            n_transition_samples=50,
            base_seed=123,
        )
        r_first = b_first.build(top_k=5)
        assert r_first.scenarios_by_period[0][0].to_dict()["B"] == "High"

        b_all = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "High", "B": "Low"},
            threshold_rules=[rule1, rule2],
            threshold_match_policy="all_matches",
            max_states_to_enumerate=10_000,
            n_transition_samples=50,
            base_seed=123,
        )
        r_all = b_all.build(top_k=5)
        assert r_all.scenarios_by_period[0][0].to_dict()["B"] == "Low"

    def test_threshold_modifier_inplace_mutation_is_isolated_in_branching(self) -> None:
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

        builder = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "High", "B": "Low"},
            threshold_rules=[
                ThresholdRule(
                    name="MutatingThreshold",
                    condition=lambda s: s.get_state("A") == "High",
                    modifier=mutating_modifier,
                )
            ],
            max_states_to_enumerate=10_000,
            n_transition_samples=10,
            base_seed=123,
        )
        result = builder.build(top_k=5)

        assert len(result.scenarios_by_period[0]) >= 1
        assert m.get_impact("A", "High", "B", "Low") == 0.0
        assert m.get_impact("A", "High", "B", "High") == 0.0

    def test_regime_aware_branching_tracks_regime_identity(self) -> None:
        m = _coordination_matrix()
        boosted = CIBMatrix(m.descriptors)
        boosted.set_impacts(dict(m.iter_impacts()))
        boosted.set_impact("A", "High", "B", "Low", -3.0)
        boosted.set_impact("A", "High", "B", "High", 3.0)

        builder = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "High", "B": "Low"},
            regimes=[RegimeSpec(name="boosted", base_matrix=boosted)],
            initial_regime="baseline",
            regime_transition_rule=lambda **kwargs: (
                "boosted" if kwargs["realized_scenario"].get_state("A") == "High" else kwargs["current_regime"]
            ),
            max_states_to_enumerate=10_000,
            n_transition_samples=50,
            base_seed=123,
        )

        result = builder.build(top_k=5)

        assert len(result.active_regimes) == 2
        assert result.active_regimes[0][0] == "boosted"

    def test_regime_aware_branching_keeps_identical_scenarios_distinct_by_regime(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        baseline = _coordination_matrix()

        def low_fixed_matrix() -> CIBMatrix:
            matrix = CIBMatrix(descriptors)
            matrix.set_impact("A", "Low", "B", "Low", 2.0)
            matrix.set_impact("A", "Low", "B", "High", -2.0)
            matrix.set_impact("A", "High", "B", "Low", 2.0)
            matrix.set_impact("A", "High", "B", "High", -2.0)
            matrix.set_impact("B", "Low", "A", "Low", 2.0)
            matrix.set_impact("B", "Low", "A", "High", -2.0)
            matrix.set_impact("B", "High", "A", "Low", 2.0)
            matrix.set_impact("B", "High", "A", "High", -2.0)
            return matrix

        builder = BranchingPathwayBuilder(
            base_matrix=baseline,
            periods=[1, 2, 3],
            initial={"A": "Low", "B": "High"},
            regimes=[RegimeSpec(name="high_path", base_matrix=low_fixed_matrix())],
            initial_regime="baseline",
            regime_transition_rule=lambda **kwargs: (
                "high_path"
                if kwargs["realized_scenario"].get_state("A") == "High"
                else kwargs["current_regime"]
            ),
            max_states_to_enumerate=10_000,
            n_transition_samples=20,
            base_seed=123,
        )

        result = builder.build(top_k=10)

        final_layer_pairs = [
            (scenario.to_dict(), regime)
            for scenario, regime in zip(
                result.scenarios_by_period[2],
                result.active_regimes[2],
            )
            if scenario.to_dict() == {"A": "Low", "B": "Low"}
        ]
        assert len(final_layer_pairs) >= 2
        assert {regime for _, regime in final_layer_pairs} >= {
            "baseline",
            "high_path",
        }

    def test_regime_aware_top_path_details_preserve_regime_context(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        baseline = _coordination_matrix()

        def low_fixed_matrix() -> CIBMatrix:
            matrix = CIBMatrix(descriptors)
            matrix.set_impact("A", "Low", "B", "Low", 2.0)
            matrix.set_impact("A", "Low", "B", "High", -2.0)
            matrix.set_impact("A", "High", "B", "Low", 2.0)
            matrix.set_impact("A", "High", "B", "High", -2.0)
            matrix.set_impact("B", "Low", "A", "Low", 2.0)
            matrix.set_impact("B", "Low", "A", "High", -2.0)
            matrix.set_impact("B", "High", "A", "Low", 2.0)
            matrix.set_impact("B", "High", "A", "High", -2.0)
            return matrix

        builder = BranchingPathwayBuilder(
            base_matrix=baseline,
            periods=[1, 2, 3],
            initial={"A": "Low", "B": "High"},
            regimes=[RegimeSpec(name="high_path", base_matrix=low_fixed_matrix())],
            initial_regime="baseline",
            regime_transition_rule=lambda **kwargs: (
                "high_path"
                if kwargs["realized_scenario"].get_state("A") == "High"
                else kwargs["current_regime"]
            ),
            max_states_to_enumerate=10_000,
            n_transition_samples=20,
            base_seed=123,
        )

        result = builder.build(top_k=10)
        details = result.top_path_details()

        matching_paths = [
            path
            for path in details
            if path["records"][-1]["scenario"] == {"A": "Low", "B": "Low"}
        ]
        final_regimes = {path["records"][-1]["regime"] for path in matching_paths}
        assert final_regimes >= {"baseline", "high_path"}

    def test_threshold_can_trigger_regime_transition_in_branching(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)
        boosted = CIBMatrix(descriptors)
        boosted.set_impact("A", "High", "B", "Low", -2.0)
        boosted.set_impact("A", "High", "B", "High", 2.0)

        builder = BranchingPathwayBuilder(
            base_matrix=baseline,
            periods=[1, 2],
            initial={"A": "High", "B": "Low"},
            regimes=[RegimeSpec(name="boosted", base_matrix=boosted)],
            threshold_rules=[
                ThresholdRule(
                    name="HighAEntersBoostedRegime",
                    condition=lambda s: s.get_state("A") == "High",
                    target_regime="boosted",
                )
            ],
            max_states_to_enumerate=10_000,
            n_transition_samples=20,
            base_seed=123,
        )

        result = builder.build(top_k=5)

        assert result.active_regimes[0] == ("boosted",)

    def test_memory_aware_branching_records_memory_and_contract(self) -> None:
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

        builder = BranchingPathwayBuilder(
            base_matrix=m,
            periods=[1, 2],
            initial={"A": "Low", "B": "Low"},
            max_states_to_enumerate=10_000,
            n_transition_samples=40,
            base_seed=123,
            memory_state=MemoryState(
                period=0,
                values={"locked_descriptors": {"B": "High"}},
                flags={"locked_in": True},
                export_label="memory",
            ),
            transition_kernel=DefaultTransitionKernel(),
        )

        result = builder.build(top_k=5)

        assert result.transition_method[1] == "sample"
        assert "memory_aware_sampling" in result.approximation_contract
        assert len(result.memory_states_by_period) == 2
        assert len(result.memory_states_by_period[0]) == 1
        assert all(
            memory.flags.get("locked_in", False) is True
            for memory in result.memory_states_by_period[1]
        )
        assert all(
            scenario.to_dict()["B"] == "High"
            for scenario in result.scenarios_by_period[1]
        )

    def test_memory_aware_root_enforces_cyclic_locks(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        cyclic_b = CyclicDescriptor(
            name="B",
            transition={"Low": {"Low": 1.0}, "High": {"High": 1.0}},
        )

        def violating_root_kernel(**kwargs):
            current = kwargs["current_scenario"]
            active = kwargs["active_matrix"]
            memory = kwargs["memory_state"]
            state = current.to_dict()
            state["B"] = "High" if state["B"] == "Low" else "Low"
            return Scenario(state, active), memory, {"forced_violation": True}

        builder = BranchingPathwayBuilder(
            base_matrix=matrix,
            periods=[1, 2],
            initial={"A": "Low", "B": "Low"},
            cyclic_descriptors=[cyclic_b],
            max_states_to_enumerate=10_000,
            n_transition_samples=5,
            base_seed=123,
            memory_state=MemoryState(period=0, values={}, flags={}, export_label="memory"),
            transition_kernel=violating_root_kernel,
        )

        result = builder.build(top_k=3)

        assert result.scenarios_by_period[0][0].to_dict()["B"] == "Low"

    def test_memory_aware_sampling_enforces_cyclic_locks_after_kernel_step(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"], "C": ["Low", "High"]}
        matrix = CIBMatrix(descriptors)
        cyclic_b = CyclicDescriptor(
            name="B",
            transition={"Low": {"Low": 1.0}, "High": {"High": 1.0}},
        )

        def violating_kernel(**kwargs):
            current = kwargs["current_scenario"]
            active = kwargs["active_matrix"]
            memory = kwargs["memory_state"]
            state = current.to_dict()
            state["B"] = "High" if state["B"] == "Low" else "Low"
            return Scenario(state, active), memory, {"forced_violation": True}

        builder = BranchingPathwayBuilder(
            base_matrix=matrix,
            periods=[1, 2],
            initial={"A": "Low", "B": "Low", "C": "Low"},
            cyclic_descriptors=[cyclic_b],
            max_states_to_enumerate=1,
            n_transition_samples=3,
            base_seed=123,
            memory_state=MemoryState(period=0, values={}, flags={}, export_label="memory"),
            transition_kernel=violating_kernel,
        )

        result = builder.build(top_k=3)

        assert result.transition_method[1] == "sample"
        assert all(
            scenario.to_dict()["B"] == "Low"
            for scenario in result.scenarios_by_period[1]
        )

    def test_sampling_regime_aware_branching_uses_resolved_regime_matrix(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"], "C": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)
        baseline.set_impact("A", "Low", "B", "Low", 2.0)
        baseline.set_impact("A", "Low", "B", "High", -2.0)
        baseline.set_impact("A", "High", "B", "Low", 2.0)
        baseline.set_impact("A", "High", "B", "High", -2.0)

        boosted = CIBMatrix(descriptors)
        boosted.set_impacts(dict(baseline.iter_impacts()))
        boosted.set_impact("A", "Low", "B", "Low", -2.0)
        boosted.set_impact("A", "Low", "B", "High", 2.0)
        boosted.set_impact("A", "High", "B", "Low", -2.0)
        boosted.set_impact("A", "High", "B", "High", 2.0)

        builder = BranchingPathwayBuilder(
            base_matrix=baseline,
            periods=[1, 2],
            initial={"A": "High", "B": "Low", "C": "Low"},
            regimes=[RegimeSpec(name="boosted", base_matrix=boosted)],
            initial_regime="baseline",
            regime_transition_rule=lambda **kwargs: "boosted",
            max_states_to_enumerate=1,
            n_transition_samples=20,
            base_seed=123,
        )

        result = builder.build(top_k=5)

        assert result.transition_method[1] == "sample"
        assert result.active_regimes[1] == ("boosted",)
        assert result.scenarios_by_period[1][0].to_dict()["B"] == "High"

    def test_branching_root_respects_period_zero_regime_resolution(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)
        boosted = CIBMatrix(descriptors)
        boosted.set_impacts(dict(baseline.iter_impacts()))
        boosted.set_impact("A", "Low", "B", "Low", -2.0)
        boosted.set_impact("A", "Low", "B", "High", 2.0)

        builder = BranchingPathwayBuilder(
            base_matrix=baseline,
            periods=[2025, 2030],
            initial={"A": "Low", "B": "Low"},
            regimes=[RegimeSpec(name="boosted", base_matrix=boosted)],
            initial_regime="baseline",
            regime_transition_rule=lambda **kwargs: "boosted",
            max_states_to_enumerate=1,
            n_transition_samples=1,
            base_seed=123,
        )

        result = builder.build(top_k=3)

        assert result.active_regimes[0] == ("boosted",)
        assert result.regime_states_by_period[0][0].entered_regime is True
        assert result.regime_states_by_period[0][0].regime_entry_period == 2025

    def test_memory_aware_branching_passes_memory_into_regime_resolution(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)
        boosted = CIBMatrix(descriptors)
        boosted.set_impact("A", "Low", "B", "Low", -2.0)
        boosted.set_impact("A", "Low", "B", "High", 2.0)

        builder = BranchingPathwayBuilder(
            base_matrix=baseline,
            periods=[1, 2],
            initial={"A": "Low", "B": "Low"},
            regimes=[RegimeSpec(name="boosted", base_matrix=boosted)],
            initial_regime="baseline",
            regime_transition_rule=lambda **kwargs: (
                "boosted"
                if kwargs["memory_state"] is not None
                and kwargs["memory_state"].values.get("required_regime") == "boosted"
                else kwargs["current_regime"]
            ),
            max_states_to_enumerate=10_000,
            n_transition_samples=20,
            base_seed=123,
            memory_state=MemoryState(
                period=0,
                values={"required_regime": "boosted"},
                flags={},
                export_label="memory",
            ),
            transition_kernel=DefaultTransitionKernel(),
        )

        result = builder.build(top_k=5)

        assert result.active_regimes[1] == ("boosted",)

    def test_memory_aware_branching_uses_retained_history_in_kernel(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)

        def history_kernel(**kwargs):
            current_scenario = kwargs["current_scenario"]
            active_matrix = kwargs["active_matrix"]
            previous_path = kwargs["previous_path"]
            memory_state = kwargs["memory_state"]
            next_state = current_scenario.to_dict()
            next_state["B"] = (
                "High"
                if len(previous_path) >= 2
                and previous_path[0].get_state("A") == "Low"
                else "Low"
            )
            return Scenario(next_state, active_matrix), memory_state, {}

        builder = BranchingPathwayBuilder(
            base_matrix=baseline,
            periods=[1, 2, 3],
            initial={"A": "Low", "B": "Low"},
            max_states_to_enumerate=10_000,
            n_transition_samples=1,
            base_seed=123,
            memory_state=MemoryState(
                period=0,
                values={},
                flags={},
                export_label="memory",
            ),
            transition_kernel=history_kernel,
        )

        result = builder.build(top_k=5)

        assert result.transition_method[1] == "sample"
        assert result.scenarios_by_period[2][0].to_dict()["B"] == "High"
        assert "retained_history_signature" in result.approximation_contract
        assert "history_horizon=full" in result.approximation_contract
        assert len(result.history_signatures_by_period[2][0]) == 3
        assert result.node_records(2)[0]["history_signature"] == result.history_signatures_by_period[2][0]

    def test_memory_aware_branching_uses_retained_history_in_regime_resolution(self) -> None:
        descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)
        baseline.set_impact("A", "Low", "B", "Low", 2.0)
        baseline.set_impact("A", "Low", "B", "High", -2.0)
        baseline.set_impact("A", "High", "B", "Low", 2.0)
        baseline.set_impact("A", "High", "B", "High", -2.0)
        baseline.set_impact("B", "Low", "A", "Low", 2.0)
        baseline.set_impact("B", "Low", "A", "High", -2.0)
        baseline.set_impact("B", "High", "A", "Low", 2.0)
        baseline.set_impact("B", "High", "A", "High", -2.0)

        boosted = CIBMatrix(descriptors)
        boosted.set_impact("A", "Low", "B", "Low", -2.0)
        boosted.set_impact("A", "Low", "B", "High", 2.0)
        boosted.set_impact("A", "High", "B", "Low", -2.0)
        boosted.set_impact("A", "High", "B", "High", 2.0)
        boosted.set_impact("B", "Low", "A", "Low", 2.0)
        boosted.set_impact("B", "Low", "A", "High", -2.0)
        boosted.set_impact("B", "High", "A", "Low", 2.0)
        boosted.set_impact("B", "High", "A", "High", -2.0)

        builder = BranchingPathwayBuilder(
            base_matrix=baseline,
            periods=[1, 2, 3],
            initial={"A": "Low", "B": "Low"},
            regimes=[RegimeSpec(name="boosted", base_matrix=boosted)],
            initial_regime="baseline",
            regime_transition_rule=lambda **kwargs: (
                "boosted"
                if len(kwargs["previous_scenarios"]) >= 2
                and kwargs["previous_scenarios"][0].get_state("A") == "Low"
                else kwargs["current_regime"]
            ),
            max_states_to_enumerate=10_000,
            n_transition_samples=1,
            base_seed=123,
            memory_state=MemoryState(
                period=0,
                values={},
                flags={},
                export_label="memory",
            ),
            transition_kernel=DefaultTransitionKernel(),
        )

        result = builder.build(top_k=5)

        assert result.active_regimes[2] == ("boosted",)
        assert result.scenarios_by_period[2][0].to_dict() == {"A": "Low", "B": "High"}

    def test_branching_records_threshold_regime_reaffirmation_metadata(self) -> None:
        descriptors = {"A": ["Low", "High"]}
        baseline = CIBMatrix(descriptors)

        builder = BranchingPathwayBuilder(
            base_matrix=baseline,
            periods=[2025, 2030],
            initial={"A": "Low"},
            threshold_rules=[
                ThresholdRule(
                    name="StayBaseline",
                    condition=lambda s: s.get_state("A") == "Low",
                    target_regime="baseline",
                )
            ],
            max_states_to_enumerate=10_000,
            n_transition_samples=5,
            base_seed=123,
        )

        result = builder.build(top_k=3)

        assert result.regime_states_by_period[0][0].threshold_regime_reaffirmations == (
            "StayBaseline",
        )
        timelines = branching_regime_residence_timelines(result)
        assert timelines[2025]["threshold_regime_reaffirmations"]["StayBaseline"] == 1.0

