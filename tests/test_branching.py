"""
Unit tests for hybrid branching pathway construction.
"""

from __future__ import annotations

from cib.branching import BranchingPathwayBuilder
from cib.core import CIBMatrix, ConsistencyChecker
from cib.cyclic import CyclicDescriptor
from cib.threshold import ThresholdRule


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

    def test_min_edge_weight_pruning_falls_back_when_all_removed(self) -> None:
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
        assert len(out) == 1
        assert abs(sum(out.values()) - 1.0) < 1e-9

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

