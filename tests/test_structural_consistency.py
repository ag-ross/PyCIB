"""
Unit tests for structural consistency diagnostics.

These tests validate check_structural_consistency for regime, memory, and
path-history consistency (e.g. required_regime, locked_descriptors, cycle phase).
"""

from __future__ import annotations

from cib.core import CIBMatrix, Scenario
from cib.pathway import MemoryState, TransitionEvent
from cib.structural_consistency import check_structural_consistency


def test_structural_consistency_detects_required_regime_mismatch() -> None:
    matrix = CIBMatrix({"A": ["Low", "High"]})
    scenario = Scenario({"A": "Low"}, matrix)
    memory = MemoryState(
        period=0,
        values={"required_regime": "boosted"},
        flags={},
        export_label="m",
    )

    state = check_structural_consistency(
        period=0,
        realized_scenario=scenario,
        regime_name="baseline",
        memory_state=memory,
        previous_scenarios=(),
        transition_events=(),
    )

    assert state.is_structurally_consistent is False
    assert "required_regime" in state.summary


def test_structural_consistency_accepts_matching_lock_event() -> None:
    matrix = CIBMatrix({"A": ["Low", "High"]})
    scenario = Scenario({"A": "Low"}, matrix)
    memory = MemoryState(
        period=0,
        values={"locked_descriptors": {"A": "Low"}},
        flags={"locked_in": True},
        export_label="m",
    )
    events = (
        TransitionEvent(
            period=0,
            event_type="lock_in",
            label="lock_in",
            source="test",
            metadata={},
        ),
    )

    state = check_structural_consistency(
        period=0,
        realized_scenario=scenario,
        regime_name="baseline",
        memory_state=memory,
        previous_scenarios=(scenario,),
        transition_events=events,
    )

    assert state.is_structurally_consistent is True


def test_structural_consistency_detects_regime_transition_target_mismatch() -> None:
    matrix = CIBMatrix({"A": ["Low", "High"]})
    scenario = Scenario({"A": "Low"}, matrix)
    events = (
        TransitionEvent(
            period=0,
            event_type="regime_transition",
            label="baseline->boosted",
            source="test",
            metadata={"from": "baseline", "to": "boosted"},
        ),
    )

    state = check_structural_consistency(
        period=0,
        realized_scenario=scenario,
        regime_name="baseline",
        memory_state=None,
        previous_scenarios=(),
        transition_events=events,
    )

    assert state.is_structurally_consistent is False
    assert "regime_transition target" in state.summary


def test_structural_consistency_accepts_regime_reaffirmation_without_transition() -> None:
    matrix = CIBMatrix({"A": ["Low", "High"]})
    scenario = Scenario({"A": "Low"}, matrix)
    events = (
        TransitionEvent(
            period=0,
            event_type="threshold_activation",
            label="StayBaseline",
            source="threshold_rule",
            metadata={
                "regime": "baseline",
                "activation_kind": "regime_reaffirmation",
                "threshold_rule": "StayBaseline",
            },
        ),
    )

    state = check_structural_consistency(
        period=0,
        realized_scenario=scenario,
        regime_name="baseline",
        memory_state=None,
        previous_scenarios=(),
        transition_events=events,
    )

    assert state.is_structurally_consistent is True


def test_structural_consistency_detects_irreversible_event_without_lock_flag() -> None:
    matrix = CIBMatrix({"A": ["Low", "High"]})
    scenario = Scenario({"A": "Low"}, matrix)
    memory = MemoryState(period=0, values={}, flags={}, export_label="m")
    events = (
        TransitionEvent(
            period=0,
            event_type="irreversible_transition",
            label="lock",
            source="test",
            metadata={},
        ),
    )

    state = check_structural_consistency(
        period=0,
        realized_scenario=scenario,
        regime_name="baseline",
        memory_state=memory,
        previous_scenarios=(scenario,),
        transition_events=events,
    )

    assert state.is_structurally_consistent is False
    assert "locked_in memory flag" in state.summary


def test_structural_consistency_detects_cycle_phase_mismatch() -> None:
    matrix = CIBMatrix({"A": ["Low", "High"]})
    scenario = Scenario({"A": "Low"}, matrix)
    memory = MemoryState(
        period=0,
        values={"cycle_signature": [(1,), (0,)], "cycle_phase": 0},
        flags={},
        export_label="m",
    )

    state = check_structural_consistency(
        period=0,
        realized_scenario=scenario,
        regime_name="baseline",
        memory_state=memory,
        previous_scenarios=(scenario,),
        transition_events=(),
    )

    assert state.is_structurally_consistent is False
    assert "cycle phase" in state.summary


def test_structural_consistency_accepts_persistent_lock_without_repeated_event() -> None:
    matrix = CIBMatrix({"A": ["Low", "High"]})
    previous = Scenario({"A": "Low"}, matrix)
    scenario = Scenario({"A": "Low"}, matrix)
    memory = MemoryState(
        period=1,
        values={"locked_descriptors": {"A": "Low"}},
        flags={"locked_in": True},
        export_label="m",
    )

    state = check_structural_consistency(
        period=1,
        realized_scenario=scenario,
        regime_name="baseline",
        memory_state=memory,
        previous_scenarios=(previous,),
        transition_events=(),
    )

    assert state.is_structurally_consistent is True


def test_structural_consistency_detects_cycle_predecessor_mismatch() -> None:
    matrix = CIBMatrix({"A": ["Low", "High"]})
    previous = Scenario({"A": "High"}, matrix)
    scenario = Scenario({"A": "High"}, matrix)
    memory = MemoryState(
        period=1,
        values={"cycle_signature": [(1,), (0,)], "cycle_phase": 0},
        flags={},
        export_label="m",
    )

    state = check_structural_consistency(
        period=1,
        realized_scenario=scenario,
        regime_name="baseline",
        memory_state=memory,
        previous_scenarios=(previous,),
        transition_events=(),
    )

    assert state.is_structurally_consistent is False
    assert "predecessor implied by cycle_signature" in state.summary


def test_structural_consistency_handles_malformed_cycle_metadata() -> None:
    matrix = CIBMatrix({"A": ["Low", "High"]})
    scenario = Scenario({"A": "Low"}, matrix)
    memory = MemoryState(
        period=0,
        values={"cycle_signature": [["x"]], "cycle_phase": "bad"},
        flags={},
        export_label="m",
    )

    state = check_structural_consistency(
        period=0,
        realized_scenario=scenario,
        regime_name="baseline",
        memory_state=memory,
        previous_scenarios=(),
        transition_events=(),
    )

    assert state.is_structurally_consistent is False
    assert "integer-compatible values" in state.summary
