"""
Unit tests for path dependence helpers.

These tests validate AdaptiveCIMUpdater, HysteresisRule, and IrreversibilityRule
for history-dependent matrix updates and memory flags.
"""

from __future__ import annotations

from cib.core import CIBMatrix, Scenario
from cib.path_dependence import (
    CallableAdaptiveCIMUpdater,
    HysteresisRule,
    IrreversibilityRule,
)
from cib.pathway import MemoryState, TransitionEvent


def test_callable_adaptive_cim_updater_returns_labels_and_events() -> None:
    descriptors = {"A": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)
    updater = CallableAdaptiveCIMUpdater(
        lambda **kwargs: (
            kwargs["active_matrix"],
            ("history_adjustment",),
            (
                TransitionEvent(
                    period=0,
                    event_type="adaptive_matrix_update",
                    label="history_adjustment",
                    source="test",
                    metadata={},
                ),
            ),
        )
    )

    updated, labels, events = updater.update(
        active_matrix=matrix,
        current_regime="baseline",
        realized_scenario=Scenario({"A": "Low"}, matrix),
        previous_scenarios=(),
        memory_state=None,
    )

    assert updated is matrix
    assert labels == ("history_adjustment",)
    assert events[0].event_type == "adaptive_matrix_update"


def test_hysteresis_rule_sets_memory_flag() -> None:
    rule = HysteresisRule(name="h", trigger_key="pressure", activation_threshold=2.0)
    memory = MemoryState(period=0, values={"pressure": 3.0}, flags={}, export_label="m")

    updated = rule.apply(memory)

    assert updated.flags["hysteresis_active"] is True


def test_irreversibility_rule_latches_memory_flag() -> None:
    rule = IrreversibilityRule(name="irrev", trigger_flag="crossed")
    memory = MemoryState(
        period=0,
        values={},
        flags={"crossed": True},
        export_label="m",
    )

    updated = rule.apply(memory)

    assert updated.flags["locked_in"] is True
