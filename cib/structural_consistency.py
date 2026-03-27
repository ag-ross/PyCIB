"""
Structural consistency diagnostics for path-dependent dynamic CIB.

This module provides check_structural_consistency for evaluating regime,
memory, and path-history consistency independently from local CIB impact-balance
consistency (e.g. lock-in semantics and cycle-phase continuity).
"""

from __future__ import annotations

from typing import Optional, Sequence

from cib.core import Scenario
from cib.pathway import MemoryState, StructuralConsistencyState, TransitionEvent


def check_structural_consistency(
    *,
    period: int,
    realized_scenario: Scenario,
    regime_name: str,
    memory_state: Optional[MemoryState],
    previous_scenarios: Sequence[Scenario],
    transition_events: Sequence[TransitionEvent],
) -> StructuralConsistencyState:
    """
    Structural consistency is evaluated independently from local CIB consistency.

    The default implementation is intentionally conservative: only explicit
    path-dependent flags and event contracts are checked; domain-specific
    structural criteria are left to user-supplied rules layered on top.
    """

    violations = []
    previous_scenario = previous_scenarios[-1] if previous_scenarios else None
    previous_state_dict = (
        previous_scenario.to_dict() if previous_scenario is not None else {}
    )
    realized_state_dict = realized_scenario.to_dict()
    regime_transition_targets = [
        event.metadata.get("to")
        for event in transition_events
        if event.event_type == "regime_transition"
    ]
    if regime_transition_targets:
        latest_target = regime_transition_targets[-1]
        if latest_target is not None and str(latest_target) != str(regime_name):
            violations.append(
                f"latest regime_transition target {latest_target!r} does not match regime {regime_name!r}"
            )

    has_irreversible_event = any(
        event.event_type == "irreversible_transition" for event in transition_events
    )
    has_lock_event = any(
        event.event_type in {"lock_in", "irreversible_transition"}
        for event in transition_events
    )
    if memory_state is not None:
        if int(memory_state.period) != int(period):
            violations.append(
                f"memory_state.period={memory_state.period!r} does not match period {period!r}"
            )
        required_regime = memory_state.values.get("required_regime")
        if required_regime is not None and str(required_regime) != str(regime_name):
            violations.append(
                f"required_regime={required_regime!r} does not match regime {regime_name!r}"
            )
        locked_descriptors = memory_state.values.get("locked_descriptors", {})
        if isinstance(locked_descriptors, dict):
            for descriptor, expected_state in locked_descriptors.items():
                if realized_state_dict.get(str(descriptor)) != str(expected_state):
                    violations.append(
                        f"locked descriptor {descriptor!r} is not in expected state {expected_state!r}"
                    )
        if bool(memory_state.flags.get("locked_in", False)):
            lock_became_binding = False
            if isinstance(locked_descriptors, dict) and locked_descriptors and previous_scenario is not None:
                lock_became_binding = any(
                    previous_state_dict.get(str(descriptor)) != realized_state_dict.get(str(descriptor))
                    for descriptor in locked_descriptors
                )
            if not has_lock_event and (
                (previous_scenario is not None and lock_became_binding)
                or (previous_scenarios and not locked_descriptors)
            ):
                violations.append("locked_in memory flag is set without a corresponding event")
        elif has_irreversible_event:
            violations.append(
                "irreversible_transition event is present but locked_in memory flag is not set"
            )

        cycle_signature = memory_state.values.get("cycle_signature")
        cycle_phase = memory_state.values.get("cycle_phase")
        if cycle_signature is not None or cycle_phase is not None:
            if not isinstance(cycle_signature, (list, tuple)) or cycle_phase is None:
                violations.append("cycle_signature and cycle_phase must be provided together")
            else:
                try:
                    frozen_signature = tuple(
                        tuple(int(v) for v in state) for state in cycle_signature
                    )
                    phase = int(cycle_phase)
                except (TypeError, ValueError):
                    violations.append(
                        "cycle_signature and cycle_phase must contain integer-compatible values"
                    )
                    frozen_signature = tuple()
                    phase = -1
                if phase < 0 or phase >= len(frozen_signature):
                    violations.append("cycle_phase is outside the cycle_signature bounds")
                else:
                    realized_indices = tuple(int(v) for v in realized_scenario.to_indices())
                    if realized_indices != frozen_signature[phase]:
                        violations.append(
                            "realised scenario does not match the memory-reported cycle phase"
                        )
                    if previous_scenario is not None:
                        previous_indices = tuple(
                            int(v) for v in previous_scenario.to_indices()
                        )
                        if previous_indices in frozen_signature:
                            expected_previous = frozen_signature[
                                (phase - 1) % len(frozen_signature)
                            ]
                            if previous_indices != expected_previous:
                                violations.append(
                                    "previous scenario does not match the predecessor implied by cycle_signature"
                                )
    elif has_irreversible_event:
        violations.append(
            "irreversible_transition event is present without an accompanying memory state"
        )

    summary = (
        "No structural consistency violations detected"
        if not violations
        else "; ".join(violations)
    )
    return StructuralConsistencyState(
        period=int(period),
        is_structurally_consistent=(len(violations) == 0),
        violations=tuple(violations),
        summary=summary,
    )
