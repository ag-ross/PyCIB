"""
Transition kernel interfaces for path-dependent dynamic CIB.

This module provides the TransitionKernel interface for path-dependent step
updates (scenario, memory, metadata) and DefaultTransitionKernel as a
succession-based implementation with cycle and lock-in handling.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

from cib.core import CIBMatrix, Scenario
from cib.pathway import MemoryState, PathDependentState, TransitionEvent
from cib.succession import GlobalSuccession, SuccessionOperator


class TransitionKernel(ABC):
    """
    Base class for path-dependent transition kernels.
    """

    @abstractmethod
    def step(
        self,
        *,
        current_scenario: Scenario,
        active_matrix: CIBMatrix,
        regime: str,
        memory_state: Optional[MemoryState],
        rng: Optional["object"],
        previous_path: Sequence[Scenario],
    ) -> Tuple[Scenario, Optional[MemoryState], Dict[str, Any]]:
        """
        The next realised scenario, updated memory, and metadata are produced.
        """

    @staticmethod
    def memory_states_match(
        expected: Optional[MemoryState], observed: Optional[MemoryState]
    ) -> bool:
        if expected is None and observed is None:
            return True
        if expected is None or observed is None:
            return False
        return (
            int(expected.period) == int(observed.period)
            and dict(expected.values) == dict(observed.values)
            and dict(expected.flags) == dict(observed.flags)
            and str(expected.export_label) == str(observed.export_label)
        )

    @staticmethod
    def _copy_memory_state(memory_state: Optional[MemoryState]) -> Optional[MemoryState]:
        if memory_state is None:
            return None
        return MemoryState(
            period=int(memory_state.period),
            values=copy.deepcopy(memory_state.values),
            flags=copy.deepcopy(memory_state.flags),
            export_label=str(memory_state.export_label),
        )

    def replay_path(
        self,
        *,
        initial_scenario: Scenario,
        active_matrices: Sequence[CIBMatrix],
        active_regimes: Sequence[str],
        initial_memory_state: Optional[MemoryState] = None,
        initial_history: Sequence[Scenario] = (),
        periods: Optional[Sequence[int]] = None,
        expected_scenarios: Optional[Sequence[Scenario]] = None,
        expected_memory_states: Optional[Sequence[MemoryState]] = None,
        seed: Optional[int] = None,
        first_period_output_mode: Literal["attractor", "initial"] = "attractor",
    ) -> "TransitionReplayResult":
        """
        A realised path is replayed under the supplied transition law.
        """

        n_steps = int(len(active_matrices))
        if len(active_regimes) != n_steps:
            raise ValueError("active_regimes must align to active_matrices")
        if periods is not None and len(periods) != n_steps:
            raise ValueError("periods must align to active_matrices")
        if expected_scenarios is not None and len(expected_scenarios) != n_steps:
            raise ValueError("expected_scenarios must align to active_matrices")
        if expected_memory_states is not None and len(expected_memory_states) != n_steps:
            raise ValueError("expected_memory_states must align to active_matrices")
        if first_period_output_mode not in {"attractor", "initial"}:
            raise ValueError("first_period_output_mode must be 'attractor' or 'initial'")

        rng = None
        if seed is not None:
            import numpy as np

            rng = np.random.default_rng(int(seed))

        current_scenario = initial_scenario
        current_memory = self._copy_memory_state(initial_memory_state)
        history = tuple(initial_history)
        records = []

        start_idx = 0
        if first_period_output_mode == "initial" and n_steps > 0:
            initial_period = int(periods[0]) if periods is not None else 0
            initial_memory = (
                MemoryState(
                    period=int(initial_period),
                    values=dict(initial_memory_state.values),
                    flags=dict(initial_memory_state.flags),
                    export_label=str(initial_memory_state.export_label),
                )
                if initial_memory_state is not None
                else None
            )
            expected_scenario = (
                expected_scenarios[0] if expected_scenarios is not None else None
            )
            expected_memory = (
                expected_memory_states[0]
                if expected_memory_states is not None
                else None
            )
            initial_history_signature = tuple(
                tuple(int(v) for v in scenario.to_indices())
                for scenario in history + (current_scenario,)
            )
            records.append(
                TransitionReplayRecord(
                    period=initial_period,
                    state=PathDependentState(
                        period=initial_period,
                        scenario=current_scenario,
                        regime_name=str(active_regimes[0]),
                        active_matrix=active_matrices[0],
                        memory_state=initial_memory,
                        history_signature=initial_history_signature,
                        transition_events=(),
                    ),
                    expected_scenario=expected_scenario,
                    expected_memory_state=expected_memory,
                    scenario_matches=(
                        expected_scenario == current_scenario
                        if expected_scenario is not None
                        else None
                    ),
                    memory_matches=(
                        self.memory_states_match(expected_memory, initial_memory)
                        if expected_memory_states is not None
                        else None
                    ),
                    metadata={"record_kind": "initial_snapshot"},
                )
            )
            next_scenario, next_memory, _metadata = self.step(
                current_scenario=current_scenario,
                active_matrix=active_matrices[0],
                regime=str(active_regimes[0]),
                memory_state=current_memory,
                rng=rng,
                previous_path=history,
            )
            current_scenario = next_scenario
            current_memory = (
                MemoryState(
                    period=(
                        int(periods[0])
                        if periods is not None and next_memory is not None
                        else int(next_memory.period)
                    ),
                    values=dict(next_memory.values),
                    flags=dict(next_memory.flags),
                    export_label=str(next_memory.export_label),
                )
                if next_memory is not None
                else None
            )
            history = history + (current_scenario,)
            start_idx = 1

        for idx in range(start_idx, n_steps):
            next_scenario, next_memory, metadata = self.step(
                current_scenario=current_scenario,
                active_matrix=active_matrices[idx],
                regime=str(active_regimes[idx]),
                memory_state=current_memory,
                rng=rng,
                previous_path=history,
            )
            if next_memory is not None:
                next_memory = MemoryState(
                    period=(
                        int(periods[idx])
                        if periods is not None
                        else int(next_memory.period)
                    ),
                    values=dict(next_memory.values),
                    flags=dict(next_memory.flags),
                    export_label=str(next_memory.export_label),
                )
            expected_scenario = (
                expected_scenarios[idx] if expected_scenarios is not None else None
            )
            expected_memory = (
                expected_memory_states[idx]
                if expected_memory_states is not None
                else None
            )
            history_signature = tuple(
                tuple(int(v) for v in scenario.to_indices())
                for scenario in history + (next_scenario,)
            )
            state = PathDependentState(
                period=(
                    int(periods[idx]) if periods is not None else int(idx)
                ),
                scenario=next_scenario,
                regime_name=str(active_regimes[idx]),
                active_matrix=active_matrices[idx],
                memory_state=next_memory,
                history_signature=history_signature,
                transition_events=tuple(
                    event
                    for event in metadata.get("transition_events", ())
                    if isinstance(event, TransitionEvent)
                ),
            )
            records.append(
                TransitionReplayRecord(
                    period=(
                        int(periods[idx]) if periods is not None else int(idx)
                    ),
                    state=state,
                    expected_scenario=expected_scenario,
                    expected_memory_state=expected_memory,
                    scenario_matches=(
                        expected_scenario == next_scenario
                        if expected_scenario is not None
                        else None
                    ),
                    memory_matches=(
                        self.memory_states_match(expected_memory, next_memory)
                        if expected_memory_states is not None
                        else None
                    ),
                    metadata={
                        str(key): value
                        for key, value in metadata.items()
                        if str(key) != "transition_events"
                    },
                )
            )
            current_scenario = next_scenario
            current_memory = next_memory
            history = history + (next_scenario,)

        return TransitionReplayResult(records=tuple(records))


@dataclass(frozen=True)
class TransitionReplayRecord:
    """
    One replayed transition step and its match diagnostics.
    """

    period: int
    state: PathDependentState
    expected_scenario: Optional[Scenario]
    expected_memory_state: Optional[MemoryState]
    scenario_matches: Optional[bool]
    memory_matches: Optional[bool]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class TransitionReplayResult:
    """
    Replay diagnostics across a path segment.
    """

    records: Tuple[TransitionReplayRecord, ...]

    @property
    def all_scenarios_match(self) -> bool:
        relevant = [
            bool(record.scenario_matches)
            for record in self.records
            if record.scenario_matches is not None
        ]
        return bool(all(relevant)) if relevant else False

    @property
    def all_memory_states_match(self) -> bool:
        relevant = [
            bool(record.memory_matches)
            for record in self.records
            if record.memory_matches is not None
        ]
        return bool(all(relevant)) if relevant else False


@dataclass(frozen=True)
class DefaultTransitionKernel(TransitionKernel):
    """
    Transition kernel that reuses a succession operator as the path update law.
    """

    succession_operator: Optional[SuccessionOperator] = None
    max_iterations: int = 1000
    allow_partial: bool = False

    @staticmethod
    def _copy_memory(memory_state: Optional[MemoryState]) -> MemoryState:
        if memory_state is None:
            return MemoryState(period=0, values={}, flags={}, export_label="memory")
        return MemoryState(
            period=int(memory_state.period),
            values=copy.deepcopy(memory_state.values),
            flags=copy.deepcopy(memory_state.flags),
            export_label=str(memory_state.export_label),
        )

    @staticmethod
    def _cycle_signature(cycle: Sequence[Scenario]) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(member.to_indices()) for member in cycle)

    @staticmethod
    def _scenario_from_state_dict(
        scenario: Scenario, active_matrix: CIBMatrix, state_dict: Dict[str, str]
    ) -> Scenario:
        try:
            return Scenario(state_dict, active_matrix)
        except (ValueError, TypeError):
            fallback_descriptors = {
                descriptor: list(scenario._descriptor_states[descriptor])
                for descriptor in scenario.descriptors
            }
            return Scenario(state_dict, CIBMatrix(fallback_descriptors))

    def step(
        self,
        *,
        current_scenario: Scenario,
        active_matrix: CIBMatrix,
        regime: str,
        memory_state: Optional[MemoryState],
        rng: Optional["object"],
        previous_path: Sequence[Scenario],
    ) -> Tuple[Scenario, Optional[MemoryState], Dict[str, Any]]:
        operator = self.succession_operator or GlobalSuccession()
        result = operator.find_attractor(
            current_scenario,
            active_matrix,
            max_iterations=int(self.max_iterations),
            allow_partial=self.allow_partial,
        )
        updated_memory = self._copy_memory(memory_state)
        transition_events = []
        if result.is_cycle:
            attractor = result.attractor
            if not isinstance(attractor, list):
                raise TypeError("cycle attractor must be a list of scenarios")
            signature = self._cycle_signature(attractor)
            anchor = None
            if previous_path and previous_path[-1] in attractor:
                anchor = previous_path[-1]
            elif current_scenario in attractor:
                anchor = current_scenario
            elif (
                updated_memory.values.get("cycle_signature") == list(signature)
                and "cycle_phase" in updated_memory.values
            ):
                phase = int(updated_memory.values["cycle_phase"])
                next_index = (phase + 1) % len(attractor)
                next_scenario = attractor[next_index]
                updated_memory.values["cycle_phase"] = int(next_index)
                updated_memory.values["cycle_signature"] = list(signature)
                metadata = {
                    "iterations": int(result.iterations),
                    "is_cycle": bool(result.is_cycle),
                    "regime": str(regime),
                    "used_history": True,
                }
                return next_scenario, updated_memory, metadata
            if anchor is not None:
                anchor_index = attractor.index(anchor)
                next_index = (anchor_index + 1) % len(attractor)
                next_scenario = attractor[next_index]
                updated_memory.values["cycle_phase"] = int(next_index)
                updated_memory.values["cycle_signature"] = list(signature)
            else:
                next_scenario = attractor[0]
                updated_memory.values["cycle_phase"] = 0
                updated_memory.values["cycle_signature"] = list(signature)
        else:
            attractor = result.attractor
            if not isinstance(attractor, Scenario):
                raise TypeError("fixed-point attractor must be a Scenario")
            next_scenario = attractor
            updated_memory.values.pop("cycle_phase", None)
            updated_memory.values.pop("cycle_signature", None)

        locked_descriptors = updated_memory.values.get("locked_descriptors", {})
        if isinstance(locked_descriptors, dict) and locked_descriptors:
            state_dict = next_scenario.to_dict()
            applied_locks = []
            for descriptor, expected_state in locked_descriptors.items():
                descriptor_name = str(descriptor)
                locked_state = str(expected_state)
                if descriptor_name in state_dict and state_dict[descriptor_name] != locked_state:
                    state_dict[descriptor_name] = locked_state
                    applied_locks.append(descriptor_name)
            if applied_locks:
                next_scenario = self._scenario_from_state_dict(
                    next_scenario, active_matrix, state_dict
                )
                event_type = (
                    "irreversible_transition"
                    if bool(updated_memory.flags.get("locked_in", False))
                    else "memory_update"
                )
                transition_events.append(
                    TransitionEvent(
                        period=int(updated_memory.period),
                        event_type=event_type,
                        label="locked_descriptors_applied",
                        source="transition_kernel",
                        metadata={"descriptors": list(applied_locks)},
                    )
                )
        metadata = {
            "iterations": int(result.iterations),
            "is_cycle": bool(result.is_cycle),
            "regime": str(regime),
            "used_history": bool(result.is_cycle),
        }
        if transition_events:
            metadata["transition_events"] = transition_events
        if (
            memory_state is None
            and not updated_memory.values
            and not updated_memory.flags
            and str(updated_memory.export_label) == "memory"
        ):
            return next_scenario, None, metadata
        return next_scenario, updated_memory, metadata


@dataclass(frozen=True)
class CallableTransitionKernel(TransitionKernel):
    """
    Adapter that wraps a plain callable as a transition kernel.
    """

    fn: Any

    def step(
        self,
        *,
        current_scenario: Scenario,
        active_matrix: CIBMatrix,
        regime: str,
        memory_state: Optional[MemoryState],
        rng: Optional["object"],
        previous_path: Sequence[Scenario],
    ) -> Tuple[Scenario, Optional[MemoryState], Dict[str, Any]]:
        return self.fn(
            current_scenario=current_scenario,
            active_matrix=active_matrix,
            regime=regime,
            memory_state=memory_state,
            rng=rng,
            previous_path=previous_path,
        )
