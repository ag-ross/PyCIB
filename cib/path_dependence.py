"""
Path-dependent state and adaptive matrix machinery for dynamic CIB.

This module provides the AdaptiveCIMUpdater interface for history-dependent
matrix updates, and helper rules (HysteresisRule, IrreversibilityRule) for
path-dependent memory flags used in Capability 3 workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from cib.core import CIBMatrix, Scenario
from cib.pathway import MemoryState, TransitionEvent


class AdaptiveCIMUpdater(ABC):
    """
    Base class for adaptive matrix update rules.
    """

    @abstractmethod
    def update(
        self,
        *,
        active_matrix: CIBMatrix,
        current_regime: str,
        realized_scenario: Scenario,
        previous_scenarios: Sequence[Scenario],
        memory_state: Optional[MemoryState],
    ) -> Tuple[CIBMatrix, Tuple[str, ...], Tuple[TransitionEvent, ...]]:
        """
        An updated matrix plus provenance labels and events are returned.
        """


@dataclass(frozen=True)
class CallableAdaptiveCIMUpdater(AdaptiveCIMUpdater):
    """
    Adapter that wraps a plain callable as an adaptive matrix updater.
    """

    fn: Any

    def update(
        self,
        *,
        active_matrix: CIBMatrix,
        current_regime: str,
        realized_scenario: Scenario,
        previous_scenarios: Sequence[Scenario],
        memory_state: Optional[MemoryState],
    ) -> Tuple[CIBMatrix, Tuple[str, ...], Tuple[TransitionEvent, ...]]:
        result = self.fn(
            active_matrix=active_matrix,
            current_regime=current_regime,
            realized_scenario=realized_scenario,
            previous_scenarios=previous_scenarios,
            memory_state=memory_state,
        )
        if isinstance(result, tuple) and len(result) == 3:
            updated, labels, events = result
            return updated, tuple(labels), tuple(events)
        return result, (), ()


@dataclass(frozen=True)
class HysteresisRule:
    """
    Minimal hysteresis rule expressed as memory flag thresholds.
    """

    name: str
    trigger_key: str
    activation_threshold: float
    release_threshold: Optional[float] = None
    memory_flag: str = "hysteresis_active"

    def apply(self, memory_state: Optional[MemoryState]) -> MemoryState:
        values = dict(memory_state.values if memory_state is not None else {})
        flags = dict(memory_state.flags if memory_state is not None else {})
        level = float(values.get(self.trigger_key, 0.0))
        release = (
            float(self.release_threshold)
            if self.release_threshold is not None
            else float(self.activation_threshold)
        )
        if level >= float(self.activation_threshold):
            flags[self.memory_flag] = True
        elif level <= release:
            flags[self.memory_flag] = False
        period = int(memory_state.period) if memory_state is not None else 0
        label = memory_state.export_label if memory_state is not None else "memory"
        return MemoryState(period=period, values=values, flags=flags, export_label=label)


@dataclass(frozen=True)
class IrreversibilityRule:
    """
    Minimal irreversibility rule that latches a named memory flag once triggered.
    """

    name: str
    trigger_flag: str
    latched_flag: str = "locked_in"

    def apply(self, memory_state: Optional[MemoryState]) -> MemoryState:
        values = dict(memory_state.values if memory_state is not None else {})
        flags = dict(memory_state.flags if memory_state is not None else {})
        if bool(flags.get(self.trigger_flag, False)):
            flags[self.latched_flag] = True
        period = int(memory_state.period) if memory_state is not None else 0
        label = memory_state.export_label if memory_state is not None else "memory"
        return MemoryState(period=period, values=values, flags=flags, export_label=label)
