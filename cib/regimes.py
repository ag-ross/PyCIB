"""
Regime definitions and transition rules for dynamic CIB extensions.

This module provides RegimeSpec for named regime definitions (with optional
matrix modifiers) and the RegimeTransitionRule interface for resolving the
active regime from scenario and path history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from cib.core import CIBMatrix, Scenario
from cib.pathway import TransitionEvent
from cib.threshold import apply_modifier_copy_on_write


@dataclass(frozen=True)
class RegimeSpec:
    """
    Named regime definition for regime-aware dynamic runs.
    """

    name: str
    base_matrix: Optional[CIBMatrix] = None
    matrix_ref: Optional[str] = None
    modifiers: Tuple[Any, ...] = ()
    activation_metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def resolve_matrix(self, fallback_matrix: CIBMatrix) -> CIBMatrix:
        """
        The regime matrix is materialised from a base matrix plus modifiers.
        """

        active = self.base_matrix or fallback_matrix
        for modifier in self.modifiers:
            active, _ = apply_modifier_copy_on_write(modifier, active)
        return active


class RegimeTransitionRule(ABC):
    """
    Base class for regime transition rules.
    """

    @abstractmethod
    def resolve_next_regime(
        self,
        *,
        current_regime: str,
        realized_scenario: Scenario,
        previous_scenarios: Sequence[Scenario],
        memory_state: Optional["object"] = None,
        rng: Optional["object"] = None,
    ) -> Tuple[str, Tuple[TransitionEvent, ...]]:
        """
        The next regime is resolved and transition events are returned.
        """


@dataclass(frozen=True)
class CallableRegimeTransitionRule(RegimeTransitionRule):
    """
    Adapter that wraps a plain callable as a regime transition rule.
    """

    fn: Any

    def resolve_next_regime(
        self,
        *,
        current_regime: str,
        realized_scenario: Scenario,
        previous_scenarios: Sequence[Scenario],
        memory_state: Optional["object"] = None,
        rng: Optional["object"] = None,
    ) -> Tuple[str, Tuple[TransitionEvent, ...]]:
        result = self.fn(
            current_regime=current_regime,
            realized_scenario=realized_scenario,
            previous_scenarios=previous_scenarios,
            memory_state=memory_state,
            rng=rng,
        )
        if isinstance(result, tuple) and len(result) == 2:
            regime_name, events = result
            return str(regime_name), tuple(events)
        return str(result), ()
