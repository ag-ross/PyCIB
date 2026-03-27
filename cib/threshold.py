"""
Threshold descriptor logic for dynamic (multi-period) CIB.

Threshold rules allow switching the active CIM when a condition on the current
scenario is met (e.g., policy regime change).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from cib.core import CIBMatrix, Scenario


MatrixModifier = Callable[[CIBMatrix], CIBMatrix]
ScenarioPredicate = Callable[[Scenario], bool]


@dataclass(frozen=True)
class ThresholdRule:
    """
    A threshold rule that conditionally modifies the active CIM or switches regime.
    """

    name: str
    condition: ScenarioPredicate
    modifier: Optional[MatrixModifier] = None
    target_regime: Optional[str] = None
    activation_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_modifier = self.modifier is not None
        has_target_regime = self.target_regime is not None
        if has_modifier == has_target_regime:
            raise ValueError(
                "ThresholdRule must define exactly one of modifier or target_regime"
            )

    @property
    def activation_kind(self) -> str:
        return "regime_transition" if self.target_regime is not None else "modifier"


def apply_modifier_copy_on_write(
    modifier: MatrixModifier,
    matrix: CIBMatrix,
) -> Tuple[CIBMatrix, bool]:
    """
    Apply a matrix modifier against a defensive clone.

    Returns:
        A tuple of (result_matrix, returned_distinct_object), where the boolean
        indicates whether the modifier returned a new object distinct from its
        clone input.
    """

    cloned_matrix = copy.deepcopy(matrix)
    modified_matrix = modifier(cloned_matrix)
    if not isinstance(modified_matrix, CIBMatrix):
        raise TypeError("Threshold modifier must return a CIBMatrix instance")
    return modified_matrix, bool(modified_matrix is not cloned_matrix)

