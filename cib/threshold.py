"""
Threshold descriptor logic for dynamic (multi-period) CIB.

Threshold rules allow switching the active CIM when a condition on the current
scenario is met (e.g., policy regime change).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cib.core import CIBMatrix, Scenario


MatrixModifier = Callable[[CIBMatrix], CIBMatrix]
ScenarioPredicate = Callable[[Scenario], bool]


@dataclass(frozen=True)
class ThresholdRule:
    """
    A threshold rule that conditionally modifies the active CIM.
    """

    name: str
    condition: ScenarioPredicate
    modifier: MatrixModifier

