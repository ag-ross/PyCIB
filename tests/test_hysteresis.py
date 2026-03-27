"""
Unit tests for hysteresis helper behaviour.

These tests validate HysteresisRule activation and release thresholds and
flag behaviour.
"""

from __future__ import annotations

from cib.path_dependence import HysteresisRule
from cib.pathway import MemoryState


def test_hysteresis_rule_releases_flag_below_release_threshold() -> None:
    rule = HysteresisRule(
        name="h",
        trigger_key="pressure",
        activation_threshold=2.0,
        release_threshold=1.0,
    )
    memory = MemoryState(
        period=0,
        values={"pressure": 0.5},
        flags={"hysteresis_active": True},
        export_label="m",
    )

    updated = rule.apply(memory)

    assert updated.flags["hysteresis_active"] is False
