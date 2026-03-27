"""
Unit tests for irreversibility helper behaviour.

These tests validate IrreversibilityRule latch preservation and trigger-flag
behaviour.
"""

from __future__ import annotations

from cib.path_dependence import IrreversibilityRule
from cib.pathway import MemoryState


def test_irreversibility_rule_preserves_existing_latch() -> None:
    rule = IrreversibilityRule(name="irrev", trigger_flag="crossed")
    memory = MemoryState(
        period=0,
        values={},
        flags={"crossed": False, "locked_in": True},
        export_label="m",
    )

    updated = rule.apply(memory)

    assert updated.flags["locked_in"] is True
