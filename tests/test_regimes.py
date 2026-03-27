"""
Unit tests for regime definitions and transition rules.

These tests validate RegimeSpec resolution with modifiers and
CallableRegimeTransitionRule behaviour.
"""

from __future__ import annotations

from cib.core import CIBMatrix, Scenario
from cib.pathway import TransitionEvent
from cib.regimes import CallableRegimeTransitionRule, RegimeSpec


def test_regime_spec_resolves_matrix_with_modifiers() -> None:
    descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)

    def modifier(base: CIBMatrix) -> CIBMatrix:
        out = CIBMatrix(base.descriptors)
        out.set_impacts(dict(base.iter_impacts()))
        out.set_impact("A", "High", "B", "Low", -2.0)
        out.set_impact("A", "High", "B", "High", 2.0)
        return out

    regime = RegimeSpec(name="boosted", modifiers=(modifier,))
    resolved = regime.resolve_matrix(matrix)

    assert resolved.get_impact("A", "High", "B", "High") == 2.0


def test_regime_spec_modifier_inplace_mutation_is_isolated() -> None:
    descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)
    matrix.set_impact("A", "High", "B", "Low", 0.0)
    matrix.set_impact("A", "High", "B", "High", 0.0)

    def mutating_modifier(base: CIBMatrix) -> CIBMatrix:
        base.set_impact("A", "High", "B", "Low", -2.0)
        base.set_impact("A", "High", "B", "High", 2.0)
        return base

    regime = RegimeSpec(name="boosted", modifiers=(mutating_modifier,))
    resolved = regime.resolve_matrix(matrix)

    assert resolved.get_impact("A", "High", "B", "High") == 2.0
    assert matrix.get_impact("A", "High", "B", "High") == 0.0


def test_callable_regime_transition_rule_returns_events() -> None:
    descriptors = {"A": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)
    scenario = Scenario({"A": "High"}, matrix)
    rule = CallableRegimeTransitionRule(
        lambda **kwargs: (
            "boosted",
            (
                TransitionEvent(
                    period=0,
                    event_type="regime_transition",
                    label="baseline->boosted",
                    source="test",
                    metadata={},
                ),
            ),
        )
    )

    next_regime, events = rule.resolve_next_regime(
        current_regime="baseline",
        realized_scenario=scenario,
        previous_scenarios=(scenario,),
        memory_state=None,
        rng=None,
    )

    assert next_regime == "boosted"
    assert events[0].event_type == "regime_transition"
