"""
Unit tests for path-dependent transition kernels.

These tests validate TransitionKernel interface and DefaultTransitionKernel
(succession, cycle, and lock-in handling).
"""

from __future__ import annotations

import pytest

from cib.core import CIBMatrix, Scenario
from cib.pathway import MemoryState
from cib.transition_kernel import CallableTransitionKernel, DefaultTransitionKernel


def _simple_matrix() -> CIBMatrix:
    desc = {"A": ["Low", "High"], "B": ["Low", "High"]}
    m = CIBMatrix(desc)
    m.set_impact("A", "Low", "B", "Low", 2.0)
    m.set_impact("A", "Low", "B", "High", -2.0)
    m.set_impact("A", "High", "B", "Low", -2.0)
    m.set_impact("A", "High", "B", "High", 2.0)
    m.set_impact("B", "Low", "A", "Low", 2.0)
    m.set_impact("B", "Low", "A", "High", -2.0)
    m.set_impact("B", "High", "A", "Low", -2.0)
    m.set_impact("B", "High", "A", "High", 2.0)
    return m


def test_default_transition_kernel_uses_succession_operator() -> None:
    m = _simple_matrix()
    scenario = Scenario({"A": "Low", "B": "Low"}, m)
    kernel = DefaultTransitionKernel()

    nxt, memory, metadata = kernel.step(
        current_scenario=scenario,
        active_matrix=m,
        regime="baseline",
        memory_state=None,
        rng=None,
        previous_path=(),
    )

    assert nxt.to_dict() == {"A": "Low", "B": "Low"}
    assert memory is None
    assert metadata["is_cycle"] is False


def test_callable_transition_kernel_can_update_memory() -> None:
    m = _simple_matrix()
    scenario = Scenario({"A": "Low", "B": "Low"}, m)
    kernel = CallableTransitionKernel(
        lambda **kwargs: (
            kwargs["current_scenario"],
            MemoryState(period=0, values={"seen": 1}, flags={"active": True}, export_label="m"),
            {"custom": True},
        )
    )

    nxt, memory, metadata = kernel.step(
        current_scenario=scenario,
        active_matrix=m,
        regime="baseline",
        memory_state=None,
        rng=None,
        previous_path=(),
    )

    assert nxt == scenario
    assert memory is not None
    assert memory.values["seen"] == 1
    assert metadata["custom"] is True


def test_default_transition_kernel_uses_history_to_progress_cycle() -> None:
    m = _simple_matrix()
    scenario = Scenario({"A": "Low", "B": "High"}, m)
    previous = Scenario({"A": "High", "B": "Low"}, m)
    kernel = DefaultTransitionKernel()

    nxt_without_history, memory_without_history, metadata_without_history = kernel.step(
        current_scenario=scenario,
        active_matrix=m,
        regime="baseline",
        memory_state=None,
        rng=None,
        previous_path=(),
    )
    nxt_with_history, memory_with_history, metadata_with_history = kernel.step(
        current_scenario=scenario,
        active_matrix=m,
        regime="baseline",
        memory_state=None,
        rng=None,
        previous_path=(previous,),
    )

    assert nxt_without_history.to_dict() == {"A": "High", "B": "Low"}
    assert nxt_with_history.to_dict() == {"A": "Low", "B": "High"}
    assert memory_without_history is not None
    assert memory_with_history is not None
    assert metadata_without_history["used_history"] is True
    assert metadata_with_history["used_history"] is True


def test_default_transition_kernel_applies_locked_descriptors() -> None:
    m = _simple_matrix()
    scenario = Scenario({"A": "Low", "B": "Low"}, m)
    kernel = DefaultTransitionKernel()

    nxt, memory, metadata = kernel.step(
        current_scenario=scenario,
        active_matrix=m,
        regime="baseline",
        memory_state=MemoryState(
            period=0,
            values={"locked_descriptors": {"B": "High"}},
            flags={"locked_in": True},
            export_label="m",
        ),
        rng=None,
        previous_path=(),
    )

    assert nxt.to_dict()["B"] == "High"
    assert memory is not None
    assert metadata["transition_events"][0].event_type == "irreversible_transition"


def test_default_transition_kernel_can_replay_a_path() -> None:
    m = _simple_matrix()
    kernel = DefaultTransitionKernel()

    replay = kernel.replay_path(
        initial_scenario=Scenario({"A": "Low", "B": "Low"}, m),
        active_matrices=(m, m),
        active_regimes=("baseline", "baseline"),
        periods=(0, 1),
        expected_scenarios=(
            Scenario({"A": "Low", "B": "Low"}, m),
            Scenario({"A": "Low", "B": "Low"}, m),
        ),
        expected_memory_states=(
            None,
            None,
        ),
    )

    assert len(replay.records) == 2
    assert replay.all_scenarios_match is True
    assert replay.all_memory_states_match is True
    assert replay.records[0].state.history_signature[-1] == (0, 0)


def test_default_transition_kernel_replay_preserves_supplied_period_labels() -> None:
    m = _simple_matrix()
    kernel = DefaultTransitionKernel()

    replay = kernel.replay_path(
        initial_scenario=Scenario({"A": "Low", "B": "Low"}, m),
        active_matrices=(m,),
        active_regimes=("baseline",),
        periods=(2025,),
        expected_scenarios=(Scenario({"A": "Low", "B": "Low"}, m),),
        expected_memory_states=(None,),
    )

    assert replay.records[0].period == 2025
    assert replay.records[0].state.period == 2025


def test_default_transition_kernel_replay_can_record_initial_snapshot() -> None:
    m = _simple_matrix()
    kernel = DefaultTransitionKernel()
    initial_memory = MemoryState(period=0, values={"phase": 0}, flags={}, export_label="m")

    replay = kernel.replay_path(
        initial_scenario=Scenario({"A": "Low", "B": "Low"}, m),
        active_matrices=(m, m),
        active_regimes=("baseline", "baseline"),
        initial_memory_state=initial_memory,
        periods=(2025, 2030),
        expected_scenarios=(
            Scenario({"A": "Low", "B": "Low"}, m),
            Scenario({"A": "Low", "B": "Low"}, m),
        ),
        expected_memory_states=(
            MemoryState(period=2025, values={"phase": 0}, flags={}, export_label="m"),
            None,
        ),
        first_period_output_mode="initial",
    )

    assert replay.records[0].metadata["record_kind"] == "initial_snapshot"
    assert replay.records[0].scenario_matches is True
    assert replay.records[0].memory_matches is True


def test_transition_kernel_replay_does_not_alias_initial_memory_state() -> None:
    m = _simple_matrix()
    kernel = CallableTransitionKernel(
        lambda **kwargs: (
            kwargs["current_scenario"],
            (
                kwargs["memory_state"]
                if kwargs["memory_state"] is None
                else (
                    kwargs["memory_state"].values["phase"].__setitem__("value", 42)
                    or kwargs["memory_state"]
                )
            ),
            {},
        )
    )
    initial_memory = MemoryState(
        period=0,
        values={"phase": {"value": 0}},
        flags={},
        export_label="m",
    )

    _ = kernel.replay_path(
        initial_scenario=Scenario({"A": "Low", "B": "Low"}, m),
        active_matrices=(m,),
        active_regimes=("baseline",),
        initial_memory_state=initial_memory,
        periods=(2025,),
        expected_scenarios=(Scenario({"A": "Low", "B": "Low"}, m),),
        expected_memory_states=(
            MemoryState(
                period=2025,
                values={"phase": {"value": 42}},
                flags={},
                export_label="m",
            ),
        ),
    )

    assert initial_memory.values["phase"]["value"] == 0


def test_scenario_from_state_dict_raises_for_incompatible_active_matrix() -> None:
    matrix_original = _simple_matrix()
    matrix_incompatible = CIBMatrix({"A": ["L", "H"], "B": ["X", "Y"]})
    scenario = Scenario({"A": "Low", "B": "High"}, matrix_original)

    with pytest.raises(ValueError, match="Invalid state"):
        _ = DefaultTransitionKernel._scenario_from_state_dict(
            scenario,
            matrix_incompatible,
            {"A": "Low", "B": "High"},
        )


def test_default_transition_kernel_rejects_invalid_locked_state_for_active_matrix() -> None:
    m = _simple_matrix()
    scenario = Scenario({"A": "Low", "B": "Low"}, m)
    kernel = DefaultTransitionKernel()

    with pytest.raises(ValueError, match="Invalid locked state"):
        _ = kernel.step(
            current_scenario=scenario,
            active_matrix=m,
            regime="baseline",
            memory_state=MemoryState(
                period=0,
                values={"locked_descriptors": {"B": "NotAState"}},
                flags={},
                export_label="m",
            ),
            rng=None,
            previous_path=(),
        )
