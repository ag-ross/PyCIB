from __future__ import annotations

import pytest

from cib.cyclic import CyclicDescriptor


def test_cyclic_descriptor_rejects_transition_to_undefined_state() -> None:
    descriptor = CyclicDescriptor(
        name="Cycle",
        transition={
            "A": {"A": 0.5, "B": 0.5},
        },
    )

    with pytest.raises(ValueError, match="references unknown next state"):
        descriptor.validate()


def test_cyclic_descriptor_rejects_type_mismatched_next_state_key() -> None:
    descriptor = CyclicDescriptor(
        name="Cycle",
        transition={
            "1": {1: 1.0},
        },
    )

    with pytest.raises(ValueError, match="references unknown next state"):
        descriptor.validate()
