"""
Example: Path-dependent disequilibrium with explicit lock-in memory

The following is demonstrated:
- Extension mode "path_dependent" with MemoryState and DefaultTransitionKernel
- Lock-in memory (locked_descriptors, locked_in flag) and required_regime
- Transition events and memory states recorded per period
- Structural consistency diagnostics alongside the pathway
- An irreversible descriptor lock being carried through the realised path
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cib import CIBMatrix, DynamicCIB, DefaultTransitionKernel, MemoryState


def build_matrix() -> CIBMatrix:
    descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)
    matrix.set_impact("B", "Low", "A", "Low", 2.0)
    matrix.set_impact("B", "Low", "A", "High", -2.0)
    matrix.set_impact("B", "High", "A", "Low", 2.0)
    matrix.set_impact("B", "High", "A", "High", -2.0)
    return matrix


if __name__ == "__main__":
    dyn = DynamicCIB(build_matrix(), periods=[2025, 2030, 2035])
    path = dyn.simulate_path_extended(
        initial={"A": "Low", "B": "Low"},
        extension_mode="path_dependent",
        initial_regime="baseline",
        memory_state=MemoryState(
            period=0,
            values={"required_regime": "baseline", "locked_descriptors": {"B": "High"}},
            flags={"locked_in": True},
            export_label="path_memory",
        ),
        transition_kernel=DefaultTransitionKernel(),
    )
    print("realised", [scenario.to_dict() for scenario in path.realised_scenarios])
    print("memory", [memory.values | memory.flags for memory in path.memory_states])
    print(
        "locked_descriptor_path",
        [scenario.to_dict()["B"] for scenario in path.realised_scenarios],
    )
    print(
        "events",
        [(event.event_type, event.label) for event in path.transition_events],
    )
    print("structural", [state.summary for state in path.structural_consistency])
