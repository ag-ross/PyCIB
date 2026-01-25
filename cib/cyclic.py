"""
Cyclic descriptor logic for dynamic (multi-period) CIB.

Cyclic descriptors have explicit transition dynamics between periods, modeled as
a transition probability matrix over discrete states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class CyclicDescriptor:
    """
    A descriptor with a discrete transition matrix between periods.

    The transition matrix is represented as:
      transition[current_state][next_state] = probability
    with probabilities summing to 1 for each current_state.
    """

    name: str
    transition: Mapping[str, Mapping[str, float]]

    def validate(self) -> None:
        if not self.transition:
            raise ValueError("transition matrix cannot be empty")
        for cur, row in self.transition.items():
            if not row:
                raise ValueError(f"transition row for {cur!r} cannot be empty")
            s = sum(float(p) for p in row.values())
            if not np.isclose(s, 1.0):
                raise ValueError(f"transition probabilities for {cur!r} must sum to 1.0")
            for nxt, p in row.items():
                if float(p) < 0:
                    raise ValueError(
                        f"transition probability must be non-negative: {cur!r}->{nxt!r}"
                    )

    def sample_next(self, current_state: str, rng: np.random.Generator) -> str:
        if current_state not in self.transition:
            raise ValueError(f"Unknown current_state {current_state!r} for {self.name!r}")
        row: Dict[str, float] = {k: float(v) for k, v in self.transition[current_state].items()}
        states = list(row.keys())
        probs = np.array([row[s] for s in states], dtype=float)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(states), p=probs))
        return states[idx]

