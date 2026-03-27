"""
Fast numeric scoring for CIB matrices.

In this module, a dense (v1) scoring backend is provided, where impact-score
computations are accelerated by converting a `CIBMatrix` into a numeric tensor
representation.

The backend is intended as an internal optimisation. The public API of
`CIBMatrix` and `ConsistencyChecker` is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from cib.core import CIBMatrix, Scenario


@dataclass(frozen=True)
class FastCIBScorer:
    """
    Dense numeric scorer for a `CIBMatrix`.

    Notes:
    - A padded dense tensor is used: shape (n_desc, max_states, n_desc, max_states).
    - States beyond a descriptor's arity are treated as padding and must be ignored.
    """

    descriptors: Sequence[str]
    state_labels: Sequence[Sequence[str]]
    state_counts: np.ndarray
    max_states: int
    impact: np.ndarray
    descriptor_index: Mapping[str, int]
    state_index: Sequence[Mapping[str, int]]

    @staticmethod
    def from_matrix(
        matrix: CIBMatrix,
        *,
        max_workspace_bytes: Optional[int] = None,
    ) -> "FastCIBScorer":
        descriptors = list(matrix.descriptors.keys())
        state_labels: List[List[str]] = [list(matrix.descriptors[d]) for d in descriptors]
        state_counts = np.asarray([len(s) for s in state_labels], dtype=np.int64)
        max_states = int(state_counts.max()) if len(state_counts) else 0
        n_desc = len(descriptors)

        if max_workspace_bytes is not None and n_desc > 0 and max_states > 0:
            est = n_desc * max_states * n_desc * max_states * 8
            if int(est) > int(max_workspace_bytes):
                raise MemoryError(
                    "FastCIBScorer dense impact tensor would exceed max_workspace_bytes: "
                    f"estimated {est} bytes for shape "
                    f"({n_desc}, {max_states}, {n_desc}, {max_states}) float64 "
                    f"(limit {int(max_workspace_bytes)}). "
                    "Use a sparse backend, reduce descriptors/states, or raise the limit."
                )

        desc_to_idx: Dict[str, int] = {d: i for i, d in enumerate(descriptors)}
        state_to_idx: List[Dict[str, int]] = []
        for states in state_labels:
            state_to_idx.append({s: i for i, s in enumerate(states)})

        impact = np.zeros(
            (len(descriptors), max_states, len(descriptors), max_states),
            dtype=np.float64,
        )

        # Impacts are stored sparsely in the matrix and are copied into the dense tensor.
        for (src_desc, src_state, tgt_desc, tgt_state), v in matrix.iter_impacts():
            i = desc_to_idx[src_desc]
            j = desc_to_idx[tgt_desc]
            k = state_to_idx[i][src_state]
            l = state_to_idx[j][tgt_state]
            impact[i, k, j, l] = float(v)

        return FastCIBScorer(
            descriptors=descriptors,
            state_labels=state_labels,
            state_counts=state_counts,
            max_states=max_states,
            impact=impact,
            descriptor_index=desc_to_idx,
            state_index=state_to_idx,
        )

    def scenario_to_indices(self, scenario: Scenario) -> np.ndarray:
        """
        Convert a `Scenario` to a numeric index vector.
        """
        return np.asarray(scenario.to_indices(), dtype=np.int64)

    def scores_for_scenario(self, z_idx: np.ndarray) -> np.ndarray:
        """
        Compute impact scores for all descriptors and candidate states.

        Args:
            z_idx: Scenario state indices, shape (n_desc,).

        Returns:
            A score matrix of shape (n_desc, max_states). Only the first
            `state_counts[j]` entries are meaningful for each descriptor j.
        """
        z = np.asarray(z_idx, dtype=np.int64)
        n_desc = int(len(self.descriptors))
        if z.shape != (n_desc,):
            raise ValueError(
                f"z_idx must have shape ({n_desc},), but shape {tuple(z.shape)} was provided"
            )
        for j in range(n_desc):
            idx = int(z[j])
            max_idx = int(self.state_counts[j]) - 1
            if idx < 0 or idx > max_idx:
                raise ValueError(
                    f"State index {idx} out of range [0, {max_idx}] for descriptor "
                    f"'{self.descriptors[j]}'"
                )

        contrib = self.impact[np.arange(n_desc), z, :, :]
        scores = contrib.sum(axis=0, dtype=np.float64)
        return scores

    def apply_change(
        self,
        scores: np.ndarray,
        *,
        changed_descriptor_idx: int,
        old_state_idx: int,
        new_state_idx: int,
    ) -> np.ndarray:
        """
        An incremental score update is applied for a single descriptor state change.

        The score matrix is modified in place and returned for convenience.
        """
        j = int(changed_descriptor_idx)
        old = int(old_state_idx)
        new = int(new_state_idx)
        n_desc = int(len(self.descriptors))
        if j < 0 or j >= n_desc:
            raise ValueError(
                f"changed_descriptor_idx {j} out of range [0, {n_desc - 1}]"
            )
        max_idx = int(self.state_counts[j]) - 1
        if old < 0 or old > max_idx:
            raise ValueError(
                f"old_state_idx {old} out of range [0, {max_idx}] for descriptor "
                f"'{self.descriptors[j]}'"
            )
        if new < 0 or new > max_idx:
            raise ValueError(
                f"new_state_idx {new} out of range [0, {max_idx}] for descriptor "
                f"'{self.descriptors[j]}'"
            )
        scores += self.impact[j, new, :, :] - self.impact[j, old, :, :]
        return scores

    def is_consistent(
        self,
        z_idx: np.ndarray,
        *,
        float_atol: float = 1e-08,
        float_rtol: float = 1e-05,
    ) -> bool:
        """
        CIB consistency of a scenario is checked using the fast scoring backend.

        Ties are handled using `np.isclose` with configurable tolerances so that
        behaviour can be matched to the reference slow path.
        """
        scores = self.scores_for_scenario(z_idx)
        z = np.asarray(z_idx, dtype=np.int64)
        for j in range(int(len(self.descriptors))):
            n_states = int(self.state_counts[j])
            chosen_idx = int(z[j])
            chosen = float(scores[j, chosen_idx])
            best = float(scores[j, :n_states].max(initial=float("-inf")))
            if best > chosen and not np.isclose(
                best, chosen, atol=float(float_atol), rtol=float(float_rtol)
            ):
                return False
        return True

    def global_successor_indices(self, z_idx: np.ndarray) -> np.ndarray:
        """
        Compute the global succession successor in index space.
        """
        scores = self.scores_for_scenario(z_idx)
        n_desc = int(len(self.descriptors))
        nxt = np.empty((n_desc,), dtype=np.int64)
        for j in range(n_desc):
            n_states = int(self.state_counts[j])
            nxt[j] = int(np.argmax(scores[j, :n_states]))
        return nxt

    def local_successor_indices(self, z_idx: np.ndarray) -> np.ndarray:
        """
        Compute the local succession successor in index space.
        """
        scores = self.scores_for_scenario(z_idx)
        z = np.asarray(z_idx, dtype=np.int64)
        n_desc = int(len(self.descriptors))

        target = None
        best_gap = float("-inf")
        best_state = None
        for j in range(n_desc):
            n_states = int(self.state_counts[j])
            chosen_idx = int(z[j])
            chosen = float(scores[j, chosen_idx])
            best_idx = int(np.argmax(scores[j, :n_states]))
            best = float(scores[j, best_idx])
            gap = best - chosen
            if gap > best_gap:
                best_gap = gap
                target = j
                best_state = best_idx

        if target is None or best_state is None or best_gap <= 0.0:
            return np.asarray(z, dtype=np.int64).copy()

        out = np.asarray(z, dtype=np.int64).copy()
        out[int(target)] = int(best_state)
        return out

