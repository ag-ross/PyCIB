"""
Exact enumeration with pruning (scaling Mode A).

In this module, a depth-first search over partial scenarios is provided with an
optional sound upper-bound pruning rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cib.constraints import ConstraintIndex
from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.fast_scoring import FastCIBScorer
from cib.solvers.config import ExactSolverConfig


@dataclass(frozen=True)
class ExactSolverResult:
    """
    Results of exact consistent-scenario enumeration.
    """

    scenarios: Tuple[Scenario, ...]
    status: str
    is_complete: bool
    diagnostics: Dict[str, Any]
    runtime_s: float


def _descriptor_order(matrix: CIBMatrix, *, ordering: str, seed: int) -> List[int]:
    n = int(len(matrix.descriptors))
    if ordering == "given":
        return list(range(n))
    if ordering == "random":
        rng = np.random.default_rng(int(seed))
        order = list(range(n))
        rng.shuffle(order)
        return order
    if ordering == "connectivity":
        names = list(matrix.descriptors.keys())
        counts = {d: 0 for d in names}
        for (src_desc, _src_state, tgt_desc, _tgt_state), v in matrix.iter_impacts():
            if float(v) == 0.0:
                continue
            counts[str(src_desc)] = int(counts.get(str(src_desc), 0)) + 1
            counts[str(tgt_desc)] = int(counts.get(str(tgt_desc), 0)) + 1
        ranked = sorted(
            range(n),
            key=lambda i: (int(counts.get(names[i], 0)), -i),
            reverse=True,
        )
        return ranked
    raise ValueError("ordering must be 'given', 'random', or 'connectivity'")


def _scenario_space_size(matrix: CIBMatrix) -> int:
    space_size = 1
    for n in matrix.state_counts:
        space_size *= int(n)
    return int(space_size)


def _bruteforce_consistent(
    matrix: CIBMatrix,
    *,
    cidx: Optional[ConstraintIndex],
    time_limit_s: Optional[float],
    max_solutions: Optional[int],
    float_atol: float,
    float_rtol: float,
) -> Tuple[List[Scenario], str, bool, Dict[str, Any]]:
    """
    A slow brute-force fallback is provided.
    """
    desc_names = list(matrix.descriptors.keys())
    state_lists = [matrix.descriptors[d] for d in desc_names]
    out: List[Scenario] = []
    started_at = perf_counter()
    checked = 0
    constraint_pruned = 0
    status = "ok"
    is_complete = True
    for comb in product(*state_lists):
        if time_limit_s is not None and (perf_counter() - started_at) >= float(time_limit_s):
            status = "timeout"
            is_complete = False
            break
        if max_solutions is not None and len(out) >= int(max_solutions):
            status = "max_solutions"
            is_complete = False
            break
        sdict = dict(zip(desc_names, comb))
        s = Scenario(sdict, matrix)
        checked += 1
        z = np.array(s.to_indices(), dtype=np.int64)
        if cidx is not None and not bool(cidx.is_full_valid(z)):
            constraint_pruned += 1
            continue
        if ConsistencyChecker.check_consistency(
            s,
            matrix,
            float_atol=float(float_atol),
            float_rtol=float(float_rtol),
        ):
            out.append(s)
    diagnostics = {
        "nodes_visited": int(checked),
        "constraint_pruned_nodes": int(constraint_pruned),
    }
    return out, str(status), bool(is_complete), diagnostics


def find_all_consistent_exact(
    *,
    matrix: CIBMatrix,
    config: ExactSolverConfig,
) -> ExactSolverResult:
    """
    Enumerate consistent scenarios using a pruned search (or bruteforce fallback).

    Always check :attr:`ExactSolverResult.is_complete` and ``status``: timeouts
    and ``max_solutions`` caps can return **partial** results. Bruteforce fallback
    for large scenario spaces requires ``ExactSolverConfig.allow_bruteforce=True``.
    """
    config.validate()
    t0 = perf_counter()

    fast_scorer_fallback = False
    fast_scorer_fallback_reason: Optional[str] = None
    fast_scorer_fallback_exception_type: Optional[str] = None
    try:
        scorer = (
            FastCIBScorer.from_matrix(
                matrix, max_workspace_bytes=int(config.max_delta_array_bytes)
            )
            if bool(config.use_fast_scoring)
            else None
        )
    except Exception as exc:
        if bool(config.strict_fast):
            raise
        if not isinstance(exc, (MemoryError, ValueError, TypeError, OSError)):
            raise
        fast_scorer_fallback = True
        fast_scorer_fallback_reason = repr(exc)
        fast_scorer_fallback_exception_type = str(type(exc).__name__)
        warnings.warn(
            "Exact solver fast scoring initialization failed; falling back to slow path. "
            f"reason={type(exc).__name__}: {exc}",
            UserWarning,
            stacklevel=2,
        )
        scorer = None

    cidx = ConstraintIndex.from_specs(matrix, config.constraints)

    if scorer is None:
        total_space = _scenario_space_size(matrix)
        if (
            total_space > int(config.bruteforce_max_scenarios)
            and not bool(config.allow_bruteforce)
        ):
            if not bool(config.use_fast_scoring):
                raise ValueError(
                    f"Exact solver would enumerate {total_space} scenarios via bruteforce "
                    f"(use_fast_scoring=False), which exceeds "
                    f"bruteforce_max_scenarios={config.bruteforce_max_scenarios}. "
                    "Set ExactSolverConfig.allow_bruteforce=True to proceed (unsafe for large "
                    "spaces), or reduce the problem / raise bruteforce_max_scenarios."
                )
            raise ValueError(
                f"Exact solver would enumerate {total_space} scenarios via bruteforce after "
                f"fast scorer failure, which exceeds "
                f"bruteforce_max_scenarios={config.bruteforce_max_scenarios}. "
                "Set ExactSolverConfig.allow_bruteforce=True to proceed (unsafe for large "
                "spaces), fix the fast-scorer error, or use strict_fast=True to surface it."
            )
        max_solutions = int(config.max_solutions) if config.max_solutions is not None else None
        time_limit_s = float(config.time_limit_s) if config.time_limit_s is not None else None
        bf_solutions, bf_status, bf_is_complete, bf_metrics = _bruteforce_consistent(
            matrix,
            cidx=cidx,
            time_limit_s=time_limit_s,
            max_solutions=max_solutions,
            float_atol=float(config.float_atol),
            float_rtol=float(config.float_rtol),
        )
        scenarios = tuple(bf_solutions)
        runtime_s = float(perf_counter() - t0)
        bf_diag: Dict[str, Any] = {
            "fallback": "bruteforce",
            "fast_scorer_fallback": bool(fast_scorer_fallback),
            "intentional_slow_scoring_path": not bool(config.use_fast_scoring),
            "fallback_stage": "fast_scorer_initialization",
            "fallback_from": "fast_scorer",
            "fallback_to": "bruteforce",
            "float_atol": float(config.float_atol),
            "float_rtol": float(config.float_rtol),
        }
        bf_diag.update(bf_metrics)
        if fast_scorer_fallback_reason is not None:
            bf_diag["fast_scorer_fallback_reason"] = str(fast_scorer_fallback_reason)
        if fast_scorer_fallback_exception_type is not None:
            bf_diag["fallback_exception_type"] = str(fast_scorer_fallback_exception_type)
        return ExactSolverResult(
            scenarios=scenarios,
            status=str(bf_status),
            is_complete=bool(bf_is_complete),
            diagnostics=bf_diag,
            runtime_s=float(runtime_s),
        )

    n_desc = int(len(scorer.descriptors))
    order = _descriptor_order(matrix, ordering=str(config.ordering), seed=int(config.seed))

    max_states = int(scorer.max_states)
    impact = scorer.impact

    delta_nbytes = (
        int(n_desc)
        * int(n_desc)
        * int(max_states)
        * int(max_states)
        * int(np.dtype(np.float64).itemsize)
    )
    if delta_nbytes > int(config.max_delta_array_bytes):
        raise ValueError(
            f"delta_max workspace would require ~{delta_nbytes} bytes "
            f"(limit max_delta_array_bytes={config.max_delta_array_bytes}). "
            "Reduce descriptor count or max descriptor cardinality, or raise the limit explicitly."
        )

    # Precompute per-unassigned-descriptor maximum advantage contributions:
    # delta_max[u, j, c, l] = max_k impact[u, k, j, c] - impact[u, k, j, l]
    delta_max = np.zeros((n_desc, n_desc, max_states, max_states), dtype=np.float64)
    for u in range(n_desc):
        n_u = int(scorer.state_counts[u])
        if n_u <= 0:
            continue
        for j in range(n_desc):
            n_j = int(scorer.state_counts[j])
            if n_j <= 0:
                continue
            block = impact[u, :n_u, j, :n_j]  # shape: (n_u, n_j)
            # Shape: (n_u, n_j, n_j); then maximum over k is taken.
            diff = block[:, :, None] - block[:, None, :]
            delta_max[u, j, :n_j, :n_j] = diff.max(axis=0)

    remaining_delta = delta_max.sum(axis=0, dtype=np.float64)
    scores = np.zeros((n_desc, max_states), dtype=np.float64)
    z = np.full((n_desc,), -1, dtype=np.int64)

    solutions: List[Scenario] = []
    nodes_visited = 0
    pruned_nodes = 0
    constraint_pruned_nodes = 0
    status = "ok"
    is_complete = True

    max_solutions = int(config.max_solutions) if config.max_solutions is not None else None
    time_limit_s = float(config.time_limit_s) if config.time_limit_s is not None else None

    def timed_out() -> bool:
        return time_limit_s is not None and (perf_counter() - t0) >= float(time_limit_s)

    def should_prune() -> bool:
        if str(config.bound) == "none":
            return False
        for j in range(n_desc):
            if int(z[j]) < 0:
                continue
            n_j = int(scorer.state_counts[j])
            chosen_idx = int(z[j])
            for l in range(n_j):
                if int(l) == int(chosen_idx):
                    continue
                # Upper bound on the best achievable margin (chosen - alternative).
                margin_ub = float(scores[j, chosen_idx]) - float(scores[j, l])
                margin_ub += float(remaining_delta[j, chosen_idx, l])
                if margin_ub < 0.0 and not np.isclose(
                    float(margin_ub), 0.0, atol=float(config.float_atol), rtol=float(config.float_rtol)
                ):
                    return True
        return False

    def dfs(pos: int) -> None:
        nonlocal nodes_visited, pruned_nodes, status, is_complete
        nonlocal constraint_pruned_nodes
        if timed_out():
            status = "timeout"
            is_complete = False
            return
        if max_solutions is not None and len(solutions) >= int(max_solutions):
            status = "max_solutions"
            is_complete = False
            return

        nodes_visited += 1

        if should_prune():
            pruned_nodes += 1
            return

        if pos >= len(order):
            # Full assignment reached; exact consistency is checked.
            if cidx is not None and not bool(cidx.is_full_valid(z)):
                constraint_pruned_nodes += 1
                return
            if scorer.is_consistent(
                z, float_atol=float(config.float_atol), float_rtol=float(config.float_rtol)
            ):
                solutions.append(Scenario(list(int(x) for x in z), matrix))
            return

        d = int(order[pos])
        n_states = int(scorer.state_counts[d])
        prev_scores = scores.copy()
        prev_remaining = remaining_delta.copy()

        for s_idx in range(n_states):
            z[d] = int(s_idx)
            if cidx is not None and not bool(cidx.is_partial_valid(z, just_set=int(d))):
                constraint_pruned_nodes += 1
                continue
            scores[:, :] = prev_scores
            remaining_delta[:, :, :] = prev_remaining

            remaining_delta[:, :, :] = remaining_delta - delta_max[d, :, :, :]
            scores[:, :] = scores + impact[d, int(s_idx), :, :]

            dfs(pos + 1)
            if not is_complete:
                return

        z[d] = -1

    dfs(0)

    runtime_s = float(perf_counter() - t0)
    diagnostics: Dict[str, Any] = {
        "nodes_visited": int(nodes_visited),
        "pruned_nodes": int(pruned_nodes),
        "constraint_pruned_nodes": int(constraint_pruned_nodes),
        "ordering": str(config.ordering),
        "bound": str(config.bound),
        "fast_scorer_fallback": bool(fast_scorer_fallback),
        "intentional_slow_scoring_path": False,
    }
    if fast_scorer_fallback_reason is not None:
        diagnostics["fast_scorer_fallback_reason"] = str(fast_scorer_fallback_reason)
    if fast_scorer_fallback_exception_type is not None:
        diagnostics["fallback_exception_type"] = str(fast_scorer_fallback_exception_type)
        diagnostics["fallback_stage"] = "fast_scorer_initialization"
        diagnostics["fallback_from"] = "fast_scorer"
        diagnostics["fallback_to"] = "pruned_exact_search"
    return ExactSolverResult(
        scenarios=tuple(solutions),
        status=str(status),
        is_complete=bool(is_complete),
        diagnostics=diagnostics,
        runtime_s=float(runtime_s),
    )

