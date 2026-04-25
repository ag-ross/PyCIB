"""
Monte Carlo attractor discovery solver (scaling Mode B).

In this solver, attractor weights are estimated by repeatedly sampling initial
scenarios and running succession to a fixed point or a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from time import perf_counter
import warnings
from typing import Any, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

from cib.core import CIBMatrix, Scenario
from cib.fast_scoring import FastCIBScorer
from cib.sparse_scoring import SparseCIBScorer
from cib.fast_succession import run_to_attractor_indices
from cib.solvers.config import MonteCarloAttractorConfig


@dataclass(frozen=True)
class AttractorKey:
    """
    Hashable attractor identifier.

    The `kind` field is used to disambiguate fixed points from cycle keys.
    """

    kind: Literal["fixed", "cycle"]
    value: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]


@dataclass(frozen=True)
class MonteCarloAttractorResult:
    """
    Results of Monte Carlo attractor discovery.

    ``weights`` maps each attractor key to ``counts[key] / n_completed`` where
    ``n_completed`` excludes runs that timed out (succession hit
    ``max_iterations``). Diagnostics expose both completed-run and
    requested-run normalization views so conditioning is explicit.
    """

    counts: Dict[AttractorKey, int]
    weights: Dict[AttractorKey, float]
    attractor_keys_ranked: Tuple[AttractorKey, ...]
    top_attractors: Tuple[Scenario, ...]
    cycles: Optional[Dict[AttractorKey, Tuple[Tuple[int, ...], ...]]]
    diagnostics: Dict[str, Any]
    status: str
    runtime_s: float

    def __repr__(self) -> str:
        d = self.diagnostics
        return (
            f"MonteCarloAttractorResult(status={self.status!r}, "
            f"n_completed_runs={d.get('n_completed_runs')!r}, "
            f"n_timeouts={d.get('n_timeouts')!r}, "
            f"fast_scorer_fallback={d.get('fast_scorer_fallback')!r}, "
            f"intentional_slow_scoring_path={d.get('intentional_slow_scoring_path')!r}, "
            f"runtime_s={float(self.runtime_s):.4f})"
        )


class _SamplerBackend(Protocol):
    """
    A sampler backend interface is defined for initial-state sampling.
    """

    descriptors: Sequence[str]
    state_counts: np.ndarray
    state_index: Sequence[Mapping[str, int]]


def _chunk_seeds(
    run_seeds: Sequence[np.random.SeedSequence], *, n_chunks: int
) -> Tuple[Tuple[np.random.SeedSequence, ...], ...]:
    if int(n_chunks) <= 0:
        raise ValueError("n_chunks must be positive")
    chunks: List[List[np.random.SeedSequence]] = [[] for _ in range(int(n_chunks))]
    for i, ss in enumerate(run_seeds):
        chunks[int(i) % int(n_chunks)].append(ss)
    return tuple(tuple(c) for c in chunks)


def _run_batch_worker(
    args: Tuple[CIBMatrix, MonteCarloAttractorConfig, Tuple[np.random.SeedSequence, ...]]
) -> Tuple[
    Dict["AttractorKey", int],
    Optional[Dict["AttractorKey", Tuple[Tuple[int, ...], ...]]],
    Dict[str, Any],
]:
    matrix, config, run_seeds = args
    return _run_batch(matrix=matrix, config=config, run_seeds=run_seeds)


def _canonicalise_cycle_key(
    cycle: Sequence[Tuple[int, ...]],
    *,
    policy: str,
) -> AttractorKey:
    if not cycle:
        raise ValueError("cycle cannot be empty")

    if policy == "min_state":
        return AttractorKey(kind="cycle", value=min(tuple(s) for s in cycle))

    if policy == "rotate_min":
        c = [tuple(s) for s in cycle]
        rots = []
        for i in range(len(c)):
            rots.append(tuple(c[i:] + c[:i]))
        return AttractorKey(kind="cycle", value=min(rots))

    raise ValueError("cycle_key_policy must be 'min_state' or 'rotate_min'")


def _canonicalise_cycle_storage(
    cycle: Sequence[Tuple[int, ...]],
    *,
    policy: str,
) -> Tuple[Tuple[int, ...], ...]:
    """
    A canonical cycle representation is returned for storage.

    Note: for "min_state", the cycle is rotated so that the lexicographically
    smallest state is first. For "rotate_min", the canonical rotated cycle is
    returned (matching the key representation).
    """
    c = [tuple(s) for s in cycle]
    if not c:
        raise ValueError("cycle cannot be empty")

    if policy == "min_state":
        m = min(c)
        i0 = c.index(m)
        return tuple(c[i0:] + c[:i0])
    if policy == "rotate_min":
        rots = []
        for i in range(len(c)):
            rots.append(tuple(c[i:] + c[:i]))
        return min(rots)
    raise ValueError("cycle_key_policy must be 'min_state' or 'rotate_min'")


def _cycle_key_from_mode(
    rng: np.random.Generator,
    cycle: Sequence[Tuple[int, ...]],
    *,
    cycle_mode: str,
    cycle_key_policy: str,
) -> AttractorKey:
    """
    Cycle key selection is performed according to cycle_mode.
    """
    c = [tuple(s) for s in cycle]
    if not c:
        raise ValueError("cycle cannot be empty")

    if cycle_mode == "keep_cycle":
        return _canonicalise_cycle_key(c, policy=str(cycle_key_policy))
    if cycle_mode == "representative_first":
        return AttractorKey(kind="cycle", value=tuple(c[0]))
    if cycle_mode == "representative_random":
        idx = int(rng.integers(0, len(c)))
        return AttractorKey(kind="cycle", value=tuple(c[idx]))
    raise ValueError("cycle_mode is not recognised")


def _key_sort_token(key: AttractorKey) -> Tuple[int, Tuple]:
    """
    A deterministic tie-break token is returned for attractor keys.
    """
    kind_rank = 0 if key.kind == "fixed" else 1
    v = key.value
    if isinstance(v, tuple) and (not v or isinstance(v[0], int)):
        return (kind_rank, ("state", tuple(v)))
    return (kind_rank, ("cycle", tuple(v)))  # type: ignore[arg-type]


def _make_weighted_sampler(
    *,
    scorer: _SamplerBackend,
    sampler_weights: Mapping[str, Mapping[str, float]],
) -> Tuple[np.ndarray, ...]:
    """
    Per-descriptor categorical probabilities are prepared.
    """
    probs: List[np.ndarray] = []
    for j, d in enumerate(scorer.descriptors):
        n_states = int(scorer.state_counts[j])
        w_map = sampler_weights.get(str(d), {})
        w = np.zeros((n_states,), dtype=np.float64)
        for state_label, idx in scorer.state_index[j].items():
            if int(idx) >= n_states:
                continue
            w[int(idx)] = float(w_map.get(str(state_label), 0.0))
        if float(w.sum()) <= 0.0:
            raise ValueError(
                f"Non-zero sampler weights were required for descriptor {str(d)!r}"
            )
        p = w / float(w.sum())
        probs.append(p)
    return tuple(probs)


def _sample_initial_state(
    rng: np.random.Generator,
    *,
    scorer: _SamplerBackend,
    sampler: str,
    weighted_probs: Optional[Sequence[np.ndarray]],
) -> np.ndarray:
    n_desc = int(len(scorer.descriptors))
    z = np.empty((n_desc,), dtype=np.int64)
    if sampler == "uniform":
        for j in range(n_desc):
            n_states = int(scorer.state_counts[j])
            z[j] = int(rng.integers(0, n_states))
        return z

    if sampler == "weighted":
        if weighted_probs is None:
            raise ValueError("weighted_probs must be provided for weighted sampling")
        for j in range(n_desc):
            p = weighted_probs[j]
            z[j] = int(rng.choice(np.arange(len(p), dtype=np.int64), p=p))
        return z

    raise ValueError("sampler must be 'uniform' or 'weighted'")


def _run_batch(
    *,
    matrix: CIBMatrix,
    config: MonteCarloAttractorConfig,
    run_seeds: Sequence[np.random.SeedSequence],
) -> Tuple[
    Dict[AttractorKey, int],
    Optional[Dict[AttractorKey, Tuple[Tuple[int, ...], ...]]],
    Dict[str, Any],
]:
    scorer: Optional[Union[FastCIBScorer, SparseCIBScorer]]
    fast_scorer_fallback = False
    fast_scorer_fallback_reason: Optional[str] = None
    fast_scorer_fallback_exception_type: Optional[str] = None
    if bool(config.use_fast_scoring):
        try:
            if str(config.fast_backend) == "sparse":
                scorer = SparseCIBScorer.from_matrix(matrix)
            else:
                scorer = FastCIBScorer.from_matrix(
                    matrix,
                    max_workspace_bytes=config.max_fast_scorer_workspace_bytes,
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
                "Monte Carlo fast scoring initialization failed; falling back to slow path. "
                f"reason={type(exc).__name__}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            scorer = None
    else:
        scorer = None

    weighted_probs = None
    if scorer is not None and config.sampler == "weighted":
        weighted_probs = _make_weighted_sampler(
            scorer=scorer,
            sampler_weights=config.sampler_weights or {},
        )

    counts: Dict[AttractorKey, int] = {}
    cycles: Optional[Dict[AttractorKey, Tuple[Tuple[int, ...], ...]]] = (
        {} if config.cycle_mode == "keep_cycle" else None
    )

    n_timeouts = 0
    n_cycles = 0
    total_iters = 0

    for ss in run_seeds:
        rng = np.random.Generator(np.random.PCG64(ss))

        if scorer is None:
            # Slow path: Scenario objects are used directly.
            state_dict: Dict[str, str] = {}
            desc_names = list(matrix.descriptors.keys())
            for d in desc_names:
                states = matrix.descriptors[d]
                if config.sampler == "uniform":
                    state_dict[d] = states[int(rng.integers(0, len(states)))]
                else:
                    w_map = (config.sampler_weights or {}).get(str(d), {})
                    w = np.asarray([float(w_map.get(str(s), 0.0)) for s in states], dtype=np.float64)
                    if float(w.sum()) <= 0.0:
                        raise ValueError(
                            f"Non-zero sampler weights were required for descriptor {str(d)!r}"
                        )
                    p = w / float(w.sum())
                    state_dict[d] = str(rng.choice(np.asarray(states, dtype=object), p=p))
            initial = Scenario(state_dict, matrix)

            if str(config.succession) == "global":
                from cib.succession import GlobalSuccession

                op = GlobalSuccession(
                    float_atol=float(config.float_atol),
                    float_rtol=float(config.float_rtol),
                )
            elif str(config.succession) == "local":
                from cib.succession import LocalSuccession

                op = LocalSuccession(
                    float_atol=float(config.float_atol),
                    float_rtol=float(config.float_rtol),
                )
            else:
                raise ValueError("succession must be 'global' or 'local'")

            try:
                res_slow = op.find_attractor(
                    initial, matrix, max_iterations=int(config.max_iterations)
                )
            except RuntimeError:
                n_timeouts += 1
                continue

            total_iters += int(res_slow.iterations)

            if res_slow.is_cycle:
                n_cycles += 1
                cyc = res_slow.attractor
                if not isinstance(cyc, list):
                    raise TypeError("cycle attractor must be a list of scenarios")
                cycle_idx = [tuple(s.to_indices()) for s in cyc]
                cycle_key = _cycle_key_from_mode(
                    rng,
                    cycle_idx,
                    cycle_mode=str(config.cycle_mode),
                    cycle_key_policy=str(config.cycle_key_policy),
                )
                counts[cycle_key] = int(counts.get(cycle_key, 0)) + 1
                if cycles is not None:
                    cycles.setdefault(
                        cycle_key,
                        _canonicalise_cycle_storage(
                            cycle_idx, policy=str(config.cycle_key_policy)
                        ),
                    )
                continue

            attractor = res_slow.attractor
            if not isinstance(attractor, Scenario):
                raise TypeError("fixed-point attractor must be a Scenario")
            fixed_key = AttractorKey(kind="fixed", value=tuple(attractor.to_indices()))
            counts[fixed_key] = int(counts.get(fixed_key, 0)) + 1
            continue

        z0 = _sample_initial_state(
            rng,
            scorer=scorer,
            sampler=str(config.sampler),
            weighted_probs=weighted_probs,
        )
        try:
            res = run_to_attractor_indices(
                scorer=scorer,
                initial_z_idx=z0,
                rule=str(config.succession),
                max_iterations=int(config.max_iterations),
                float_atol=float(config.float_atol),
                float_rtol=float(config.float_rtol),
            )
        except RuntimeError:
            n_timeouts += 1
            continue

        total_iters += int(res.iterations)

        if res.is_cycle:
            n_cycles += 1
            cycle = res.attractor
            if not isinstance(cycle, tuple):
                raise TypeError("cycle attractor indices must be a tuple")
            cycle_key = _cycle_key_from_mode(
                rng,
                cycle,
                cycle_mode=str(config.cycle_mode),
                cycle_key_policy=str(config.cycle_key_policy),
            )
            counts[cycle_key] = int(counts.get(cycle_key, 0)) + 1
            if cycles is not None:
                cycles.setdefault(
                    cycle_key,
                    _canonicalise_cycle_storage(
                        cycle, policy=str(config.cycle_key_policy)
                    ),
                )
            continue

        attractor = res.attractor
        if not isinstance(attractor, tuple):
            raise TypeError("fixed-point attractor indices must be a tuple")
        fixed_key = AttractorKey(kind="fixed", value=tuple(attractor))
        counts[fixed_key] = int(counts.get(fixed_key, 0)) + 1

    diag: Dict[str, Any] = {
        "n_completed_runs": int(len(run_seeds)) - int(n_timeouts),
        "n_timeouts": int(n_timeouts),
        "n_cycles": int(n_cycles),
        "mean_iterations": float(total_iters / max(1, (len(run_seeds) - n_timeouts))),
        "fast_scorer_fallback": bool(fast_scorer_fallback),
        "intentional_slow_scoring_path": not bool(config.use_fast_scoring),
        "float_atol": float(config.float_atol),
        "float_rtol": float(config.float_rtol),
    }
    if fast_scorer_fallback_reason is not None:
        diag["fast_scorer_fallback_reason"] = str(fast_scorer_fallback_reason)
    if fast_scorer_fallback_exception_type is not None:
        diag["fallback_exception_type"] = str(fast_scorer_fallback_exception_type)
        diag["fallback_stage"] = "fast_scorer_initialization"
        diag["fallback_from"] = "fast_scorer"
        diag["fallback_to"] = "slow_succession_path"
    return counts, cycles, diag


def _merge_counts(
    parts: Sequence[Dict[AttractorKey, int]]
) -> Dict[AttractorKey, int]:
    out: Dict[AttractorKey, int] = {}
    for d in parts:
        for k, v in d.items():
            out[k] = int(out.get(k, 0)) + int(v)
    return out


def _merge_cycles(
    parts: Sequence[Optional[Dict[AttractorKey, Tuple[Tuple[int, ...], ...]]]]
) -> Optional[Dict[AttractorKey, Tuple[Tuple[int, ...], ...]]]:
    if not parts:
        return None
    if all(p is None for p in parts):
        return None
    out: Dict[AttractorKey, Tuple[Tuple[int, ...], ...]] = {}
    for p in parts:
        if p is None:
            continue
        for k, cyc in p.items():
            out.setdefault(k, cyc)
    return out


def _merge_diagnostics(parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not parts:
        return {}
    n_completed = int(sum(int(p.get("n_completed_runs", 0)) for p in parts))
    n_timeouts = int(sum(int(p.get("n_timeouts", 0)) for p in parts))
    n_cycles = int(sum(int(p.get("n_cycles", 0)) for p in parts))
    total_iters_weighted = 0.0
    for p in parts:
        n = int(p.get("n_completed_runs", 0))
        mean = float(p.get("mean_iterations", 0.0))
        total_iters_weighted += float(n) * float(mean)
    mean_iters = float(total_iters_weighted / max(1, n_completed))
    fast_fb = any(bool(p.get("fast_scorer_fallback")) for p in parts)
    float_atol = float(parts[0].get("float_atol", 0.0))
    float_rtol = float(parts[0].get("float_rtol", 0.0))
    for p in parts[1:]:
        if not np.isclose(float(p.get("float_atol", float_atol)), float_atol):
            raise RuntimeError("Inconsistent float_atol values across Monte Carlo workers")
        if not np.isclose(float(p.get("float_rtol", float_rtol)), float_rtol):
            raise RuntimeError("Inconsistent float_rtol values across Monte Carlo workers")
    out: Dict[str, Any] = {
        "n_completed_runs": int(n_completed),
        "n_timeouts": int(n_timeouts),
        "n_cycles": int(n_cycles),
        "mean_iterations": float(mean_iters),
        "fast_scorer_fallback": bool(fast_fb),
        "intentional_slow_scoring_path": bool(
            parts and parts[0].get("intentional_slow_scoring_path")
        ),
        "float_atol": float(float_atol),
        "float_rtol": float(float_rtol),
    }
    for p in parts:
        r = p.get("fast_scorer_fallback_reason")
        if r:
            out["fast_scorer_fallback_reason"] = str(r)
            break
    return out


def find_attractors_monte_carlo(
    *,
    matrix: CIBMatrix,
    config: MonteCarloAttractorConfig,
) -> MonteCarloAttractorResult:
    """
    Monte Carlo attractor discovery is performed.

    Reported ``weights`` are normalized over **completed** runs only; timed-out
    runs (see ``diagnostics['n_timeouts']``) are excluded from that denominator.
    Diagnostics also include requested-run-normalized frequencies. Configure
    ``fail_on_timeout`` and ``min_completion_fraction`` for stricter completion
    quality requirements.
    """
    config.validate()
    t0 = perf_counter()

    root = np.random.SeedSequence(int(config.seed))

    run_seeds = root.spawn(int(config.runs))
    if int(config.n_jobs) == 1:
        counts, cycles, diag = _run_batch(
            matrix=matrix, config=config, run_seeds=run_seeds
        )
    else:
        n_jobs = int(config.n_jobs)
        chunks = _chunk_seeds(run_seeds, n_chunks=n_jobs)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            results = pool.map(
                _run_batch_worker, [(matrix, config, c) for c in chunks]
            )
        counts_parts = [r[0] for r in results]
        cycles_parts = [r[1] for r in results]
        diag_parts = [r[2] for r in results]
        counts = _merge_counts(counts_parts)
        cycles = _merge_cycles(cycles_parts)
        diag = _merge_diagnostics(diag_parts)

    runtime_s = float(perf_counter() - t0)
    n_runs_cfg = int(config.runs)
    n_completed = int(diag.get("n_completed_runs", 0))
    n_timeouts = int(diag.get("n_timeouts", 0))
    if bool(config.fail_on_timeout) and n_timeouts > 0:
        raise RuntimeError(
            "Monte Carlo attractor discovery: fail_on_timeout is True but "
            f"n_timeouts={n_timeouts} (completed={n_completed}, runs={n_runs_cfg}). "
            "Increase max_iterations or runs, or disable fail_on_timeout."
        )
    if config.min_completion_fraction is not None:
        frac = float(n_completed) / float(max(1, n_runs_cfg))
        need = float(config.min_completion_fraction)
        if frac + 1e-15 < need:
            raise RuntimeError(
                "Monte Carlo attractor discovery: completion fraction "
                f"{frac:.6g} is below min_completion_fraction={need:.6g} "
                f"(completed={n_completed}, n_timeouts={n_timeouts}, runs={n_runs_cfg})."
            )

    n_eff = n_completed
    weights = {k: float(v) / float(max(1, n_eff)) for k, v in counts.items()}
    weights_over_requested_runs = {
        k: float(v) / float(max(1, n_runs_cfg)) for k, v in counts.items()
    }
    completion_fraction = float(n_completed) / float(max(1, n_runs_cfg))
    ranked = tuple(
        sorted(
            counts.keys(),
            key=lambda k: (-int(counts[k]), _key_sort_token(k)),
        )
    )

    top_attractors: List[Scenario] = []
    if config.result_storage == "topk_scenarios":
        for key in ranked[: int(config.top_k)]:
            if key.kind != "fixed":
                continue
            v = key.value
            if not isinstance(v, tuple):
                raise TypeError("fixed attractor key value must be a tuple of indices")
            top_attractors.append(Scenario(list(v), matrix))

    completion_target = float(config.completion_status_target_fraction)
    if n_completed <= 0:
        status = "incomplete"
    elif completion_fraction + 1e-15 < completion_target:
        status = "partial_timeout" if counts else "incomplete"
    else:
        status = "ok" if counts else "no_attractors"
    diagnostics = dict(diag)
    diagnostics["runs"] = int(config.runs)
    diagnostics["requested_runs"] = int(config.runs)
    diagnostics["weights_normalization"] = "completed_runs_only"
    diagnostics["weights_requested_runs"] = dict(weights_over_requested_runs)
    diagnostics["weights_requested_runs_serialized"] = {
        repr(k): float(v) for k, v in weights_over_requested_runs.items()
    }
    diagnostics["requested_runs_normalization"] = "requested_runs"
    diagnostics["completion_fraction"] = float(completion_fraction)
    diagnostics["completion_status_target_fraction"] = float(completion_target)
    diagnostics["status_interpretation"] = (
        "ok: completion meets target and attractors found; "
        "no_attractors: completion meets target but none found; "
        "partial_timeout: attractors found but completion below target; "
        "incomplete: completion below target with no attractor evidence from completed runs."
    )
    diagnostics["seed"] = int(config.seed)
    diagnostics["succession"] = str(config.succession)
    diagnostics["runtime_s"] = float(runtime_s)

    return MonteCarloAttractorResult(
        counts=counts,
        weights=weights,
        attractor_keys_ranked=ranked,
        top_attractors=tuple(top_attractors),
        cycles=cycles,
        diagnostics=diagnostics,
        status=status,
        runtime_s=float(runtime_s),
    )

