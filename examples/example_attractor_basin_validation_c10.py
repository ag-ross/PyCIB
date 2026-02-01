"""
Example: attractor basin validation on DATASET_C10.

In this example, global succession is run from every initial scenario in the full
DATASET_C10 configuration space (3^10 = 59,049) so that exact attractor basin
weights can be computed. These weights are then compared to Monte Carlo attractor
weights estimated by `ScenarioAnalyzer.find_attractors_monte_carlo`.

A comparison plot is written to `results/`.
"""

from __future__ import annotations

import os
import sys
from itertools import product
from time import perf_counter

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The repository root is added to the path so that examples can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib.analysis import ScenarioAnalyzer  # noqa: E402
from cib.core import CIBMatrix  # noqa: E402
from cib.example_data import DATASET_C10_DESCRIPTORS, DATASET_C10_IMPACTS  # noqa: E402
from cib.fast_scoring import FastCIBScorer  # noqa: E402
from cib.fast_succession import run_to_attractor_indices  # noqa: E402
from cib.solvers.config import MonteCarloAttractorConfig  # noqa: E402
from cib.solvers.monte_carlo_attractors import AttractorKey  # noqa: E402


def _canonicalise_cycle_key(
    cycle: tuple[tuple[int, ...], ...],
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
    cycle: tuple[tuple[int, ...], ...],
    *,
    policy: str,
) -> tuple[tuple[int, ...], ...]:
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


def _total_space_size(matrix: CIBMatrix) -> int:
    total = 1
    for n in matrix.state_counts:
        total *= int(n)
    return int(total)


def main() -> None:
    matrix = CIBMatrix(DATASET_C10_DESCRIPTORS)
    matrix.set_impacts(DATASET_C10_IMPACTS)

    total = _total_space_size(matrix)
    print(f"DATASET_C10 total scenarios: {total}")

    mc_runs = int(os.environ.get("CIB_C10_MC_RUNS", "20000"))
    mc_seed = int(os.environ.get("CIB_C10_MC_SEED", "123"))
    cycle_key_policy = str(os.environ.get("CIB_C10_CYCLE_KEY_POLICY", "min_state"))
    if cycle_key_policy not in {"min_state", "rotate_min"}:
        raise ValueError("CIB_C10_CYCLE_KEY_POLICY must be 'min_state' or 'rotate_min'")

    scorer = FastCIBScorer.from_matrix(matrix)

    t0 = perf_counter()
    exact_counts: dict[AttractorKey, int] = {}
    exact_cycles: dict[AttractorKey, tuple[tuple[int, ...], ...]] = {}

    n_desc = int(len(scorer.descriptors))
    ranges = [range(int(scorer.state_counts[j])) for j in range(n_desc)]
    for z in product(*ranges):
        res = run_to_attractor_indices(
            scorer=scorer,
            initial_z_idx=z,
            rule="global",
            max_iterations=500,
        )
        if not bool(res.is_cycle):
            key = AttractorKey(kind="fixed", value=tuple(int(x) for x in res.attractor))  # type: ignore[arg-type]
        else:
            cycle = res.attractor  # type: ignore[assignment]
            assert isinstance(cycle, tuple)
            key = _canonicalise_cycle_key(cycle, policy=cycle_key_policy)
            exact_cycles[key] = _canonicalise_cycle_storage(cycle, policy=cycle_key_policy)

        exact_counts[key] = int(exact_counts.get(key, 0)) + 1

    t1 = perf_counter()
    exact_weights = {k: float(v) / float(total) for k, v in exact_counts.items()}
    print(f"Exact unique attractors: {len(exact_counts)}")
    print(f"Exact basin computation runtime: {t1 - t0:.2f}s")

    analyzer = ScenarioAnalyzer(matrix)
    mc_cfg = MonteCarloAttractorConfig(
        runs=int(mc_runs),
        seed=int(mc_seed),
        succession="global",
        max_iterations=500,
        n_jobs=1,
        cycle_mode="keep_cycle",
        cycle_key_policy=cycle_key_policy,  # type: ignore[arg-type]
    )
    t2 = perf_counter()
    mc = analyzer.find_attractors_monte_carlo(config=mc_cfg)
    t3 = perf_counter()
    print(f"Monte Carlo runtime: {t3 - t2:.2f}s (runs={mc_runs})")
    print(f"Monte Carlo unique attractors: {len(mc.weights)}")

    keys = set(exact_weights.keys()) | set(mc.weights.keys())
    l1 = float(sum(abs(float(exact_weights.get(k, 0.0)) - float(mc.weights.get(k, 0.0))) for k in keys))

    ranked = sorted(exact_weights.items(), key=lambda kv: (-float(kv[1]), kv[0].kind, kv[0].value))
    top_n = int(min(12, len(ranked)))
    top_keys = [k for (k, _) in ranked[:top_n]]

    print(f"L1 error (exact vs Monte Carlo): {l1:.4f}")
    for k in top_keys[:8]:
        print("  ", k.kind, exact_weights.get(k, 0.0), mc.weights.get(k, 0.0))

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "example_attractor_basin_validation_c10_plot_1.png")

    labels = [f"{k.kind}:{i}" for i, k in enumerate(top_keys)]
    exact_vals = [float(exact_weights.get(k, 0.0)) for k in top_keys]
    mc_vals = [float(mc.weights.get(k, 0.0)) for k in top_keys]

    x = np.arange(len(top_keys), dtype=float)
    w = 0.42
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.bar(x - w / 2.0, exact_vals, width=w, label="exact", color="#4C78A8", alpha=0.9)
    ax.bar(x + w / 2.0, mc_vals, width=w, label="monte_carlo", color="#F58518", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("basin weight")
    ax.set_title("DATASET_C10 attractor basin weights (exact vs Monte Carlo)")
    ax.legend()
    ax.text(
        0.01,
        0.98,
        f"exact attractors={len(exact_weights)}\n"
        f"mc runs={mc_runs}\n"
        f"L1 error={l1:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()

