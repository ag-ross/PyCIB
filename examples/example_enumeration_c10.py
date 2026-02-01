"""
Example: full enumeration on DATASET_C10.

The full configuration space for DATASET_C10 (10 descriptors × 3 states) is enumerated
and the complete set of consistent scenarios is computed.

This example is intended as a correctness and interpretability check for cases where
full enumeration is feasible. A summary plot is written to `results/`.
"""

from __future__ import annotations

import os
import sys
from time import perf_counter

import numpy as np

# The repository root is added to the path so that examples can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib.analysis import ScenarioAnalyzer  # noqa: E402
from cib.core import CIBMatrix, ConsistencyChecker  # noqa: E402
from cib.constraints import AllowedStates, ForbiddenPair, Implies  # noqa: E402
from cib.example_data import DATASET_C10_DESCRIPTORS, DATASET_C10_IMPACTS  # noqa: E402
from cib.scoring import scenario_diagnostics  # noqa: E402
from cib.solvers.config import ExactSolverConfig  # noqa: E402


def main() -> None:
    matrix = CIBMatrix(DATASET_C10_DESCRIPTORS)
    matrix.set_impacts(DATASET_C10_IMPACTS)
    analyzer = ScenarioAnalyzer(matrix)

    total = 1
    for n in matrix.state_counts:
        total *= int(n)
    print(f"DATASET_C10 total scenarios: {int(total)}")

    t0 = perf_counter()
    scenarios = analyzer.enumerate_scenarios()
    t1 = perf_counter()
    print(f"Enumerated scenarios: {len(scenarios)}")

    diag_t0 = perf_counter()
    margins: list[float] = []
    inconsistent_descriptor_counts: list[int] = []
    consistent: list = []

    for s in scenarios:
        diag = scenario_diagnostics(s, matrix)
        margins.append(float(diag.consistency_margin))

        inconsistent_desc = 0
        for d, bal in diag.balances.items():
            chosen_state = diag.chosen_states.get(d)
            if chosen_state is None:
                continue
            chosen_score = float(bal.get(chosen_state, 0.0))
            best_score = max(float(v) for v in bal.values())
            if chosen_score < best_score:
                inconsistent_desc += 1
        inconsistent_descriptor_counts.append(int(inconsistent_desc))

        if bool(diag.is_consistent):
            consistent.append(s)

    diag_t1 = perf_counter()
    t2 = perf_counter()

    print(f"Enumeration runtime: {t1 - t0:.2f}s")
    print(f"Diagnostics runtime: {diag_t1 - diag_t0:.2f}s")
    print(f"Consistent scenarios (brute force): {len(consistent)}")

    cfg = ExactSolverConfig(ordering="connectivity", bound="safe_upper_bound_v1")
    t3 = perf_counter()
    res = analyzer.find_all_consistent_exact(config=cfg)
    t4 = perf_counter()
    print(f"Exact solver runtime: {t4 - t3:.2f}s")
    print(f"Consistent scenarios (exact solver): {len(res.scenarios)}")
    print(f"Exact solver complete: {res.is_complete}")
    print("Exact solver diagnostics:", res.diagnostics)

    brute_set = {tuple(s.to_indices()) for s in consistent}
    exact_set = {tuple(s.to_indices()) for s in res.scenarios}
    print(f"Parity (brute force vs exact): {brute_set == exact_set}")

    # -----------------------------
    # The same workflow is repeated with feasibility constraints.
    # These constraints are treated as domain feasibility rules and are external to CIB consistency.
    # -----------------------------
    constraints = [
        # Example rule: high electrification implies at least moderate renewables deployment.
        Implies(
            a_desc="Electrification_Demand",
            a_state="High",
            b_desc="Renewables_Deployment",
            b_state="Moderate",
        ),
        # Example rule: high policy stringency cannot coincide with low public acceptance.
        ForbiddenPair(
            a_desc="Policy_Stringency",
            a_state="High",
            b_desc="Public_Acceptance",
            b_state="Low",
        ),
        # Example rule: technology costs cannot be "High" (e.g., assume cost decline has occurred).
        AllowedStates(desc="Technology_Costs", allowed={"Moderate", "Low"}),
    ]

    cfg_constrained = ExactSolverConfig(
        ordering="connectivity",
        bound="safe_upper_bound_v1",
        constraints=constraints,
    )
    t5 = perf_counter()
    res_c = analyzer.find_all_consistent_exact(config=cfg_constrained)
    t6 = perf_counter()
    print("")
    print("Constrained exact solver runtime:", f"{t6 - t5:.2f}s")
    print("Constrained consistent scenarios:", len(res_c.scenarios))
    print("Constrained exact solver complete:", res_c.is_complete)
    print("Constrained exact solver diagnostics:", res_c.diagnostics)

    # A subset check is performed: constrained results are a subset of unconstrained results.
    constrained_set = {tuple(s.to_indices()) for s in res_c.scenarios}
    print("Constrained subset of unconstrained:", constrained_set.issubset(exact_set))

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "example_enumeration_c10_plot_1.png")

    import matplotlib.pyplot as plt  # noqa: E402

    descriptor_names = list(matrix.descriptors.keys())
    state_names = [list(matrix.descriptors[d]) for d in descriptor_names]
    max_states = max(len(s) for s in state_names)

    freq = np.zeros((len(descriptor_names), max_states), dtype=float)
    if len(consistent) > 0:
        for s in consistent:
            for i, d in enumerate(descriptor_names):
                st = s.get_state(d)
                j = state_names[i].index(st)
                freq[i, j] += 1.0
        freq /= float(len(consistent))

    margins_arr = np.asarray(margins, dtype=float)
    inconsistent_arr = np.asarray(inconsistent_descriptor_counts, dtype=int)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    ax0.hist(margins_arr, bins=40, color="#4C78A8", alpha=0.85)
    ax0.axvline(0.0, color="black", linewidth=1.0)
    ax0.set_title("Consistency margin across all scenarios")
    ax0.set_xlabel("min_d (chosen_score - best_score)")
    ax0.set_ylabel("count")
    ax0.text(
        0.02,
        0.98,
        f"total={len(margins_arr)}\nconsistent={len(consistent)}",
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )

    im = ax1.imshow(freq, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax1.set_title("State frequencies in consistent scenarios")
    ax1.set_yticks(np.arange(len(descriptor_names)))
    ax1.set_yticklabels(descriptor_names, fontsize=8)
    ax1.set_xticks(np.arange(max_states))
    ax1.set_xticklabels([f"s{k}" for k in range(max_states)])
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("proportion")

    fig.suptitle(
        "DATASET_C10 full enumeration diagnostics\n"
        f"mean inconsistent descriptors per scenario: {float(np.mean(inconsistent_arr)):.2f}"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()

