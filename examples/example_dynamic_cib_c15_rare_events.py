"""
Example: dynamic CIB on DATASET_C15 with rare events and regime switching.

In this example, a heavier workshop-scale dataset (15 descriptors × 4 states) is
simulated with heavy-tailed innovations and rare jumps. Threshold activations
are recorded to quantify regime-switch frequency.

Runtime is controlled by environment variables so that the example can be run
quickly by default while still supporting heavier research runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The repository root is added to the path so that examples can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib.dynamic import DynamicCIB  # noqa: E402
from cib.pathway import numeric_quantile_timelines, state_probability_timelines  # noqa: E402
from cib.uncertainty import UncertainCIBMatrix  # noqa: E402
from cib.visualization import DynamicVisualizer  # noqa: E402
from cib.example_data import (  # noqa: E402
    DATASET_C15_CONFIDENCE,
    DATASET_C15_DESCRIPTORS,
    DATASET_C15_IMPACTS,
    DATASET_C15_INITIAL_SCENARIO,
    DATASET_C15_NUMERIC_MAPPING,
    DEFAULT_PERIODS,
    dataset_c15_cyclic_descriptors,
    dataset_c15_threshold_rule_accelerated_transition,
    seeds_for_run,
)
from cib.shocks import ShockModel  # noqa: E402


def main() -> None:
    periods = list(DEFAULT_PERIODS)
    descriptor = "Electrification_Demand"

    n_runs = int(os.environ.get("CIB_C15_RUNS", "800"))
    base_seed = int(os.environ.get("CIB_C15_SEED", "123"))

    matrix = UncertainCIBMatrix(DATASET_C15_DESCRIPTORS)
    matrix.set_impacts(DATASET_C15_IMPACTS, confidence=DATASET_C15_CONFIDENCE)

    dyn = DynamicCIB(matrix, periods=periods)
    for cd in dataset_c15_cyclic_descriptors():
        dyn.add_cyclic_descriptor(cd)
    dyn.add_threshold_rule(dataset_c15_threshold_rule_accelerated_transition())

    paths = []
    iters: list[int] = []
    cycles: list[bool] = []
    applied: list[list[str]] = []

    t0 = perf_counter()
    for m in range(int(n_runs)):
        seeds = seeds_for_run(base_seed, m)

        sm = ShockModel(matrix)
        sm.add_dynamic_shocks(
            periods=periods,
            tau=0.28,
            rho=0.65,
            innovation_dist="student_t",
            innovation_df=5.0,
            jump_prob=0.02,
            jump_scale=0.85,
        )
        dynamic_shocks = sm.sample_dynamic_shocks(seeds["dynamic_shock_seed"])

        sigma_by_period = {int(t): 1.0 + 0.8 * i for i, t in enumerate(periods)}
        diag = {}
        paths.append(
            dyn.simulate_path(
                initial=DATASET_C15_INITIAL_SCENARIO,
                seed=seeds["dynamic_shock_seed"],
                dynamic_shocks_by_period=dynamic_shocks,
                judgment_sigma_scale_by_period=sigma_by_period,
                structural_sigma=0.15,
                structural_seed_base=seeds["structural_shock_seed"],
                equilibrium_mode="relax_unshocked",
                diagnostics=diag,
                max_iterations=800,
            )
        )

        iters.extend(int(x) for x in diag.get("iterations", []))
        cycles.extend(bool(x) for x in diag.get("is_cycle", []))
        applied.extend(list(x) for x in diag.get("threshold_rules_applied", []))
    t1 = perf_counter()

    # Threshold activations are summarised.
    fired_periods = [bool(p) for p in applied]
    fired_any = 0
    for r in range(int(n_runs)):
        start = r * len(periods)
        end = start + len(periods)
        if any(fired_periods[start:end]):
            fired_any += 1
    regime_rate = float(fired_any) / float(max(1, n_runs))

    timelines = state_probability_timelines(paths, scenario_mode="realized")
    mapping = DATASET_C15_NUMERIC_MAPPING[descriptor]
    quantiles = numeric_quantile_timelines(
        paths,
        descriptor=descriptor,
        numeric_mapping=mapping,
        quantiles=(0.05, 0.5, 0.95),
        scenario_mode="realized",
    )
    expectation = {
        int(t): sum(float(p) * float(mapping[s]) for s, p in timelines[t][descriptor].items())
        for t in timelines
    }

    plt.figure(figsize=(14, 10))
    DynamicVisualizer.plot_descriptor_stochastic_summary(
        timelines=timelines,
        quantiles_by_period=quantiles,
        numeric_expectation_by_period=expectation,
        descriptor=descriptor,
        title="DATASET_C15: Electrification_Demand (realised) — rare events + regime switching",
        spaghetti_paths=paths,
        spaghetti_numeric_mapping=mapping,
        spaghetti_max_runs=200,
    )
    plt.tight_layout()

    repo_root = Path(__file__).resolve().parent.parent
    out = repo_root / "results" / "example_dynamic_c15_rare_events_plot_1.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()

    mean_iter = float(sum(iters)) / float(max(1, len(iters)))
    cycle_rate = float(sum(1 for x in cycles if x)) / float(max(1, len(cycles)))

    print(f"Saved plot to {out}")
    print(f"Runtime (simulation): {t1 - t0:.2f}s (runs={n_runs}, periods={len(periods)})")
    print(f"Regime switch rate (any period): {regime_rate:.3f}")
    print(
        "Succession (realised): "
        f"mean_iterations_per_period={mean_iter:.2f}, cycle_rate={cycle_rate:.3f}"
    )


if __name__ == "__main__":
    main()

