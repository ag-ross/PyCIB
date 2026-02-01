"""
Example: dynamic CIB on DATASET_C10 (workshop-scale dataset).

In this example, a Monte Carlo ensemble is simulated for DATASET_C10 and a
probability-band plot with numeric summaries is written to the results
directory.
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
from cib.sensitivity import compute_global_sensitivity_dynamic  # noqa: E402
from cib.uncertainty import UncertainCIBMatrix  # noqa: E402
from cib.visualization import DynamicVisualizer  # noqa: E402
from cib.rare_events import near_miss_rate  # noqa: E402
from cib.attribution import attribute_scenario, flip_candidates_for_descriptor  # noqa: E402
from cib.example_data import (
    DATASET_C10_CONFIDENCE,
    DATASET_C10_DESCRIPTORS,
    DATASET_C10_IMPACTS,
    DATASET_C10_INITIAL_SCENARIO,
    DATASET_C10_NUMERIC_MAPPING,
    DEFAULT_PERIODS,
    dataset_c10_cyclic_descriptors,
    dataset_c10_threshold_rule_accelerated_transition,
    seeds_for_run,
)
from cib.shocks import ShockModel


def main() -> None:
    periods = list(DEFAULT_PERIODS)
    descriptor = "Electrification_Demand"

    matrix = UncertainCIBMatrix(DATASET_C10_DESCRIPTORS)
    matrix.set_impacts(DATASET_C10_IMPACTS, confidence=DATASET_C10_CONFIDENCE)

    dyn = DynamicCIB(matrix, periods=periods)
    for cd in dataset_c10_cyclic_descriptors():
        dyn.add_cyclic_descriptor(cd)
    dyn.add_threshold_rule(dataset_c10_threshold_rule_accelerated_transition())

    base_seed = 123
    n_runs = 400
    paths = []
    diagnostics_by_run: list[dict] = []
    iters: list[int] = []
    cycles: list[bool] = []
    eq_iters: list[int] = []
    eq_cycles: list[bool] = []

    t_sim0 = perf_counter()
    for m in range(int(n_runs)):
        seeds = seeds_for_run(base_seed, m)

        sm = ShockModel(matrix)
        sm.add_dynamic_shocks(
            periods=periods,
            tau=0.22,
            rho=0.6,
            innovation_dist="normal",
        )
        dynamic_shocks = sm.sample_dynamic_shocks(seeds["dynamic_shock_seed"])

        sigma_by_period = {int(t): 1.0 + 0.6 * i for i, t in enumerate(periods)}
        diag = {}
        paths.append(
            dyn.simulate_path(
                initial=DATASET_C10_INITIAL_SCENARIO,
                seed=seeds["dynamic_shock_seed"],
                dynamic_shocks_by_period=dynamic_shocks,
                judgment_sigma_scale_by_period=sigma_by_period,
                structural_sigma=0.12,
                structural_seed_base=seeds["structural_shock_seed"],
                equilibrium_mode="relax_unshocked",
                diagnostics=diag,
            )
        )
        diagnostics_by_run.append(dict(diag))
        iters.extend(int(x) for x in diag.get("iterations", []))
        cycles.extend(bool(x) for x in diag.get("is_cycle", []))
        eq_iters.extend(int(x) for x in diag.get("equilibrium_iterations", []))
        eq_cycles.extend(bool(x) for x in diag.get("equilibrium_is_cycle", []))
    t_sim1 = perf_counter()

    t_post0 = perf_counter()
    timelines = state_probability_timelines(paths, scenario_mode="realized")
    mapping = DATASET_C10_NUMERIC_MAPPING[descriptor]
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
    t_post1 = perf_counter()

    t_plot0 = perf_counter()
    plt.figure(figsize=(14, 10))
    DynamicVisualizer.plot_descriptor_stochastic_summary(
        timelines=timelines,
        quantiles_by_period=quantiles,
        numeric_expectation_by_period=expectation,
        descriptor=descriptor,
        title="DATASET_C10: Electrification_Demand (realised) — probability bands + fan + spaghetti",
        spaghetti_paths=paths,
        spaghetti_numeric_mapping=mapping,
        spaghetti_max_runs=200,
    )
    plt.tight_layout()

    repo_root = Path(__file__).resolve().parent.parent
    out = repo_root / "results" / "example_dynamic_c10_plot_1.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    t_plot1 = perf_counter()

    # A compact sensitivity and diagnostics summary is written to the results directory.
    report = compute_global_sensitivity_dynamic(
        paths,
        diagnostics_by_run=diagnostics_by_run,
        cyclic_descriptor_names=[cd.name for cd in dataset_c10_cyclic_descriptors()],
        key_descriptors=[descriptor],
        numeric_mappings={descriptor: mapping},
        bootstrap=100,
        seed=123,
        scenario_mode="realized",
    )

    # The near-miss rate is computed against the base CIM only (per-run sampled CIMs are not retained).
    final_scenarios = [p.scenarios[-1] for p in paths]
    nm = near_miss_rate(final_scenarios, matrix, epsilon=0.25)

    # An attribution example is produced for one representative final scenario (illustrative under the base CIM).
    attr = attribute_scenario(final_scenarios[0], matrix, top_k_sources=10)
    d_attr = attr.by_descriptor().get(descriptor)
    flips = flip_candidates_for_descriptor(d_attr, k=5) if d_attr is not None else tuple()

    summary_path = repo_root / "results" / "example_dynamic_c10_sensitivity_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("DATASET_C10: sensitivity and diagnostics summary\n")
        fh.write(f"Runs: {len(paths)}\n")
        fh.write(f"Near-miss rate (base CIM, epsilon=0.25): {nm:.4f}\n")
        fh.write("\nRare-outcome warnings:\n")
        for w in report.rare_outcome_warnings[:20]:
            fh.write(f"- {w}\n")
        fh.write("\nTop importances (first 5 outcomes shown):\n")
        for osens in report.outcome_sensitivity[:5]:
            fh.write(f"\nOutcome: {osens.outcome}\n")
            for imp in osens.driver_importance[:8]:
                fh.write(f"  - {imp.name}: {imp.estimate:.6g}  (CI [{imp.ci_low:.6g}, {imp.ci_high:.6g}])\n")
        fh.write("\nAttribution example (illustrative, base CIM):\n")
        fh.write(f"Min margin to switching: {attr.min_margin_to_switch:.6g}\n")
        if d_attr is not None:
            fh.write(f"Descriptor: {descriptor}\n")
            fh.write(f"  Margin to switching: {d_attr.margin_to_switch:.6g}\n")
            fh.write("  Flip candidates (top 5, single-cell, clipped to [-3, +3]):\n")
            for fc in flips:
                fh.write(
                    f"    - {fc.action} for {fc.src_descriptor}[{fc.src_state}] "
                    f"({fc.chosen_state} vs {fc.alternative_state}): "
                    f"required={fc.required_change:.6g}, available={fc.available_change:.6g}, feasible={fc.feasible_under_clip}\n"
                )

    print(f"Saved plot to {out}")
    print(f"Saved summary to {summary_path}")
    print(f"Runtime: simulate={t_sim1 - t_sim0:.2f}s, post={t_post1 - t_post0:.2f}s, plot={t_plot1 - t_plot0:.2f}s")
    if iters:
        mean_iter = float(sum(iters)) / float(len(iters))
        frac_cycles = float(sum(1 for x in cycles if x)) / float(len(cycles))
        print(f"Succession (realised): mean_iterations_per_period={mean_iter:.2f}, cycle_rate={frac_cycles:.3f}")
    if eq_iters:
        mean_eq = float(sum(eq_iters)) / float(len(eq_iters))
        frac_eq_cycles = float(sum(1 for x in eq_cycles if x)) / float(len(eq_cycles))
        print(f"Succession (equilibrium): mean_iterations_per_period={mean_eq:.2f}, cycle_rate={frac_eq_cycles:.3f}")


if __name__ == "__main__":
    main()

