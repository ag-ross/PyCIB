"""
Example: scaling solver modes on DATASET_C10 (10 descriptors × 3 states).

In this example, the scaling solvers are exercised on a workshop-scale dataset:
  - Monte Carlo attractor discovery is performed to identify dominant attractors.
  - Exact enumeration is attempted with a short time limit to demonstrate graceful
    partial results behaviour when full enumeration is not desired.
"""

from __future__ import annotations

import os
import sys

# The repository root is added to the path so that examples can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib.analysis import ScenarioAnalyzer
from cib.core import CIBMatrix
from cib.constraints import ForbiddenPair, Implies
from cib.example_data import (
    DATASET_C10_DESCRIPTORS,
    DATASET_C10_IMPACTS,
)
from cib.solvers.config import ExactSolverConfig, MonteCarloAttractorConfig


def main() -> None:
    matrix = CIBMatrix(DATASET_C10_DESCRIPTORS)
    matrix.set_impacts(DATASET_C10_IMPACTS)
    analyzer = ScenarioAnalyzer(matrix)

    # Strict (reporting) profile.
    mc_cfg = MonteCarloAttractorConfig(
        runs=1500,
        seed=123,
        succession="global",
        max_iterations=250,
        n_jobs=1,
        cycle_mode="keep_cycle",
        fast_backend="dense",
        min_completion_fraction=0.995,
    )
    mc = analyzer.find_attractors_monte_carlo(config=mc_cfg)
    print("Monte Carlo status:", mc.status)
    print("Monte Carlo completed runs:", mc.diagnostics.get("n_completed_runs"))
    print("Monte Carlo completion fraction:", mc.diagnostics.get("completion_fraction"))
    print("Monte Carlo unique attractors:", len(mc.counts))
    top = mc.attractor_keys_ranked[:5]
    for k in top:
        print("  ", k.kind, mc.counts[k])

    # Permissive (exploratory) profile.
    mc_cfg_sparse = MonteCarloAttractorConfig(
        runs=1500,
        seed=123,
        succession="global",
        max_iterations=250,
        n_jobs=1,
        cycle_mode="keep_cycle",
        fast_backend="sparse",
        min_completion_fraction=None,
        fail_on_timeout=False,
    )
    mc_sparse = analyzer.find_attractors_monte_carlo(config=mc_cfg_sparse)
    print("Monte Carlo (sparse backend) status:", mc_sparse.status)
    print(
        "Monte Carlo (sparse backend) unique attractors:",
        len(mc_sparse.counts),
    )
    print(
        "Monte Carlo (sparse backend) n_timeouts:",
        mc_sparse.diagnostics.get("n_timeouts"),
    )

    exact_cfg = ExactSolverConfig(
        ordering="connectivity",
        bound="safe_upper_bound_v1",
        time_limit_s=2.0,
    )
    exact = analyzer.find_all_consistent_exact(config=exact_cfg)
    print("Exact solver status:", exact.status)
    print("Exact solver complete:", exact.is_complete)
    print("Exact scenarios found:", len(exact.scenarios))
    print("Exact diagnostics:", exact.diagnostics)

    # A constrained exact run is shown (same solver, smaller feasible space).
    constrained_cfg = ExactSolverConfig(
        ordering="connectivity",
        bound="safe_upper_bound_v1",
        time_limit_s=2.0,
        constraints=[
            Implies(
                a_desc="Electrification_Demand",
                a_state="High",
                b_desc="Renewables_Deployment",
                b_state="Moderate",
            ),
            ForbiddenPair(
                a_desc="Policy_Stringency",
                a_state="High",
                b_desc="Public_Acceptance",
                b_state="Low",
            ),
        ],
    )
    constrained = analyzer.find_all_consistent_exact(config=constrained_cfg)
    print("Constrained exact solver status:", constrained.status)
    print("Constrained exact solver complete:", constrained.is_complete)
    print("Constrained exact scenarios found:", len(constrained.scenarios))
    print("Constrained exact diagnostics:", constrained.diagnostics)


if __name__ == "__main__":
    main()

