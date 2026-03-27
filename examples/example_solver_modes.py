"""
Example: scaling solver modes (exact and Monte Carlo attractors).

The following is demonstrated:
  - exact enumeration via the pruned exact solver (Mode A), and
  - Monte Carlo attractor discovery with estimated weights (Mode B).
"""

from __future__ import annotations

import os
import sys

# The repository root is added to the path so that examples can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib.analysis import ScenarioAnalyzer
from cib.benchmark_data import benchmark_matrix_b1
from cib.solvers.config import ExactSolverConfig, MonteCarloAttractorConfig
from cib.sensitivity import compute_global_sensitivity_attractors


def main() -> None:
    matrix = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(matrix)

    exact_cfg = ExactSolverConfig(ordering="connectivity", bound="safe_upper_bound_v1")
    exact = analyzer.find_all_consistent_exact(config=exact_cfg)
    print("Exact solver status:", exact.status)
    print("Exact consistent scenarios:", len(exact.scenarios))

    # A strict profile is appropriate for reporting:
    # completion quality is enforced and diagnostics are inspected explicitly.
    mc_cfg_strict = MonteCarloAttractorConfig(
        runs=500,
        seed=123,
        succession="global",
        n_jobs=1,
        min_completion_fraction=0.995,
    )
    mc_strict = analyzer.find_attractors_monte_carlo(config=mc_cfg_strict)
    print("Monte Carlo (strict) status:", mc_strict.status)
    print(
        "Monte Carlo (strict) completion fraction:",
        mc_strict.diagnostics.get("completion_fraction"),
    )
    print("Monte Carlo (strict) unique attractors:", len(mc_strict.counts))
    top = mc_strict.attractor_keys_ranked[:3]
    for k in top:
        print("  ", k.kind, mc_strict.counts[k])

    # A permissive profile can be useful for exploratory iteration:
    # threshold checks are disabled, and timeout diagnostics must be reviewed.
    mc_cfg_explore = MonteCarloAttractorConfig(
        runs=500,
        seed=123,
        succession="global",
        n_jobs=1,
        min_completion_fraction=None,
        fail_on_timeout=False,
    )
    mc_explore = analyzer.find_attractors_monte_carlo(config=mc_cfg_explore)
    print(
        "Monte Carlo (explore) n_timeouts:",
        mc_explore.diagnostics.get("n_timeouts"),
    )

    rep = compute_global_sensitivity_attractors(mc_strict, top_k=10)
    if rep.rare_outcome_warnings:
        print("Rare-attractor warnings (first 10):")
        for w in rep.rare_outcome_warnings[:10]:
            print("  -", w)


if __name__ == "__main__":
    main()

