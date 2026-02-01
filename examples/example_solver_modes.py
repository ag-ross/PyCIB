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

    mc_cfg = MonteCarloAttractorConfig(runs=500, seed=123, succession="global", n_jobs=1)
    mc = analyzer.find_attractors_monte_carlo(config=mc_cfg)
    print("Monte Carlo status:", mc.status)
    print("Monte Carlo unique attractors:", len(mc.counts))
    top = mc.attractor_keys_ranked[:3]
    for k in top:
        print("  ", k.kind, mc.counts[k])

    rep = compute_global_sensitivity_attractors(mc, top_k=10)
    if rep.rare_outcome_warnings:
        print("Rare-attractor warnings (first 10):")
        for w in rep.rare_outcome_warnings[:10]:
            print("  -", w)


if __name__ == "__main__":
    main()

