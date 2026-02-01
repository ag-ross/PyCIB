"""
Research-oriented probabilistic CIA example (scaling using the iterative method).

The iterative method is intended for large scenario spaces where dense enumeration is not desired.
An approximate distribution is returned as a sampled support representation.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

# The repository root is added to the path so that examples can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib.prob import DiagnosticsReport, FactorSpec, ProbabilisticCIAModel


def _uniform_marginals(factors: list[FactorSpec]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for f in factors:
        p = 1.0 / float(len(f.outcomes))
        out[f.name] = {o: float(p) for o in f.outcomes}
    return out


def main() -> None:
    factors = [FactorSpec(f"X{i}", ["Low", "Medium", "High"]) for i in range(10)]
    marginals = _uniform_marginals(factors)

    # A coherent sparse multiplier set is constructed for X0 given X1.
    outcomes = ["Low", "Medium", "High"]
    cond_by_x1 = {
        "Low": {"Low": 0.50, "Medium": 0.25, "High": 0.25},
        "Medium": {"Low": 0.25, "Medium": 0.50, "High": 0.25},
        "High": {"Low": 0.25, "Medium": 0.25, "High": 0.50},
    }
    multipliers: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float] = {}
    for b in outcomes:
        for a in outcomes:
            pi = float(marginals["X0"][a])
            multipliers[(("X0", a), ("X1", b))] = float(cond_by_x1[b][a] / pi)

    model = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers=multipliers)
    dist = model.fit_joint(
        method="iterative",
        random_seed=123,
        iterative_burn_in_sweeps=2500,
        iterative_n_samples=15000,
        iterative_thinning=3,
        with_report=True,
    )

    diag = DiagnosticsReport.from_distribution(dist, marginals=marginals, multipliers=multipliers)
    print("Diagnostics (approximate):")
    print("  marginal_max_abs_error =", diag.marginal_max_abs_error)
    print("  pairwise_target_max_abs_error =", diag.pairwise_target_max_abs_error)

    print("\nTop sampled scenarios:")
    for p, s in dist.top_scenarios(10):
        print(f"  p={p:.4f}  {s}")


if __name__ == "__main__":
    main()

