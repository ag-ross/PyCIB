"""
Research-oriented probabilistic CIA example (dynamic predict–update).

The predict–update mode is demonstrated as a regularised dynamic fit where:
- each period has its own marginal and multiplier inputs, and
- the fitted distribution from period t-1 is used as a KL baseline for period t.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

# The repository root is added to the path so that examples can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib.prob import DynamicProbabilisticCIA, FactorSpec, ProbabilisticCIAModel


def _multipliers_for_a_given_b(marginals, *, p_a1_b0: float, p_a1_b1: float):
    p_a0_b0 = 1.0 - float(p_a1_b0)
    p_a0_b1 = 1.0 - float(p_a1_b1)
    return {
        (("A", "a1"), ("B", "b0")): float(p_a1_b0) / float(marginals["A"]["a1"]),
        (("A", "a0"), ("B", "b0")): float(p_a0_b0) / float(marginals["A"]["a0"]),
        (("A", "a1"), ("B", "b1")): float(p_a1_b1) / float(marginals["A"]["a1"]),
        (("A", "a0"), ("B", "b1")): float(p_a0_b1) / float(marginals["A"]["a0"]),
    }


def main() -> None:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}

    multipliers_t1 = _multipliers_for_a_given_b(marginals, p_a1_b0=0.3142857142857143, p_a1_b1=0.9)
    multipliers_t2 = _multipliers_for_a_given_b(marginals, p_a1_b0=0.3142857142857143, p_a1_b1=0.1)

    model1 = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers=multipliers_t1)
    model2 = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers=multipliers_t2)

    dyn = DynamicProbabilisticCIA(periods=[2025, 2030], models_by_period={2025: model1, 2030: model2})

    refit = dyn.fit_distributions(mode="refit", method="direct", weight_by_target=False, solver_maxiter=4000)
    pu = dyn.fit_distributions(
        mode="predict-update",
        method="direct",
        kl_weight=50.0,
        weight_by_target=False,
        solver_maxiter=4000,
    )

    p1 = refit[2025].p
    d_refit = float(np.sum(np.abs(refit[2030].p - p1)))
    d_pu = float(np.sum(np.abs(pu[2030].p - p1)))

    print("Distance to previous period (L1):")
    print("  refit =", d_refit)
    print("  predict_update =", d_pu)


if __name__ == "__main__":
    main()

