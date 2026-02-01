"""
Research-oriented probabilistic CIA example (strict versus repair feasibility).

An intentionally incoherent multiplier is provided to demonstrate:
- strict feasibility rejection, and
- repair feasibility projection with auditable adjustments.
"""

from __future__ import annotations

import os
import sys

# The repository root is added to the path so that examples can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib.prob import ProbabilisticCIAModel
from cib.prob.types import FactorSpec


def main() -> None:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.5, "a1": 0.5}, "B": {"b0": 0.5, "b1": 0.5}}

    # This multiplier implies P(A=a1,B=b1)=1.25, which is infeasible.
    multipliers = {(("A", "a1"), ("B", "b1")): 5.0}

    print("Strict feasibility:")
    try:
        _model_strict = ProbabilisticCIAModel(
            factors=factors,
            marginals=marginals,
            multipliers=multipliers,
            feasibility_mode="strict",
        )
        print("  A strict model was constructed (unexpected).")
    except ValueError as exc:
        print("  A strict model was rejected as expected.")
        print("  error =", str(exc))

    print("\nRepair feasibility:")
    model_repair = ProbabilisticCIAModel(
        factors=factors,
        marginals=marginals,
        multipliers=multipliers,
        feasibility_mode="repair",
    )
    dist = model_repair.fit_joint(method="direct", kl_weight=1e-8, solver_maxiter=3000, with_report=True)
    if dist.fit_report is None:
        print("  No fit report was returned.")
        return

    print("  feasibility_mode =", dist.fit_report.feasibility_mode)
    print("  number_of_adjustments =", len(dist.fit_report.feasibility_adjustments))
    if dist.fit_report.feasibility_adjustments:
        a = dist.fit_report.feasibility_adjustments[0]
        print("  example_adjustment =", (a.i, a.a, a.j, a.b, a.original_value, a.adjusted_value))


if __name__ == "__main__":
    main()

