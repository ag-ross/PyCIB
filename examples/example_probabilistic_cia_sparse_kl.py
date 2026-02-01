"""
Research-oriented probabilistic CIA example (sparse constraints + KL regularisation).

The following workflow is demonstrated:
- a small factor system is specified,
- a synthetic coherent joint distribution is generated,
- sparse multipliers are derived from that distribution,
- a joint distribution is fitted from marginals and sparse multipliers,
- a fit report and diagnostics are printed,
- an identification bound is computed for a selected event.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

# The repository root is added to the path so that examples can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib.prob import DiagnosticsReport, FactorSpec, ProbabilisticCIAModel
from cib.prob.types import ScenarioIndex


def _marginals_from_joint(index: ScenarioIndex, p: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for f in index.factors:
        out[f.name] = {o: 0.0 for o in f.outcomes}
    for k in range(index.size):
        scen = index.scenario_at(k)
        for fname, outcome in zip(index.factor_names, scen.assignment):
            out[fname][outcome] += float(p[k])
    return out


def _multipliers_from_joint(
    index: ScenarioIndex,
    p: np.ndarray,
    *,
    target_factor: str,
    given_factor: str,
) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
    target_factor = str(target_factor)
    given_factor = str(given_factor)
    pos_i = list(index.factor_names).index(target_factor)
    pos_j = list(index.factor_names).index(given_factor)

    marg = _marginals_from_joint(index, p)
    pair: Dict[Tuple[str, str], float] = {}
    for a in index.factors[pos_i].outcomes:
        for b in index.factors[pos_j].outcomes:
            pair[(a, b)] = 0.0
    for k in range(index.size):
        scen = index.scenario_at(k)
        a = scen.assignment[pos_i]
        b = scen.assignment[pos_j]
        pair[(a, b)] += float(p[k])

    out: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float] = {}
    for a in index.factors[pos_i].outcomes:
        for b in index.factors[pos_j].outcomes:
            pj = float(marg[given_factor][b])
            pi = float(marg[target_factor][a])
            pij = float(pair[(a, b)])
            cond = pij / pj
            out[((target_factor, a), (given_factor, b))] = float(cond / pi)
    return out


def main() -> None:
    factors = [
        FactorSpec("A", ["Low", "Medium", "High"]),
        FactorSpec("B", ["Low", "Medium", "High"]),
        FactorSpec("C", ["Low", "Medium", "High"]),
        FactorSpec("D", ["Low", "Medium", "High"]),
    ]
    index = ScenarioIndex(factors)

    rng = np.random.default_rng(123)
    p_true = rng.dirichlet(alpha=np.ones(index.size, dtype=float))
    marginals = _marginals_from_joint(index, p_true)

    # A sparse multiplier set is constructed from one directed relation only.
    multipliers = _multipliers_from_joint(index, p_true, target_factor="C", given_factor="A")

    model = ProbabilisticCIAModel(
        factors=factors,
        marginals=marginals,
        multipliers=multipliers,
        feasibility_mode="strict",
    )
    dist = model.fit_joint(
        method="direct",
        kl_weight=1e-6,
        weight_by_target=False,
        random_seed=123,
        solver_maxiter=8000,
        with_report=True,
    )
    report = dist.fit_report
    diag = DiagnosticsReport.from_distribution(dist, marginals=marginals, multipliers=multipliers)

    print("Fit report:")
    if report is None:
        print("  (no report)")
    else:
        print("  method =", report.method)
        print("  solver =", report.solver)
        print("  success =", report.success)
        print("  message =", report.message)
        print("  wls_value =", report.wls_value)
        print("  kl_value =", report.kl_value)
        print("  kl_weight =", report.kl_weight)
        print("  max_abs_marginal_residual =", report.max_abs_marginal_residual)
        print("  max_abs_pairwise_residual =", report.max_abs_pairwise_residual)

    print("\nDiagnostics:")
    print("  sum_to_one_error =", diag.sum_to_one_error)
    print("  marginal_max_abs_error =", diag.marginal_max_abs_error)
    print("  pairwise_target_max_abs_error =", diag.pairwise_target_max_abs_error)
    if diag.multiplier_normalization_issues:
        print("  multiplier_normalization_issues =", len(diag.multiplier_normalization_issues))

    # An identification bound is computed for a selected event (small-space LP).
    bounds = model.event_probability_bounds(event={"C": "High", "A": "High"}, include_pairwise_targets=True)
    print("\nIdentification bounds:")
    print("  P(C=High, A=High) in [", bounds.lower, ",", bounds.upper, "]")


if __name__ == "__main__":
    main()

