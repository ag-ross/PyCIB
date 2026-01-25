from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from cib.prob.constraints import (
    Marginals,
    Multipliers,
    MultiplierFeasibilityIssue,
    multiplier_normalization_issues,
    multiplier_pairwise_targets,
    pairwise_target_frechet_violations,
)
from cib.prob.model import JointDistribution


@dataclass(frozen=True)
class DiagnosticsReport:
    sum_to_one_error: float
    min_probability: float
    max_probability: float
    marginal_max_abs_error: float
    pairwise_target_max_abs_error: float
    multiplier_normalization_issues: Tuple[MultiplierFeasibilityIssue, ...]
    pairwise_target_frechet_violations: Tuple[Tuple[Tuple[str, str, str, str], float, float, float], ...]

    @staticmethod
    def from_distribution(
        dist: JointDistribution,
        *,
        marginals: Marginals,
        multipliers: Optional[Multipliers] = None,
        tol: float = 1e-9,
    ) -> "DiagnosticsReport":
        multipliers = multipliers or {}
        p = np.asarray(dist.p, dtype=float)
        sum_err = float(abs(float(np.sum(p)) - 1.0))
        min_p = float(np.min(p)) if p.size else float("nan")
        max_p = float(np.max(p)) if p.size else float("nan")

        # Marginal agreement is checked.
        max_marg_err = 0.0
        for f in dist.index.factors:
            implied = dist.marginal(f.name)
            for o in f.outcomes:
                max_marg_err = max(max_marg_err, abs(float(implied[o]) - float(marginals[f.name][o])))

        targets = multiplier_pairwise_targets(marginals, multipliers)
        max_pair_err = 0.0
        for (i, a, j, b), t in targets.items():
            implied = dist.pairwise_marginal(i, a, j, b)
            max_pair_err = max(max_pair_err, abs(float(implied) - float(t)))

        norm_issues = tuple(
            multiplier_normalization_issues(dist.index.factors, marginals, multipliers, tol=1e-6)
        )

        frechet = pairwise_target_frechet_violations(marginals, targets, tol=tol)
        frechet_items = tuple((k, v, lo, hi) for k, (v, lo, hi) in frechet.items())

        return DiagnosticsReport(
            sum_to_one_error=float(sum_err),
            min_probability=float(min_p),
            max_probability=float(max_p),
            marginal_max_abs_error=float(max_marg_err),
            pairwise_target_max_abs_error=float(max_pair_err),
            multiplier_normalization_issues=norm_issues,
            pairwise_target_frechet_violations=frechet_items,
        )

