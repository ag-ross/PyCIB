from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from cib.prob.constraints import (
    Marginals,
    Multipliers,
    multiplier_pairwise_targets,
    pairwise_target_frechet_violations,
    validate_marginals,
)
from cib.prob.fit_direct import fit_joint_direct
from cib.prob.types import AssignmentLike, FactorSpec, ScenarioIndex


@dataclass(frozen=True)
class JointDistribution:
    """
    Dense joint distribution over all scenarios (small scenario spaces).
    """

    index: ScenarioIndex
    p: np.ndarray  # shape (index.size,)

    def __post_init__(self) -> None:
        if self.p.ndim != 1:
            raise ValueError("p must be 1D")
        if int(self.p.shape[0]) != int(self.index.size):
            raise ValueError("p has wrong length")

    def scenario_prob(self, assignment: AssignmentLike) -> float:
        return float(self.p[self.index.index_of(assignment)])

    def marginal(self, factor: str) -> Dict[str, float]:
        factor = str(factor)
        if factor not in self.index.factor_names:
            raise ValueError(f"Unknown factor: {factor!r}")
        pos = list(self.index.factor_names).index(factor)
        outs = list(self.index.factors[pos].outcomes)
        out: Dict[str, float] = {o: 0.0 for o in outs}
        for idx in range(self.index.size):
            scen = self.index.scenario_at(idx)
            out[scen.assignment[pos]] += float(self.p[idx])
        return out

    def pairwise_marginal(self, i: str, a: str, j: str, b: str) -> float:
        i = str(i)
        j = str(j)
        pos_i = list(self.index.factor_names).index(i)
        pos_j = list(self.index.factor_names).index(j)
        total = 0.0
        for idx in range(self.index.size):
            scen = self.index.scenario_at(idx)
            if scen.assignment[pos_i] == str(a) and scen.assignment[pos_j] == str(b):
                total += float(self.p[idx])
        return float(total)

    def conditional(self, target: Tuple[str, str], given: Tuple[str, str], *, eps: float = 1e-15) -> float:
        """
        Return P(target_factor=target_outcome | given_factor=given_outcome).
        """
        (i, a) = (str(target[0]), str(target[1]))
        (j, b) = (str(given[0]), str(given[1]))
        num = self.pairwise_marginal(i, a, j, b)
        denom = self.marginal(j).get(b, 0.0)
        if denom <= float(eps):
            return float("nan")
        return float(num / denom)


class ProbabilisticCIAModel:
    """
    Static joint-distribution probabilistic CIA model (point marginals + multipliers).
    """

    def __init__(
        self,
        *,
        factors: Sequence[FactorSpec],
        marginals: Marginals,
        multipliers: Optional[Multipliers] = None,
    ) -> None:
        if not factors:
            raise ValueError("factors cannot be empty")
        self.factors = tuple(factors)
        self.index = ScenarioIndex(self.factors)
        self.marginals: Marginals = marginals
        self.multipliers: Multipliers = multipliers or {}

        validate_marginals(self.factors, self.marginals)
        # A soft pre-check is performed: Fréchet violations typically indicate infeasible constraints.
        targets = multiplier_pairwise_targets(self.marginals, self.multipliers)
        violations = pairwise_target_frechet_violations(self.marginals, targets)
        if violations:
            # This check is kept strict to protect users from impossible inputs.
            # If a "soft constraint" mode is desired later, this can be relaxed.
            k = next(iter(violations.keys()))
            v, lo, hi = violations[k]
            raise ValueError(
                "Multiplier-implied pairwise target violates Fréchet bounds: "
                f"{k!r} has {v}, but bounds are [{lo}, {hi}]"
            )

    def fit_joint(
        self,
        *,
        method: str = "direct",
        max_scenarios: int = 20_000,
        kl_weight: float = 0.0,
        weight_by_target: bool = True,
        random_seed: Optional[int] = None,
        solver_maxiter: int = 2_000,
    ) -> JointDistribution:
        method = str(method).strip().lower()
        if method not in {"direct", "iterative"}:
            raise ValueError(f"Unknown method: {method!r}")

        if self.index.size > int(max_scenarios):
            raise ValueError(
                f"Scenario space too large for dense fit (size={self.index.size}, "
                f"max_scenarios={int(max_scenarios)})."
            )

        if method == "iterative":
            # In Phase-1 implementation, iterative methods are only meaningful for large spaces.
            # Direct methods are used in the small-space regime.
            method = "direct"

        p = fit_joint_direct(
            index=self.index,
            marginals=self.marginals,
            multipliers=self.multipliers,
            kl_weight=float(kl_weight),
            weight_by_target=bool(weight_by_target),
            random_seed=random_seed,
            solver_maxiter=int(solver_maxiter),
        )
        return JointDistribution(index=self.index, p=p)

