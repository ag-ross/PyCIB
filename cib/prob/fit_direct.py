from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from cib.prob.constraints import Marginals, Multipliers, multiplier_pairwise_targets, validate_marginals
from cib.prob.types import ScenarioIndex


def _product_baseline(index: ScenarioIndex, marginals: Marginals) -> np.ndarray:
    """
    Baseline q(x) = Π_i P(X_i = x_i) implied by marginals (independence baseline).
    """
    q = np.zeros(index.size, dtype=float)
    for idx in range(index.size):
        scen = index.scenario_at(idx)
        prod = 1.0
        for fname, outcome in zip(index.factor_names, scen.assignment):
            prod *= float(marginals[fname][outcome])
        q[idx] = float(prod)
    s = float(np.sum(q))
    if s <= 0.0:
        raise ValueError("Baseline distribution is degenerate (sum <= 0)")
    q = q / s
    return q


def _build_linear_constraints(index: ScenarioIndex, marginals: Marginals) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build equality constraints matrix C p = d for:
      - sum_x p_x = 1
      - per-factor marginals (k-1 outcomes per factor, last implied)
    """
    rows = []
    rhs = []

    # The sum-to-one constraint is enforced.
    rows.append(np.ones(index.size, dtype=float))
    rhs.append(1.0)

    # Only (k-1) outcomes per factor are used for marginals to avoid dependent equalities.
    for pos, factor in enumerate(index.factors):
        outcomes = list(factor.outcomes)
        for outcome in outcomes[:-1]:
            row = np.zeros(index.size, dtype=float)
            for idx in range(index.size):
                scen = index.scenario_at(idx)
                if scen.assignment[pos] == outcome:
                    row[idx] = 1.0
            rows.append(row)
            rhs.append(float(marginals[factor.name][outcome]))

    C = np.vstack(rows)
    d = np.array(rhs, dtype=float)
    return C, d


def _build_pairwise_matrix(
    index: ScenarioIndex,
    targets: Mapping[Tuple[str, str, str, str], float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build A p ~ t for pairwise targets P(i=a, j=b).

    Returns:
      A: (K, M)
      t: (K,)
      w: (K,) weights (default 1.0)
    """
    keys = list(targets.keys())
    A = np.zeros((len(keys), index.size), dtype=float)
    t = np.array([float(targets[k]) for k in keys], dtype=float)

    fname_to_pos = {n: i for i, n in enumerate(index.factor_names)}
    for r, (i, a, j, b) in enumerate(keys):
        pi = fname_to_pos[str(i)]
        pj = fname_to_pos[str(j)]
        a = str(a)
        b = str(b)
        for idx in range(index.size):
            scen = index.scenario_at(idx)
            if scen.assignment[pi] == a and scen.assignment[pj] == b:
                A[r, idx] = 1.0
    w = np.ones(len(keys), dtype=float)
    return A, t, w


def fit_joint_direct(
    *,
    index: ScenarioIndex,
    marginals: Marginals,
    multipliers: Optional[Multipliers] = None,
    kl_weight: float = 0.0,
    weight_by_target: bool = True,
    random_seed: Optional[int] = None,
    solver_maxiter: int = 2_000,
) -> np.ndarray:
    """
    Fit a dense joint distribution p over all scenarios.

    Objective (default):
      minimize Σ_k w_k (A_k p - t_k)^2  + kl_weight * KL(p || q)

    subject to:
      p >= 0, sum(p)=1, and exact marginal matching.
    """
    multipliers = multipliers or {}
    validate_marginals(index.factors, marginals)

    # Fast path: if there are no multiplier constraints, the independent baseline implied by marginals is already a valid joint distribution.
    if not multipliers:
        return _product_baseline(index, marginals)

    targets = multiplier_pairwise_targets(marginals, multipliers)
    A, t, w = _build_pairwise_matrix(index, targets)
    if weight_by_target and len(w) > 0:
        # Relative errors are weighted more evenly across different scales.
        w = 1.0 / np.maximum(1e-12, np.abs(t))
        # Extreme weights are capped to avoid numerical overflow when targets are tiny.
        w = np.minimum(w, 1e8)

    C, d = _build_linear_constraints(index, marginals)
    lin = LinearConstraint(C, d, d)

    kl_weight = float(kl_weight)
    use_kl = kl_weight > 0.0
    if use_kl:
        q = _product_baseline(index, marginals)
        if np.any(q <= 0.0):
            raise ValueError("KL baseline has zeros; cannot use KL regularization")
        eps = 1e-12
        bounds = Bounds(eps * np.ones(index.size), np.ones(index.size))
    else:
        q = np.ones(index.size, dtype=float) / float(index.size)
        bounds = Bounds(np.zeros(index.size), np.ones(index.size))

    rng = np.random.default_rng(int(random_seed) if random_seed is not None else 0)

    # Feasible initialization: baseline q generally matches marginals by construction.
    # The product of marginals is used so marginals match exactly; sum-to-one is also satisfied.
    x0 = _product_baseline(index, marginals)
    if not use_kl:
        # A small random perturbation is applied to help trust-constr avoid some degenerate Hessians.
        noise = rng.normal(scale=1e-6, size=index.size)
        x0 = np.clip(x0 + noise, 0.0, None)
        x0 = x0 / float(np.sum(x0))

    w = np.asarray(w, dtype=float)
    t = np.asarray(t, dtype=float)
    A = np.asarray(A, dtype=float)

    def fun(p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        # Weighted least squares are used in residual form to avoid forming large quadratic matrices.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            r = A @ p - t
        if not np.all(np.isfinite(r)):
            return 1e100
        val = float(np.sum(w * r * r))
        if use_kl:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
                kl = float(np.sum(p * (np.log(p) - np.log(q))))
            if not np.isfinite(kl):
                return 1e100
            val += float(kl_weight) * kl
        return float(val)

    def jac(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            r = A @ p - t
        if not np.all(np.isfinite(r)):
            return np.zeros(index.size, dtype=float)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            grad = (2.0 * (A.T @ (w * r))).astype(float)
        if use_kl:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
                grad = grad + float(kl_weight) * (np.log(p) - np.log(q) + 1.0)
        return grad

    res = minimize(
        fun,
        x0,
        method="trust-constr",
        jac=jac,
        constraints=[lin],
        bounds=bounds,
        options={
            "maxiter": int(solver_maxiter),
            "verbose": 0,
            "gtol": 1e-10,
            "xtol": 1e-12,
        },
    )
    if not res.success:
        raise RuntimeError(f"Direct fit failed: {res.message}")

    p = np.asarray(res.x, dtype=float)
    # Numerical cleanliness is enforced.
    p[p < 0.0] = 0.0
    s = float(np.sum(p))
    if s <= 0.0:
        raise RuntimeError("Direct fit returned degenerate distribution")
    p = p / s
    return p

