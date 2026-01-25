"""
Shock modeling and robustness testing for CIB analysis.

This module provides classes for applying structural shocks to CIB matrices
and testing scenario robustness under perturbations.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.succession import SuccessionOperator

from cib.example_data import (
    sample_dynamic_shocks_ar1,
    sample_dynamic_shocks_ar1_time_varying,
    sample_structural_shock,
    seeds_for_run,
)


class ShockModel:
    """
    Model for structural shocks to CIB matrices.

    Structural shocks represent permanent perturbations to impact values,
    modeling unexpected changes to system relationships.
    """

    def __init__(self, base_matrix: CIBMatrix) -> None:
        """
        Initialize shock model with base matrix.

        Args:
            base_matrix: Original CIB matrix to apply shocks to.
        """
        self.base_matrix = base_matrix
        self.structural_sigma: Optional[float] = None
        self.correlation_matrix: Optional[np.ndarray] = None
        self.structural_dist: str = "normal"
        self.structural_df: Optional[float] = None
        self.structural_jump_prob: float = 0.0
        self.structural_jump_scale: Optional[float] = None
        self.dynamic_tau: Optional[float] = None
        self.dynamic_rho: Optional[float] = None
        self.dynamic_periods: Optional[List[int]] = None
        self.dynamic_innovation_dist: str = "normal"
        self.dynamic_innovation_df: Optional[float] = None
        self.dynamic_jump_prob: float = 0.0
        self.dynamic_jump_scale: Optional[float] = None

    def add_structural_shocks(
        self,
        sigma: float,
        correlation_matrix: Optional[np.ndarray] = None,
        dist: str = "normal",
        df: Optional[float] = None,
        jump_prob: float = 0.0,
        jump_scale: Optional[float] = None,
    ) -> None:
        """
        Configure structural shock parameters.

        Args:
            sigma: Standard deviation for structural shocks.
            correlation_matrix: Optional correlation matrix for shocks.
                If None, shocks are independent.

        Raises:
            ValueError: If sigma is non-positive.
        """
        if sigma <= 0:
            raise ValueError("Sigma must be positive")

        self.structural_sigma = sigma
        self.correlation_matrix = correlation_matrix
        self.structural_dist = str(dist)
        self.structural_df = df
        self.structural_jump_prob = float(jump_prob)
        self.structural_jump_scale = jump_scale

    def add_dynamic_shocks(
        self,
        *,
        periods: Sequence[int],
        tau: float,
        rho: float,
        innovation_dist: str = "normal",
        innovation_df: Optional[float] = None,
        jump_prob: float = 0.0,
        jump_scale: Optional[float] = None,
    ) -> None:
        """
        Configure dynamic shocks (AR(1)) for within-period impact-balance perturbations.

        Args:
            periods: Period labels used in a dynamic simulation.
            tau: Long-run standard deviation of dynamic shocks.
            rho: AR(1) persistence parameter in [-1, 1].
        """
        tau = float(tau)
        rho = float(rho)
        if tau <= 0:
            raise ValueError("tau must be positive")
        if rho < -1.0 or rho > 1.0:
            raise ValueError("rho must be in [-1, 1]")
        self.dynamic_tau = tau
        self.dynamic_rho = rho
        self.dynamic_periods = [int(p) for p in periods]
        self.dynamic_innovation_dist = str(innovation_dist)
        self.dynamic_innovation_df = innovation_df
        self.dynamic_jump_prob = float(jump_prob)
        self.dynamic_jump_scale = jump_scale

    def sample_dynamic_shocks_time_varying(
        self, *, seed: int, tau_by_period: Mapping[int, float]
    ) -> Dict[int, Dict[Tuple[str, str], float]]:
        """
        Sample AR(1) dynamic shocks with a time-varying tau(t).
        """
        if self.dynamic_rho is None or self.dynamic_periods is None:
            raise ValueError("Dynamic shocks not configured. Call add_dynamic_shocks first.")
        return sample_dynamic_shocks_ar1_time_varying(
            descriptors=self.base_matrix.descriptors,
            periods=self.dynamic_periods,
            tau_by_period=tau_by_period,
            rho=self.dynamic_rho,
            seed=int(seed),
            innovation_dist=self.dynamic_innovation_dist,
            innovation_df=self.dynamic_innovation_df,
            jump_prob=self.dynamic_jump_prob,
            jump_scale=self.dynamic_jump_scale,
        )

    def sample_dynamic_shocks(self, seed: int) -> Dict[int, Dict[Tuple[str, str], float]]:
        """
        Sample AR(1) dynamic shocks for the configured periods.
        """
        if self.dynamic_tau is None or self.dynamic_rho is None or self.dynamic_periods is None:
            raise ValueError("Dynamic shocks not configured. Call add_dynamic_shocks first.")
        return sample_dynamic_shocks_ar1(
            descriptors=self.base_matrix.descriptors,
            periods=self.dynamic_periods,
            dynamic_tau=self.dynamic_tau,
            rho=self.dynamic_rho,
            seed=int(seed),
            innovation_dist=self.dynamic_innovation_dist,
            innovation_df=self.dynamic_innovation_df,
            jump_prob=self.dynamic_jump_prob,
            jump_scale=self.dynamic_jump_scale,
        )

    def sample_shocked_matrix(self, seed: int) -> CIBMatrix:
        """
        Sample a CIB matrix with structural shocks applied.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            New CIBMatrix with shocks applied to impact values.

        Raises:
            ValueError: If structural shocks are not configured.
        """
        if self.structural_sigma is None:
            raise ValueError("Structural shocks not configured. Call add_structural_shocks first.")

        impact_keys = []
        for src_desc in self.base_matrix.descriptors:
            for src_state in self.base_matrix.descriptors[src_desc]:
                for tgt_desc in self.base_matrix.descriptors:
                    if src_desc == tgt_desc:
                        continue
                    for tgt_state in self.base_matrix.descriptors[tgt_desc]:
                        key = (src_desc, src_state, tgt_desc, tgt_state)
                        impact_keys.append(key)

        if self.correlation_matrix is None:
            shock_values = sample_structural_shock(
                impacts_keys=impact_keys,
                structural_sigma=self.structural_sigma,
                seed=seed,
                dist=self.structural_dist,
                df=self.structural_df,
                jump_prob=self.structural_jump_prob,
                jump_scale=self.structural_jump_scale,
            )
        else:
            corr = np.asarray(self.correlation_matrix, dtype=float)
            n = len(impact_keys)
            if corr.shape != (n, n):
                raise ValueError(
                    f"correlation_matrix must have shape ({n}, {n}), got {corr.shape}"
                )
            if not np.allclose(np.diag(corr), 1.0):
                raise ValueError("correlation_matrix must have 1.0 on the diagonal")
            cov = (float(self.structural_sigma) ** 2) * corr
            rng = np.random.default_rng(int(seed))
            # Correlated shocks remain Gaussian by construction.
            eps = rng.multivariate_normal(mean=np.zeros(n), cov=cov)
            shock_values = {k: float(eps[i]) for i, k in enumerate(impact_keys)}

        shocked_matrix = CIBMatrix(self.base_matrix.descriptors)
        for key in impact_keys:
            base_value = self.base_matrix.get_impact(
                key[0], key[1], key[2], key[3]
            )
            shock = shock_values.get(key, 0.0)
            new_value = np.clip(base_value + shock, -3.0, 3.0)
            shocked_matrix.set_impact(key[0], key[1], key[2], key[3], new_value)

        return shocked_matrix


class ShockAwareGlobalSuccession(SuccessionOperator):
    """
    Global succession with additive dynamic shocks on impact balances.

    Dynamic shocks are provided as:
      shocks[(descriptor, candidate_state)] -> eta
    and are applied as:
      theta'[descriptor, state] = theta[descriptor, state] + eta
    """

    def __init__(self, shocks: Mapping[Tuple[str, str], float]) -> None:
        self.shocks = {k: float(v) for k, v in shocks.items()}

    def find_successor(self, scenario: Scenario, matrix: CIBMatrix) -> Scenario:
        new_state_dict: Dict[str, str] = {}
        for descriptor, states in matrix.descriptors.items():
            # Perturbed impact scores are computed for each candidate state.
            best_state: Optional[str] = None
            best_score = float("-inf")
            for state in states:
                score = matrix.calculate_impact_score(scenario, descriptor, state)
                score += self.shocks.get((descriptor, state), 0.0)
                if score > best_score:
                    best_score = score
                    best_state = state
            if best_state is None:
                raise ValueError(f"No states found for descriptor '{descriptor}'")
            new_state_dict[descriptor] = best_state

        return Scenario(new_state_dict, matrix)


class RobustnessTester:
    """
    Tests scenario robustness under structural shocks.

    Evaluates how often scenarios remain consistent when impact values
    are perturbed by structural shocks.
    """

    def __init__(
        self,
        matrix: CIBMatrix,
        shock_model: ShockModel,
    ) -> None:
        """
        Initialize robustness tester.

        Args:
            matrix: Base CIB matrix (can be CIBMatrix or UncertainCIBMatrix).
            shock_model: ShockModel configured with shock parameters.
        """
        self.matrix = matrix
        self.shock_model = shock_model

    def test_scenario(
        self,
        scenario: Scenario,
        n_simulations: int = 2000,
        seed: Optional[int] = None,
    ) -> float:
        """
        Test robustness of a single scenario.

        Args:
            scenario: Scenario to test.
            n_simulations: Number of shock simulations to run.
            seed: Random seed for reproducibility.

        Returns:
            Robustness score (fraction of simulations where scenario
            remains consistent), between 0 and 1.
        """
        if seed is None:
            base_seed = np.random.randint(0, 2**31)
        else:
            base_seed = seed

        survives = 0

        for sim_idx in range(n_simulations):
            if seeds_for_run is not None:
                seeds = seeds_for_run(base_seed, sim_idx)
                shock_seed = seeds["structural_shock_seed"]
            else:
                shock_seed = base_seed + sim_idx

            shocked_matrix = self.shock_model.sample_shocked_matrix(shock_seed)
            is_consistent = ConsistencyChecker.check_consistency(
                scenario, shocked_matrix
            )

            if is_consistent:
                survives += 1

        return survives / n_simulations

    def test_scenarios(
        self,
        scenarios: List[Scenario],
        n_simulations: int = 2000,
        seed: Optional[int] = None,
    ) -> Dict[Scenario, float]:
        """
        Test robustness of multiple scenarios.

        Args:
            scenarios: List of scenarios to test.
            n_simulations: Number of shock simulations per scenario.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary mapping scenarios to their robustness scores.
        """
        scores: Dict[Scenario, float] = {}

        for scenario in scenarios:
            score = self.test_scenario(scenario, n_simulations, seed)
            scores[scenario] = score

        return scores

    def rank_by_robustness(
        self, scores: Dict[Scenario, float]
    ) -> List[Tuple[Scenario, float]]:
        """
        Rank scenarios by robustness score.

        Args:
            scores: Dictionary mapping scenarios to robustness scores.

        Returns:
            List of (scenario, score) tuples, sorted by score in
            descending order.
        """
        ranked = list(scores.items())
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
