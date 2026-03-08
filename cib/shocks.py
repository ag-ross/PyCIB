"""
Shock modeling and robustness testing for CIB analysis.

This module provides classes for applying structural shocks to CIB matrices
and testing scenario robustness under perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.rare_events import BinomialInterval, wilson_interval_from_count
from cib.succession import GlobalSuccession, SuccessionOperator

from cib.example_data import (
    sample_dynamic_shocks_ar1,
    sample_dynamic_shocks_ar1_time_varying,
    sample_structural_shock,
    seeds_for_run,
)

StateKey = Tuple[str, str]


@dataclass(frozen=True)
class RobustnessMetrics:
    """
    Extended robustness summary for a scenario under structural shocks.

    Notes:
        - The legacy robustness interpretation is reproduced by `consistency_rate`.
        - Attractor retention and switch rate are assessed against the canonical
          base attractor reached from the tested scenario under the unshocked matrix.
    """

    n_simulations: int
    consistency_rate: float
    consistency_interval: BinomialInterval
    attractor_retention_rate: float
    attractor_retention_interval: BinomialInterval
    switch_rate: float
    switch_rate_interval: BinomialInterval
    mean_hamming_to_base_attractor: float


def calibrate_structural_sigma_from_confidence(
    confidence_codes: Sequence[int], *, method: str = "mean"
) -> float:
    """
    A structural sigma candidate is derived from confidence codes.

    Args:
        confidence_codes: Sequence of confidence codes in {1, 2, 3, 4, 5}.
        method: Aggregation method over confidence-derived sigmas:
            `"mean"`, `"median"`, or `"p75"`.

    Returns:
        Suggested structural sigma (positive float).
    """
    codes = [int(c) for c in confidence_codes]
    if not codes:
        raise ValueError("confidence_codes must be non-empty")
    from cib.uncertainty import ConfidenceMapper

    sigmas = np.asarray(
        [float(ConfidenceMapper.sigma_from_confidence(c)) for c in codes], dtype=float
    )
    if method == "mean":
        return float(np.mean(sigmas))
    if method == "median":
        return float(np.median(sigmas))
    if method == "p75":
        return float(np.quantile(sigmas, 0.75))
    raise ValueError("method must be 'mean', 'median', or 'p75'")


def suggest_dynamic_tau_bounds(
    structural_sigma: float, *, low_ratio: float = 0.5, high_ratio: float = 1.0
) -> Tuple[float, float]:
    """
    A dynamic tau range is suggested from structural sigma.

    This helper is intentionally simple and explicit: tau candidates are
    generated as proportions of the selected structural sigma.
    """
    sigma = float(structural_sigma)
    lo = float(low_ratio)
    hi = float(high_ratio)
    if sigma <= 0:
        raise ValueError("structural_sigma must be positive")
    if lo <= 0 or hi <= 0:
        raise ValueError("low_ratio and high_ratio must be positive")
    if lo > hi:
        raise ValueError("low_ratio must be <= high_ratio")
    return float(sigma * lo), float(sigma * hi)


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
        self.structural_scaling_mode: str = "additive"
        self.structural_scaling_alpha: float = 0.0
        self.structural_scale_by_descriptor: Dict[str, float] = {}
        self.structural_scale_by_state: Dict[StateKey, float] = {}
        self.dynamic_tau: Optional[float] = None
        self.dynamic_rho: Optional[float] = None
        self.dynamic_periods: Optional[List[int]] = None
        self.dynamic_innovation_dist: str = "normal"
        self.dynamic_innovation_df: Optional[float] = None
        self.dynamic_jump_prob: float = 0.0
        self.dynamic_jump_scale: Optional[float] = None
        self.dynamic_scale_by_descriptor: Dict[str, float] = {}
        self.dynamic_scale_by_state: Dict[StateKey, float] = {}

    def _validate_descriptor_scale_map(
        self, scale_map: Mapping[str, float], *, name: str
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for descriptor, multiplier in scale_map.items():
            if descriptor not in self.base_matrix.descriptors:
                raise ValueError(f"{name} contains unknown descriptor '{descriptor}'")
            m = float(multiplier)
            if not np.isfinite(m) or m < 0:
                raise ValueError(f"{name} multipliers must be finite and non-negative")
            out[str(descriptor)] = m
        return out

    def _validate_state_scale_map(
        self, scale_map: Mapping[StateKey, float], *, name: str
    ) -> Dict[StateKey, float]:
        out: Dict[StateKey, float] = {}
        for key, multiplier in scale_map.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(
                    f"{name} keys must be (descriptor, state) tuples, got {key!r}"
                )
            descriptor, state = key
            if descriptor not in self.base_matrix.descriptors:
                raise ValueError(f"{name} contains unknown descriptor '{descriptor}'")
            if state not in self.base_matrix.descriptors[descriptor]:
                raise ValueError(
                    f"{name} contains unknown state '{state}' for descriptor '{descriptor}'"
                )
            m = float(multiplier)
            if not np.isfinite(m) or m < 0:
                raise ValueError(f"{name} multipliers must be finite and non-negative")
            out[(str(descriptor), str(state))] = m
        return out

    def add_structural_shocks(
        self,
        sigma: float,
        correlation_matrix: Optional[np.ndarray] = None,
        dist: str = "normal",
        df: Optional[float] = None,
        jump_prob: float = 0.0,
        jump_scale: Optional[float] = None,
        scaling_mode: str = "additive",
        scaling_alpha: float = 0.0,
        scale_by_descriptor: Optional[Mapping[str, float]] = None,
        scale_by_state: Optional[Mapping[StateKey, float]] = None,
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
        if scaling_mode not in {"additive", "multiplicative_magnitude"}:
            raise ValueError(
                "scaling_mode must be 'additive' or 'multiplicative_magnitude'"
            )
        if float(scaling_alpha) < 0:
            raise ValueError("scaling_alpha must be non-negative")

        self.structural_sigma = sigma
        self.correlation_matrix = correlation_matrix
        self.structural_dist = str(dist)
        self.structural_df = df
        self.structural_jump_prob = float(jump_prob)
        self.structural_jump_scale = jump_scale
        self.structural_scaling_mode = str(scaling_mode)
        self.structural_scaling_alpha = float(scaling_alpha)
        self.structural_scale_by_descriptor = (
            self._validate_descriptor_scale_map(
                scale_by_descriptor, name="scale_by_descriptor"
            )
            if scale_by_descriptor is not None
            else {}
        )
        self.structural_scale_by_state = (
            self._validate_state_scale_map(scale_by_state, name="scale_by_state")
            if scale_by_state is not None
            else {}
        )

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
        scale_by_descriptor: Optional[Mapping[str, float]] = None,
        scale_by_state: Optional[Mapping[StateKey, float]] = None,
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
        self.dynamic_scale_by_descriptor = (
            self._validate_descriptor_scale_map(
                scale_by_descriptor, name="scale_by_descriptor"
            )
            if scale_by_descriptor is not None
            else {}
        )
        self.dynamic_scale_by_state = (
            self._validate_state_scale_map(scale_by_state, name="scale_by_state")
            if scale_by_state is not None
            else {}
        )

    def _dynamic_shock_multiplier(self, descriptor: str, state: str) -> float:
        by_desc = self.dynamic_scale_by_descriptor.get(descriptor, 1.0)
        by_state = self.dynamic_scale_by_state.get((descriptor, state), 1.0)
        return float(by_desc) * float(by_state)

    def _structural_shock_multiplier(self, src_descriptor: str, src_state: str) -> float:
        by_desc = self.structural_scale_by_descriptor.get(src_descriptor, 1.0)
        by_state = self.structural_scale_by_state.get((src_descriptor, src_state), 1.0)
        return float(by_desc) * float(by_state)

    def sample_dynamic_shocks_time_varying(
        self, *, seed: int, tau_by_period: Mapping[int, float]
    ) -> Dict[int, Dict[Tuple[str, str], float]]:
        """
        Sample AR(1) dynamic shocks with a time-varying tau(t).
        """
        if self.dynamic_rho is None or self.dynamic_periods is None:
            raise ValueError("Dynamic shocks not configured. Call add_dynamic_shocks first.")
        shocks = sample_dynamic_shocks_ar1_time_varying(
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
        for t, field in shocks.items():
            shocks[t] = {
                k: float(v) * self._dynamic_shock_multiplier(k[0], k[1])
                for k, v in field.items()
            }
        return shocks

    def sample_dynamic_shocks(self, seed: int) -> Dict[int, Dict[Tuple[str, str], float]]:
        """
        Sample AR(1) dynamic shocks for the configured periods.
        """
        if self.dynamic_tau is None or self.dynamic_rho is None or self.dynamic_periods is None:
            raise ValueError("Dynamic shocks not configured. Call add_dynamic_shocks first.")
        shocks = sample_dynamic_shocks_ar1(
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
        for t, field in shocks.items():
            shocks[t] = {
                k: float(v) * self._dynamic_shock_multiplier(k[0], k[1])
                for k, v in field.items()
            }
        return shocks

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
            effective_shock = float(shock)
            if self.structural_scaling_mode == "multiplicative_magnitude":
                # Scale structural shocks by impact magnitude while preserving sign.
                scale = 1.0 + float(self.structural_scaling_alpha) * (
                    abs(float(base_value)) / 3.0
                )
                effective_shock = float(shock) * float(scale)
            effective_shock *= self._structural_shock_multiplier(key[0], key[1])
            new_value = np.clip(base_value + effective_shock, -3.0, 3.0)
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
        if int(n_simulations) <= 0:
            raise ValueError("n_simulations must be positive")
        if seed is None:
            base_seed = np.random.randint(0, 2**31)
        else:
            base_seed = seed

        survives = 0

        for sim_idx in range(int(n_simulations)):
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

        return survives / int(n_simulations)

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
        if int(n_simulations) <= 0:
            raise ValueError("n_simulations must be positive")
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

    @staticmethod
    def _canonical_attractor(
        result: object,
    ) -> Scenario:
        from cib.succession import AttractorResult

        if not isinstance(result, AttractorResult):
            raise TypeError("result must be an AttractorResult")
        if result.is_cycle:
            cycle = result.attractor
            if not isinstance(cycle, list) or not cycle:
                raise ValueError("Cycle attractor must be a non-empty list of scenarios")
            return min(cycle, key=lambda s: tuple(s.to_indices()))
        attractor = result.attractor
        if not isinstance(attractor, Scenario):
            raise ValueError("Fixed-point attractor must be a Scenario")
        return attractor

    @staticmethod
    def _hamming_distance(a: Scenario, b: Scenario) -> int:
        ia = a.to_indices()
        ib = b.to_indices()
        if len(ia) != len(ib):
            raise ValueError("Scenarios must have matching dimensionality")
        return int(sum(1 for x, y in zip(ia, ib) if x != y))

    def evaluate_scenario(
        self,
        scenario: Scenario,
        *,
        n_simulations: int = 2000,
        seed: Optional[int] = None,
        succession_operator: Optional[SuccessionOperator] = None,
        max_iterations: int = 1000,
        interval_level: float = 0.95,
    ) -> RobustnessMetrics:
        """
        Evaluate extended robustness metrics for a scenario.
        """
        if int(n_simulations) <= 0:
            raise ValueError("n_simulations must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if succession_operator is None:
            succession_operator = GlobalSuccession()

        if seed is None:
            base_seed = np.random.randint(0, 2**31)
        else:
            base_seed = int(seed)

        base_result = succession_operator.find_attractor(
            scenario, self.matrix, max_iterations=max_iterations
        )
        base_attractor = self._canonical_attractor(base_result)

        consistency_hits = 0
        retention_hits = 0
        switch_hits = 0
        hamming_sum = 0.0

        for sim_idx in range(int(n_simulations)):
            if seeds_for_run is not None:
                seeds = seeds_for_run(base_seed, sim_idx)
                shock_seed = seeds["structural_shock_seed"]
            else:
                shock_seed = base_seed + sim_idx

            shocked_matrix = self.shock_model.sample_shocked_matrix(shock_seed)
            is_consistent = ConsistencyChecker.check_consistency(scenario, shocked_matrix)
            if is_consistent:
                consistency_hits += 1

            scenario_on_shocked = Scenario(scenario.to_dict(), shocked_matrix)
            shocked_result = succession_operator.find_attractor(
                scenario_on_shocked,
                shocked_matrix,
                max_iterations=max_iterations,
            )
            shocked_attractor = self._canonical_attractor(shocked_result)
            retained = shocked_attractor == base_attractor
            if retained:
                retention_hits += 1
            else:
                switch_hits += 1
            hamming_sum += float(self._hamming_distance(shocked_attractor, base_attractor))

        n = int(n_simulations)
        consistency_rate = consistency_hits / n
        retention_rate = retention_hits / n
        switch_rate = switch_hits / n
        mean_hamming = hamming_sum / n
        return RobustnessMetrics(
            n_simulations=n,
            consistency_rate=float(consistency_rate),
            consistency_interval=wilson_interval_from_count(
                consistency_hits, n, level=float(interval_level)
            ),
            attractor_retention_rate=float(retention_rate),
            attractor_retention_interval=wilson_interval_from_count(
                retention_hits, n, level=float(interval_level)
            ),
            switch_rate=float(switch_rate),
            switch_rate_interval=wilson_interval_from_count(
                switch_hits, n, level=float(interval_level)
            ),
            mean_hamming_to_base_attractor=float(mean_hamming),
        )
