"""
Transformation matrix computation for scenario transition analysis.

This module provides functionality to build transformation matrices showing
which perturbations cause transitions from one scenario to another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cib.core import CIBMatrix, Scenario
from cib.succession import GlobalSuccession, SuccessionOperator
from cib.shocks import ShockModel, ShockAwareGlobalSuccession
from cib.uncertainty import UncertainCIBMatrix
from cib.example_data import sample_dynamic_shocks_ar1


@dataclass
class PerturbationInfo:
    """
    Information about a perturbation that causes a scenario transformation.

    Attributes:
        perturbation_type: Type of perturbation ("structural", "dynamic", "judgment").
        magnitude: Magnitude of the perturbation (type-specific).
        success_rate: Fraction of trials where this perturbation caused the transformation.
        details: Type-specific parameters (sigma, tau, rho, etc.).
    """

    perturbation_type: str
    magnitude: float
    success_rate: float
    details: Dict[str, Any]


@dataclass
class TransformationMatrix:
    """
    Transformation matrix showing scenario-to-scenario transitions.

    Attributes:
        scenarios: List of scenarios (rows/columns of the matrix).
        transformations: Dictionary mapping (source_scenario, target_scenario) tuples
            to lists of PerturbationInfo objects that cause that transformation.
        summary_stats: Summary statistics about the transformation matrix.
    """

    scenarios: List[Scenario]
    transformations: Dict[Tuple[Scenario, Scenario], List[PerturbationInfo]]
    summary_stats: Dict[str, Any]


def _hamming_distance(scenario_a: Scenario, scenario_b: Scenario) -> int:
    """
    Compute Hamming distance between two scenarios.

    Args:
        scenario_a: First scenario.
        scenario_b: Second scenario.

    Returns:
        Number of descriptors that differ between the scenarios.
    """
    return int(
        sum(
            x != y
            for x, y in zip(scenario_a.to_indices(), scenario_b.to_indices())
        )
    )


def _scenarios_match(
    scenario_a: Scenario, scenario_b: Scenario, max_hamming: int = 1
) -> bool:
    """
    Check if two scenarios match (equal or close within Hamming distance).

    Args:
        scenario_a: First scenario.
        scenario_b: Second scenario.
        max_hamming: Maximum Hamming distance for "close" match. Defaults to 1.

    Returns:
        True if scenarios are equal or within max_hamming distance.
    """
    if scenario_a == scenario_b:
        return True
    return _hamming_distance(scenario_a, scenario_b) <= max_hamming


class TransformationMatrixBuilder:
    """
    Builder for transformation matrices from scenario sets.

    Tests different perturbation types to determine which perturbations
    cause transitions between scenarios.
    """

    def __init__(
        self,
        base_matrix: CIBMatrix,
        *,
        succession_operator: Optional[SuccessionOperator] = None,
    ) -> None:
        """
        Initialize transformation matrix builder.

        Args:
            base_matrix: Base CIB matrix for analysis.
            succession_operator: Succession operator to use. Defaults to GlobalSuccession.
        """
        self.base_matrix = base_matrix
        if succession_operator is None:
            succession_operator = GlobalSuccession()
        self.succession_operator = succession_operator

    def build_matrix(
        self,
        scenarios: List[Scenario],
        *,
        perturbation_types: Optional[List[str]] = None,
        n_trials_per_pair: int = 100,
        structural_sigma_values: Optional[List[float]] = None,
        dynamic_tau_values: Optional[List[float]] = None,
        dynamic_rho: float = 0.5,
        judgment_sigma_scale_values: Optional[List[float]] = None,
        seed: Optional[int] = None,
        max_iterations: int = 1000,
    ) -> TransformationMatrix:
        """
        Build transformation matrix for given scenarios.

        For each scenario pair (i, j), tests different perturbation types
        to determine which perturbations cause transitions from i to j.

        Args:
            scenarios: List of scenarios to analyze.
            perturbation_types: Types of perturbations to test. Defaults to
                ["structural", "dynamic", "judgment"].
            n_trials_per_pair: Number of trials per scenario pair per perturbation type.
            structural_sigma_values: Sigma values to test for structural shocks.
            dynamic_tau_values: Tau values to test for dynamic shocks.
            dynamic_rho: AR(1) persistence parameter for dynamic shocks in [-1, 1].
            judgment_sigma_scale_values: Sigma scale values to test for judgment uncertainty.
            seed: Random seed for reproducibility.
            max_iterations: Maximum iterations for succession.

        Returns:
            TransformationMatrix with all detected transformations.

        Raises:
            ValueError: If scenarios list is empty or invalid parameters provided.
        """
        if not scenarios:
            raise ValueError("scenarios list cannot be empty")
        if int(n_trials_per_pair) <= 0:
            raise ValueError("n_trials_per_pair must be positive")
        if float(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        if float(dynamic_rho) < -1.0 or float(dynamic_rho) > 1.0:
            raise ValueError("dynamic_rho must be in [-1, 1]")

        if perturbation_types is None:
            perturbation_types = ["structural", "dynamic", "judgment"]

        if structural_sigma_values is None:
            structural_sigma_values = [0.1, 0.15, 0.2, 0.25, 0.3]

        if dynamic_tau_values is None:
            dynamic_tau_values = [0.1, 0.2, 0.3, 0.4, 0.5]

        if judgment_sigma_scale_values is None:
            judgment_sigma_scale_values = [0.5, 1.0, 1.5, 2.0]

        base_seed = int(seed) if seed is not None else 0

        transformations: Dict[Tuple[Scenario, Scenario], List[PerturbationInfo]] = {}

        # Each scenario pair is tested.
        total_pairs = len(scenarios) * len(scenarios)
        pair_count = 0

        for source_idx, source in enumerate(scenarios):
            for target_idx, target in enumerate(scenarios):
                pair_count += 1
                if source == target:
                    continue

                # Structural shock testing is performed.
                if "structural" in perturbation_types:
                    for sigma in structural_sigma_values:
                        info = self._test_structural_shock_transformation(
                            source=source,
                            target=target,
                            sigma=sigma,
                            n_trials=n_trials_per_pair,
                            seed=base_seed + pair_count * 1000,
                            max_iterations=max_iterations,
                        )
                        if info is not None:
                            key = (source, target)
                            if key not in transformations:
                                transformations[key] = []
                            transformations[key].append(info)

                # Dynamic shock testing is performed.
                if "dynamic" in perturbation_types:
                    for tau in dynamic_tau_values:
                        info = self._test_dynamic_shock_transformation(
                            source=source,
                            target=target,
                            tau=tau,
                            rho=float(dynamic_rho),
                            n_trials=n_trials_per_pair,
                            seed=base_seed + pair_count * 2000,
                            max_iterations=max_iterations,
                        )
                        if info is not None:
                            key = (source, target)
                            if key not in transformations:
                                transformations[key] = []
                            transformations[key].append(info)

                # Judgment uncertainty testing is performed.
                if "judgment" in perturbation_types:
                    if not isinstance(self.base_matrix, UncertainCIBMatrix):
                        # Judgment uncertainty requires UncertainCIBMatrix.
                        continue

                    for sigma_scale in judgment_sigma_scale_values:
                        info = self._test_judgment_uncertainty_transformation(
                            source=source,
                            target=target,
                            sigma_scale=sigma_scale,
                            n_trials=n_trials_per_pair,
                            seed=base_seed + pair_count * 3000,
                            max_iterations=max_iterations,
                        )
                        if info is not None:
                            key = (source, target)
                            if key not in transformations:
                                transformations[key] = []
                            transformations[key].append(info)

        # Summary statistics are computed.
        summary_stats = {
            "total_scenarios": len(scenarios),
            "total_pairs_tested": total_pairs,
            "pairs_with_transformations": len(transformations),
            "total_transformations": sum(len(v) for v in transformations.values()),
        }

        return TransformationMatrix(
            scenarios=scenarios,
            transformations=transformations,
            summary_stats=summary_stats,
        )

    def _test_structural_shock_transformation(
        self,
        source: Scenario,
        target: Scenario,
        sigma: float,
        n_trials: int,
        seed: int,
        max_iterations: int,
    ) -> Optional[PerturbationInfo]:
        """
        Test if structural shocks cause transformation from source to target.

        Returns PerturbationInfo if transformation occurs, None otherwise.

        Args:
            source: Source scenario.
            target: Target scenario.
            sigma: Structural shock sigma value.
            n_trials: Number of trials to run.
            seed: Random seed.
            max_iterations: Maximum iterations for succession.

        Returns:
            PerturbationInfo if transformation is successful, None otherwise.
        """
        success_count = 0

        for trial in range(n_trials):
            trial_seed = seed + trial
            shock_model = ShockModel(self.base_matrix)
            shock_model.add_structural_shocks(sigma=sigma)
            shocked_matrix = shock_model.sample_shocked_matrix(trial_seed)

            result = self.succession_operator.find_attractor(
                source, shocked_matrix, max_iterations=max_iterations
            )

            # All cycle states are checked against the target.
            if result.is_cycle:
                cycle = result.attractor
                assert isinstance(cycle, list)
                matched = any(_scenarios_match(s, target) for s in cycle)
            else:
                attractor = result.attractor
                assert isinstance(attractor, Scenario)
                matched = _scenarios_match(attractor, target)

            if matched:
                success_count += 1

        success_rate = float(success_count) / float(n_trials)
        if success_rate > 0.0:
            return PerturbationInfo(
                perturbation_type="structural",
                magnitude=sigma,
                success_rate=success_rate,
                details={"sigma": sigma},
            )
        return None

    def _test_dynamic_shock_transformation(
        self,
        source: Scenario,
        target: Scenario,
        tau: float,
        rho: float,
        n_trials: int,
        seed: int,
        max_iterations: int,
    ) -> Optional[PerturbationInfo]:
        """
        Test if dynamic shocks cause transformation from source to target.

        Args:
            source: Source scenario.
            target: Target scenario.
            tau: Dynamic shock tau value.
            n_trials: Number of trials to run.
            seed: Random seed.
            max_iterations: Maximum iterations for succession.

        Returns:
            PerturbationInfo if transformation is successful, None otherwise.
        """
        success_count = 0

        for trial in range(n_trials):
            trial_seed = seed + trial

            # AR(1) dynamic shocks are sampled using the repository's reference generator.
            # A single pseudo-period is used here because this is a static (one-step) test.
            dynamic_shocks = sample_dynamic_shocks_ar1(
                descriptors=self.base_matrix.descriptors,
                periods=[0],
                dynamic_tau=float(tau),
                rho=float(rho),
                seed=int(trial_seed),
            )[0]

            # Succession with dynamic shocks is performed.
            op = ShockAwareGlobalSuccession(dynamic_shocks)
            result = op.find_attractor(
                source, self.base_matrix, max_iterations=max_iterations
            )

            # All cycle states are checked against the target.
            if result.is_cycle:
                cycle = result.attractor
                assert isinstance(cycle, list)
                matched = any(_scenarios_match(s, target) for s in cycle)
            else:
                attractor = result.attractor
                assert isinstance(attractor, Scenario)
                matched = _scenarios_match(attractor, target)

            if matched:
                success_count += 1

        success_rate = float(success_count) / float(n_trials)
        if success_rate > 0.0:
            return PerturbationInfo(
                perturbation_type="dynamic",
                magnitude=tau,
                success_rate=success_rate,
                details={"tau": tau, "rho": rho},
            )
        return None

    def _test_judgment_uncertainty_transformation(
        self,
        source: Scenario,
        target: Scenario,
        sigma_scale: float,
        n_trials: int,
        seed: int,
        max_iterations: int,
    ) -> Optional[PerturbationInfo]:
        """
        Test if judgment uncertainty causes transformation from source to target.

        Args:
            source: Source scenario.
            target: Target scenario.
            sigma_scale: Sigma scale for judgment uncertainty.
            n_trials: Number of trials to run.
            seed: Random seed.
            max_iterations: Maximum iterations for succession.

        Returns:
            PerturbationInfo if transformation is successful, None otherwise.
        """
        if not isinstance(self.base_matrix, UncertainCIBMatrix):
            return None

        success_count = 0

        for trial in range(n_trials):
            trial_seed = seed + trial
            sampled_matrix = self.base_matrix.sample_matrix(
                trial_seed, sigma_scale=sigma_scale
            )

            result = self.succession_operator.find_attractor(
                source, sampled_matrix, max_iterations=max_iterations
            )

            # All cycle states are checked against the target.
            if result.is_cycle:
                cycle = result.attractor
                assert isinstance(cycle, list)
                matched = any(_scenarios_match(s, target) for s in cycle)
            else:
                attractor = result.attractor
                assert isinstance(attractor, Scenario)
                matched = _scenarios_match(attractor, target)

            if matched:
                success_count += 1

        success_rate = float(success_count) / float(n_trials)
        if success_rate > 0.0:
            return PerturbationInfo(
                perturbation_type="judgment",
                magnitude=sigma_scale,
                success_rate=success_rate,
                details={"sigma_scale": sigma_scale},
            )
        return None
