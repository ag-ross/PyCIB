"""
Transformation matrix computation for scenario transition analysis.

This module provides functionality for building transformation matrices that
show which perturbations cause transitions from one scenario to another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix, Scenario
from cib.pathway import MemoryState
from cib.succession import GlobalSuccession, SuccessionOperator
from cib.shocks import ShockModel, ShockAwareGlobalSuccession
from cib.transition_kernel import (
    CallableTransitionKernel,
    DefaultTransitionKernel,
    TransitionKernel,
)
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
    The Hamming distance between two scenarios is computed.

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
    Two scenarios are checked for match (equal or close within Hamming distance).

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

    Different perturbation types are tested to determine which perturbations
    cause transitions between scenarios.
    """

    def __init__(
        self,
        base_matrix: CIBMatrix,
        *,
        succession_operator: Optional[SuccessionOperator] = None,
    ) -> None:
        """
        The transformation matrix builder is initialised.

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
        extension_mode: str = "scenario",
        regime_name: Optional[str] = None,
        active_matrix: Optional[CIBMatrix] = None,
        perturbation_types: Optional[List[str]] = None,
        n_trials_per_pair: int = 100,
        structural_sigma_values: Optional[List[float]] = None,
        dynamic_tau_values: Optional[List[float]] = None,
        dynamic_rho: float = 0.5,
        judgment_sigma_scale_values: Optional[List[float]] = None,
        seed: Optional[int] = None,
        max_iterations: int = 1000,
        max_hamming_match: int = 1,
    ) -> TransformationMatrix:
        """
        The transformation matrix for given scenarios is built.

        For each scenario pair (i, j), different perturbation types are tested
        to determine which perturbations cause transitions from i to j.

        Args:
            scenarios: List of scenarios to analyse.
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
        analysis_matrix = active_matrix or self.base_matrix

        transformations: Dict[Tuple[Scenario, Scenario], List[PerturbationInfo]] = {}

        # Each scenario pair is tested; only non-self index pairs are counted.
        total_pairs_tested = 0
        pair_count = 0

        for source_idx, source in enumerate(scenarios):
            for target_idx, target in enumerate(scenarios):
                if source_idx == target_idx:
                    continue
                total_pairs_tested += 1
                pair_count += 1

                # Structural shock testing is performed.
                if "structural" in perturbation_types:
                    for sigma in structural_sigma_values:
                        info = self._test_structural_shock_transformation(
                            matrix=analysis_matrix,
                            source=source,
                            target=target,
                            sigma=sigma,
                            n_trials=n_trials_per_pair,
                            seed=base_seed + pair_count * 1000,
                            max_iterations=max_iterations,
                            max_hamming_match=int(max_hamming_match),
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
                            matrix=analysis_matrix,
                            source=source,
                            target=target,
                            tau=tau,
                            rho=float(dynamic_rho),
                            n_trials=n_trials_per_pair,
                            seed=base_seed + pair_count * 2000,
                            max_iterations=max_iterations,
                            max_hamming_match=int(max_hamming_match),
                        )
                        if info is not None:
                            key = (source, target)
                            if key not in transformations:
                                transformations[key] = []
                            transformations[key].append(info)

                # Judgment uncertainty testing is performed.
                if "judgment" in perturbation_types:
                    if not isinstance(analysis_matrix, UncertainCIBMatrix):
                        # Judgment uncertainty requires UncertainCIBMatrix.
                        continue

                    for sigma_scale in judgment_sigma_scale_values:
                        info = self._test_judgment_uncertainty_transformation(
                            matrix=analysis_matrix,
                            source=source,
                            target=target,
                            sigma_scale=sigma_scale,
                            n_trials=n_trials_per_pair,
                            seed=base_seed + pair_count * 3000,
                            max_iterations=max_iterations,
                            max_hamming_match=int(max_hamming_match),
                        )
                        if info is not None:
                            key = (source, target)
                            if key not in transformations:
                                transformations[key] = []
                            transformations[key].append(info)

        # Summary statistics are computed.
        summary_stats = {
            "total_scenarios": len(scenarios),
            "total_pairs_tested": int(total_pairs_tested),
            "pairs_with_transformations": len(transformations),
            "total_transformations": sum(len(v) for v in transformations.values()),
            "extension_mode": str(extension_mode),
            "regime_name": regime_name,
            "analysis_matrix_class": analysis_matrix.__class__.__name__,
            "uses_active_matrix": bool(active_matrix is not None),
            "max_hamming_match": int(max_hamming_match),
        }

        return TransformationMatrix(
            scenarios=scenarios,
            transformations=transformations,
            summary_stats=summary_stats,
        )

    def analyze_path_to_path_transformations(
        self,
        source_path: Sequence[Scenario],
        target_path: Sequence[Scenario],
        *,
        periods: Optional[Sequence[int]] = None,
        active_matrices: Optional[Sequence[CIBMatrix]] = None,
        active_regimes: Optional[Sequence[str]] = None,
        source_memory_states: Optional[Sequence[MemoryState]] = None,
        target_memory_states: Optional[Sequence[MemoryState]] = None,
        perturbation_types: Optional[List[str]] = None,
        n_trials_per_pair: int = 100,
        structural_sigma_values: Optional[List[float]] = None,
        dynamic_tau_values: Optional[List[float]] = None,
        dynamic_rho: float = 0.5,
        judgment_sigma_scale_values: Optional[List[float]] = None,
        seed: Optional[int] = None,
        max_iterations: int = 1000,
        initial_scenario: Optional[Scenario] = None,
        initial_memory_state: Optional[MemoryState] = None,
        transition_kernel: Optional[TransitionKernel | Any] = None,
        first_period_output_mode: Literal["attractor", "initial"] = "attractor",
    ) -> Dict[str, Any]:
        """
        A source path is analysed against a target path period by period.

        The returned payload extends the lightweight path summary with
        per-period perturbation evidence generated under each period's active
        matrix and regime context.
        """

        if len(source_path) != len(target_path):
            raise ValueError("source_path and target_path must have the same length")
        n_periods = int(len(source_path))
        if periods is not None and len(periods) != n_periods:
            raise ValueError("periods must match the path length")
        if active_matrices is not None and len(active_matrices) != n_periods:
            raise ValueError("active_matrices must match the path length")
        if active_regimes is not None and len(active_regimes) != n_periods:
            raise ValueError("active_regimes must match the path length")
        if source_memory_states is not None and len(source_memory_states) != n_periods:
            raise ValueError("source_memory_states must match the path length")
        if target_memory_states is not None and len(target_memory_states) != n_periods:
            raise ValueError("target_memory_states must match the path length")

        summary = summarize_path_to_path_transformations(
            source_path,
            target_path,
            source_memory_states=source_memory_states,
            target_memory_states=target_memory_states,
            active_regimes=active_regimes,
        )

        memory_changed_periods = set(summary["memory_changed_periods"])
        changed_periods = tuple(
            sorted(
                {
                    int(idx)
                    for idx in summary["changed_periods"]
                }
                | {int(idx) for idx in memory_changed_periods}
            )
        )
        period_analyses = []
        supported_periods = []
        period_labels = (
            tuple(int(period) for period in periods)
            if periods is not None
            else tuple(int(memory.period) for memory in target_memory_states)
            if target_memory_states is not None
            else tuple(int(memory.period) for memory in source_memory_states)
            if source_memory_states is not None
            else tuple(int(idx) for idx in range(n_periods))
        )

        active_matrices_for_replay = (
            tuple(active_matrices)
            if active_matrices is not None
            else tuple(self.base_matrix for _ in range(n_periods))
        )
        active_regimes_for_replay = (
            tuple(str(regime) for regime in active_regimes)
            if active_regimes is not None
            else tuple("baseline" for _ in range(n_periods))
        )
        replay_records_by_index: Tuple[object, ...] = ()
        replay_result = None
        if initial_scenario is not None:
            if (
                (source_memory_states is not None or target_memory_states is not None)
                and initial_memory_state is None
            ):
                raise ValueError(
                    "initial_memory_state must be provided when replay is requested for memory-aware paths"
                )
            if transition_kernel is None:
                active_kernel: TransitionKernel = DefaultTransitionKernel(
                    succession_operator=self.succession_operator,
                    max_iterations=int(max_iterations),
                )
            elif isinstance(transition_kernel, TransitionKernel):
                active_kernel = transition_kernel
            else:
                active_kernel = CallableTransitionKernel(transition_kernel)
            replay_result = active_kernel.replay_path(
                initial_scenario=initial_scenario,
                active_matrices=active_matrices_for_replay,
                active_regimes=active_regimes_for_replay,
                initial_memory_state=initial_memory_state,
                periods=period_labels,
                expected_scenarios=target_path,
                expected_memory_states=target_memory_states,
                seed=seed,
                first_period_output_mode=first_period_output_mode,
            )
            replay_records_by_index = tuple(replay_result.records)

        for period_idx in changed_periods:
            active_matrix = (
                active_matrices[period_idx]
                if active_matrices is not None
                else self.base_matrix
            )
            regime_name = (
                str(active_regimes[period_idx])
                if active_regimes is not None
                else None
            )
            pair_key = (source_path[period_idx], target_path[period_idx])
            scenarios_differ = source_path[period_idx] != target_path[period_idx]
            if scenarios_differ:
                matrix_result = self.build_matrix(
                    scenarios=[source_path[period_idx], target_path[period_idx]],
                    extension_mode="path_dependent",
                    regime_name=regime_name,
                    active_matrix=active_matrix,
                    perturbation_types=perturbation_types,
                    n_trials_per_pair=n_trials_per_pair,
                    structural_sigma_values=structural_sigma_values,
                    dynamic_tau_values=dynamic_tau_values,
                    dynamic_rho=dynamic_rho,
                    judgment_sigma_scale_values=judgment_sigma_scale_values,
                    seed=None if seed is None else int(seed) + 5000 * int(period_idx),
                    max_iterations=max_iterations,
                    max_hamming_match=0,
                )
                perturbation_support = pair_key in matrix_result.transformations
                perturbations = tuple(
                    matrix_result.transformations.get(pair_key, ())
                )
                summary_stats = dict(matrix_result.summary_stats)
            else:
                perturbation_support = False
                perturbations = ()
                summary_stats = {
                    "total_scenarios": 2,
                    "total_pairs_tested": 0,
                    "pairs_with_transformations": 0,
                    "total_transformations": 0,
                    "extension_mode": "path_dependent",
                    "regime_name": regime_name,
                    "analysis_matrix_class": active_matrix.__class__.__name__,
                    "uses_active_matrix": bool(active_matrices is not None),
                    "max_hamming_match": 0,
                }
            period_label = int(period_labels[period_idx])
            replay_record = (
                replay_records_by_index[period_idx]
                if period_idx < len(replay_records_by_index)
                else None
            )
            if replay_record is not None:
                scenario_matches = bool(getattr(replay_record, "scenario_matches"))
                memory_matches = getattr(replay_record, "memory_matches")
                prefix_replay_support = (
                    scenario_matches
                    and (
                        memory_matches is None
                        or bool(memory_matches)
                    )
                )
            else:
                prefix_replay_support = None

            source_state_replay_support = None
            source_state_replay_metadata: Dict[str, Any] = {}
            if initial_scenario is not None:
                step_initial_scenario = (
                    source_path[period_idx]
                    if not (period_idx == 0 and first_period_output_mode == "initial")
                    else initial_scenario
                )
                step_initial_memory = (
                    source_memory_states[period_idx]
                    if source_memory_states is not None
                    else initial_memory_state if period_idx == 0 else None
                )
                step_expected_memory = (
                    (target_memory_states[period_idx],)
                    if target_memory_states is not None
                    else None
                )
                step_replay = active_kernel.replay_path(
                    initial_scenario=step_initial_scenario,
                    active_matrices=(active_matrix,),
                    active_regimes=((regime_name or "baseline"),),
                    initial_memory_state=step_initial_memory,
                    initial_history=tuple(source_path[:period_idx]),
                    periods=(period_label,),
                    expected_scenarios=(target_path[period_idx],),
                    expected_memory_states=step_expected_memory,
                    seed=seed,
                    first_period_output_mode=(
                        "initial"
                        if period_idx == 0 and first_period_output_mode == "initial"
                        else "attractor"
                    ),
                )
                step_record = step_replay.records[0]
                source_state_replay_support = bool(step_record.scenario_matches) and (
                    step_record.memory_matches is None or bool(step_record.memory_matches)
                )
                source_state_replay_metadata = dict(step_record.metadata)

            supports_period = (
                bool(source_state_replay_support)
                if source_state_replay_support is not None
                else bool(prefix_replay_support)
                if prefix_replay_support is not None
                else bool(perturbation_support)
            )
            if supports_period:
                supported_periods.append(int(period_idx))
            period_analyses.append(
                {
                    "period_index": int(period_idx),
                    "source": source_path[period_idx].to_dict(),
                    "target": target_path[period_idx].to_dict(),
                    "hamming_distance": _hamming_distance(
                        source_path[period_idx], target_path[period_idx]
                    ),
                    "regime_name": regime_name,
                    "memory_changed": bool(period_idx in memory_changed_periods),
                    "supported_by_replay": prefix_replay_support,
                    "supported_by_perturbations": bool(perturbation_support),
                    "supported_by_source_state_replay": source_state_replay_support,
                    "supported_by_transition_law": bool(supports_period),
                    "perturbations": perturbations,
                    "replay_metadata": (
                        dict(getattr(replay_record, "metadata"))
                        if replay_record is not None
                        else {}
                    ),
                    "source_state_replay_metadata": source_state_replay_metadata,
                    "replay_history_signature": (
                        tuple(getattr(replay_record, "state").history_signature)
                        if replay_record is not None
                        else ()
                    ),
                    "summary_stats": summary_stats,
                }
            )

        return {
            **summary,
            "analysis_mode": "path_to_path",
            "active_matrices_supplied": bool(active_matrices is not None),
            "periods": period_labels,
            "path_replay_available": bool(initial_scenario is not None),
            "path_replay_matches_target": (
                bool(replay_result.all_scenarios_match)
                if replay_result is not None
                else None
            ),
            "path_replay_matches_target_memory": (
                bool(replay_result.all_memory_states_match)
                if replay_result is not None and target_memory_states is not None
                else None
            ),
            "path_replay_matches_full_segment": (
                bool(replay_result.all_scenarios_match)
                and (
                    target_memory_states is None
                    or bool(replay_result.all_memory_states_match)
                )
                if replay_result is not None
                else None
            ),
            "analyzed_periods": changed_periods,
            "supported_changed_periods": tuple(supported_periods),
            "all_changed_periods_supported": (
                len(supported_periods) == len(changed_periods)
                if changed_periods
                else True
            ),
            "period_analyses": tuple(period_analyses),
        }

    def _test_structural_shock_transformation(
        self,
        matrix: CIBMatrix,
        source: Scenario,
        target: Scenario,
        sigma: float,
        n_trials: int,
        seed: int,
        max_iterations: int,
        max_hamming_match: int,
    ) -> Optional[PerturbationInfo]:
        """
        Structural shocks are tested for whether they cause transformation from source to target.

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
            shock_model = ShockModel(matrix)
            shock_model.add_structural_shocks(sigma=sigma)
            shocked_matrix = shock_model.sample_shocked_matrix(trial_seed)

            result = self.succession_operator.find_attractor(
                source, shocked_matrix, max_iterations=max_iterations
            )

            # All cycle states are checked against the target.
            if result.is_cycle:
                cycle = result.attractor
                if not isinstance(cycle, list):
                    raise TypeError("cycle attractor must be a list of scenarios")
                matched = any(
                    _scenarios_match(s, target, max_hamming=int(max_hamming_match))
                    for s in cycle
                )
            else:
                attractor = result.attractor
                if not isinstance(attractor, Scenario):
                    raise TypeError("fixed-point attractor must be a Scenario")
                matched = _scenarios_match(
                    attractor, target, max_hamming=int(max_hamming_match)
                )

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
        matrix: CIBMatrix,
        source: Scenario,
        target: Scenario,
        tau: float,
        rho: float,
        n_trials: int,
        seed: int,
        max_iterations: int,
        max_hamming_match: int,
    ) -> Optional[PerturbationInfo]:
        """
        Dynamic shocks are tested for whether they cause transformation from source to target.

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
                descriptors=matrix.descriptors,
                periods=[0],
                dynamic_tau=float(tau),
                rho=float(rho),
                seed=int(trial_seed),
            )[0]

            # Succession with dynamic shocks is performed.
            op = ShockAwareGlobalSuccession(dynamic_shocks)
            result = op.find_attractor(
                source, matrix, max_iterations=max_iterations
            )

            # All cycle states are checked against the target.
            if result.is_cycle:
                cycle = result.attractor
                if not isinstance(cycle, list):
                    raise TypeError("cycle attractor must be a list of scenarios")
                matched = any(
                    _scenarios_match(s, target, max_hamming=int(max_hamming_match))
                    for s in cycle
                )
            else:
                attractor = result.attractor
                if not isinstance(attractor, Scenario):
                    raise TypeError("fixed-point attractor must be a Scenario")
                matched = _scenarios_match(
                    attractor, target, max_hamming=int(max_hamming_match)
                )

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
        matrix: UncertainCIBMatrix,
        source: Scenario,
        target: Scenario,
        sigma_scale: float,
        n_trials: int,
        seed: int,
        max_iterations: int,
        max_hamming_match: int,
    ) -> Optional[PerturbationInfo]:
        """
        Judgment uncertainty is tested for whether it causes transformation from source to target.

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
        if not isinstance(matrix, UncertainCIBMatrix):
            return None

        success_count = 0

        for trial in range(n_trials):
            trial_seed = seed + trial
            sampled_matrix = matrix.sample_matrix(
                trial_seed, sigma_scale=sigma_scale
            )

            result = self.succession_operator.find_attractor(
                source, sampled_matrix, max_iterations=max_iterations
            )

            # All cycle states are checked against the target.
            if result.is_cycle:
                cycle = result.attractor
                if not isinstance(cycle, list):
                    raise TypeError("cycle attractor must be a list of scenarios")
                matched = any(
                    _scenarios_match(s, target, max_hamming=int(max_hamming_match))
                    for s in cycle
                )
            else:
                attractor = result.attractor
                if not isinstance(attractor, Scenario):
                    raise TypeError("fixed-point attractor must be a Scenario")
                matched = _scenarios_match(
                    attractor, target, max_hamming=int(max_hamming_match)
                )

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


def summarize_path_to_path_transformations(
    source_path: Sequence[Scenario],
    target_path: Sequence[Scenario],
    *,
    source_memory_states: Optional[Sequence[MemoryState]] = None,
    target_memory_states: Optional[Sequence[MemoryState]] = None,
    active_regimes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    A path-to-path transformation is summarised at the sequence level.
    """

    if len(source_path) != len(target_path):
        raise ValueError("source_path and target_path must have the same length")
    if source_memory_states is not None and len(source_memory_states) != len(source_path):
        raise ValueError("source_memory_states must align to source_path")
    if target_memory_states is not None and len(target_memory_states) != len(target_path):
        raise ValueError("target_memory_states must align to target_path")
    if active_regimes is not None and len(active_regimes) != len(source_path):
        raise ValueError("active_regimes must align to source_path")
    per_step_hamming = [
        _hamming_distance(source, target)
        for source, target in zip(source_path, target_path)
    ]
    changed_periods = tuple(
        idx for idx, value in enumerate(per_step_hamming) if int(value) > 0
    )

    memory_changed_periods: Tuple[int, ...] = ()
    if source_memory_states is not None or target_memory_states is not None:
        src_mem = source_memory_states or [
            MemoryState(period=idx, values={}, flags={}, export_label="memory")
            for idx in range(len(source_path))
        ]
        tgt_mem = target_memory_states or [
            MemoryState(period=idx, values={}, flags={}, export_label="memory")
            for idx in range(len(target_path))
        ]
        memory_changed_periods = tuple(
            idx
            for idx, (source_memory, target_memory) in enumerate(zip(src_mem, tgt_mem))
            if (
                int(source_memory.period) != int(target_memory.period)
                or dict(source_memory.values) != dict(target_memory.values)
                or dict(source_memory.flags) != dict(target_memory.flags)
                or str(source_memory.export_label) != str(target_memory.export_label)
            )
        )

    return {
        "n_periods": int(len(source_path)),
        "per_step_hamming": tuple(int(value) for value in per_step_hamming),
        "total_hamming": int(sum(per_step_hamming)),
        "max_step_hamming": int(max(per_step_hamming) if per_step_hamming else 0),
        "changed_periods": changed_periods,
        "memory_active": bool(
            source_memory_states is not None or target_memory_states is not None
        ),
        "memory_changed_periods": memory_changed_periods,
        "active_regimes": (
            tuple(str(regime) for regime in active_regimes)
            if active_regimes is not None
            else ()
        ),
    }


def explain_regime_transformation(
    *,
    source: Scenario,
    target: Scenario,
    regime_name: str,
    active_matrix_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    A lightweight regime-aware explanation payload for a transition is returned.
    """

    return {
        "regime_name": str(regime_name),
        "active_matrix_id": active_matrix_id,
        "hamming_distance": _hamming_distance(source, target),
        "source": source.to_dict(),
        "target": target.to_dict(),
    }
