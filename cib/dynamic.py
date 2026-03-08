"""
Dynamic (multi-period) CIB simulation framework (simulation-first).

This module implements a practical dynamic CIB mode:
  - simulate discrete paths across a small number of periods,
  - optionally sample uncertain CIMs per run (Monte Carlo ensemble),
  - optionally apply threshold-triggered CIM modifiers,
  - optionally evolve cyclic descriptors between periods via transition matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from cib.constraints import ConstraintIndex, ConstraintSpec
from cib.core import CIBMatrix, Scenario
from cib.example_data import seeds_for_run
from cib.succession import ConstrainedGlobalSuccession, GlobalSuccession, SuccessionOperator
from cib.cyclic import CyclicDescriptor
from cib.pathway import TransformationPathway
from cib.threshold import ThresholdRule


class _LockedSuccessionOperator(SuccessionOperator):
    """
    Wrapper that prevents selected descriptors from being updated by succession.

    DynamicCIB uses CyclicDescriptor transitions to evolve some descriptors
    between periods. Those descriptors represent exogenous/inertial dynamics and
    should remain fixed during within-period succession, otherwise the successor
    step will immediately overwrite the cyclic transition.
    """

    def __init__(self, inner: SuccessionOperator, locked: Dict[str, str]) -> None:
        """
        Initialize locked succession operator.

        Args:
            inner: Base succession operator to wrap.
            locked: Dictionary mapping descriptor names to state values that
                should remain fixed during succession.
        """
        self.inner = inner
        self.locked = dict(locked)

    def find_successor(self, scenario: Scenario, matrix: CIBMatrix) -> Scenario:
        """
        Find successor scenario while preserving locked descriptor states.

        Args:
            scenario: Current scenario to find successor for.
            matrix: CIB matrix for computing impacts.

        Returns:
            Successor scenario with locked descriptors preserved at their
            specified values.
        """
        nxt = self.inner.find_successor(scenario, matrix)
        if not self.locked:
            return nxt
        state = nxt.to_dict()
        for d, v in self.locked.items():
            if d in state:
                state[d] = v
        return Scenario(state, matrix)


class _ConstraintAwareSuccessionOperator(SuccessionOperator):
    """
    Wrapper that enforces feasibility on successors returned by an inner operator.

    In dynamic workflows, repair must preserve any descriptor states that were
    fixed for the current period (for example cyclic/exogenous descriptors).
    """

    def __init__(
        self,
        inner: SuccessionOperator,
        constraint_index: ConstraintIndex,
        *,
        constraint_mode: Literal["strict", "repair"],
        constrained_top_k: int,
        constrained_backtracking_depth: int,
        locked_states: Optional[Dict[str, str]] = None,
    ) -> None:
        self.inner = inner
        self.constraint_index = constraint_index
        self.constraint_mode = constraint_mode
        self.constrained_top_k = int(constrained_top_k)
        self.constrained_backtracking_depth = int(constrained_backtracking_depth)
        self.locked_states = dict(locked_states or {})

    def find_successor(self, scenario: Scenario, matrix: CIBMatrix) -> Scenario:
        successor = self.inner.find_successor(scenario, matrix)
        z = np.asarray(successor.to_indices(), dtype=np.int64)
        if bool(self.constraint_index.is_full_valid(z)):
            return successor
        if self.constraint_mode == "strict":
            raise ValueError("Constraint infeasibility was encountered during succession")
        repaired = DynamicCIB._repair_to_valid(
            successor,
            matrix,
            self.constraint_index,
            constrained_top_k=self.constrained_top_k,
            constrained_backtracking_depth=self.constrained_backtracking_depth,
            locked_states=self.locked_states,
        )
        if repaired is None:
            raise ValueError("Constraint repair failed during succession")
        return repaired


@dataclass
class DynamicCIB:
    """
    Simulation-first dynamic CIB wrapper.
    """

    base_matrix: CIBMatrix
    periods: List[int]
    threshold_match_policy: Literal["first_match", "all_matches"] = "all_matches"

    def __post_init__(self) -> None:
        """
        Initialize dynamic CIB instance after dataclass creation.

        Raises:
            ValueError: If periods list is empty.
        """
        if not self.periods:
            raise ValueError("periods cannot be empty")
        if self.threshold_match_policy not in {"first_match", "all_matches"}:
            raise ValueError("threshold_match_policy must be 'first_match' or 'all_matches'")
        self.threshold_rules: List[ThresholdRule] = []
        self.cyclic_descriptors: Dict[str, CyclicDescriptor] = {}

    def add_threshold_rule(self, rule: ThresholdRule) -> None:
        """
        Add a threshold rule to modify the CIM conditionally.

        Args:
            rule: Threshold rule to add. Rules are evaluated in order during
                simulation. When multiple rules match, application is controlled
                by `threshold_match_policy`.
        """
        self.threshold_rules.append(rule)

    def add_cyclic_descriptor(self, cyclic: CyclicDescriptor) -> None:
        """
        Add a cyclic descriptor for exogenous/inertial dynamics.

        Args:
            cyclic: Cyclic descriptor defining transition probabilities between
                periods. The descriptor will evolve between periods but remain
                fixed during within-period succession.

        Raises:
            ValueError: If the cyclic descriptor fails validation.
        """
        cyclic.validate()
        self.cyclic_descriptors[cyclic.name] = cyclic

    def _apply_thresholds(self, matrix: CIBMatrix, scenario: Scenario) -> CIBMatrix:
        """
        Apply threshold rules to modify the active CIM based on scenario state.

        Args:
            matrix: Base CIB matrix to potentially modify.
            scenario: Current scenario to evaluate threshold conditions against.

        Returns:
            Modified CIB matrix if any threshold rule matches, otherwise
            returns the original matrix.
        """
        active = matrix
        for rule in self.threshold_rules:
            if rule.condition(scenario):
                active = rule.modifier(active)
                if self.threshold_match_policy == "first_match":
                    break
        return active

    def _apply_cyclic_transitions(
        self, scenario_dict: Dict[str, str], rng: np.random.Generator
    ) -> Dict[str, str]:
        """
        Apply cyclic descriptor transitions to evolve exogenous variables.

        Args:
            scenario_dict: Current scenario state as dictionary.
            rng: Random number generator for sampling transitions.

        Returns:
            Updated scenario dictionary with cyclic descriptors evolved to
            their next states according to transition probabilities.

        Raises:
            ValueError: If a cyclic descriptor is missing from the scenario.
        """
        out = dict(scenario_dict)
        for name, cyclic in self.cyclic_descriptors.items():
            if name not in out:
                raise ValueError(f"Cyclic descriptor {name!r} missing from scenario")
            out[name] = cyclic.sample_next(out[name], rng)
        return out

    @staticmethod
    def _is_constraint_valid(scenario: Scenario, cidx: Optional[ConstraintIndex]) -> bool:
        if cidx is None:
            return True
        z = np.asarray(scenario.to_indices(), dtype=np.int64)
        return bool(cidx.is_full_valid(z))

    @staticmethod
    def _repair_to_valid(
        scenario: Scenario,
        matrix: CIBMatrix,
        cidx: ConstraintIndex,
        *,
        constrained_top_k: int,
        constrained_backtracking_depth: int,
        locked_states: Optional[Dict[str, str]] = None,
    ) -> Optional[Scenario]:
        """
        Repair a scenario to feasibility while preserving any locked states.

        DynamicCIB uses this helper for initial-state repair, post-cyclic
        repair, post-attractor repair, and equilibrium repair so the same
        feasibility contract applies throughout the simulation.
        """
        helper = ConstrainedGlobalSuccession(
            cidx,
            constraint_mode="repair",
            constrained_top_k=int(constrained_top_k),
            constrained_backtracking_depth=int(constrained_backtracking_depth),
        )
        return helper.repair_to_valid(
            scenario, matrix, locked_states=locked_states
        )

    def _locked_states_for_period(self, scenario_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Return descriptor states that must remain fixed during the current period.

        Cyclic descriptors are treated as exogenous/inertial once a period
        begins: succession and any feasibility recovery may react to them,
        but may not rewrite them until the next between-period transition.
        """
        return {
            name: scenario_dict[name]
            for name in self.cyclic_descriptors.keys()
            if name in scenario_dict
        }

    @staticmethod
    def _uses_builtin_global_succession(operator: SuccessionOperator) -> bool:
        """
        Return whether the operator is the built-in `GlobalSuccession`.

        Some dynamic features replace the supplied operator with specialised
        global operators. Those modes therefore accept only the concrete
        built-in `GlobalSuccession` path, not subclasses or custom wrappers
        whose semantics would otherwise be silently discarded.
        """
        return type(operator) is GlobalSuccession

    @staticmethod
    def _resolve_constraint_index(
        matrix: CIBMatrix,
        *,
        constraint_index: Optional[ConstraintIndex],
        constraints: Optional[Sequence[ConstraintSpec]],
    ) -> Optional[ConstraintIndex]:
        if constraint_index is not None and constraints is not None:
            raise ValueError("Provide either constraint_index or constraints, not both")
        if constraint_index is not None:
            return constraint_index
        return ConstraintIndex.from_specs(matrix, constraints)

    def simulate_path(
        self,
        *,
        initial: Dict[str, str],
        seed: Optional[int] = None,
        succession_operator: Optional[SuccessionOperator] = None,
        dynamic_shocks_by_period: Optional[Dict[int, Dict[tuple[str, str], float]]] = None,
        judgment_sigma_scale_by_period: Optional[Dict[int, float]] = None,
        structural_sigma: Optional[float] = None,
        structural_seed_base: Optional[int] = None,
        structural_shock_scaling_mode: Literal["additive", "multiplicative_magnitude"] = "additive",
        structural_shock_scaling_alpha: float = 0.0,
        structural_shock_scale_by_descriptor: Optional[Dict[str, float]] = None,
        structural_shock_scale_by_state: Optional[Dict[Tuple[str, str], float]] = None,
        constraints: Optional[Sequence[ConstraintSpec]] = None,
        constraint_index: Optional[ConstraintIndex] = None,
        constraint_mode: Literal["none", "strict", "repair"] = "none",
        constrained_top_k: int = 2,
        constrained_backtracking_depth: int = 2,
        cyclic_infeasible_retries: int = 0,
        first_period_output_mode: Literal["attractor", "initial"] = "attractor",
        max_iterations: int = 1000,
        tie_break: str = "deterministic_first",
        equilibrium_mode: Literal["none", "relax_unshocked"] = "none",
        diagnostics: Optional[Dict[str, List[object]]] = None,
    ) -> TransformationPathway:
        """
        A single discrete pathway across periods is simulated.

        Threshold rules and cyclic descriptors:
            - Cyclic transitions (if configured) are applied at the start of each new period
              (except the first), evolving exogenous/inertial descriptors between periods.
            - Threshold rules are evaluated after any cyclic transitions are applied, using
              the resulting scenario state to determine the active CIM used for within-period
              succession.

        Args:
            initial: Initial scenario as a descriptor -> state mapping.
            seed: Seed used for stochastic elements (cyclic transitions and tie breaks).
            succession_operator: Succession operator used within each period.
                When `dynamic_shocks_by_period` is provided, dynamic shocks are
                currently supported with the built-in `GlobalSuccession`
                operator only.
            dynamic_shocks_by_period: Optional per-period shock field used for within-period
                succession (score perturbations at the descriptor-state level).
                Dynamic shocks are applied via `ShockAwareGlobalSuccession`, so
                custom operators, subclasses, or local succession are not
                currently supported with this option.
            judgment_sigma_scale_by_period: Optional per-period sigma scale used when the base
                matrix supports `sample_matrix(...)`.
            structural_sigma: Optional structural shock magnitude applied to the per-period
                matrix prior to within-period succession.
            structural_seed_base: Optional base seed used for structural shocks.
            structural_shock_scaling_mode: Optional structural shock scaling mode.
                Baseline behaviour is reproduced by `"additive"`.
                `"multiplicative_magnitude"` scales shocks by impact magnitude.
            structural_shock_scaling_alpha: Non-negative scaling strength used when
                `structural_shock_scaling_mode="multiplicative_magnitude"`.
            structural_shock_scale_by_descriptor: Optional non-negative multipliers
                applied by source descriptor for structural shocks.
            structural_shock_scale_by_state: Optional non-negative multipliers
                applied by source (descriptor, state) for structural shocks.
            constraints: Optional constraint specifications compiled to a feasibility index.
            constraint_index: Optional precompiled feasibility index. When provided,
                `constraints` should not be provided.
            constraint_mode: Optional feasibility mode:
                - `"none"` disables feasibility handling (default),
                - `"strict"` raises when infeasibility is encountered,
                - `"repair"` applies bounded repair search while preserving
                  period-locked cyclic descriptors.
            constrained_top_k: Positive top-k cap used during feasibility repair search.
            constrained_backtracking_depth: Non-negative maximum number of descriptors
                allowed to deviate from unconstrained maxima during repair.
            cyclic_infeasible_retries: Non-negative number of retries when a cyclic
                transition produces an infeasible state.
            first_period_output_mode: Output policy for the first period:
                - `"attractor"` records the period attractor (default),
                - `"initial"` records the validated/adjusted initial state.
            max_iterations: Maximum number of succession iterations per period.
            tie_break: Cycle representative selection policy when a cycle is detected.
            equilibrium_mode: Optional equilibrium output mode. When set to
                `"relax_unshocked"`, an unshocked relaxation is performed after the realised
                attractor is selected, and `equilibrium_scenarios` is populated on the returned
                pathway.
            diagnostics: Optional diagnostics sink. When provided, per-period iteration counts
                and cycle flags are appended to the provided lists. When threshold rules are
                configured, the applied rule names are recorded per period.

        Returns:
            A `TransformationPathway` containing the realised per-period scenarios, and optionally
            equilibrium scenarios when `equilibrium_mode` is enabled.

        Raises:
            ValueError: If input parameters are invalid.
            RuntimeError: If succession does not converge within `max_iterations`.
        """
        if succession_operator is None:
            succession_operator = GlobalSuccession()
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tie_break not in {"deterministic_first", "random"}:
            raise ValueError("Unsupported tie_break")
        if equilibrium_mode not in {"none", "relax_unshocked"}:
            raise ValueError("Unsupported equilibrium_mode")
        if structural_shock_scaling_mode not in {"additive", "multiplicative_magnitude"}:
            raise ValueError("Unsupported structural_shock_scaling_mode")
        if float(structural_shock_scaling_alpha) < 0:
            raise ValueError("structural_shock_scaling_alpha must be non-negative")
        if constraint_mode not in {"none", "strict", "repair"}:
            raise ValueError("Unsupported constraint_mode")
        if int(constrained_top_k) <= 0:
            raise ValueError("constrained_top_k must be positive")
        if int(constrained_backtracking_depth) < 0:
            raise ValueError("constrained_backtracking_depth must be non-negative")
        if int(cyclic_infeasible_retries) < 0:
            raise ValueError("cyclic_infeasible_retries must be non-negative")
        if first_period_output_mode not in {"attractor", "initial"}:
            raise ValueError("Unsupported first_period_output_mode")

        cidx = self._resolve_constraint_index(
            self.base_matrix,
            constraint_index=constraint_index,
            constraints=constraints,
        )
        if constraint_mode != "none" and cidx is None:
            raise ValueError("constraint_mode requires constraints or constraint_index")
        if constraint_mode == "repair" and not self._uses_builtin_global_succession(
            succession_operator
        ):
            raise ValueError(
                "constraint_mode='repair' currently supports only the built-in "
                "GlobalSuccession operator"
            )
        if dynamic_shocks_by_period is not None and not self._uses_builtin_global_succession(
            succession_operator
        ):
            raise ValueError(
                "dynamic_shocks_by_period currently supports only the built-in "
                "GlobalSuccession operator"
            )

        rng = np.random.default_rng(seed)

        scenarios: List[Scenario] = []
        equilibrium_scenarios: Optional[List[Scenario]] = (
            [] if equilibrium_mode == "relax_unshocked" else None
        )
        current = Scenario(dict(initial), self.base_matrix)
        if constraint_mode != "none" and cidx is not None:
            if not self._is_constraint_valid(current, cidx):
                if constraint_mode == "strict":
                    raise ValueError("Initial scenario is infeasible under provided constraints")
                repaired_initial = self._repair_to_valid(
                    current,
                    self.base_matrix,
                    cidx,
                    constrained_top_k=int(constrained_top_k),
                    constrained_backtracking_depth=int(constrained_backtracking_depth),
                    locked_states=self._locked_states_for_period(current.to_dict()),
                )
                if repaired_initial is None:
                    raise ValueError("Initial scenario could not be repaired to feasibility")
                current = repaired_initial
        current_state = current.to_dict()

        for period_idx, t in enumerate(self.periods):
            # Cyclic transitions are applied at the start of each new period (except the first).
            if period_idx > 0 and self.cyclic_descriptors:
                prior_state = dict(current_state)
                current_state = self._apply_cyclic_transitions(current_state, rng)
                if constraint_mode != "none" and cidx is not None:
                    trial = Scenario(current_state, self.base_matrix)
                    retries_used = 0
                    while (
                        not self._is_constraint_valid(trial, cidx)
                        and retries_used < int(cyclic_infeasible_retries)
                    ):
                        retries_used += 1
                        current_state = self._apply_cyclic_transitions(prior_state, rng)
                        trial = Scenario(current_state, self.base_matrix)
                    if diagnostics is not None:
                        diagnostics.setdefault("cyclic_infeasible_retries_used", []).append(
                            int(retries_used)
                        )
                    if not self._is_constraint_valid(trial, cidx):
                        if constraint_mode == "strict":
                            raise ValueError(
                                "Cyclic transition produced infeasible state and retries were exhausted"
                            )
                        repaired = self._repair_to_valid(
                            trial,
                            self.base_matrix,
                            cidx,
                            constrained_top_k=int(constrained_top_k),
                            constrained_backtracking_depth=int(constrained_backtracking_depth),
                            locked_states=self._locked_states_for_period(current_state),
                        )
                        if repaired is None:
                            raise ValueError(
                                "Cyclic transition infeasibility could not be repaired"
                            )
                        current_state = repaired.to_dict()
                        if diagnostics is not None:
                            diagnostics.setdefault("constraint_repairs_applied", []).append(
                                "post_cyclic_transition"
                            )

            current = Scenario(current_state, self.base_matrix)

            # Optionally, the CIM is re-sampled per period with a time-varying sigma scale.
            matrix_period: CIBMatrix = self.base_matrix
            if judgment_sigma_scale_by_period is not None:
                scale = float(judgment_sigma_scale_by_period.get(int(t), 1.0))
                if hasattr(self.base_matrix, "sample_matrix"):
                    try:
                        matrix_period = self.base_matrix.sample_matrix(  # type: ignore[attr-defined]
                            int(seed or 0) + 1000 * period_idx, sigma_scale=scale
                        )
                    except TypeError:
                        # Backward compatibility: sampling is performed without sigma_scale.
                        matrix_period = self.base_matrix.sample_matrix(  # type: ignore[attr-defined]
                            int(seed or 0) + 1000 * period_idx
                        )

            if structural_sigma is not None:
                from cib.shocks import ShockModel

                if structural_seed_base is None:
                    structural_seed_base = int(seed or 0) + 50_000
                sm = ShockModel(matrix_period)
                sm.add_structural_shocks(
                    sigma=float(structural_sigma),
                    scaling_mode=structural_shock_scaling_mode,
                    scaling_alpha=float(structural_shock_scaling_alpha),
                    scale_by_descriptor=structural_shock_scale_by_descriptor,
                    scale_by_state=structural_shock_scale_by_state,
                )
                matrix_period = sm.sample_shocked_matrix(
                    int(structural_seed_base) + int(period_idx)
                )

            # Threshold rules are evaluated to select the active CIM for this period.
            # Note: for period_idx > 0, cyclic descriptors (if configured) have already
            # advanced at the start of the period, so thresholds “see” the post-cyclic
            # scenario state.
            if diagnostics is not None and self.threshold_rules:
                applied = [r.name for r in self.threshold_rules if r.condition(current)]
                diagnostics.setdefault("threshold_rules_applied", []).append(list(applied))
            matrix_t = self._apply_thresholds(matrix_period, current)

            op = succession_operator
            locked = self._locked_states_for_period(current_state)
            # If cyclic descriptors are configured, they are treated as exogenous/inertial
            # variables for this period: they evolve via transitions between periods,
            # but remain fixed during within-period succession.
            if locked:
                op = _LockedSuccessionOperator(op, locked)
            if dynamic_shocks_by_period is not None and int(t) in dynamic_shocks_by_period:
                from cib.shocks import ShockAwareGlobalSuccession

                op = ShockAwareGlobalSuccession(dynamic_shocks_by_period[int(t)])
                if locked:
                    op = _LockedSuccessionOperator(op, locked)
            if constraint_mode != "none" and cidx is not None:
                op = _ConstraintAwareSuccessionOperator(
                    op,
                    cidx,
                    constraint_mode=constraint_mode,  # type: ignore[arg-type]
                    constrained_top_k=int(constrained_top_k),
                    constrained_backtracking_depth=int(constrained_backtracking_depth),
                    locked_states=locked,
                )

            result = op.find_attractor(
                current, matrix_t, max_iterations=max_iterations
            )
            if diagnostics is not None:
                diagnostics.setdefault("iterations", []).append(int(result.iterations))
                diagnostics.setdefault("is_cycle", []).append(bool(result.is_cycle))

            chosen: Scenario
            if result.is_cycle:
                cycle = result.attractor
                assert isinstance(cycle, list)
                if tie_break == "deterministic_first":
                    chosen = cycle[0]
                else:
                    chosen = cycle[int(rng.integers(0, len(cycle)))]
            else:
                attractor = result.attractor
                assert isinstance(attractor, Scenario)
                chosen = attractor

            if constraint_mode != "none" and cidx is not None:
                if not self._is_constraint_valid(chosen, cidx):
                    if constraint_mode == "strict":
                        raise ValueError("Selected attractor is infeasible under constraints")
                    repaired_chosen = self._repair_to_valid(
                        chosen,
                        matrix_t,
                        cidx,
                        constrained_top_k=int(constrained_top_k),
                        constrained_backtracking_depth=int(constrained_backtracking_depth),
                        locked_states=locked,
                    )
                    if repaired_chosen is None:
                        raise ValueError("Selected attractor could not be repaired")
                    chosen = repaired_chosen
                    if diagnostics is not None:
                        diagnostics.setdefault("constraint_repairs_applied", []).append(
                            "post_attractor_selection"
                        )
            if period_idx == 0 and first_period_output_mode == "initial":
                scenarios.append(Scenario(dict(current_state), matrix_t))
            else:
                scenarios.append(chosen)
                current_state = chosen.to_dict()
            if period_idx == 0 and first_period_output_mode == "initial":
                current_state = chosen.to_dict()

            if equilibrium_scenarios is not None:
                # An unshocked relaxation is performed to obtain a matrix-consistent
                # equilibrium scenario for the active period matrix.
                eq_op: SuccessionOperator = succession_operator
                if locked:
                    eq_op = _LockedSuccessionOperator(eq_op, locked)
                eq_start = Scenario(chosen.to_dict(), matrix_t)
                eq_result = eq_op.find_attractor(
                    eq_start, matrix_t, max_iterations=max_iterations
                )
                if diagnostics is not None:
                    diagnostics.setdefault("equilibrium_iterations", []).append(int(eq_result.iterations))
                    diagnostics.setdefault("equilibrium_is_cycle", []).append(bool(eq_result.is_cycle))
                if eq_result.is_cycle:
                    eq_cycle = eq_result.attractor
                    assert isinstance(eq_cycle, list)
                    eq_chosen = eq_cycle[0]
                else:
                    eq_attractor = eq_result.attractor
                    assert isinstance(eq_attractor, Scenario)
                    eq_chosen = eq_attractor
                if constraint_mode != "none" and cidx is not None:
                    if not self._is_constraint_valid(eq_chosen, cidx):
                        if constraint_mode == "strict":
                            raise ValueError("Equilibrium scenario is infeasible under constraints")
                        repaired_eq = self._repair_to_valid(
                            eq_chosen,
                            matrix_t,
                            cidx,
                            constrained_top_k=int(constrained_top_k),
                            constrained_backtracking_depth=int(constrained_backtracking_depth),
                            locked_states=locked,
                        )
                        if repaired_eq is None:
                            raise ValueError("Equilibrium scenario could not be repaired")
                        eq_chosen = repaired_eq
                        if diagnostics is not None:
                            diagnostics.setdefault("constraint_repairs_applied", []).append(
                                "post_equilibrium_selection"
                            )
                equilibrium_scenarios.append(eq_chosen)

        eq_out = tuple(equilibrium_scenarios) if equilibrium_scenarios is not None else None
        return TransformationPathway(
            periods=tuple(int(t) for t in self.periods),
            scenarios=tuple(scenarios),
            equilibrium_scenarios=eq_out,
        )

    def simulate_ensemble(
        self,
        *,
        initial: Dict[str, str],
        n_runs: int,
        base_seed: int,
        structural_sigma: Optional[float] = None,
        structural_shock_scaling_mode: Literal["additive", "multiplicative_magnitude"] = "additive",
        structural_shock_scaling_alpha: float = 0.0,
        structural_shock_scale_by_descriptor: Optional[Dict[str, float]] = None,
        structural_shock_scale_by_state: Optional[Dict[Tuple[str, str], float]] = None,
        dynamic_tau: Optional[float] = None,
        dynamic_shock_scale_by_descriptor: Optional[Dict[str, float]] = None,
        dynamic_shock_scale_by_state: Optional[Dict[Tuple[str, str], float]] = None,
        dynamic_tau_growth: float = 0.0,
        judgment_sigma_growth: float = 0.0,
        dynamic_rho: float = 0.5,
        constraints: Optional[Sequence[ConstraintSpec]] = None,
        constraint_index: Optional[ConstraintIndex] = None,
        constraint_mode: Literal["none", "strict", "repair"] = "none",
        constrained_top_k: int = 2,
        constrained_backtracking_depth: int = 2,
        cyclic_infeasible_retries: int = 0,
        first_period_output_mode: Literal["attractor", "initial"] = "attractor",
        succession_operator: Optional[SuccessionOperator] = None,
        max_iterations: int = 1000,
        equilibrium_mode: Literal["none", "relax_unshocked"] = "none",
    ) -> List[TransformationPathway]:
        """
        An ensemble of pathways is simulated with reproducible seeding.

        If the base matrix supports `sample_matrix(seed)`, each run samples a
        static CIM once (judgment uncertainty). Otherwise, runs reuse base_matrix.

        Stochastic dynamic behaviour (recommended for workshop realism):
          - If dynamic_tau is provided, each run samples AR(1) dynamic shocks per period and
            applies them during within-period succession.
          - If structural_sigma is provided, each run applies a structural shock to the CIM
            before simulating the path.

        Args:
            initial: Initial scenario as a descriptor -> state mapping.
            n_runs: Number of Monte Carlo runs.
            base_seed: Base seed used to generate per-run seeds.
            structural_sigma: Optional structural shock magnitude applied per run.
            structural_shock_scaling_mode: Optional structural shock scaling mode.
                Baseline behaviour is reproduced by `"additive"`.
                `"multiplicative_magnitude"` scales shocks by impact magnitude.
            structural_shock_scaling_alpha: Non-negative scaling strength used when
                `structural_shock_scaling_mode="multiplicative_magnitude"`.
            structural_shock_scale_by_descriptor: Optional non-negative multipliers
                applied by source descriptor for structural shocks.
            structural_shock_scale_by_state: Optional non-negative multipliers
                applied by source (descriptor, state) for structural shocks.
            dynamic_tau: Optional long-run scale for AR(1) dynamic shocks.
                Dynamic shocks are currently applied with the built-in
                `GlobalSuccession` operator only.
            dynamic_shock_scale_by_descriptor: Optional non-negative multipliers
                applied by descriptor for dynamic shocks.
            dynamic_shock_scale_by_state: Optional non-negative multipliers
                applied by (descriptor, candidate_state) for dynamic shocks.
            dynamic_tau_growth: Non-negative growth used to increase tau over the horizon.
            judgment_sigma_growth: Non-negative growth used to increase judgment sigma scales
                over the horizon when the base matrix supports sampling.
            dynamic_rho: AR(1) persistence parameter in [-1, 1].
            constraints: Optional constraint specifications compiled to a feasibility index.
            constraint_index: Optional precompiled feasibility index.
            constraint_mode: Optional feasibility mode (`"none"`, `"strict"`, `"repair"`).
                Repair mode preserves period-locked cyclic descriptors and is
                currently supported with the built-in `GlobalSuccession`
                operator only.
            constrained_top_k: Positive top-k cap used during feasibility repair.
            constrained_backtracking_depth: Non-negative repair backtracking depth.
            cyclic_infeasible_retries: Non-negative cyclic infeasibility retry count.
            first_period_output_mode: First-period output policy (`"attractor"` or `"initial"`).
            succession_operator: Succession operator used within each period.
                When `dynamic_tau` is provided, dynamic shocks are currently
                supported with the built-in `GlobalSuccession` operator only.
            max_iterations: Maximum number of succession iterations per period.
            equilibrium_mode: Optional equilibrium output mode passed through to `simulate_path`.

        Returns:
            A list of `TransformationPathway` objects.

        Raises:
            ValueError: If input parameters are invalid.
            RuntimeError: If succession does not converge within `max_iterations`.
        """
        if n_runs <= 0:
            raise ValueError("n_runs must be positive")
        if structural_sigma is not None and float(structural_sigma) <= 0:
            raise ValueError("structural_sigma must be positive if provided")
        if dynamic_tau is not None and float(dynamic_tau) <= 0:
            raise ValueError("dynamic_tau must be positive if provided")
        if float(dynamic_tau_growth) < 0:
            raise ValueError("dynamic_tau_growth must be non-negative")
        if float(judgment_sigma_growth) < 0:
            raise ValueError("judgment_sigma_growth must be non-negative")
        if structural_shock_scaling_mode not in {"additive", "multiplicative_magnitude"}:
            raise ValueError("Unsupported structural_shock_scaling_mode")
        if float(structural_shock_scaling_alpha) < 0:
            raise ValueError("structural_shock_scaling_alpha must be non-negative")
        if constraint_mode not in {"none", "strict", "repair"}:
            raise ValueError("Unsupported constraint_mode")
        if int(constrained_top_k) <= 0:
            raise ValueError("constrained_top_k must be positive")
        if int(constrained_backtracking_depth) < 0:
            raise ValueError("constrained_backtracking_depth must be non-negative")
        if int(cyclic_infeasible_retries) < 0:
            raise ValueError("cyclic_infeasible_retries must be non-negative")
        if first_period_output_mode not in {"attractor", "initial"}:
            raise ValueError("Unsupported first_period_output_mode")
        cidx = self._resolve_constraint_index(
            self.base_matrix,
            constraint_index=constraint_index,
            constraints=constraints,
        )
        if constraint_mode != "none" and cidx is None:
            raise ValueError("constraint_mode requires constraints or constraint_index")
        if succession_operator is None:
            succession_operator = GlobalSuccession()
        if constraint_mode == "repair" and not self._uses_builtin_global_succession(
            succession_operator
        ):
            raise ValueError(
                "constraint_mode='repair' currently supports only the built-in "
                "GlobalSuccession operator"
            )
        if dynamic_tau is not None and not self._uses_builtin_global_succession(
            succession_operator
        ):
            raise ValueError(
                "dynamic_tau currently supports only the built-in "
                "GlobalSuccession operator"
            )

        pathways: List[TransformationPathway] = []
        for m in range(int(n_runs)):
            seeds = seeds_for_run(int(base_seed), int(m))

            # 1) A per-run CIM is started from (judgment uncertainty if available).
            matrix_run: CIBMatrix
            if hasattr(self.base_matrix, "sample_matrix"):
                if float(judgment_sigma_growth) > 0:
                    # The sampling-capable matrix is kept so resampling can be performed per period with
                    # increasing sigma scales (response-driven widening uncertainty).
                    matrix_run = self.base_matrix
                else:
                    # Sampling is performed once per run at baseline scale.
                    try:
                        matrix_run = self.base_matrix.sample_matrix(  # type: ignore[attr-defined]
                            seeds["judgment_uncertainty_seed"], sigma_scale=1.0
                        )
                    except TypeError:
                        matrix_run = self.base_matrix.sample_matrix(  # type: ignore[attr-defined]
                            seeds["judgment_uncertainty_seed"]
                        )
            else:
                matrix_run = self.base_matrix

            # 2) A structural shock is applied to the CIM (optional).
            if structural_sigma is not None and float(judgment_sigma_growth) <= 0:
                from cib.shocks import ShockModel

                sm = ShockModel(matrix_run)
                sm.add_structural_shocks(
                    sigma=float(structural_sigma),
                    scaling_mode=structural_shock_scaling_mode,
                    scaling_alpha=float(structural_shock_scaling_alpha),
                    scale_by_descriptor=structural_shock_scale_by_descriptor,
                    scale_by_state=structural_shock_scale_by_state,
                )
                matrix_run = sm.sample_shocked_matrix(seeds["structural_shock_seed"])

            # 3) Dynamic shocks are sampled (optional) and applied during succession.
            dynamic_shocks_by_period = None
            if dynamic_tau is not None:
                from cib.shocks import ShockModel

                dm = ShockModel(matrix_run)
                dm.add_dynamic_shocks(
                    periods=self.periods,
                    tau=float(dynamic_tau),
                    rho=float(dynamic_rho),
                    scale_by_descriptor=dynamic_shock_scale_by_descriptor,
                    scale_by_state=dynamic_shock_scale_by_state,
                )
                if float(dynamic_tau_growth) > 0:
                    # Tau is increased over the horizon: tau(t_i) = tau * (1 + growth * i).
                    tau_by_period = {
                        int(t): float(dynamic_tau) * (1.0 + float(dynamic_tau_growth) * idx)
                        for idx, t in enumerate(self.periods)
                    }
                    dynamic_shocks_by_period = dm.sample_dynamic_shocks_time_varying(
                        seed=seeds["dynamic_shock_seed"], tau_by_period=tau_by_period
                    )
                else:
                    dynamic_shocks_by_period = dm.sample_dynamic_shocks(
                        seeds["dynamic_shock_seed"]
                    )

            dyn = DynamicCIB(matrix_run, periods=list(self.periods))
            dyn.threshold_rules = list(self.threshold_rules)
            dyn.cyclic_descriptors = dict(self.cyclic_descriptors)

            judgment_sigma_scale_by_period = None
            if float(judgment_sigma_growth) > 0:
                judgment_sigma_scale_by_period = {
                    int(t): 1.0 + float(judgment_sigma_growth) * idx
                    for idx, t in enumerate(self.periods)
                }

            p = dyn.simulate_path(
                initial=initial,
                seed=seeds["dynamic_shock_seed"],  # used for cyclic draws/tie breaks
                succession_operator=succession_operator,
                dynamic_shocks_by_period=dynamic_shocks_by_period,
                judgment_sigma_scale_by_period=judgment_sigma_scale_by_period,
                structural_sigma=structural_sigma if float(judgment_sigma_growth) > 0 else None,
                structural_seed_base=seeds["structural_shock_seed"],
                structural_shock_scaling_mode=structural_shock_scaling_mode,
                structural_shock_scaling_alpha=float(structural_shock_scaling_alpha),
                structural_shock_scale_by_descriptor=structural_shock_scale_by_descriptor,
                structural_shock_scale_by_state=structural_shock_scale_by_state,
                constraint_index=cidx,
                constraint_mode=constraint_mode,
                constrained_top_k=int(constrained_top_k),
                constrained_backtracking_depth=int(constrained_backtracking_depth),
                cyclic_infeasible_retries=int(cyclic_infeasible_retries),
                first_period_output_mode=first_period_output_mode,
                max_iterations=max_iterations,
                equilibrium_mode=equilibrium_mode,
            )
            pathways.append(p)

        return pathways

