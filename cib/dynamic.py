"""
Dynamic (multi-period) CIB simulation framework (simulation-first).

This module implements a practical dynamic CIB mode:
  - simulate discrete paths across a small number of periods,
  - optionally sample uncertain CIMs per run (Monte Carlo ensemble),
  - optionally apply threshold-triggered CIM modifiers,
  - optionally evolve cyclic descriptors between periods via transition matrices.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import traceback
import warnings
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.constraints import ConstraintIndex, ConstraintSpec
from cib.core import CIBMatrix, Scenario
from cib.example_data import seeds_for_run
from cib.cyclic import CyclicDescriptor
from cib.path_dependence import AdaptiveCIMUpdater, CallableAdaptiveCIMUpdater
from cib.pathway import (
    ActiveMatrixState,
    ExtendedTransformationPathway,
    MemoryState,
    PathDependentState,
    PerPeriodDisequilibriumMetrics,
    StructuralConsistencyState,
    TransformationPathway,
    TransitionEvent,
)
from cib.regimes import CallableRegimeTransitionRule, RegimeSpec, RegimeTransitionRule
from cib.scoring import attractor_distance, consistent_set_distance, equilibrium_distance, scenario_diagnostics
from cib.structural_consistency import check_structural_consistency
from cib.succession import ConstrainedGlobalSuccession, GlobalSuccession, LocalSuccession, SuccessionOperator
from cib.threshold import ThresholdRule, apply_modifier_copy_on_write
from cib.transition_kernel import CallableTransitionKernel, DefaultTransitionKernel, TransitionKernel


def _format_dist_metric_exc(exc: BaseException) -> str:
    """Single-line exception summary for disequilibrium distance diagnostics."""
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


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
        The locked succession operator is initialised.

        Args:
            inner: Base succession operator to wrap.
            locked: Dictionary mapping descriptor names to state values that
                should remain fixed during succession.
        """
        self.inner = inner
        self.locked = dict(locked)

    def find_successor(self, scenario: Scenario, matrix: CIBMatrix) -> Scenario:
        """
        The successor scenario is found whilst locked descriptor states are preserved.

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
        The dynamic CIB instance is initialised after dataclass creation.

        Raises:
            ValueError: If periods list is empty.
        """
        if not self.periods:
            raise ValueError("periods cannot be empty")
        if self.threshold_match_policy not in {"first_match", "all_matches"}:
            raise ValueError("threshold_match_policy must be 'first_match' or 'all_matches'")
        self.threshold_rules: List[ThresholdRule] = []
        self.cyclic_descriptors: Dict[str, CyclicDescriptor] = {}
        self.regimes: Dict[str, RegimeSpec] = {
            "baseline": RegimeSpec(
                name="baseline",
                base_matrix=self.base_matrix,
                activation_metadata={},
                description="Baseline regime",
            )
        }
        self.regime_transition_rule: Optional[RegimeTransitionRule] = None

    def add_threshold_rule(self, rule: ThresholdRule) -> None:
        """
        A threshold rule is added to modify the CIM conditionally.

        Args:
            rule: Threshold rule to add. Rules are evaluated in order during
                simulation. When multiple rules match, application is controlled
                by `threshold_match_policy`.
        """
        self.threshold_rules.append(rule)

    def add_cyclic_descriptor(self, cyclic: CyclicDescriptor) -> None:
        """
        A cyclic descriptor is added for exogenous/inertial dynamics.

        Args:
            cyclic: Cyclic descriptor defining transition probabilities between
                periods. The descriptor will evolve between periods but remain
                fixed during within-period succession.

        Raises:
            ValueError: If the cyclic descriptor fails validation.
        """
        cyclic.validate()
        self.cyclic_descriptors[cyclic.name] = cyclic

    def add_regime(self, regime: RegimeSpec) -> None:
        """
        A named regime definition is added or replaced.
        """

        self.regimes[str(regime.name)] = regime

    def set_regime_transition_rule(
        self, rule: RegimeTransitionRule | Any
    ) -> None:
        """
        The active regime transition rule is set.
        """

        if isinstance(rule, RegimeTransitionRule):
            self.regime_transition_rule = rule
        else:
            self.regime_transition_rule = CallableRegimeTransitionRule(rule)

    def _apply_thresholds(self, matrix: CIBMatrix, scenario: Scenario) -> CIBMatrix:
        """
        Threshold rules are applied to modify the active CIM based on scenario state.

        Args:
            matrix: Base CIB matrix to potentially modify.
            scenario: Current scenario to evaluate threshold conditions against.

        Returns:
            Modified CIB matrix if any threshold rule matches, otherwise
            the original matrix is returned.
        """
        active = matrix
        for rule in self.threshold_rules:
            if rule.condition(scenario):
                if rule.modifier is None:
                    warnings.warn(
                        (
                            f"Threshold rule {rule.name!r} has target_regime={rule.target_regime!r} "
                            "but no modifier; regime-transition threshold rules are ignored "
                            "in simulate_path() matrix threshold application."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                active, _ = apply_modifier_copy_on_write(rule.modifier, active)
                if self.threshold_match_policy == "first_match":
                    break
        return active

    def _apply_cyclic_transitions(
        self, scenario_dict: Dict[str, str], rng: np.random.Generator
    ) -> Dict[str, str]:
        """
        Cyclic descriptor transitions are applied to evolve exogenous variables.

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
        The descriptor states that must remain fixed during the current period are returned.

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
        Whether the operator is the built-in `GlobalSuccession` is returned.

        Some dynamic features replace the supplied operator with specialised
        global operators. Those modes therefore accept only the concrete
        built-in `GlobalSuccession` path, not subclasses or custom wrappers
        whose semantics would otherwise be silently discarded.
        """
        return type(operator) is GlobalSuccession

    @staticmethod
    def _accepts_dynamic_shocks(operator: SuccessionOperator) -> bool:
        """
        Whether the operator is a built-in type that supports shock-aware wrapping.

        When dynamic_shocks_by_period or dynamic_tau is used, the period operator
        is replaced with ShockAwareGlobalSuccession or ShockAwareLocalSuccession.
        Only the concrete built-in GlobalSuccession and LocalSuccession are supported.
        """
        return type(operator) is GlobalSuccession or type(operator) is LocalSuccession

    @staticmethod
    def _period_shock_operator(
        succession_operator: SuccessionOperator,
        shocks: Mapping[Tuple[str, str], float],
    ) -> SuccessionOperator:
        """Return the shock-aware operator for this period; raises if not supported."""
        from cib.shocks import ShockAwareGlobalSuccession, ShockAwareLocalSuccession

        if type(succession_operator) is GlobalSuccession:
            return ShockAwareGlobalSuccession(shocks)
        if type(succession_operator) is LocalSuccession:
            return ShockAwareLocalSuccession(shocks)
        raise ValueError(
            "dynamic_shocks_by_period supports only the built-in GlobalSuccession "
            "or LocalSuccession operator"
        )

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

    @staticmethod
    def _matrix_identity(matrix: CIBMatrix) -> str:
        """
        A stable identifier for a matrix payload is returned.
        """

        try:
            descriptor_payload = tuple(
                (str(name), tuple(str(state) for state in states))
                for name, states in sorted(matrix.descriptors.items())
            )
            impact_payload = tuple(
                (
                    str(src_desc),
                    str(src_state),
                    str(tgt_desc),
                    str(tgt_state),
                    float(value),
                )
                for (src_desc, src_state, tgt_desc, tgt_state), value in sorted(
                    matrix.iter_impacts()
                )
            )
            digest = hashlib.sha1(
                repr(
                    (
                        matrix.__class__.__name__,
                        descriptor_payload,
                        impact_payload,
                    )
                ).encode("utf-8")
            ).hexdigest()[:12]
            return f"{matrix.__class__.__name__}:{digest}"
        except (TypeError, ValueError, KeyError):
            return (
                f"{matrix.__class__.__name__}:weak_id_fallback:{id(matrix)}"
            )

    @staticmethod
    def _matrix_diff_summary(
        base_matrix: CIBMatrix, active_matrix: CIBMatrix
    ) -> Dict[str, float]:
        """
        The extent to which the active matrix differs from the base matrix is summarised.
        """

        if base_matrix is active_matrix:
            return {
                "n_changed_cells": 0.0,
                "sum_abs_delta": 0.0,
                "max_abs_delta": 0.0,
                "weak_coarse_diff": 0.0,
            }
        changes = 0
        sum_abs_delta = 0.0
        max_abs_delta = 0.0
        try:
            base_impacts = {
                tuple(key): float(value)
                for key, value in base_matrix.iter_impacts()
            }
            active_impacts = {
                tuple(key): float(value)
                for key, value in active_matrix.iter_impacts()
            }
            for key in set(base_impacts) | set(active_impacts):
                base_value = float(base_impacts.get(key, 0.0))
                active_value = float(active_impacts.get(key, 0.0))
                delta = float(active_value) - float(base_value)
                if delta != 0.0:
                    changes += 1
                    sum_abs_delta += abs(delta)
                    max_abs_delta = max(max_abs_delta, abs(delta))
        except (TypeError, ValueError, KeyError):
            # A coarse summary is returned when matrix iteration APIs differ.
            changes = int(base_matrix is not active_matrix)
            return {
                "n_changed_cells": float(changes),
                "sum_abs_delta": float(sum_abs_delta),
                "max_abs_delta": float(max_abs_delta),
                "weak_coarse_diff": 1.0,
            }
        return {
            "n_changed_cells": float(changes),
            "sum_abs_delta": float(sum_abs_delta),
            "max_abs_delta": float(max_abs_delta),
            "weak_coarse_diff": 0.0,
        }

    def _resolve_active_regime(
        self,
        *,
        current_regime: str,
        realized_scenario: Scenario,
        previous_scenarios: Sequence[Scenario],
        memory_state: Optional[MemoryState],
        rng: np.random.Generator,
        regime_transition_rule: Optional[RegimeTransitionRule],
    ) -> Tuple[str, Tuple[TransitionEvent, ...]]:
        """
        The active regime for the current period is resolved.
        """

        if regime_transition_rule is None:
            return str(current_regime), ()
        next_regime, events = regime_transition_rule.resolve_next_regime(
            current_regime=str(current_regime),
            realized_scenario=realized_scenario,
            previous_scenarios=previous_scenarios,
            memory_state=memory_state,
            rng=rng,
        )
        return str(next_regime), tuple(events)

    def _resolve_threshold_regime_transitions(
        self,
        *,
        period: int,
        current_regime: str,
        scenario: Scenario,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[TransitionEvent, ...]]:
        """
        Threshold-triggered regime transitions are resolved before matrix assembly.
        """

        resolved_regime = str(current_regime)
        transition_rules: List[str] = []
        reaffirmation_rules: List[str] = []
        events: List[TransitionEvent] = []
        for rule in self.threshold_rules:
            if rule.target_regime is None or not rule.condition(scenario):
                continue
            target_regime = str(rule.target_regime)
            if target_regime != resolved_regime:
                transition_rules.append(str(rule.name))
                metadata = {
                    "from": resolved_regime,
                    "to": target_regime,
                    "activation_kind": "regime_transition",
                    "threshold_rule": str(rule.name),
                }
                metadata.update(dict(rule.activation_metadata))
                events.append(
                    TransitionEvent(
                        period=int(period),
                        event_type="regime_transition",
                        label=f"{resolved_regime}->{target_regime}",
                        source="threshold_rule",
                        metadata=metadata,
                    )
                )
                resolved_regime = target_regime
            else:
                reaffirmation_rules.append(str(rule.name))
                metadata = {
                    "regime": resolved_regime,
                    "activation_kind": "regime_reaffirmation",
                    "threshold_rule": str(rule.name),
                }
                metadata.update(dict(rule.activation_metadata))
                events.append(
                    TransitionEvent(
                        period=int(period),
                        event_type="threshold_activation",
                        label=str(rule.name),
                        source="threshold_rule",
                        metadata=metadata,
                    )
                )
            if self.threshold_match_policy == "first_match":
                break
        return (
            resolved_regime,
            tuple(transition_rules),
            tuple(reaffirmation_rules),
            tuple(events),
        )

    def _apply_threshold_modifiers(
        self,
        *,
        period: int,
        matrix: CIBMatrix,
        scenario: Scenario,
        regime_name: str,
    ) -> Tuple[CIBMatrix, Tuple[str, ...], Tuple[TransitionEvent, ...]]:
        """
        Threshold-triggered in-regime matrix modifiers are applied.
        """

        active_matrix = matrix
        applied_rules: List[str] = []
        events: List[TransitionEvent] = []
        for rule in self.threshold_rules:
            if rule.modifier is None or not rule.condition(scenario):
                continue
            active_matrix, modifier_returned_distinct_object = apply_modifier_copy_on_write(
                rule.modifier, active_matrix
            )
            applied_rules.append(str(rule.name))
            metadata = {
                "regime": str(regime_name),
                "activation_kind": "modifier",
                "threshold_rule": str(rule.name),
                "modifier_returned_distinct_object": bool(
                    modifier_returned_distinct_object
                ),
            }
            metadata.update(dict(rule.activation_metadata))
            events.append(
                TransitionEvent(
                    period=int(period),
                    event_type="threshold_activation",
                    label=str(rule.name),
                    source="threshold_rule",
                    metadata=metadata,
                )
            )
            if self.threshold_match_policy == "first_match":
                break
        return active_matrix, tuple(applied_rules), tuple(events)

    def _compute_period_disequilibrium(
        self,
        *,
        period: int,
        scenario: Scenario,
        matrix: CIBMatrix,
        succession_operator: SuccessionOperator,
        max_iterations: int,
        seen_consistent_before: bool,
    ) -> PerPeriodDisequilibriumMetrics:
        """
        Per-period disequilibrium metrics for one realised scenario are computed.
        """

        diagnostics = scenario_diagnostics(scenario, matrix)
        descriptor_margins: Dict[str, float] = {}
        for descriptor, states in diagnostics.balances.items():
            chosen_state = diagnostics.chosen_states[descriptor]
            chosen_score = float(states[chosen_state])
            best_alt = float("-inf")
            for state_name, score in states.items():
                if state_name == chosen_state:
                    continue
                best_alt = max(best_alt, float(score))
            descriptor_margins[descriptor] = (
                0.0 if best_alt == float("-inf") else float(chosen_score - best_alt)
            )
        time_to_equilibrium = self._time_to_consistent_set(
            scenario=scenario,
            matrix=matrix,
            succession_operator=succession_operator,
            max_iterations=max_iterations,
        )
        distance_to_consistent_set: Optional[float] = None
        consistent_set_distance_error: Optional[str] = None
        try:
            distance_to_consistent_set = float(
                consistent_set_distance(
                    scenario,
                    matrix,
                    succession_operator=succession_operator,
                    max_iterations=max_iterations,
                )
            )
        except (ValueError, RuntimeError, TypeError, ArithmeticError) as exc:
            consistent_set_distance_error = _format_dist_metric_exc(exc)

        attractor_relation = None
        attractor_distance_error: Optional[str] = None
        try:
            attractor_relation = attractor_distance(
                scenario,
                matrix,
                succession_operator=succession_operator,
                max_iterations=max_iterations,
            )
        except (ValueError, RuntimeError, TypeError, ArithmeticError) as exc:
            attractor_distance_error = _format_dist_metric_exc(exc)

        return PerPeriodDisequilibriumMetrics(
            period=int(period),
            is_consistent=bool(diagnostics.is_consistent),
            consistency_margin=float(diagnostics.consistency_margin),
            descriptor_margins=descriptor_margins,
            brink_descriptors=tuple(diagnostics.brink_descriptors()),
            distance_to_equilibrium=(
                float(attractor_relation.distance_to_attractor)
                if attractor_relation is not None
                else None
            ),
            time_to_equilibrium=time_to_equilibrium,
            entered_consistent_set=(bool(diagnostics.is_consistent) and not seen_consistent_before),
            distance_to_consistent_set=(
                float(distance_to_consistent_set)
                if distance_to_consistent_set is not None
                else None
            ),
            distance_to_attractor=(
                float(attractor_relation.distance_to_attractor)
                if attractor_relation is not None
                else None
            ),
            nearest_attractor_kind=(
                str(attractor_relation.attractor_kind)
                if attractor_relation is not None
                else None
            ),
            attractor_size=(
                int(attractor_relation.attractor_size)
                if attractor_relation is not None
                else None
            ),
            is_on_attractor=(
                bool(attractor_relation.is_on_attractor)
                if attractor_relation is not None
                else None
            ),
            consistent_set_distance_error=consistent_set_distance_error,
            attractor_distance_error=attractor_distance_error,
        )

    @staticmethod
    def _time_to_consistent_set(
        *,
        scenario: Scenario,
        matrix: CIBMatrix,
        succession_operator: SuccessionOperator,
        max_iterations: int,
    ) -> Optional[int]:
        """
        The first successor step at which the consistent set is entered is returned.
        """

        current = scenario
        if bool(scenario_diagnostics(current, matrix).is_consistent):
            return 0
        for step in range(1, int(max_iterations) + 1):
            current = succession_operator.find_successor(current, matrix)
            if bool(scenario_diagnostics(current, matrix).is_consistent):
                return int(step)
        return None

    def _build_active_matrix_state(
        self,
        *,
        period: int,
        regime_name: str,
        base_matrix: CIBMatrix,
        active_matrix: CIBMatrix,
        threshold_rules_applied: Sequence[str],
        structural_labels: Sequence[str],
        judgment_labels: Sequence[str],
        adaptive_labels: Sequence[str],
        threshold_regime_reaffirmations: Sequence[str],
        entered_regime: bool,
        regime_entry_period: Optional[int],
        regime_spell_index: int,
        provenance_labels: Sequence[str],
    ) -> ActiveMatrixState:
        """
        Structured provenance for the active period matrix is built.
        """

        return ActiveMatrixState(
            period=int(period),
            regime_name=str(regime_name),
            base_matrix_id=self._matrix_identity(base_matrix),
            active_matrix_id=self._matrix_identity(active_matrix),
            applied_threshold_rules=tuple(str(label) for label in threshold_rules_applied),
            applied_structural_shocks=tuple(str(label) for label in structural_labels),
            applied_judgment_sampling=tuple(str(label) for label in judgment_labels),
            adaptive_updates=tuple(str(label) for label in adaptive_labels),
            threshold_regime_reaffirmations=tuple(
                str(label) for label in threshold_regime_reaffirmations
            ),
            entered_regime=bool(entered_regime),
            regime_entry_period=(
                int(regime_entry_period)
                if regime_entry_period is not None
                else None
            ),
            regime_spell_index=int(regime_spell_index),
            diff_summary=self._matrix_diff_summary(base_matrix, active_matrix),
            provenance_labels=tuple(str(label) for label in provenance_labels),
        )

    def _compute_structural_consistency_state(
        self,
        *,
        period: int,
        realized_scenario: Scenario,
        regime_name: str,
        memory_state: Optional[MemoryState],
        previous_scenarios: Sequence[Scenario],
        transition_events: Sequence[TransitionEvent],
    ) -> StructuralConsistencyState:
        """
        Structural consistency for one realised period is computed.
        """

        return check_structural_consistency(
            period=int(period),
            realized_scenario=realized_scenario,
            regime_name=str(regime_name),
            memory_state=memory_state,
            previous_scenarios=previous_scenarios,
            transition_events=transition_events,
        )

    @staticmethod
    def _copy_memory_state(memory_state: Optional[MemoryState]) -> Optional[MemoryState]:
        if memory_state is None:
            return None
        return MemoryState(
            period=int(memory_state.period),
            values=copy.deepcopy(memory_state.values),
            flags=copy.deepcopy(memory_state.flags),
            export_label=str(memory_state.export_label),
        )

    @staticmethod
    def _memory_state_changed(
        before: Optional[MemoryState], after: Optional[MemoryState]
    ) -> bool:
        if before is None and after is None:
            return False
        if before is None or after is None:
            return True
        return (
            int(before.period) != int(after.period)
            or dict(before.values) != dict(after.values)
            or dict(before.flags) != dict(after.flags)
            or str(before.export_label) != str(after.export_label)
        )

    def _build_path_dependent_state(
        self,
        *,
        period: int,
        scenario: Scenario,
        regime_name: str,
        active_matrix: CIBMatrix,
        memory_state: Optional[MemoryState],
        previous_scenarios: Sequence[Scenario],
        transition_events: Sequence[TransitionEvent],
    ) -> PathDependentState:
        return PathDependentState(
            period=int(period),
            scenario=scenario,
            regime_name=str(regime_name),
            active_matrix=active_matrix,
            memory_state=self._copy_memory_state(memory_state),
            history_signature=tuple(
                tuple(int(value) for value in item.to_indices())
                for item in tuple(previous_scenarios) + (scenario,)
            ),
            transition_events=tuple(transition_events),
        )

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
        allow_partial: bool = False,
        equilibrium_max_iterations: Optional[int] = None,
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
                When `dynamic_shocks_by_period` is provided, only the built-in
                `GlobalSuccession` or `LocalSuccession` operators are supported.
            dynamic_shocks_by_period: Optional per-period shock field used for within-period
                succession (score perturbations at the descriptor-state level).
                Dynamic shocks are applied via ShockAwareGlobalSuccession or
                ShockAwareLocalSuccession; other operators are not supported.
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
            allow_partial: When True, within-period succession may be capped without raising;
                the last state is used as the period outcome.
            equilibrium_max_iterations: When allow_partial is True and equilibrium is requested,
                this cap is used for the unshocked relaxation only; if None, a default
                (max(max_iterations, 100)) is used so that equilibrium is fully relaxed.
            diagnostics: Optional diagnostics sink. When provided, per-period iteration counts
                and cycle flags are appended to the provided lists. When threshold rules are
                configured, the applied rule names are recorded per period. When allow_partial
                is True, a "converged" list is populated per period.

        Returns:
            A `TransformationPathway` containing the realised per-period scenarios, and optionally
            equilibrium scenarios when `equilibrium_mode` is enabled.

        Raises:
            ValueError: If input parameters are invalid.
            RuntimeError: If succession does not converge within `max_iterations` when
                allow_partial is False.
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
        if dynamic_shocks_by_period is not None and not self._accepts_dynamic_shocks(
            succession_operator
        ):
            raise ValueError(
                "dynamic_shocks_by_period currently supports only the built-in "
                "GlobalSuccession or LocalSuccession operator"
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
                op = self._period_shock_operator(
                    succession_operator, dynamic_shocks_by_period[int(t)]
                )
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
                current, matrix_t, max_iterations=max_iterations,
                allow_partial=allow_partial,
            )
            if diagnostics is not None:
                diagnostics.setdefault("iterations", []).append(int(result.iterations))
                diagnostics.setdefault("is_cycle", []).append(bool(result.is_cycle))
                diagnostics.setdefault("converged", []).append(bool(result.converged))

            chosen: Scenario
            if result.is_cycle:
                cycle = result.attractor
                if not isinstance(cycle, list):
                    raise TypeError("cycle attractor must be a list of scenarios")
                if tie_break == "deterministic_first":
                    chosen = cycle[0]
                else:
                    chosen = cycle[int(rng.integers(0, len(cycle)))]
            else:
                attractor = result.attractor
                if not isinstance(attractor, Scenario):
                    raise TypeError("fixed-point attractor must be a Scenario")
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
                # equilibrium scenario for the active period matrix. The base succession
                # operator is used (no constraint wrapper); the result is then validated
                # and repaired below if constraints are in use.
                eq_op: SuccessionOperator = succession_operator
                if locked:
                    eq_op = _LockedSuccessionOperator(eq_op, locked)
                eq_start = Scenario(chosen.to_dict(), matrix_t)
                eq_max = (
                    equilibrium_max_iterations
                    if equilibrium_max_iterations is not None
                    else max(max_iterations, 100)
                ) if allow_partial else max_iterations
                eq_result = eq_op.find_attractor(
                    eq_start, matrix_t, max_iterations=eq_max
                )
                if diagnostics is not None:
                    diagnostics.setdefault("equilibrium_iterations", []).append(int(eq_result.iterations))
                    diagnostics.setdefault("equilibrium_is_cycle", []).append(bool(eq_result.is_cycle))
                if eq_result.is_cycle:
                    eq_cycle = eq_result.attractor
                    if not isinstance(eq_cycle, list):
                        raise TypeError("cycle attractor must be a list of scenarios")
                    eq_chosen = eq_cycle[0]
                else:
                    eq_attractor = eq_result.attractor
                    if not isinstance(eq_attractor, Scenario):
                        raise TypeError("fixed-point attractor must be a Scenario")
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

    def simulate_path_extended(
        self,
        *,
        initial: Dict[str, str],
        extension_mode: Literal["transient", "regime", "path_dependent"] = "transient",
        initial_regime: str = "baseline",
        equilibrium_mode: Literal["none", "relax_unshocked"] = "none",
        memory_state: Optional[MemoryState] = None,
        regime_transition_rule: Optional[RegimeTransitionRule | Any] = None,
        transition_kernel: Optional[TransitionKernel | Any] = None,
        adaptive_matrix_updater: Optional[AdaptiveCIMUpdater | Any] = None,
        return_disequilibrium: bool = True,
        return_active_matrices: bool = True,
        return_transition_events: bool = True,
        return_regime_history: bool = True,
        return_structural_consistency: bool = True,
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
        allow_partial: bool = False,
        equilibrium_max_iterations: Optional[int] = None,
        diagnostics: Optional[Dict[str, List[object]]] = None,
        structural_sigma_by_period: Optional[Dict[int, float]] = None,
    ) -> ExtendedTransformationPathway:
        """
        A disequilibrium-aware dynamic pathway is simulated.

        When allow_partial is True, within-period succession may be capped; the last
        state is used as the period outcome and disequilibrium metrics for that period
        may be sparse (e.g. distance_to_attractor or time_to_equilibrium may be None).
        """

        if extension_mode not in {"transient", "regime", "path_dependent"}:
            raise ValueError("Unsupported extension_mode")
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
        if dynamic_shocks_by_period is not None and not self._accepts_dynamic_shocks(
            succession_operator
        ):
            raise ValueError(
                "dynamic_shocks_by_period currently supports only the built-in "
                "GlobalSuccession or LocalSuccession operator"
            )

        if regime_transition_rule is None:
            active_rule = self.regime_transition_rule
        elif isinstance(regime_transition_rule, RegimeTransitionRule):
            active_rule = regime_transition_rule
        else:
            active_rule = CallableRegimeTransitionRule(regime_transition_rule)

        active_kernel: Optional[TransitionKernel]
        if transition_kernel is None:
            active_kernel = None
        elif isinstance(transition_kernel, TransitionKernel):
            active_kernel = transition_kernel
        else:
            active_kernel = CallableTransitionKernel(transition_kernel)

        active_updater: Optional[AdaptiveCIMUpdater]
        if adaptive_matrix_updater is None:
            active_updater = None
        elif isinstance(adaptive_matrix_updater, AdaptiveCIMUpdater):
            active_updater = adaptive_matrix_updater
        else:
            active_updater = CallableAdaptiveCIMUpdater(adaptive_matrix_updater)

        if str(initial_regime) not in self.regimes:
            raise ValueError(f"Unknown initial_regime {initial_regime!r}")

        if extension_mode in {"regime", "path_dependent"}:
            return_disequilibrium = True
            return_active_matrices = True
            return_regime_history = True
            return_transition_events = True
        if extension_mode == "path_dependent":
            return_structural_consistency = True

        rng = np.random.default_rng(seed)
        realised_scenarios: List[Scenario] = []
        realised_scenarios_internal: List[Scenario] = []
        equilibrium_scenarios: Optional[List[Scenario]] = (
            [] if equilibrium_mode == "relax_unshocked" else None
        )
        disequilibrium_metrics: List[PerPeriodDisequilibriumMetrics] = []
        active_regimes: List[str] = []
        active_matrices: List[ActiveMatrixState] = []
        transition_events: List[TransitionEvent] = []
        memory_states: List[MemoryState] = []
        structural_consistency_states: List[StructuralConsistencyState] = []

        current = Scenario(dict(initial), self.base_matrix)
        if constraint_mode != "none" and cidx is not None and not self._is_constraint_valid(current, cidx):
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
        current_regime = str(initial_regime)
        current_memory = self._copy_memory_state(memory_state)
        seen_consistent = False
        current_regime_entry_period: Optional[int] = None
        current_regime_spell_index = -1

        for period_idx, t in enumerate(self.periods):
            period_events: List[TransitionEvent] = []

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

            current = Scenario(current_state, self.base_matrix)

            if extension_mode in {"regime", "path_dependent"} or active_rule is not None:
                resolved_regime, resolved_events = self._resolve_active_regime(
                    current_regime=current_regime,
                    realized_scenario=current,
                    previous_scenarios=tuple(realised_scenarios_internal),
                    memory_state=current_memory,
                    rng=rng,
                    regime_transition_rule=active_rule,
                )
                for event in resolved_events:
                    period_events.append(
                        TransitionEvent(
                            period=int(t),
                            event_type=str(event.event_type),
                            label=str(event.label),
                            source=str(event.source),
                            metadata=dict(event.metadata),
                        )
                    )
                if resolved_regime != current_regime and not any(
                    event.event_type == "regime_transition" for event in period_events
                ):
                    period_events.append(
                        TransitionEvent(
                            period=int(t),
                            event_type="regime_transition",
                            label=f"{current_regime}->{resolved_regime}",
                            source="regime_transition_rule",
                            metadata={"from": current_regime, "to": resolved_regime},
                        )
                    )
                current_regime = resolved_regime

            threshold_regime_rules_applied: Tuple[str, ...] = ()
            threshold_regime_reaffirmations: Tuple[str, ...] = ()
            if self.threshold_rules:
                (
                    current_regime,
                    threshold_regime_rules_applied,
                    threshold_regime_reaffirmations,
                    threshold_regime_events,
                ) = self._resolve_threshold_regime_transitions(
                    period=int(t),
                    current_regime=current_regime,
                    scenario=current,
                )
                period_events.extend(threshold_regime_events)

            entered_regime = (
                not active_regimes or str(active_regimes[-1]) != str(current_regime)
            )
            if entered_regime:
                current_regime_entry_period = int(t)
                current_regime_spell_index += 1

            regime_spec = self.regimes[current_regime]
            regime_matrix = regime_spec.resolve_matrix(self.base_matrix)
            matrix_period: CIBMatrix = regime_matrix
            judgment_labels: List[str] = []
            if judgment_sigma_scale_by_period is not None:
                scale = float(judgment_sigma_scale_by_period.get(int(t), 1.0))
                if hasattr(regime_matrix, "sample_matrix"):
                    try:
                        matrix_period = regime_matrix.sample_matrix(  # type: ignore[attr-defined]
                            int(seed or 0) + 1000 * period_idx, sigma_scale=scale
                        )
                    except TypeError:
                        matrix_period = regime_matrix.sample_matrix(  # type: ignore[attr-defined]
                            int(seed or 0) + 1000 * period_idx
                        )
                    judgment_labels.append(f"sigma_scale={scale}")
                    period_events.append(
                        TransitionEvent(
                            period=int(t),
                            event_type="judgment_sampling",
                            label=f"judgment_sigma_scale={scale}",
                            source="simulate_path_extended",
                            metadata={"sigma_scale": scale},
                        )
                    )

            structural_labels: List[str] = []
            sigma_t = (
                structural_sigma_by_period.get(int(t), structural_sigma)
                if structural_sigma_by_period is not None
                else structural_sigma
            )
            if sigma_t is not None and float(sigma_t) > 0:
                from cib.shocks import ShockModel

                if structural_seed_base is None:
                    structural_seed_base = int(seed or 0) + 50_000
                sm = ShockModel(matrix_period)
                sm.add_structural_shocks(
                    sigma=float(sigma_t),
                    scaling_mode=structural_shock_scaling_mode,
                    scaling_alpha=float(structural_shock_scaling_alpha),
                    scale_by_descriptor=structural_shock_scale_by_descriptor,
                    scale_by_state=structural_shock_scale_by_state,
                )
                matrix_period = sm.sample_shocked_matrix(
                    int(structural_seed_base) + int(period_idx)
                )
                structural_labels.append(f"sigma={float(sigma_t)}")
                period_events.append(
                    TransitionEvent(
                        period=int(t),
                        event_type="structural_shock",
                        label=f"structural_sigma={float(sigma_t)}",
                        source="simulate_path_extended",
                        metadata={"sigma": float(sigma_t)},
                    )
                )

            (
                matrix_t,
                threshold_modifier_rules_applied,
                threshold_modifier_events,
            ) = self._apply_threshold_modifiers(
                period=int(t),
                matrix=matrix_period,
                scenario=current,
                regime_name=current_regime,
            )
            threshold_rules_applied = (
                list(threshold_regime_rules_applied)
                + list(threshold_regime_reaffirmations)
                + list(threshold_modifier_rules_applied)
            )
            if diagnostics is not None:
                diagnostics.setdefault("threshold_rules_applied", []).append(
                    list(threshold_rules_applied)
                )
            period_events.extend(threshold_modifier_events)

            adaptive_labels: List[str] = []
            if extension_mode == "path_dependent" and active_updater is not None:
                updated_matrix, updater_labels, updater_events = active_updater.update(
                    active_matrix=matrix_t,
                    current_regime=current_regime,
                    realized_scenario=current,
                    previous_scenarios=tuple(realised_scenarios_internal),
                    memory_state=current_memory,
                )
                matrix_t = updated_matrix
                adaptive_labels.extend(str(label) for label in updater_labels)
                for event in updater_events:
                    period_events.append(event)
                if updater_labels:
                    period_events.append(
                        TransitionEvent(
                            period=int(t),
                            event_type="adaptive_matrix_update",
                            label="adaptive_matrix_update",
                            source="adaptive_matrix_updater",
                            metadata={"labels": list(adaptive_labels)},
                        )
                    )

            period_operator = succession_operator
            pre_transition_memory = self._copy_memory_state(current_memory)
            pre_transition_event_count = len(period_events)
            locked = self._locked_states_for_period(current_state)
            if locked:
                period_operator = _LockedSuccessionOperator(period_operator, locked)
            if dynamic_shocks_by_period is not None and int(t) in dynamic_shocks_by_period:
                period_operator = self._period_shock_operator(
                    succession_operator, dynamic_shocks_by_period[int(t)]
                )
                if locked:
                    period_operator = _LockedSuccessionOperator(period_operator, locked)
            if constraint_mode != "none" and cidx is not None:
                period_operator = _ConstraintAwareSuccessionOperator(
                    period_operator,
                    cidx,
                    constraint_mode=constraint_mode,  # type: ignore[arg-type]
                    constrained_top_k=int(constrained_top_k),
                    constrained_backtracking_depth=int(constrained_backtracking_depth),
                    locked_states=locked,
                )

            if extension_mode == "path_dependent":
                kernel = active_kernel or DefaultTransitionKernel(
                    succession_operator=period_operator,
                    max_iterations=int(max_iterations),
                    allow_partial=allow_partial,
                )
                prior_memory = current_memory
                chosen, updated_memory, kernel_metadata = kernel.step(
                    current_scenario=current,
                    active_matrix=matrix_t,
                    regime=current_regime,
                    memory_state=prior_memory,
                    rng=rng,
                    previous_path=tuple(realised_scenarios_internal),
                )
                if kernel_metadata:
                    extra_events = kernel_metadata.get("transition_events", ())
                    for event in extra_events:
                        if isinstance(event, TransitionEvent):
                            period_events.append(
                                TransitionEvent(
                                    period=int(t),
                                    event_type=str(event.event_type),
                                    label=str(event.label),
                                    source=str(event.source),
                                    metadata=dict(event.metadata),
                                )
                            )
                    event_metadata = {
                        str(key): value
                        for key, value in kernel_metadata.items()
                        if str(key) != "transition_events"
                    }
                    if event_metadata and self._memory_state_changed(
                        prior_memory, updated_memory
                    ):
                        period_events.append(
                            TransitionEvent(
                                period=int(t),
                                event_type="memory_update",
                                label="transition_kernel",
                                source="transition_kernel",
                                metadata=dict(event_metadata),
                            )
                        )
                    elif event_metadata and diagnostics is not None:
                        diagnostics.setdefault("transition_kernel_metadata", []).append(
                            dict(event_metadata)
                        )
            else:
                result = period_operator.find_attractor(
                    current, matrix_t, max_iterations=max_iterations,
                    allow_partial=allow_partial,
                )
                if diagnostics is not None:
                    diagnostics.setdefault("iterations", []).append(int(result.iterations))
                    diagnostics.setdefault("is_cycle", []).append(bool(result.is_cycle))
                    diagnostics.setdefault("converged", []).append(bool(result.converged))
                if result.is_cycle:
                    cycle = result.attractor
                    if not isinstance(cycle, list):
                        raise TypeError("cycle attractor must be a list of scenarios")
                    if tie_break == "deterministic_first":
                        chosen = cycle[0]
                    else:
                        chosen = cycle[int(rng.integers(0, len(cycle)))]
                else:
                    attractor = result.attractor
                    if not isinstance(attractor, Scenario):
                        raise TypeError("fixed-point attractor must be a Scenario")
                    chosen = attractor
                updated_memory = current_memory

            if constraint_mode != "none" and cidx is not None and not self._is_constraint_valid(chosen, cidx):
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

            if period_idx == 0 and first_period_output_mode == "initial":
                recorded_scenario = Scenario(dict(current_state), matrix_t)
            else:
                recorded_scenario = chosen
                current_state = chosen.to_dict()
            if period_idx == 0 and first_period_output_mode == "initial":
                current_state = chosen.to_dict()
            realised_scenarios.append(recorded_scenario)
            realised_scenarios_internal.append(chosen)

            if extension_mode == "path_dependent":
                if updated_memory is None:
                    normalized_updated_memory = MemoryState(
                        period=int(t),
                        values={},
                        flags={},
                        export_label="memory",
                    )
                else:
                    normalized_updated_memory = MemoryState(
                        period=int(t),
                        values=dict(updated_memory.values),
                        flags=dict(updated_memory.flags),
                        export_label=str(updated_memory.export_label),
                    )
                if period_idx == 0 and first_period_output_mode == "initial":
                    recorded_memory = (
                        MemoryState(
                            period=int(t),
                            values=dict(pre_transition_memory.values),
                            flags=dict(pre_transition_memory.flags),
                            export_label=str(pre_transition_memory.export_label),
                        )
                        if pre_transition_memory is not None
                        else MemoryState(
                            period=int(t),
                            values={},
                            flags={},
                            export_label="memory",
                        )
                    )
                    recorded_transition_events = tuple(
                        period_events[:pre_transition_event_count]
                    )
                else:
                    recorded_memory = normalized_updated_memory
                    recorded_transition_events = tuple(period_events)
                period_state = self._build_path_dependent_state(
                    period=int(t),
                    scenario=recorded_scenario,
                    regime_name=current_regime,
                    active_matrix=matrix_t,
                    memory_state=recorded_memory,
                    previous_scenarios=tuple(realised_scenarios[:-1]),
                    transition_events=recorded_transition_events,
                )
            else:
                period_state = None

            if equilibrium_scenarios is not None:
                eq_op: SuccessionOperator = succession_operator
                if locked:
                    eq_op = _LockedSuccessionOperator(eq_op, locked)
                if constraint_mode != "none" and cidx is not None:
                    eq_op = _ConstraintAwareSuccessionOperator(
                        eq_op,
                        cidx,
                        constraint_mode=constraint_mode,  # type: ignore[arg-type]
                        constrained_top_k=int(constrained_top_k),
                        constrained_backtracking_depth=int(constrained_backtracking_depth),
                        locked_states=locked,
                    )
                eq_start = Scenario(chosen.to_dict(), matrix_t)
                eq_max = (
                    equilibrium_max_iterations
                    if equilibrium_max_iterations is not None
                    else max(max_iterations, 100)
                ) if allow_partial else max_iterations
                eq_result = eq_op.find_attractor(
                    eq_start, matrix_t, max_iterations=eq_max
                )
                if diagnostics is not None:
                    diagnostics.setdefault("equilibrium_iterations", []).append(
                        int(eq_result.iterations)
                    )
                    diagnostics.setdefault("equilibrium_is_cycle", []).append(
                        bool(eq_result.is_cycle)
                    )
                if eq_result.is_cycle:
                    eq_cycle = eq_result.attractor
                    if not isinstance(eq_cycle, list):
                        raise TypeError("cycle attractor must be a list of scenarios")
                    eq_chosen = eq_cycle[0]
                else:
                    eq_attractor = eq_result.attractor
                    if not isinstance(eq_attractor, Scenario):
                        raise TypeError("fixed-point attractor must be a Scenario")
                    eq_chosen = eq_attractor
                equilibrium_scenarios.append(eq_chosen)

            if return_disequilibrium:
                metric = self._compute_period_disequilibrium(
                    period=int(t),
                    scenario=recorded_scenario,
                    matrix=matrix_t,
                    succession_operator=period_operator,
                    max_iterations=int(max_iterations),
                    seen_consistent_before=seen_consistent,
                )
                disequilibrium_metrics.append(metric)
                seen_consistent = seen_consistent or bool(metric.is_consistent)

            if return_regime_history:
                active_regimes.append(str(current_regime))

            provenance_labels = ["baseline", f"regime:{current_regime}"]
            if threshold_regime_rules_applied:
                provenance_labels.extend(
                    f"threshold_regime_transition:{label}"
                    for label in threshold_regime_rules_applied
                )
            if threshold_regime_reaffirmations:
                provenance_labels.extend(
                    f"threshold_regime_reaffirmation:{label}"
                    for label in threshold_regime_reaffirmations
                )
            if threshold_rules_applied:
                provenance_labels.extend(
                    f"threshold_modifier:{label}"
                    for label in threshold_modifier_rules_applied
                )
            if structural_labels:
                provenance_labels.extend(
                    f"structural_shock:{label}" for label in structural_labels
                )
            if judgment_labels:
                provenance_labels.extend(
                    f"judgment_sampling:{label}" for label in judgment_labels
                )
            if adaptive_labels:
                provenance_labels.extend(
                    f"adaptive_matrix_update:{label}" for label in adaptive_labels
                )
            if return_active_matrices or extension_mode in {"regime", "path_dependent"}:
                active_matrices.append(
                    self._build_active_matrix_state(
                        period=int(t),
                        regime_name=current_regime,
                        base_matrix=self.base_matrix,
                        active_matrix=matrix_t,
                        threshold_rules_applied=threshold_rules_applied,
                        structural_labels=structural_labels,
                        judgment_labels=judgment_labels,
                        adaptive_labels=adaptive_labels,
                        threshold_regime_reaffirmations=threshold_regime_reaffirmations,
                        entered_regime=entered_regime,
                        regime_entry_period=current_regime_entry_period,
                        regime_spell_index=current_regime_spell_index,
                        provenance_labels=provenance_labels,
                    )
                )

            if extension_mode == "path_dependent":
                if period_state is None:
                    raise ValueError(
                        "path_dependent extension requires a non-null period_state"
                    )
                current_memory = normalized_updated_memory
                if period_state.memory_state is None:
                    raise ValueError(
                        "path_dependent extension requires period_state.memory_state"
                    )
                memory_states.append(period_state.memory_state)
                if return_structural_consistency:
                    state = self._compute_structural_consistency_state(
                        period=int(t),
                        realized_scenario=period_state.scenario,
                        regime_name=period_state.regime_name,
                        memory_state=period_state.memory_state,
                        previous_scenarios=tuple(realised_scenarios[:-1]),
                        transition_events=period_state.transition_events,
                    )
                    structural_consistency_states.append(state)
            elif current_memory is not None:
                snapshot = self._copy_memory_state(current_memory)
                if snapshot is None:
                    raise ValueError("memory state copy unexpectedly returned None")
                snapshot = MemoryState(
                    period=int(t),
                    values=dict(snapshot.values),
                    flags=dict(snapshot.flags),
                    export_label=str(snapshot.export_label),
                )
                memory_states.append(snapshot)

            if return_transition_events or extension_mode in {"regime", "path_dependent"}:
                transition_events.extend(period_events)

        eq_out = tuple(equilibrium_scenarios) if equilibrium_scenarios is not None else None
        diag_out = {
            str(k): tuple(v) for k, v in (diagnostics or {}).items()
        }
        return ExtendedTransformationPathway(
            periods=tuple(int(t) for t in self.periods),
            realised_scenarios=tuple(realised_scenarios),
            equilibrium_scenarios=eq_out,
            extension_mode=str(extension_mode),
            disequilibrium_metrics=tuple(disequilibrium_metrics),
            active_regimes=tuple(active_regimes),
            active_matrices=tuple(active_matrices),
            transition_events=tuple(transition_events),
            memory_states=tuple(memory_states),
            structural_consistency=tuple(structural_consistency_states),
            diagnostics=diag_out,
        )

    def trace_to_equilibrium(
        self,
        *,
        initial: Dict[str, str],
        seed: Optional[int] = None,
        succession_operator: Optional[SuccessionOperator] = None,
        max_iterations: int = 1000,
    ) -> ExtendedTransformationPathway:
        """
        A scenario is traced forward until the consistent set is entered or the
        iteration budget is exhausted.
        """

        if succession_operator is None:
            succession_operator = GlobalSuccession()
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        current = Scenario(dict(initial), self.base_matrix)
        periods: List[int] = []
        realized: List[Scenario] = []
        metrics: List[PerPeriodDisequilibriumMetrics] = []
        active_regimes = []
        active_matrices = []
        seen_consistent = False

        for step in range(int(max_iterations) + 1):
            periods.append(int(step))
            realized.append(current)
            metrics.append(
                self._compute_period_disequilibrium(
                    period=int(step),
                    scenario=current,
                    matrix=self.base_matrix,
                    succession_operator=succession_operator,
                    max_iterations=int(max_iterations),
                    seen_consistent_before=seen_consistent,
                )
            )
            seen_consistent = seen_consistent or bool(metrics[-1].is_consistent)
            active_regimes.append("baseline")
            active_matrices.append(
                self._build_active_matrix_state(
                    period=int(step),
                    regime_name="baseline",
                    base_matrix=self.base_matrix,
                    active_matrix=self.base_matrix,
                    threshold_rules_applied=(),
                    structural_labels=(),
                    judgment_labels=(),
                    adaptive_labels=(),
                    threshold_regime_reaffirmations=(),
                    entered_regime=bool(step == 0),
                    regime_entry_period=0,
                    regime_spell_index=0,
                    provenance_labels=("baseline",),
                )
            )
            if metrics[-1].is_consistent:
                break
            current = succession_operator.find_successor(current, self.base_matrix)

        return ExtendedTransformationPathway(
            periods=tuple(periods),
            realised_scenarios=tuple(realized),
            equilibrium_scenarios=None,
            extension_mode="transient",
            disequilibrium_metrics=tuple(metrics),
            active_regimes=tuple(active_regimes),
            active_matrices=tuple(active_matrices),
            transition_events=(),
            memory_states=(),
            structural_consistency=(),
            diagnostics={},
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
        if dynamic_tau is not None and not self._accepts_dynamic_shocks(
            succession_operator
        ):
            raise ValueError(
                "dynamic_tau currently supports only the built-in "
                "GlobalSuccession or LocalSuccession operator"
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

