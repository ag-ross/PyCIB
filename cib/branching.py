"""
Hybrid branching pathway construction (enumerate when feasible; sample otherwise).

This module implements a “middle way” between:
  - enumeration-based branching per-period scenario-set enumeration (A/B/C per sub-period), and
  - pure Monte Carlo simulation of single paths.

For each transition between periods, the builder can:
  - enumerate all consistent scenarios for the next period when the scenario space
    is small (guarded by `max_states_to_enumerate`), or
  - approximate the next-period branching distribution by repeated sampling of
    uncertainty/shocks and succession (Monte Carlo).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import warnings
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.cyclic import CyclicDescriptor
from cib.example_data import seeds_for_run
from cib.pathway import MemoryState, PathDependentState, TransitionEvent
from cib.transition_kernel import CallableTransitionKernel, DefaultTransitionKernel, TransitionKernel
from cib.regimes import CallableRegimeTransitionRule, RegimeSpec, RegimeTransitionRule
from cib.shocks import ShockAwareGlobalSuccession, ShockModel
from cib.succession import GlobalSuccession, SuccessionOperator
from cib.threshold import ThresholdRule, apply_modifier_copy_on_write


@dataclass(frozen=True)
class BranchRegimeState:
    """
    Regime-residence metadata for one branching node.
    """

    period: int
    regime_name: str
    entered_regime: bool
    regime_entry_period: int
    regime_spell_index: int
    threshold_regime_reaffirmations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BranchingResult:
    """
    Result of building a branching pathway.
    """

    periods: Tuple[int, ...]
    scenarios_by_period: Tuple[Tuple[Scenario, ...], ...]
    # Edges between consecutive periods are stored as index weights.
    # Format is: edges[(period_idx, src_idx)][tgt_idx] = weight.
    edges: Mapping[Tuple[int, int], Mapping[int, float]]
    # Method used for each layer transition: "enumerate" or "sample".
    transition_method: Mapping[int, str]
    # Top-K most likely node-index paths and their weights are stored.
    top_paths: Tuple[Tuple[Tuple[int, ...], float], ...]
    # Regime identity can be tracked alongside scenarios for regime-aware branching.
    active_regimes: Tuple[Tuple[str, ...], ...] = ()
    # Regime residence semantics align to nodes when regime-aware branching is active.
    regime_states_by_period: Tuple[Tuple[BranchRegimeState, ...], ...] = ()
    # Memory states align to nodes when memory-aware branching is active.
    memory_states_by_period: Tuple[Tuple[MemoryState, ...], ...] = ()
    # Retained history signatures align to nodes when history-aware branching is active.
    history_signatures_by_period: Tuple[
        Tuple[Tuple[Tuple[int, ...], ...], ...], ...
    ] = ()
    # The branching approximation contract is recorded explicitly.
    approximation_contract: str = "exact_scenario_regime_branching"

    def node_records(self, period_idx: int) -> Tuple[Dict[str, object], ...]:
        """
        Node records for one period layer with regime context attached are returned.
        """

        idx = int(period_idx)
        scenarios = self.scenarios_by_period[idx]
        regimes = (
            self.active_regimes[idx]
            if self.active_regimes
            else tuple("baseline" for _ in scenarios)
        )
        memories = (
            self.memory_states_by_period[idx]
            if self.memory_states_by_period
            else tuple(None for _ in scenarios)
        )
        regime_states = (
            self.regime_states_by_period[idx]
            if self.regime_states_by_period
            else tuple(
                BranchRegimeState(
                    period=int(self.periods[idx]),
                    regime_name=str(regime),
                    entered_regime=bool(idx == 0),
                    regime_entry_period=int(self.periods[idx]),
                    regime_spell_index=0,
                )
                for regime in regimes
            )
        )
        history_signatures = (
            self.history_signatures_by_period[idx]
            if self.history_signatures_by_period
            else tuple(() for _ in scenarios)
        )
        return tuple(
            {
                "period": int(self.periods[idx]),
                "node_index": int(node_idx),
                "scenario": scenario.to_dict(),
                "regime": str(regime),
                "regime_state": regime_state,
                "memory_state": memory,
                "history_signature": history_signature,
            }
            for node_idx, (
                scenario,
                regime,
                regime_state,
                memory,
                history_signature,
            ) in enumerate(
                zip(scenarios, regimes, regime_states, memories, history_signatures)
            )
        )

    def top_path_details(self) -> Tuple[Dict[str, object], ...]:
        """
        Regime-aware decoded top-path summaries are returned.
        """

        out = []
        for node_indices, weight in self.top_paths:
            records = []
            for period_idx, node_idx in enumerate(node_indices):
                layer_records = self.node_records(period_idx)
                records.append(dict(layer_records[int(node_idx)]))
            out.append(
                {
                    "node_indices": tuple(int(idx) for idx in node_indices),
                    "weight": float(weight),
                    "records": tuple(records),
                }
            )
        return tuple(out)


class _LockedSuccessionOperator(SuccessionOperator):
    """
    Wrapper that prevents selected descriptors from being updated by succession.
    """

    def __init__(self, inner: SuccessionOperator, locked: Mapping[str, str]) -> None:
        self.inner = inner
        self.locked = dict(locked)

    def find_successor(self, scenario: Scenario, matrix: CIBMatrix) -> Scenario:
        nxt = self.inner.find_successor(scenario, matrix)
        if not self.locked:
            return nxt
        state = nxt.to_dict()
        for d, v in self.locked.items():
            if d in state:
                state[d] = v
        return Scenario(state, matrix)


def _scenario_space_size(descriptors: Mapping[str, Sequence[str]]) -> int:
    n = 1
    for states in descriptors.values():
        n *= int(len(states))
    return int(n)


def _enumerate_consistent_with_locks(
    *,
    matrix: CIBMatrix,
    locked: Mapping[str, str],
) -> List[Scenario]:
    """
    Enumerate consistent scenarios for `matrix`, holding `locked` descriptors fixed.
    """
    var_descs = [d for d in matrix.descriptors.keys() if d not in locked]
    state_lists = [matrix.descriptors[d] for d in var_descs]

    out: List[Scenario] = []
    for comb in product(*state_lists):
        sdict = dict(locked)
        sdict.update(dict(zip(var_descs, comb)))
        s = Scenario(sdict, matrix)
        if ConsistencyChecker.check_consistency(s, matrix):
            out.append(s)
    return out


def _enumerate_states_with_locks(
    *,
    matrix: CIBMatrix,
    locked: Mapping[str, str],
) -> List[Scenario]:
    """
    Enumerate all state assignments for `matrix`, holding `locked` descriptors fixed.
    """
    var_descs = [d for d in matrix.descriptors.keys() if d not in locked]
    state_lists = [matrix.descriptors[d] for d in var_descs]

    out: List[Scenario] = []
    for comb in product(*state_lists):
        sdict = dict(locked)
        sdict.update(dict(zip(var_descs, comb)))
        out.append(Scenario(sdict, matrix))
    return out


def _matrix_state_space_signature(matrix: CIBMatrix) -> Tuple[object, ...]:
    """
    Build a stable cache signature for a matrix state space.
    """
    descriptor_signature = tuple(
        (str(name), tuple(str(state) for state in states))
        for name, states in matrix.descriptors.items()
    )
    impacts_signature = tuple(
        sorted(
            (
                str(src_desc),
                str(src_state),
                str(tgt_desc),
                str(tgt_state),
                float(value),
            )
            for (src_desc, src_state, tgt_desc, tgt_state), value in matrix.iter_impacts()
        )
    )
    return (descriptor_signature, impacts_signature)


def _prune_distribution(
    dist: Mapping[object, float],
    *,
    prune_policy: str,
    per_parent_top_k: Optional[int],
    min_edge_weight: Optional[float],
) -> Dict[object, float]:
    """
    An edge-pruning policy is applied to a single parent's outgoing distribution.
    """
    out = {k: float(v) for k, v in dist.items() if float(v) > 0.0}
    if not out:
        return {}

    def _sort_key(item: object) -> Tuple[int, ...]:
        if isinstance(item, Scenario):
            return tuple(int(v) for v in item.to_indices())
        if isinstance(item, tuple) and item and isinstance(item[0], Scenario):
            scenario = item[0]
            return tuple(int(v) for v in scenario.to_indices())
        return (0,)

    if prune_policy == "incoming_mass":
        # No per-parent pruning is applied for this policy.
        pass
    elif prune_policy == "per_parent_topk":
        if per_parent_top_k is None or int(per_parent_top_k) <= 0:
            raise ValueError("per_parent_top_k must be positive for prune_policy='per_parent_topk'")
        k = int(per_parent_top_k)
        kept = sorted(
            out.items(),
            key=lambda kv: (float(kv[1]), _sort_key(kv[0])),
            reverse=True,
        )[:k]
        out = {s: w for s, w in kept}
    elif prune_policy == "min_edge_weight":
        if min_edge_weight is None or float(min_edge_weight) < 0.0:
            raise ValueError("min_edge_weight must be non-negative for prune_policy='min_edge_weight'")
        thr = float(min_edge_weight)
        out = {s: w for s, w in out.items() if float(w) >= thr}
    else:
        raise ValueError(
            "prune_policy must be 'incoming_mass', 'per_parent_topk', or 'min_edge_weight'"
        )

    if not out:
        return {}

    s = float(sum(out.values()))
    if s <= 0.0:
        return {}
    return {k: float(v) / s for k, v in out.items()}


class BranchingPathwayBuilder:
    """
    A per-period branching pathway graph is built.

    Notes:
    - Cyclic descriptors are treated as exogenous/inertial between periods and are
      locked during within-period succession, consistent with `DynamicCIB`.
    - The active CIM for period t+1 is determined by the scenario at the start
      of period t+1 (after cyclic transitions). Threshold rules are evaluated on
      that post-cyclic scenario so that behaviour matches `DynamicCIB.simulate_path()`.

    Enumeration vs sampling:
        The builder automatically chooses between enumeration and sampling modes
        based on scenario-space size (controlled by `max_states_to_enumerate`).

        - Enumeration mode (scenario space size <= max_states_to_enumerate):
          - Enumerates all consistent scenarios for a deterministic base matrix
          - Ignores `judgment_sigma_scale_by_period` and `structural_sigma`
          - Produces complete results for the CIB-consistent scenario set, conditional on the realised cyclic transitions

        - Sampling mode (scenario space size > max_states_to_enumerate):
          - Estimates transition distributions via Monte Carlo sampling
          - Respects `judgment_sigma_scale_by_period` and `structural_sigma`
          - Produces stochastic, approximate results

        Note: if uncertainty parameters are set but enumeration mode is used,
        those parameters are ignored. To ensure uncertainty is applied, decrease
        `max_states_to_enumerate` to force sampling mode.
    """

    def __init__(
        self,
        *,
        base_matrix: CIBMatrix,
        periods: Sequence[int],
        initial: Mapping[str, str],
        cyclic_descriptors: Optional[Sequence[CyclicDescriptor]] = None,
        threshold_rules: Optional[Sequence[ThresholdRule]] = None,
        threshold_match_policy: Literal["first_match", "all_matches"] = "all_matches",
        succession_operator: Optional[SuccessionOperator] = None,
        node_mode: Literal["equilibrium", "realized"] = "equilibrium",
        max_states_to_enumerate: int = 20_000,
        n_transition_samples: int = 200,
        max_nodes_per_period: Optional[int] = None,
        prune_policy: Literal["incoming_mass", "per_parent_topk", "min_edge_weight"] = "incoming_mass",
        per_parent_top_k: Optional[int] = None,
        min_edge_weight: Optional[float] = None,
        base_seed: int = 123,
        # Optional uncertainty/shock settings are provided for sampling transitions.
        structural_sigma: Optional[float] = None,
        judgment_sigma_scale_by_period: Optional[Mapping[int, float]] = None,
        dynamic_tau: Optional[float] = None,
        dynamic_rho: float = 0.6,
        dynamic_innovation_dist: str = "normal",
        dynamic_innovation_df: Optional[float] = None,
        dynamic_jump_prob: float = 0.0,
        dynamic_jump_scale: Optional[float] = None,
        regimes: Optional[Sequence[RegimeSpec]] = None,
        initial_regime: str = "baseline",
        regime_transition_rule: Optional[RegimeTransitionRule | object] = None,
        memory_state: Optional[MemoryState] = None,
        transition_kernel: Optional[TransitionKernel | object] = None,
        history_horizon: Optional[int] = None,
    ) -> None:
        """
        A branching pathway builder is initialised.

        Args:
            base_matrix: Base CIB matrix used for transitions.
            periods: Period labels for the pathway graph.
            initial: Initial scenario as a descriptor -> state mapping.
            cyclic_descriptors: Optional cyclic descriptors (treated as exogenous/inertial).
            threshold_rules: Optional threshold rules applied between periods.
            threshold_match_policy: Threshold rule matching policy. When set to
                "first_match", only the first matching rule is applied. When set to
                "all_matches", modifiers for matching rules are applied sequentially
                (order matters).
            succession_operator: Succession operator used for within-period attractor finding.
            node_mode: Node representation mode. When set to `"equilibrium"`, an unshocked
                relaxation is performed after dynamic-shock succession so that nodes remain
                CIB-consistent with the period matrix.
            max_states_to_enumerate: Maximum scenario space size allowed for enumeration mode.
            n_transition_samples: Number of transition samples used in sampling mode.
            max_nodes_per_period: Optional cap used to prune each period layer for readability.
            prune_policy: Pruning policy applied to each parent's outgoing edge distribution.
                When set to "incoming_mass", only layer-level pruning (if configured) is applied.
                When set to "per_parent_topk", the top-K outgoing edges are kept per parent.
                When set to "min_edge_weight", edges below the given threshold are removed.
            per_parent_top_k: Number of outgoing edges kept per parent when prune_policy is "per_parent_topk".
            min_edge_weight: Minimum edge weight kept when prune_policy is "min_edge_weight".
            base_seed: Base seed used for reproducible sampling.
            structural_sigma: Optional structural shock magnitude applied to sampled matrices.
                Note: only used in sampling mode; ignored in enumeration mode.
            judgment_sigma_scale_by_period: Optional per-period sigma scales used for matrix sampling.
                Note: only used in sampling mode; ignored in enumeration mode.
            dynamic_tau: Optional long-run scale for AR(1) dynamic shocks.
            dynamic_rho: AR(1) persistence parameter in [-1, 1].
            dynamic_innovation_dist: Innovation distribution used for dynamic shocks.
            dynamic_innovation_df: Optional degrees of freedom used for Student-t innovations.
            dynamic_jump_prob: Optional jump probability used for dynamic innovations.
            dynamic_jump_scale: Optional jump scale used for dynamic innovations.

        Raises:
            ValueError: If input parameters are invalid.
        """
        if not periods:
            raise ValueError("periods cannot be empty")
        if max_states_to_enumerate <= 0:
            raise ValueError("max_states_to_enumerate must be positive")
        if n_transition_samples <= 0:
            raise ValueError("n_transition_samples must be positive")
        if max_nodes_per_period is not None and int(max_nodes_per_period) <= 0:
            raise ValueError("max_nodes_per_period must be positive when provided")
        if node_mode not in {"equilibrium", "realized"}:
            raise ValueError("node_mode must be 'equilibrium' or 'realized'")
        if threshold_match_policy not in {"first_match", "all_matches"}:
            raise ValueError("threshold_match_policy must be 'first_match' or 'all_matches'")
        if prune_policy not in {"incoming_mass", "per_parent_topk", "min_edge_weight"}:
            raise ValueError("prune_policy is not recognised")
        if prune_policy == "per_parent_topk" and (per_parent_top_k is None or int(per_parent_top_k) <= 0):
            raise ValueError("per_parent_top_k must be positive for prune_policy='per_parent_topk'")
        if prune_policy == "min_edge_weight" and (min_edge_weight is None or float(min_edge_weight) < 0.0):
            raise ValueError("min_edge_weight must be non-negative for prune_policy='min_edge_weight'")
        if history_horizon is not None and int(history_horizon) <= 0:
            raise ValueError("history_horizon must be positive when provided")

        self.base_matrix = base_matrix
        self.periods = [int(t) for t in periods]
        self.initial = dict(initial)
        self.cyclic_descriptors = list(cyclic_descriptors or [])
        for cd in self.cyclic_descriptors:
            cd.validate()
        self.threshold_rules = list(threshold_rules or [])
        self.threshold_match_policy = str(threshold_match_policy)
        self.succession_operator = succession_operator or GlobalSuccession()
        self.node_mode = str(node_mode)

        self.max_states_to_enumerate = int(max_states_to_enumerate)
        self.n_transition_samples = int(n_transition_samples)
        self.max_nodes_per_period = int(max_nodes_per_period) if max_nodes_per_period is not None else None
        self.prune_policy = str(prune_policy)
        self.per_parent_top_k = int(per_parent_top_k) if per_parent_top_k is not None else None
        self.min_edge_weight = float(min_edge_weight) if min_edge_weight is not None else None
        self.base_seed = int(base_seed)

        self.structural_sigma = structural_sigma
        self.judgment_sigma_scale_by_period = (
            {int(k): float(v) for k, v in judgment_sigma_scale_by_period.items()}
            if judgment_sigma_scale_by_period is not None
            else None
        )
        self.dynamic_tau = dynamic_tau
        self.dynamic_rho = float(dynamic_rho)
        self.dynamic_innovation_dist = str(dynamic_innovation_dist)
        self.dynamic_innovation_df = dynamic_innovation_df
        self.dynamic_jump_prob = float(dynamic_jump_prob)
        self.dynamic_jump_scale = dynamic_jump_scale
        self.regimes: Dict[str, RegimeSpec] = {
            "baseline": RegimeSpec(
                name="baseline",
                base_matrix=self.base_matrix,
                activation_metadata={},
                description="Baseline regime",
            )
        }
        for regime in regimes or ():
            self.regimes[str(regime.name)] = regime
        if str(initial_regime) not in self.regimes:
            raise ValueError(f"Unknown initial_regime {initial_regime!r}")
        self.initial_regime = str(initial_regime)
        if regime_transition_rule is None:
            self.regime_transition_rule = None
        elif isinstance(regime_transition_rule, RegimeTransitionRule):
            self.regime_transition_rule = regime_transition_rule
        else:
            self.regime_transition_rule = CallableRegimeTransitionRule(regime_transition_rule)
        self.initial_memory_state = (
            MemoryState(
                period=int(memory_state.period),
                values=dict(memory_state.values),
                flags=dict(memory_state.flags),
                export_label=str(memory_state.export_label),
            )
            if memory_state is not None
            else None
        )
        if transition_kernel is None:
            self.transition_kernel = None
        elif isinstance(transition_kernel, TransitionKernel):
            self.transition_kernel = transition_kernel
        else:
            self.transition_kernel = CallableTransitionKernel(transition_kernel)
        self.history_horizon = (
            int(history_horizon) if history_horizon is not None else None
        )

    @staticmethod
    def _copy_memory_state(memory_state: Optional[MemoryState]) -> Optional[MemoryState]:
        if memory_state is None:
            return None
        return MemoryState(
            period=int(memory_state.period),
            values=dict(memory_state.values),
            flags=dict(memory_state.flags),
            export_label=str(memory_state.export_label),
        )

    @staticmethod
    def _memory_signature(memory_state: Optional[MemoryState]) -> Tuple[object, ...]:
        if memory_state is None:
            return ("none",)

        def _freeze(value: object) -> object:
            if isinstance(value, dict):
                return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
            if isinstance(value, (list, tuple)):
                return tuple(_freeze(v) for v in value)
            return value

        return (
            int(memory_state.period),
            str(memory_state.export_label),
            _freeze(memory_state.values),
            _freeze(memory_state.flags),
        )

    def _truncate_history(
        self, history: Sequence[Scenario]
    ) -> Tuple[Scenario, ...]:
        if self.history_horizon is None:
            return tuple(history)
        return tuple(history[-int(self.history_horizon) :])

    def _history_signature(
        self, history: Sequence[Scenario]
    ) -> Tuple[Tuple[int, ...], ...]:
        truncated = self._truncate_history(history)
        return tuple(tuple(int(v) for v in scenario.to_indices()) for scenario in truncated)

    def _build_path_state(
        self,
        *,
        period: int,
        scenario: Scenario,
        regime_name: str,
        active_matrix: CIBMatrix,
        memory_state: Optional[MemoryState],
        previous_history: Sequence[Scenario],
    ) -> PathDependentState:
        return PathDependentState(
            period=int(period),
            scenario=scenario,
            regime_name=str(regime_name),
            active_matrix=active_matrix,
            memory_state=self._copy_memory_state(memory_state),
            history_signature=self._history_signature(tuple(previous_history) + (scenario,)),
            transition_events=(),
        )

    @staticmethod
    def _advance_regime_state(
        *,
        period: int,
        regime_name: str,
        parent_regime: Optional[BranchRegimeState],
        threshold_regime_reaffirmations: Sequence[str],
    ) -> BranchRegimeState:
        entered_regime = (
            parent_regime is None
            or str(parent_regime.regime_name) != str(regime_name)
        )
        regime_entry_period = (
            int(period)
            if entered_regime
            else int(parent_regime.regime_entry_period)
        )
        regime_spell_index = (
            0
            if parent_regime is None
            else int(parent_regime.regime_spell_index) + 1
            if entered_regime
            else int(parent_regime.regime_spell_index)
        )
        return BranchRegimeState(
            period=int(period),
            regime_name=str(regime_name),
            entered_regime=bool(entered_regime),
            regime_entry_period=int(regime_entry_period),
            regime_spell_index=int(regime_spell_index),
            threshold_regime_reaffirmations=tuple(
                str(label) for label in threshold_regime_reaffirmations
            ),
        )

    def _resolve_threshold_regime_transitions(
        self, *, current_regime: str, scenario: Scenario
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        resolved_regime = str(current_regime)
        transition_rules: List[str] = []
        reaffirmation_rules: List[str] = []
        for rule in self.threshold_rules:
            if rule.target_regime is None or not rule.condition(scenario):
                continue
            target_regime = str(rule.target_regime)
            if target_regime != resolved_regime:
                transition_rules.append(str(rule.name))
                resolved_regime = target_regime
            else:
                reaffirmation_rules.append(str(rule.name))
            if self.threshold_match_policy == "first_match":
                break
        return resolved_regime, tuple(transition_rules), tuple(reaffirmation_rules)

    def _apply_threshold_modifiers(
        self, matrix: CIBMatrix, scenario: Scenario
    ) -> CIBMatrix:
        active = matrix
        for rule in self.threshold_rules:
            if rule.modifier is None or not rule.condition(scenario):
                continue
            active, _ = apply_modifier_copy_on_write(rule.modifier, active)
            if self.threshold_match_policy == "first_match":
                break
        return active

    def _apply_cyclic_transitions(
        self, state: Mapping[str, str], rng: np.random.Generator
    ) -> Dict[str, str]:
        out = dict(state)
        for cd in self.cyclic_descriptors:
            if cd.name not in out:
                raise ValueError(f"Cyclic descriptor {cd.name!r} missing from scenario")
            out[cd.name] = cd.sample_next(out[cd.name], rng)
        return out

    def _lock_map(self, state: Mapping[str, str]) -> Dict[str, str]:
        return {cd.name: str(state[cd.name]) for cd in self.cyclic_descriptors}

    def _matrix_for_regime(self, regime_name: str) -> CIBMatrix:
        return self.regimes[str(regime_name)].resolve_matrix(self.base_matrix)

    @staticmethod
    def _enforce_locked_scenario(
        *,
        scenario: Scenario,
        matrix: CIBMatrix,
        locked: Mapping[str, str],
    ) -> Scenario:
        if not locked:
            return scenario
        state = scenario.to_dict()
        changed = False
        for descriptor, expected in locked.items():
            descriptor_name = str(descriptor)
            expected_state = str(expected)
            if descriptor_name in state and state[descriptor_name] != expected_state:
                state[descriptor_name] = expected_state
                changed = True
        if not changed:
            return scenario
        return Scenario(state, matrix)

    def _resolve_next_regime(
        self,
        *,
        current_regime: str,
        realized_scenario: Scenario,
        previous_scenarios: Sequence[Scenario],
        memory_state: Optional[MemoryState],
        rng: np.random.Generator,
    ) -> Tuple[str, Tuple[TransitionEvent, ...]]:
        if self.regime_transition_rule is None:
            return str(current_regime), ()
        return self.regime_transition_rule.resolve_next_regime(
            current_regime=str(current_regime),
            realized_scenario=realized_scenario,
            previous_scenarios=previous_scenarios,
            memory_state=memory_state,
            rng=rng,
        )

    def _sample_matrix_for_period(
        self,
        *,
        base_matrix: CIBMatrix,
        period_idx: int,
        seed: int,
    ) -> CIBMatrix:
        m: CIBMatrix = base_matrix
        if self.judgment_sigma_scale_by_period is not None and hasattr(m, "sample_matrix"):
            scale = float(self.judgment_sigma_scale_by_period.get(self.periods[period_idx], 1.0))
            try:
                m = m.sample_matrix(int(seed), sigma_scale=scale)  # type: ignore[attr-defined]
            except TypeError:
                m = m.sample_matrix(int(seed))  # type: ignore[attr-defined]
        return m

    def _sample_active_matrix_for_next_period(
        self,
        *,
        regime_matrix: CIBMatrix,
        scenario_for_threshold: Scenario,
        next_period_idx: int,
        seed: int,
    ) -> CIBMatrix:
        """
        Sample/perturb the matrix for the next period (optional), then apply thresholds.

        The scenario passed in must be the state at the start of the next period
        (after cyclic transitions). Threshold rules are evaluated on this scenario
        so that the active CIM is chosen consistently with `DynamicCIB.simulate_path()`.
        """
        m = self._sample_matrix_for_period(
            base_matrix=regime_matrix,
            period_idx=next_period_idx,
            seed=seed,
        )

        if self.structural_sigma is not None:
            sm = ShockModel(m)
            sm.add_structural_shocks(sigma=float(self.structural_sigma))
            m = sm.sample_shocked_matrix(int(seed) + 10_000 + int(next_period_idx))

        return self._apply_threshold_modifiers(m, scenario_for_threshold)

    def _find_attractor(
        self,
        *,
        scenario: Scenario,
        matrix: CIBMatrix,
        locked: Mapping[str, str],
        dynamic_shocks: Optional[Mapping[Tuple[str, str], float]] = None,
        max_iterations: int = 1000,
    ) -> Scenario:
        base_op: SuccessionOperator = self.succession_operator

        op_realized: SuccessionOperator = base_op
        if dynamic_shocks is not None:
            op_realized = ShockAwareGlobalSuccession(dynamic_shocks)
        if locked:
            op_realized = _LockedSuccessionOperator(op_realized, locked)

        realized_res = op_realized.find_attractor(
            scenario, matrix, max_iterations=max_iterations
        )
        if realized_res.is_cycle:
            realized_cycle = realized_res.attractor
            if not isinstance(realized_cycle, list):
                raise TypeError("cycle attractor must be a list of scenarios")
            realized = realized_cycle[0]
        else:
            realized_attractor = realized_res.attractor
            if not isinstance(realized_attractor, Scenario):
                raise TypeError("fixed-point attractor must be a Scenario")
            realized = realized_attractor

        if self.node_mode == "realized" or dynamic_shocks is None:
            return realized

        # An unshocked relaxation is performed so that nodes remain CIB-consistent
        # with the period matrix, while edge weights continue to reflect forcing.
        eq_op: SuccessionOperator = base_op
        if locked:
            eq_op = _LockedSuccessionOperator(eq_op, locked)
        eq_start = Scenario(realized.to_dict(), matrix)
        eq_res = eq_op.find_attractor(eq_start, matrix, max_iterations=max_iterations)
        if eq_res.is_cycle:
            eq_cycle = eq_res.attractor
            if not isinstance(eq_cycle, list):
                raise TypeError("cycle attractor must be a list of scenarios")
            return eq_cycle[0]
        eq_attractor = eq_res.attractor
        if not isinstance(eq_attractor, Scenario):
            raise TypeError("fixed-point attractor must be a Scenario")
        return eq_attractor

    def build(self, *, top_k: int = 10) -> BranchingResult:
        """
        A branching pathway graph across periods is built.
        """
        periods = tuple(int(t) for t in self.periods)

        has_uncertainty = (
            self.structural_sigma is not None
            or self.judgment_sigma_scale_by_period is not None
        )
        scenario_space_size = _scenario_space_size(self.base_matrix.descriptors)
        memory_aware = self.initial_memory_state is not None or self.transition_kernel is not None
        can_enumerate = scenario_space_size <= self.max_states_to_enumerate and not memory_aware
        if has_uncertainty and can_enumerate:
            warnings.warn(
                "Uncertainty parameters (structural_sigma and/or "
                "judgment_sigma_scale_by_period) were provided, but enumeration "
                f"mode will be used (scenario space size: {scenario_space_size} "
                f"<= {self.max_states_to_enumerate}). These uncertainty parameters "
                "are ignored in enumeration mode. To ensure uncertainty is applied, "
                "decrease `max_states_to_enumerate` to force sampling mode.",
                UserWarning,
                stacklevel=2,
            )

        # The initial period is resolved under the same regime/kernel semantics as later layers.
        init_state = dict(self.initial)
        init_lock = self._lock_map(init_state)
        rng0 = np.random.default_rng(int(self.base_seed))
        init_memory = self._copy_memory_state(self.initial_memory_state)
        regime0, _ = self._resolve_next_regime(
            current_regime=self.initial_regime,
            realized_scenario=Scenario(init_state, self.base_matrix),
            previous_scenarios=(),
            memory_state=init_memory,
            rng=rng0,
        )
        (
            regime0,
            _threshold_regime_transitions0,
            threshold_regime_reaffirmations0,
        ) = self._resolve_threshold_regime_transitions(
            current_regime=regime0,
            scenario=Scenario(init_state, self.base_matrix),
        )
        base_init_matrix = self._matrix_for_regime(regime0)
        init_matrix = self._sample_active_matrix_for_next_period(
            regime_matrix=base_init_matrix,
            scenario_for_threshold=Scenario(init_state, base_init_matrix),
            next_period_idx=0,
            seed=int(self.base_seed),
        )
        init_s = Scenario(init_state, init_matrix)
        init_regime_state = self._advance_regime_state(
            period=periods[0],
            regime_name=regime0,
            parent_regime=None,
            threshold_regime_reaffirmations=threshold_regime_reaffirmations0,
        )
        if memory_aware:
            root_kernel = self.transition_kernel or DefaultTransitionKernel(
                succession_operator=self.succession_operator
            )
            root, root_memory, _ = root_kernel.step(
                current_scenario=init_s,
                active_matrix=init_matrix,
                regime=regime0,
                memory_state=self._copy_memory_state(init_memory),
                rng=rng0,
                previous_path=(),
            )
            root = self._enforce_locked_scenario(
                scenario=root,
                matrix=init_matrix,
                locked=init_lock,
            )
            if root_memory is None:
                root_memory = self._copy_memory_state(init_memory)
            if root_memory is None:
                root_memory = MemoryState(
                    period=int(periods[0]),
                    values={},
                    flags={},
                    export_label="memory",
                )
            else:
                root_memory = MemoryState(
                    period=int(periods[0]),
                    values=dict(root_memory.values),
                    flags=dict(root_memory.flags),
                    export_label=str(root_memory.export_label),
                )
        else:
            root = self._find_attractor(
                scenario=init_s,
                matrix=init_matrix,
                locked=init_lock,
            )
            root_memory = MemoryState(
                period=int(periods[0]),
                values={},
                flags={},
                export_label="memory",
            )
        root_state = self._build_path_state(
            period=periods[0],
            scenario=root,
            regime_name=regime0,
            active_matrix=init_matrix,
            memory_state=root_memory if memory_aware else None,
            previous_history=(),
        )

        scenarios_by_period: List[List[Scenario]] = [[root_state.scenario]]
        regimes_by_period: List[List[str]] = [[regime0]]
        regime_states_by_period: List[List[BranchRegimeState]] = [[init_regime_state]]
        memories_by_period: List[List[MemoryState]] = (
            [[root_memory]]
            if memory_aware
            else []
        )
        histories_by_period: List[List[Tuple[Scenario, ...]]] = (
            [[self._truncate_history((root_state.scenario,))]] if memory_aware else []
        )
        edges: Dict[Tuple[int, int], Dict[int, float]] = {}
        transition_method: Dict[int, str] = {}
        basin_dist_cache: Dict[
            Tuple[Tuple[object, ...], Tuple[Tuple[str, str], ...]],
            Dict[Tuple[int, ...], float],
        ] = {}
        matrix_signature_cache: Dict[int, Tuple[object, ...]] = {}

        # Expansion is performed forward period by period.
        for p_idx in range(len(periods) - 1):
            t = periods[p_idx]
            t_next = periods[p_idx + 1]

            layer = scenarios_by_period[p_idx]
            layer_regimes = regimes_by_period[p_idx]
            layer_regime_states = regime_states_by_period[p_idx]
            layer_memories = memories_by_period[p_idx] if memory_aware else []
            layer_histories = histories_by_period[p_idx] if memory_aware else []
            next_nodes: List[Scenario] = []
            next_regimes: List[str] = []
            next_regime_states: List[BranchRegimeState] = []
            next_memories: List[MemoryState] = []
            next_histories: List[Tuple[Scenario, ...]] = []
            next_index: Dict[
                Tuple[
                    Scenario,
                    str,
                    Tuple[object, ...],
                    Tuple[object, ...],
                    Tuple[Tuple[int, ...], ...],
                ],
                int,
            ] = {}

            # Method is chosen based on scenario-space size of *unlocked* descriptors.
            # This is a global choice per transition layer (simple and predictable).
            # The worst-case space is computed (no locks), which is conservative.
            method = "enumerate" if can_enumerate else "sample"
            transition_method[int(t)] = method

            for src_idx, parent in enumerate(layer):
                parent_state = parent.to_dict()
                parent_regime = layer_regimes[src_idx]
                parent_regime_state = layer_regime_states[src_idx]
                parent_memory = layer_memories[src_idx] if memory_aware else None
                parent_history = (
                    layer_histories[src_idx] if memory_aware else tuple([parent])
                )

                # For each source node, a distribution over next-period nodes is computed.
                if method == "enumerate":
                    rng = np.random.default_rng(
                        int(self.base_seed) + 1000 * p_idx + 10 * src_idx
                    )
                    next_state = self._apply_cyclic_transitions(parent_state, rng)
                    locked = self._lock_map(next_state)
                    next_regime, _ = self._resolve_next_regime(
                        current_regime=parent_regime,
                        realized_scenario=Scenario(next_state, self.base_matrix),
                        previous_scenarios=parent_history,
                        memory_state=parent_memory,
                        rng=rng,
                    )
                    (
                        next_regime,
                        _threshold_regime_transitions,
                        threshold_regime_reaffirmations,
                    ) = self._resolve_threshold_regime_transitions(
                        current_regime=next_regime,
                        scenario=Scenario(next_state, self.base_matrix),
                    )
                    next_regime_state = self._advance_regime_state(
                        period=int(t_next),
                        regime_name=next_regime,
                        parent_regime=parent_regime_state,
                        threshold_regime_reaffirmations=threshold_regime_reaffirmations,
                    )
                    regime_matrix = self._matrix_for_regime(next_regime)
                    active = self._apply_threshold_modifiers(
                        regime_matrix, Scenario(next_state, regime_matrix)
                    )

                    # Consistent scenarios are enumerated with cyclic descriptors fixed.
                    candidates = _enumerate_consistent_with_locks(
                        matrix=active, locked=locked
                    )
                    if not candidates:
                        # The deterministic successor is used as a fallback if enumeration yields none.
                        s0 = Scenario(next_state, active)
                        chosen = self._find_attractor(scenario=s0, matrix=active, locked=locked)
                        dist_s: Dict[Scenario, float] = {chosen: 1.0}
                    else:
                        # Enumeration edges are weighted by exact basin mass under the
                        # active succession semantics (not uniform over candidates).
                        locked_signature = tuple(
                            sorted((str(k), str(v)) for k, v in locked.items())
                        )
                        matrix_id = id(active)
                        matrix_signature = matrix_signature_cache.get(matrix_id)
                        if matrix_signature is None:
                            matrix_signature = _matrix_state_space_signature(active)
                            matrix_signature_cache[matrix_id] = matrix_signature
                        cache_key = (
                            matrix_signature,
                            locked_signature,
                        )
                        cached_dist = basin_dist_cache.get(cache_key)
                        candidate_idx = {
                            tuple(int(v) for v in scenario.to_indices())
                            for scenario in candidates
                        }
                        if cached_dist is None:
                            basin_counts: Dict[Tuple[int, ...], int] = {}
                            for initial_state in _enumerate_states_with_locks(
                                matrix=active, locked=locked
                            ):
                                attractor = self._find_attractor(
                                    scenario=initial_state,
                                    matrix=active,
                                    locked=locked,
                                )
                                attractor_idx = tuple(
                                    int(v) for v in attractor.to_indices()
                                )
                                if attractor_idx in candidate_idx:
                                    basin_counts[attractor_idx] = (
                                        basin_counts.get(attractor_idx, 0) + 1
                                    )
                            if basin_counts:
                                denom = float(sum(basin_counts.values()))
                                cached_dist = {
                                    idx: float(count) / denom
                                    for idx, count in basin_counts.items()
                                }
                            else:
                                cached_dist = {}
                            basin_dist_cache[cache_key] = dict(cached_dist)
                        if not cached_dist:
                            w = 1.0 / float(len(candidates))
                            dist_s = {c: w for c in candidates}
                        else:
                            dist_s = {
                                Scenario(list(idx), active): float(weight)
                                for idx, weight in cached_dist.items()
                                if idx in candidate_idx
                            }
                            if not dist_s:
                                w = 1.0 / float(len(candidates))
                                dist_s = {c: w for c in candidates}
                    dist_s = _prune_distribution(
                        dist_s,
                        prune_policy=self.prune_policy,
                        per_parent_top_k=self.per_parent_top_k,
                        min_edge_weight=self.min_edge_weight,
                    )
                    out: Dict[int, float] = {}
                    for c, ww in dist_s.items():
                        child_history = self._truncate_history(parent_history + (c,))
                        child_state = self._build_path_state(
                            period=int(t_next),
                            scenario=c,
                            regime_name=next_regime,
                            active_matrix=active,
                            memory_state=None,
                            previous_history=parent_history,
                        )
                        key = (
                            c,
                            next_regime,
                            (
                                bool(next_regime_state.entered_regime),
                                int(next_regime_state.regime_entry_period),
                                int(next_regime_state.regime_spell_index),
                                tuple(next_regime_state.threshold_regime_reaffirmations),
                            ),
                            ("none",),
                            child_state.history_signature,
                        )
                        if key not in next_index:
                            next_index[key] = len(next_nodes)
                            next_nodes.append(c)
                            next_regimes.append(next_regime)
                            next_regime_states.append(next_regime_state)
                            next_memories.append(
                                MemoryState(
                                    period=int(t_next),
                                    values={},
                                    flags={},
                                    export_label="memory",
                                )
                            )
                            next_histories.append(self._truncate_history(parent_history + (child_state.scenario,)))
                        out[next_index[key]] = out.get(next_index[key], 0.0) + float(ww)
                    edges[(p_idx, src_idx)] = out
                    continue

                # Sampling mode: transition distribution is estimated by Monte Carlo.
                counts_s: Dict[
                    Tuple[
                        Scenario,
                        str,
                        Tuple[object, ...],
                        Tuple[object, ...],
                        Tuple[Tuple[int, ...], ...],
                    ],
                    int,
                ] = {}
                sampled_memories: Dict[
                    Tuple[
                        Scenario,
                        str,
                        Tuple[object, ...],
                        Tuple[object, ...],
                        Tuple[Tuple[int, ...], ...],
                    ],
                    MemoryState,
                ] = {}
                sampled_histories: Dict[
                    Tuple[
                        Scenario,
                        str,
                        Tuple[object, ...],
                        Tuple[object, ...],
                        Tuple[Tuple[int, ...], ...],
                    ],
                    Tuple[Scenario, ...],
                ] = {}
                sampled_regime_states: Dict[
                    Tuple[
                        Scenario,
                        str,
                        Tuple[object, ...],
                        Tuple[object, ...],
                        Tuple[Tuple[int, ...], ...],
                    ],
                    BranchRegimeState,
                ] = {}
                for m in range(self.n_transition_samples):
                    seeds = seeds_for_run(self.base_seed + 1000 * p_idx + 17 * src_idx, m)
                    rng = np.random.default_rng(int(seeds["dynamic_shock_seed"]))

                    next_state = self._apply_cyclic_transitions(parent_state, rng)
                    locked = self._lock_map(next_state)
                    next_regime, _ = self._resolve_next_regime(
                        current_regime=parent_regime,
                        realized_scenario=Scenario(next_state, self.base_matrix),
                        previous_scenarios=parent_history,
                        memory_state=parent_memory,
                        rng=rng,
                    )
                    (
                        next_regime,
                        _threshold_regime_transitions,
                        threshold_regime_reaffirmations,
                    ) = self._resolve_threshold_regime_transitions(
                        current_regime=next_regime,
                        scenario=Scenario(next_state, self.base_matrix),
                    )
                    next_regime_state = self._advance_regime_state(
                        period=int(t_next),
                        regime_name=next_regime,
                        parent_regime=parent_regime_state,
                        threshold_regime_reaffirmations=threshold_regime_reaffirmations,
                    )
                    regime_matrix = self._matrix_for_regime(next_regime)

                    active = self._sample_active_matrix_for_next_period(
                        regime_matrix=regime_matrix,
                        scenario_for_threshold=Scenario(
                            next_state, regime_matrix
                        ),
                        next_period_idx=p_idx + 1,
                        seed=int(seeds["judgment_uncertainty_seed"]),
                    )

                    dyn_shocks = None
                    if self.dynamic_tau is not None:
                        sm = ShockModel(active)
                        sm.add_dynamic_shocks(
                            periods=[int(t_next)],
                            tau=float(self.dynamic_tau),
                            rho=float(self.dynamic_rho),
                            innovation_dist=self.dynamic_innovation_dist,
                            innovation_df=self.dynamic_innovation_df,
                            jump_prob=self.dynamic_jump_prob,
                            jump_scale=self.dynamic_jump_scale,
                        )
                        dyn = sm.sample_dynamic_shocks(int(seeds["dynamic_shock_seed"]))
                        dyn_shocks = dyn[int(t_next)]

                    s0 = Scenario(next_state, active)
                    if memory_aware:
                        period_operator: SuccessionOperator = self.succession_operator
                        if dyn_shocks is not None:
                            period_operator = ShockAwareGlobalSuccession(dyn_shocks)
                        if locked:
                            period_operator = _LockedSuccessionOperator(period_operator, locked)
                        kernel = self.transition_kernel or DefaultTransitionKernel(
                            succession_operator=period_operator
                        )
                        child, child_memory, _ = kernel.step(
                            current_scenario=s0,
                            active_matrix=active,
                            regime=next_regime,
                            memory_state=self._copy_memory_state(parent_memory),
                            rng=rng,
                            previous_path=parent_history,
                        )
                        child = self._enforce_locked_scenario(
                            scenario=child,
                            matrix=active,
                            locked=locked,
                        )
                        if child_memory is None:
                            child_memory = self._copy_memory_state(parent_memory)
                        if child_memory is None:
                            child_memory = MemoryState(
                                period=int(t_next),
                                values={},
                                flags={},
                                export_label="memory",
                            )
                        else:
                            child_memory = MemoryState(
                                period=int(t_next),
                                values=dict(child_memory.values),
                                flags=dict(child_memory.flags),
                                export_label=str(child_memory.export_label),
                            )
                    else:
                        child = self._find_attractor(
                            scenario=s0,
                            matrix=active,
                            locked=locked,
                            dynamic_shocks=dyn_shocks,
                        )
                        child_memory = MemoryState(
                            period=int(t_next),
                            values={},
                            flags={},
                            export_label="memory",
                        )

                    child_history = self._truncate_history(parent_history + (child,))
                    child_state = self._build_path_state(
                        period=int(t_next),
                        scenario=child,
                        regime_name=next_regime,
                        active_matrix=active,
                        memory_state=child_memory,
                        previous_history=parent_history,
                    )
                    key = (
                        child,
                        next_regime,
                        (
                            bool(next_regime_state.entered_regime),
                            int(next_regime_state.regime_entry_period),
                            int(next_regime_state.regime_spell_index),
                            tuple(next_regime_state.threshold_regime_reaffirmations),
                        ),
                        self._memory_signature(child_memory),
                        child_state.history_signature,
                    )
                    counts_s[key] = counts_s.get(key, 0) + 1
                    sampled_memories[key] = child_memory
                    sampled_histories[key] = child_history
                    sampled_regime_states[key] = next_regime_state

                total = float(sum(counts_s.values()) or 1.0)
                dist_s = {k: float(c) / total for k, c in counts_s.items()}
                dist_s = _prune_distribution(
                    dist_s,
                    prune_policy=self.prune_policy,
                    per_parent_top_k=self.per_parent_top_k,
                    min_edge_weight=self.min_edge_weight,
                )
                out: Dict[int, float] = {}
                for key, ww in dist_s.items():
                    child, child_regime, _regime_sig, _memory_sig, _history_sig = key
                    if key not in next_index:
                        next_index[key] = len(next_nodes)
                        next_nodes.append(child)
                        next_regimes.append(child_regime)
                        next_regime_states.append(sampled_regime_states[key])
                        next_memories.append(sampled_memories[key])
                        next_histories.append(sampled_histories[key])
                    out[next_index[key]] = out.get(next_index[key], 0.0) + float(ww)
                edges[(p_idx, src_idx)] = out

            # Optional layer-level pruning is performed to prevent node explosion.
            if self.max_nodes_per_period is not None and len(next_nodes) > self.max_nodes_per_period:
                incoming: Dict[int, float] = {}
                for src_idx in range(len(layer)):
                    out = edges.get((p_idx, src_idx), {})
                    for tgt_idx, w in out.items():
                        incoming[int(tgt_idx)] = incoming.get(int(tgt_idx), 0.0) + float(w)

                keep_n = max(1, int(self.max_nodes_per_period))
                kept_old = {
                    idx for idx, _w in sorted(incoming.items(), key=lambda x: x[1], reverse=True)[:keep_n]
                }
                if not kept_old:
                    kept_old = {0}

                remap = {old: new for new, old in enumerate(sorted(kept_old))}
                new_next_nodes = [next_nodes[old] for old in sorted(kept_old)]
                new_next_regimes = [next_regimes[old] for old in sorted(kept_old)]
                new_next_regime_states = [
                    next_regime_states[old] for old in sorted(kept_old)
                ]
                new_next_memories = [next_memories[old] for old in sorted(kept_old)]
                new_next_histories = [next_histories[old] for old in sorted(kept_old)]

                # Each parent's outgoing distribution is filtered and renormalized.
                for src_idx in range(len(layer)):
                    out = edges.get((p_idx, src_idx), {})
                    filtered = {remap[int(k)]: float(v) for k, v in out.items() if int(k) in kept_old}
                    s = float(sum(filtered.values()))
                    if s <= 0.0:
                        filtered = {}
                    else:
                        filtered = {k: v / s for k, v in filtered.items()}
                    edges[(p_idx, src_idx)] = filtered

                scenarios_by_period.append(new_next_nodes)
                regimes_by_period.append(new_next_regimes)
                regime_states_by_period.append(new_next_regime_states)
                if memory_aware:
                    memories_by_period.append(new_next_memories)
                    histories_by_period.append(new_next_histories)
            else:
                scenarios_by_period.append(next_nodes)
                regimes_by_period.append(next_regimes)
                regime_states_by_period.append(next_regime_states)
                if memory_aware:
                    memories_by_period.append(next_memories)
                    histories_by_period.append(next_histories)

        # Top-K most likely paths are computed (simple beam search).
        top_k = max(1, int(top_k))
        paths: List[Tuple[Tuple[int, ...], float]] = [((0,), 1.0)]
        for p_idx in range(len(periods) - 1):
            new_paths: List[Tuple[Tuple[int, ...], float]] = []
            for node_path, w in paths:
                src = node_path[-1]
                out = edges.get((p_idx, src), {})
                for nxt, p in out.items():
                    new_paths.append((node_path + (int(nxt),), float(w) * float(p)))
            new_paths.sort(key=lambda x: x[1], reverse=True)
            paths = new_paths[:top_k]

        used_sampling = any(method == "sample" for method in transition_method.values())
        return BranchingResult(
            periods=periods,
            scenarios_by_period=tuple(tuple(layer) for layer in scenarios_by_period),
            edges=edges,
            transition_method=transition_method,
            top_paths=tuple(paths),
            active_regimes=tuple(tuple(layer) for layer in regimes_by_period),
            regime_states_by_period=tuple(
                tuple(layer) for layer in regime_states_by_period
            ),
            memory_states_by_period=tuple(tuple(layer) for layer in memories_by_period),
            history_signatures_by_period=tuple(
                tuple(self._history_signature(history) for history in layer)
                for layer in histories_by_period
            ),
            approximation_contract=(
                "memory_aware_sampling: nodes are keyed by "
                "(scenario, regime, memory_state, retained_history_signature); "
                f"transitions use retained realised history with history_horizon="
                f"{self.history_horizon if self.history_horizon is not None else 'full'}"
                if memory_aware
                else (
                    "approximate_scenario_regime_branching"
                    if used_sampling
                    else "exact_scenario_regime_branching"
                )
            ),
        )

