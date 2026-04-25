"""
Succession operators for finding consistent scenarios.

This module provides operators for iteratively adjusting scenarios to find
consistent states through succession analysis. Includes global and local
succession strategies, as well as attractor finding capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, List, Mapping, Optional, Union

import numpy as np

from cib.constraints import ConstraintIndex
from cib.core import (
    CIBMatrix,
    DEFAULT_FLOAT_ATOL,
    DEFAULT_FLOAT_RTOL,
    ImpactBalance,
    Scenario,
)


@dataclass
class AttractorResult:
    """
    Result of succession analysis finding an attractor.

    Attributes:
        attractor: Either a single Scenario (fixed point) or a list of
            Scenarios (cycle). When is_cycle is True, the list contains all
            scenarios in the periodic sequence, ordered from first occurrence
            to the point where the cycle was detected. When is_cycle is False,
            attractor is a single Scenario representing the fixed point.
        path: List of scenarios visited during succession, including the
            initial scenario and all successors up to the attractor.
        iterations: Number of iterations performed before convergence or
            cycle detection.
        is_cycle: Whether the result is a cycle (True) or fixed point (False).
            Cycles occur when succession revisits a previously encountered
            scenario, creating a periodic attractor rather than converging
            to a single state.
        converged: Whether the result is a true attractor (True) or the last
            state when the iteration cap was reached without convergence (False).
    """

    attractor: Union[Scenario, List[Scenario]]
    path: List[Scenario]
    iterations: int
    is_cycle: bool
    converged: bool = True


class SuccessionOperator(ABC):
    """
    Base class for succession operators.

    Succession operators iteratively adjust scenarios to find consistent
    states by updating descriptor states based on impact balances.
    """

    @abstractmethod
    def find_successor(
        self, scenario: Scenario, matrix: CIBMatrix
    ) -> Scenario:
        """
        Compute the next scenario in the succession sequence.

        Args:
            scenario: Current scenario.
            matrix: CIB matrix containing impact relationships.

        Returns:
            Next scenario in the succession.
        """
        pass

    def find_attractor(
        self,
        initial: Scenario,
        matrix: CIBMatrix,
        max_iterations: int = 1000,
        allow_partial: bool = False,
    ) -> AttractorResult:
        """
        Find an attractor (fixed point or cycle) from an initial scenario.

        The method detects cycles by maintaining a visited set of scenarios.
        When a successor scenario has been encountered before, a cycle is
        identified and extracted. The cycle list contains all scenarios
        from the first occurrence to the current duplicate, representing
        the periodic sequence.

        Args:
            initial: Starting scenario.
            matrix: CIB matrix containing impact relationships.
            max_iterations: Maximum number of iterations before stopping.
            allow_partial: If True, non-convergence within max_iterations does
                not raise; the last state is returned with converged=False.

        Returns:
            AttractorResult containing the attractor and path information.
            If is_cycle is True, the attractor is a list of scenarios
            representing the cycle. If is_cycle is False, the attractor
            is a single Scenario representing the fixed point. When
            converged is False, the attractor is the last state reached.

        Raises:
            RuntimeError: When allow_partial is False, raised if max_iterations
                is exceeded without convergence.
        """
        path: List[Scenario] = [initial]
        visited: dict[Scenario, int] = {initial: 0}

        current = initial
        for iteration in range(1, max_iterations + 1):
            successor = self.find_successor(current, matrix)
            path.append(successor)

            if successor == current:
                return AttractorResult(
                    attractor=successor,
                    path=path,
                    iterations=iteration,
                    is_cycle=False,
                )

            if successor in visited:
                cycle_start = visited[successor]
                # The repeated closing state is removed to avoid duplication bias.
                # Example: A -> B -> C -> B is represented as [B, C], not [B, C, B].
                cycle = path[cycle_start:-1]
                return AttractorResult(
                    attractor=cycle,
                    path=path,
                    iterations=iteration,
                    is_cycle=True,
                )

            visited[successor] = iteration
            current = successor

        if allow_partial:
            return AttractorResult(
                attractor=current,
                path=path,
                iterations=max_iterations,
                is_cycle=False,
                converged=False,
            )
        raise RuntimeError(
            f"Succession did not converge within {max_iterations} iterations"
        )


class GlobalSuccession(SuccessionOperator):
    """
    Global succession operator.

    Adjusts all inconsistent descriptors simultaneously to their
    maximum-impact states. This is the most common succession rule.
    """

    def __init__(
        self,
        *,
        float_atol: float = DEFAULT_FLOAT_ATOL,
        float_rtol: float = DEFAULT_FLOAT_RTOL,
    ) -> None:
        self.float_atol = float(float_atol)
        self.float_rtol = float(float_rtol)

    def find_successor(
        self, scenario: Scenario, matrix: CIBMatrix
    ) -> Scenario:
        """
        Compute successor by updating all descriptors to max-impact states.

        Args:
            scenario: Current scenario.
            matrix: CIB matrix containing impact relationships.

        Returns:
            New scenario with all descriptors set to their maximum-impact
            states.
        """
        balance = ImpactBalance(scenario, matrix)

        new_state_dict: dict[str, str] = {}
        for descriptor in matrix.descriptors:
            current_state = scenario.get_state(descriptor)
            current_score = float(balance.get_score(descriptor, current_state))
            state_scores = balance.balance[descriptor]
            max_state = max(state_scores, key=state_scores.get)
            max_score = float(state_scores[max_state])
            if float(self.float_atol) == 0.0 and float(self.float_rtol) == 0.0:
                # Preserve legacy behavior exactly when no tolerance is requested.
                new_state_dict[descriptor] = max_state
            elif np.isclose(
                current_score,
                max_score,
                atol=float(self.float_atol),
                rtol=float(self.float_rtol),
            ):
                new_state_dict[descriptor] = current_state
            else:
                new_state_dict[descriptor] = max_state

        return Scenario(new_state_dict, matrix)


class ConstrainedGlobalSuccession(SuccessionOperator):
    """
    Global succession operator with optional feasibility-aware repair.

    A global successor is computed first. If infeasibility is detected,
    a bounded top-k/backtracking search is applied to recover a valid
    successor that remains as close as possible to the unconstrained
    impact optimum.

    Repair is global by design: it searches within the global impact-balance
    optimum subject to feasibility constraints. Dynamic workflows may supply
    locked descriptor states so the repair search cannot rewrite exogenous
    states that were fixed for the current period.
    """

    def __init__(
        self,
        constraint_index: ConstraintIndex,
        *,
        constraint_mode: str = "strict",
        constrained_top_k: int = 2,
        constrained_backtracking_depth: int = 2,
    ) -> None:
        self.constraint_index = constraint_index
        self.constraint_mode = str(constraint_mode).strip().lower()
        self.constrained_top_k = int(constrained_top_k)
        self.constrained_backtracking_depth = int(constrained_backtracking_depth)
        if self.constraint_mode not in {"strict", "repair"}:
            raise ValueError("constraint_mode must be 'strict' or 'repair'")
        if self.constrained_top_k <= 0:
            raise ValueError("constrained_top_k must be positive")
        if self.constrained_backtracking_depth < 0:
            raise ValueError("constrained_backtracking_depth must be non-negative")

    def _is_valid(self, scenario: Scenario) -> bool:
        z = np.asarray(scenario.to_indices(), dtype=np.int64)
        return bool(self.constraint_index.is_full_valid(z))

    def _repair_to_valid(
        self,
        scenario: Scenario,
        matrix: CIBMatrix,
        *,
        locked_states: Optional[Mapping[str, str]] = None,
    ) -> Optional[Scenario]:
        balance = ImpactBalance(scenario, matrix)
        desc_names = list(matrix.descriptors.keys())
        n_desc = len(desc_names)
        locked: Dict[str, str] = {
            str(descriptor): str(state)
            for descriptor, state in (locked_states or {}).items()
        }
        unknown_locked = sorted(set(locked).difference(desc_names))
        if unknown_locked:
            raise ValueError(
                f"locked_states contains unknown descriptors: {unknown_locked}"
            )

        ranked_options: List[List[str]] = []
        base_states: List[str] = []
        mutable_positions: List[int] = []
        for descriptor in desc_names:
            if descriptor in locked:
                locked_state = locked[descriptor]
                if locked_state not in matrix.descriptors[descriptor]:
                    raise ValueError(
                        f"locked_states contains invalid state {locked_state!r} "
                        f"for descriptor {descriptor!r}"
                    )
                ranked_options.append([locked_state])
                base_states.append(locked_state)
                continue
            state_scores = list(balance.balance[descriptor].items())
            state_scores.sort(key=lambda x: float(x[1]), reverse=True)
            ranked_states = [state for state, _score in state_scores[: self.constrained_top_k]]
            if not ranked_states:
                return None
            ranked_options.append(ranked_states)
            base_states.append(ranked_states[0])
            mutable_positions.append(len(ranked_options) - 1)

        best_candidate: Optional[Scenario] = None
        best_objective = float("-inf")

        def _build_scenario(state_indices: List[int]) -> Scenario:
            out = [base_states[i] for i in range(n_desc)]
            for i, opt_idx in enumerate(state_indices):
                out[i] = ranked_options[i][opt_idx]
            return Scenario(dict(zip(desc_names, out)), matrix)

        base_idx = [0 for _ in range(n_desc)]
        base_scenario = _build_scenario(base_idx)
        if self._is_valid(base_scenario):
            return base_scenario

        max_depth = min(int(self.constrained_backtracking_depth), len(mutable_positions))
        for depth in range(1, max_depth + 1):
            for changed in combinations(mutable_positions, depth):
                nonzero_choices = []
                for d in changed:
                    k = len(ranked_options[d])
                    if k <= 1:
                        nonzero_choices = []
                        break
                    nonzero_choices.append(range(1, k))
                if not nonzero_choices:
                    continue
                for picks in product(*nonzero_choices):
                    state_idx = [0 for _ in range(n_desc)]
                    for d, pick in zip(changed, picks):
                        state_idx[d] = int(pick)
                    candidate = _build_scenario(state_idx)
                    if not self._is_valid(candidate):
                        continue
                    objective = 0.0
                    for i, descriptor in enumerate(desc_names):
                        state = candidate.get_state(descriptor)
                        objective += float(balance.get_score(descriptor, state))
                    # Deterministic ordering in ties is enforced by lexicographic index vector.
                    if objective > best_objective:
                        best_objective = objective
                        best_candidate = candidate
                    elif best_candidate is not None and np.isclose(objective, best_objective):
                        if tuple(candidate.to_indices()) < tuple(best_candidate.to_indices()):
                            best_candidate = candidate
        return best_candidate

    def repair_to_valid(
        self,
        scenario: Scenario,
        matrix: CIBMatrix,
        *,
        locked_states: Optional[Mapping[str, str]] = None,
    ) -> Optional[Scenario]:
        """
        Return a repaired feasible scenario when possible.

        When `locked_states` is provided, those descriptor assignments are
        treated as immutable during repair and must be preserved exactly.
        """
        return self._repair_to_valid(
            scenario, matrix, locked_states=locked_states
        )

    def find_successor(self, scenario: Scenario, matrix: CIBMatrix) -> Scenario:
        candidate = GlobalSuccession().find_successor(scenario, matrix)
        if self._is_valid(candidate):
            return candidate
        if self.constraint_mode == "strict":
            raise ValueError("Constraint infeasibility was encountered during succession")
        repaired = self._repair_to_valid(candidate, matrix)
        if repaired is None:
            raise ValueError("Constraint repair failed during succession")
        return repaired


class LocalSuccession(SuccessionOperator):
    """
    Local succession operator.

    Adjusts only the most inconsistent descriptor (the one with the largest
    gap between current and maximum impact score).
    """

    def __init__(
        self,
        *,
        float_atol: float = DEFAULT_FLOAT_ATOL,
        float_rtol: float = DEFAULT_FLOAT_RTOL,
    ) -> None:
        self.float_atol = float(float_atol)
        self.float_rtol = float(float_rtol)

    def find_successor(
        self, scenario: Scenario, matrix: CIBMatrix
    ) -> Scenario:
        """
        Compute successor by updating only the most inconsistent descriptor.

        Args:
            scenario: Current scenario.
            matrix: CIB matrix containing impact relationships.

        Returns:
            New scenario with the most inconsistent descriptor updated to
            its maximum-impact state.
        """
        balance = ImpactBalance(scenario, matrix)

        new_state_dict = scenario.to_dict()
        max_gap = float("-inf")
        target_descriptor = None

        for descriptor in matrix.descriptors:
            current_state = scenario.get_state(descriptor)
            current_score = balance.get_score(descriptor, current_state)
            state_scores = balance.balance[descriptor]
            max_state = max(state_scores, key=state_scores.get)
            max_score = balance.get_score(descriptor, max_state)

            gap = max_score - current_score
            if gap > max_gap:
                max_gap = gap
                target_descriptor = descriptor

        if (
            target_descriptor is not None
            and max_gap > 0
            and not np.isclose(
                max_gap,
                0.0,
                atol=float(self.float_atol),
                rtol=float(self.float_rtol),
            )
        ):
            target_scores = balance.balance[target_descriptor]
            new_state_dict[target_descriptor] = max(target_scores, key=target_scores.get)

        return Scenario(new_state_dict, matrix)


class AttractorFinder:
    """
    Finds attractors (consistent scenarios or cycles) in a CIB system.

    Provides methods to discover all attractors through exhaustive search
    or by starting from specific initial scenarios.
    """

    def __init__(self, matrix: CIBMatrix) -> None:
        """
        The attractor finder is initialised with a CIB matrix.

        Args:
            matrix: CIB matrix to analyse.
        """
        self.matrix = matrix

    def find_attractors(
        self,
        initial_scenarios: Optional[List[Scenario]] = None,
        succession_operator: Optional[SuccessionOperator] = None,
    ) -> List[AttractorResult]:
        """
        Find all attractors by running succession from initial scenarios.

        Args:
            initial_scenarios: List of starting scenarios. If None, uses
                all possible scenarios (only feasible for small systems).
            succession_operator: Succession operator to use. Defaults to
                GlobalSuccession if not specified.

        Returns:
            List of AttractorResult objects, one per unique attractor found.
        """
        if succession_operator is None:
            succession_operator = GlobalSuccession()

        if initial_scenarios is None:
            initial_scenarios = self._enumerate_all_scenarios()

        attractors: dict[Scenario, AttractorResult] = {}
        for initial in initial_scenarios:
            result = succession_operator.find_attractor(initial, self.matrix)
            if isinstance(result.attractor, Scenario):
                key = result.attractor
            else:
                cycle = result.attractor
                key = min(cycle, key=lambda s: tuple(s.to_indices()))

            if key not in attractors:
                attractors[key] = result

        return list(attractors.values())

    def get_basin(
        self, attractor: Scenario, matrix: CIBMatrix
    ) -> List[Scenario]:
        """
        Find all scenarios that converge to a given attractor.

        Args:
            attractor: The target attractor scenario.
            matrix: CIB matrix (should match the one used to find attractor).

        Returns:
            List of scenarios that converge to the attractor.

        Note:
            This method enumerates all scenarios, so it is only feasible
            for small systems.
        """
        basin: List[Scenario] = []
        succession = GlobalSuccession()

        all_scenarios = self._enumerate_all_scenarios()
        for initial in all_scenarios:
            result = succession.find_attractor(initial, matrix)
            if isinstance(result.attractor, Scenario):
                if result.attractor == attractor:
                    basin.append(initial)
            else:
                if attractor in result.attractor:
                    basin.append(initial)

        return basin

    def find_attractors_exhaustive(
        self, matrix: CIBMatrix
    ) -> List[Scenario]:
        """
        Enumerate all consistent scenarios (exhaustive search).

        Args:
            matrix: CIB matrix to analyse.

        Returns:
            List of all consistent scenarios.

        Note:
            This method is only feasible for small systems (R ≤ 50k scenarios).
        """
        from cib.core import ConsistencyChecker

        consistent: List[Scenario] = []
        all_scenarios = self._enumerate_all_scenarios()

        for scenario in all_scenarios:
            if ConsistencyChecker.check_consistency(scenario, matrix):
                consistent.append(scenario)

        return consistent

    def _enumerate_all_scenarios(self) -> List[Scenario]:
        """
        Generate all possible scenarios for the matrix.

        Returns:
            List of all possible scenario combinations.
        """
        from itertools import product

        descriptor_names = list(self.matrix.descriptors.keys())
        state_lists = [
            self.matrix.descriptors[desc] for desc in descriptor_names
        ]

        scenarios: List[Scenario] = []
        for state_combination in product(*state_lists):
            state_dict = dict(zip(descriptor_names, state_combination))
            scenario = Scenario(state_dict, self.matrix)
            scenarios.append(scenario)

        return scenarios
