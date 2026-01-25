"""
Succession operators for finding consistent scenarios.

This module provides operators for iteratively adjusting scenarios to find
consistent states through succession analysis. Includes global and local
succession strategies, as well as attractor finding capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

from cib.core import CIBMatrix, ImpactBalance, Scenario


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
    """

    attractor: Union[Scenario, List[Scenario]]
    path: List[Scenario]
    iterations: int
    is_cycle: bool


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

        Returns:
            AttractorResult containing the attractor and path information.
            If is_cycle is True, the attractor is a list of scenarios
            representing the cycle. If is_cycle is False, the attractor
            is a single Scenario representing the fixed point.

        Raises:
            RuntimeError: If max_iterations is exceeded without convergence.
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

        raise RuntimeError(
            f"Succession did not converge within {max_iterations} iterations"
        )


class GlobalSuccession(SuccessionOperator):
    """
    Global succession operator.

    Adjusts all inconsistent descriptors simultaneously to their
    maximum-impact states. This is the most common succession rule.
    """

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
        max_states = balance.get_max_states()

        new_state_dict: dict[str, str] = {}
        for descriptor in matrix.descriptors:
            new_state_dict[descriptor] = max_states[descriptor]

        return Scenario(new_state_dict, matrix)


class LocalSuccession(SuccessionOperator):
    """
    Local succession operator.

    Adjusts only the most inconsistent descriptor (the one with the largest
    gap between current and maximum impact score).
    """

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
        max_states = balance.get_max_states()

        new_state_dict = scenario.to_dict()
        max_gap = float("-inf")
        target_descriptor = None

        for descriptor in matrix.descriptors:
            current_state = scenario.get_state(descriptor)
            current_score = balance.get_score(descriptor, current_state)
            max_state = max_states[descriptor]
            max_score = balance.get_score(descriptor, max_state)

            gap = max_score - current_score
            if gap > max_gap:
                max_gap = gap
                target_descriptor = descriptor

        if target_descriptor is not None and max_gap > 0:
            new_state_dict[target_descriptor] = max_states[target_descriptor]

        return Scenario(new_state_dict, matrix)


class AttractorFinder:
    """
    Finds attractors (consistent scenarios or cycles) in a CIB system.

    Provides methods to discover all attractors through exhaustive search
    or by starting from specific initial scenarios.
    """

    def __init__(self, matrix: CIBMatrix) -> None:
        """
        Initialize attractor finder with a CIB matrix.

        Args:
            matrix: CIB matrix to analyze.
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
            matrix: CIB matrix to analyze.

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
