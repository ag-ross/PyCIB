"""
Scenario analysis tools for CIB systems.

This module provides utilities for enumerating scenarios, finding consistent
scenarios, and ranking scenarios by various metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.constraints import ConstraintIndex, ConstraintSpec

# Imports for Monte Carlo are attempted.
try:
    from cib.uncertainty import UncertainCIBMatrix
except ImportError:
    UncertainCIBMatrix = None

try:
    from cib.bayesian import GaussianCIBMatrix
except ImportError:
    GaussianCIBMatrix = None

from cib.example_data import seeds_for_run

# Default cap when :meth:`ScenarioAnalyzer.enumerate_scenarios` is called without
# ``max_scenarios`` and with ``allow_unbounded=False`` (prevents accidental OOM).
_DEFAULT_ENUMERATE_SCENARIO_CAP = 50_000


@dataclass(frozen=True)
class FindAllConsistentResult:
    """
    Result of :meth:`ScenarioAnalyzer.find_all_consistent` when ``return_metadata=True``.

    For large systems the ``random_restarts`` mode returns a **shortlist**, not
    the full consistent set.
    """

    scenarios: List[Scenario]
    enumeration_mode: Literal["exhaustive", "random_restarts"]
    total_scenario_space: int
    is_complete: bool
    requested_mode: Literal["exhaustive", "shortlist", "auto"]
    effective_mode: Literal["exhaustive", "shortlist"]
    switch_reason: str
    max_scenarios_requested: Optional[int]
    exhaustive_threshold: int
    n_restarts: int
    max_iterations: int


class ScenarioAnalyzer:
    """
    Analyzes scenarios in a CIB system.

    Provides methods to enumerate all possible scenarios, find consistent
    scenarios, and filter or rank scenarios based on consistency.
    """

    def __init__(self, matrix: CIBMatrix) -> None:
        """
        The analyser is initialised with a CIB matrix.

        Args:
            matrix: CIB matrix to analyse.
        """
        self.matrix = matrix

    def enumerate_scenarios(
        self,
        max_scenarios: Optional[int] = None,
        *,
        allow_unbounded: bool = False,
    ) -> List[Scenario]:
        """
        Generate all possible scenarios for the matrix.

        Args:
            max_scenarios: If set, raises ``ValueError`` when the Cartesian
                product size exceeds this cap (guards memory and runtime).
            allow_unbounded: When ``False`` (default) and ``max_scenarios`` is
                ``None``, the scenario count must not exceed 50,000.
                Set ``True`` to materialize
                the full Cartesian product regardless of size (dangerous for
                large matrices).

        Returns:
            List of all possible scenario combinations.

        Note:
            The number of scenarios grows exponentially with the number
            of descriptors and states. Use with caution for large systems.
        """
        total = 1
        for count in self.matrix.state_counts:
            total *= int(count)
        limit: Optional[int]
        if max_scenarios is not None:
            limit = int(max_scenarios)
        elif allow_unbounded:
            limit = None
        else:
            limit = int(_DEFAULT_ENUMERATE_SCENARIO_CAP)
        if limit is not None and total > limit:
            raise ValueError(
                f"enumerate_scenarios: scenario space has size {total}, "
                f"which exceeds max_scenarios={limit}"
                + (
                    ". Pass a larger max_scenarios, or allow_unbounded=True "
                    "if you accept the memory cost."
                    if max_scenarios is None and not allow_unbounded
                    else ""
                )
            )

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

    def find_all_consistent(
        self,
        max_scenarios: Optional[int] = None,
        n_restarts: int = 200,
        seed: Optional[int] = None,
        max_iterations: int = 1000,
        constraints: Optional[Sequence[ConstraintSpec]] = None,
        constrained_mode: Literal["strict", "repair"] = "repair",
        constrained_top_k: int = 2,
        constrained_backtracking_depth: int = 2,
        return_metadata: bool = False,
        mode: Literal["exhaustive", "shortlist", "auto"] = "exhaustive",
    ) -> Union[List[Scenario], FindAllConsistentResult]:
        """
        Find consistent scenarios in the system.

        By default, exhaustive enumeration is required. If the scenario space
        exceeds configured limits, a ``ValueError`` is raised unless shortlist
        mode is explicitly requested.

        Args:
            max_scenarios: Maximum number of scenarios to enumerate.
                If None, enumerates all scenarios (only for small systems).
            n_restarts: Number of random initial scenarios to use when the
                system is too large to enumerate.
            seed: Random seed for reproducibility of random restarts.
            max_iterations: Maximum iterations per succession run.
            constrained_mode: Constraint handling mode for shortlist succession when
                constraints are provided.
            constrained_top_k: Candidate breadth for constrained repair.
            constrained_backtracking_depth: Repair search depth for constrained repair.
            return_metadata: If True, returns :class:`FindAllConsistentResult`
                with ``enumeration_mode`` and ``total_scenario_space``.
            mode: Execution mode contract:
                - ``"exhaustive"`` (default): require exhaustive enumeration;
                  raise if infeasible under threshold/cap.
                - ``"shortlist"``: use random-restart shortlist explicitly.
                - ``"auto"``: preserve legacy auto-switch behavior.

        Returns:
            List of consistent scenarios found, or ``FindAllConsistentResult``
            when ``return_metadata`` is True.
        """
        total_scenarios = 1
        for count in self.matrix.state_counts:
            total_scenarios *= count

        if max_scenarios is not None and max_scenarios <= 0:
            raise ValueError("max_scenarios must be positive if specified")
        if n_restarts <= 0:
            raise ValueError("n_restarts must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if constrained_mode not in {"strict", "repair"}:
            raise ValueError("constrained_mode must be 'strict' or 'repair'")
        if constrained_top_k <= 0:
            raise ValueError("constrained_top_k must be positive")
        if constrained_backtracking_depth < 0:
            raise ValueError("constrained_backtracking_depth must be non-negative")
        if mode not in {"exhaustive", "shortlist", "auto"}:
            raise ValueError("mode must be 'exhaustive', 'shortlist', or 'auto'")

        exhaustive_threshold = 50_000
        threshold_exceeded = total_scenarios > exhaustive_threshold
        max_cap_exceeded = (
            max_scenarios is not None and total_scenarios > int(max_scenarios)
        )
        can_enumerate = not threshold_exceeded and not max_cap_exceeded

        if mode == "shortlist":
            out = self.find_consistent_via_random_restarts(
                n_restarts=n_restarts,
                seed=seed,
                max_iterations=max_iterations,
                constraints=constraints,
                constrained_mode=constrained_mode,
                constrained_top_k=constrained_top_k,
                constrained_backtracking_depth=constrained_backtracking_depth,
            )
            if return_metadata:
                return FindAllConsistentResult(
                    scenarios=out,
                    enumeration_mode="random_restarts",
                    total_scenario_space=int(total_scenarios),
                    is_complete=False,
                    requested_mode="shortlist",
                    effective_mode="shortlist",
                    switch_reason="explicit_shortlist_mode",
                    max_scenarios_requested=(
                        int(max_scenarios) if max_scenarios is not None else None
                    ),
                    exhaustive_threshold=int(exhaustive_threshold),
                    n_restarts=int(n_restarts),
                    max_iterations=int(max_iterations),
                )
            return out

        if mode == "exhaustive" and not can_enumerate:
            limits = [f"exhaustive_threshold={int(exhaustive_threshold)}"]
            if max_scenarios is not None:
                limits.append(f"max_scenarios={int(max_scenarios)}")
            raise ValueError(
                "find_all_consistent(mode='exhaustive') cannot be honored: "
                f"scenario space has size {int(total_scenarios)} and exceeds "
                + " and ".join(limits)
                + ". "
                "Use mode='shortlist' to opt into heuristic random-restart results, "
                "or mode='auto' for legacy auto-switch behavior."
            )

        if can_enumerate:
            all_scenarios = self.enumerate_scenarios(max_scenarios=50_000)
            out = self.filter_consistent(all_scenarios, constraints=constraints)
            if return_metadata:
                return FindAllConsistentResult(
                    scenarios=out,
                    enumeration_mode="exhaustive",
                    total_scenario_space=int(total_scenarios),
                    is_complete=True,
                    requested_mode=str(mode),  # type: ignore[arg-type]
                    effective_mode="exhaustive",
                    switch_reason="none",
                    max_scenarios_requested=(
                        int(max_scenarios) if max_scenarios is not None else None
                    ),
                    exhaustive_threshold=int(exhaustive_threshold),
                    n_restarts=int(n_restarts),
                    max_iterations=int(max_iterations),
                )
            return out

        if mode == "auto":
            warnings.warn(
                "Scenario space exceeds exhaustive threshold/cap; using random-restart "
                "shortlist — results are not a complete enumeration of consistent scenarios.",
                UserWarning,
                stacklevel=2,
            )
        out = self.find_consistent_via_random_restarts(
            n_restarts=n_restarts,
            seed=seed,
            max_iterations=max_iterations,
            constraints=constraints,
            constrained_mode=constrained_mode,
            constrained_top_k=constrained_top_k,
            constrained_backtracking_depth=constrained_backtracking_depth,
        )
        if return_metadata:
            return FindAllConsistentResult(
                scenarios=out,
                enumeration_mode="random_restarts",
                total_scenario_space=int(total_scenarios),
                is_complete=False,
                requested_mode=str(mode),  # type: ignore[arg-type]
                effective_mode="shortlist",
                switch_reason=(
                    "auto_threshold_and_max_scenarios_exceeded"
                    if threshold_exceeded and max_cap_exceeded
                    else (
                        "auto_threshold_exceeded"
                        if threshold_exceeded
                        else "auto_max_scenarios_exceeded"
                    )
                ),
                max_scenarios_requested=(
                    int(max_scenarios) if max_scenarios is not None else None
                ),
                exhaustive_threshold=int(exhaustive_threshold),
                n_restarts=int(n_restarts),
                max_iterations=int(max_iterations),
            )
        return out

    def find_consistent_via_random_restarts(
        self,
        n_restarts: int = 200,
        seed: Optional[int] = None,
        max_iterations: int = 1000,
        constraints: Optional[Sequence[ConstraintSpec]] = None,
        constrained_mode: Literal["strict", "repair"] = "repair",
        constrained_top_k: int = 2,
        constrained_backtracking_depth: int = 2,
    ) -> List[Scenario]:
        """
        Find a shortlist of consistent scenarios via succession random restarts.

        This is the recommended workflow for medium/large systems where full
        enumeration is infeasible.
        """
        consistent: List[Scenario] = []
        seen: set[Scenario] = set()
        cidx = ConstraintIndex.from_specs(self.matrix, constraints)
        if cidx is not None:
            from cib.succession import ConstrainedGlobalSuccession

            succession_operator = ConstrainedGlobalSuccession(
                cidx,
                constraint_mode=str(constrained_mode),
                constrained_top_k=int(constrained_top_k),
                constrained_backtracking_depth=int(constrained_backtracking_depth),
            )
        else:
            from cib.succession import GlobalSuccession

            succession_operator = GlobalSuccession()

        results = self.find_attractors_via_random_restarts(
            n_restarts=n_restarts,
            seed=seed,
            max_iterations=max_iterations,
            succession_operator=succession_operator,
        )

        for res in results:
            if res.is_cycle:
                continue
            attractor = res.attractor
            if not isinstance(attractor, Scenario):
                continue
            if attractor in seen:
                continue
            if cidx is not None and not bool(cidx.is_full_valid(attractor.to_indices())):
                continue
            if ConsistencyChecker.check_consistency(attractor, self.matrix):
                consistent.append(attractor)
                seen.add(attractor)

        return consistent

    def find_attractors_via_random_restarts(
        self,
        n_restarts: int = 200,
        seed: Optional[int] = None,
        max_iterations: int = 1000,
        succession_operator=None,
    ):
        """
        Find attractors (fixed points or cycles) via random restarts.

        Returns:
            List of AttractorResult objects.
        """
        from cib.succession import AttractorResult, AttractorFinder, GlobalSuccession

        if succession_operator is None:
            succession_operator = GlobalSuccession()

        rng = np.random.default_rng(seed)
        initial_scenarios: List[Scenario] = []
        descriptor_names = list(self.matrix.descriptors.keys())
        for _ in range(n_restarts):
            state_dict: Dict[str, str] = {}
            for desc in descriptor_names:
                states = self.matrix.descriptors[desc]
                state_dict[desc] = states[int(rng.integers(0, len(states)))]
            initial_scenarios.append(Scenario(state_dict, self.matrix))

        finder = AttractorFinder(self.matrix)
        results: List[AttractorResult] = []
        for initial in initial_scenarios:
            results.append(
                succession_operator.find_attractor(
                    initial, self.matrix, max_iterations=max_iterations
                )
            )
        # Results are de-duplicated by attractor identity (fixed point or first element of cycle).
        unique: dict[Scenario, AttractorResult] = {}
        for res in results:
            if isinstance(res.attractor, Scenario):
                key = res.attractor
            else:
                cycle = res.attractor
                key = min(cycle, key=lambda s: tuple(s.to_indices()))
            unique.setdefault(key, res)
        return list(unique.values())

    def find_attractors_monte_carlo(
        self,
        *,
        config=None,
    ):
        """
        Find attractors and estimate weights via Monte Carlo sampling.

        This method is intended for large scenario spaces where complete
        enumeration is infeasible.
        """
        from cib.solvers.config import MonteCarloAttractorConfig
        from cib.solvers.monte_carlo_attractors import (
            MonteCarloAttractorResult,
            find_attractors_monte_carlo,
        )

        cfg = config if config is not None else MonteCarloAttractorConfig()
        if not isinstance(cfg, MonteCarloAttractorConfig):
            raise ValueError("config must be a MonteCarloAttractorConfig")
        res: MonteCarloAttractorResult = find_attractors_monte_carlo(
            matrix=self.matrix, config=cfg
        )
        return res

    def find_all_consistent_exact(
        self,
        *,
        config=None,
    ):
        """
        Enumerate consistent scenarios using a pruned exact solver.

        Callers must inspect :attr:`~cib.solvers.exact_pruned.ExactSolverResult.is_complete`
        and ``status``: time limits, ``max_solutions``, or search interruption
        can yield **partial** solution lists. When fast scoring fails, bruteforce
        fallback is blocked for large scenario spaces unless
        ``ExactSolverConfig.allow_bruteforce`` is set.
        """
        from cib.solvers.config import ExactSolverConfig
        from cib.solvers.exact_pruned import ExactSolverResult, find_all_consistent_exact

        cfg = config if config is not None else ExactSolverConfig()
        if not isinstance(cfg, ExactSolverConfig):
            raise ValueError("config must be an ExactSolverConfig")
        res: ExactSolverResult = find_all_consistent_exact(matrix=self.matrix, config=cfg)
        if not res.is_complete:
            warnings.warn(
                f"find_all_consistent_exact returned incomplete results "
                f"(status={res.status!r}). Inspect is_complete and diagnostics.",
                UserWarning,
                stacklevel=2,
            )
        return res

    def filter_consistent(
        self,
        candidates: List[Scenario],
        *,
        constraints: Optional[Sequence[ConstraintSpec]] = None,
    ) -> List[Scenario]:
        """
        Filter consistent scenarios from a candidate list.

        Args:
            candidates: List of scenarios to check.

        Returns:
            List of scenarios that are consistent.
        """
        consistent: List[Scenario] = []
        cidx = ConstraintIndex.from_specs(self.matrix, constraints)
        for scenario in candidates:
            if cidx is not None and not bool(cidx.is_full_valid(scenario.to_indices())):
                continue
            if ConsistencyChecker.check_consistency(scenario, self.matrix):
                consistent.append(scenario)

        return consistent

    def rank_scenarios(
        self, scenarios: List[Scenario]
    ) -> List[Tuple[Scenario, float]]:
        """
        Rank scenarios by consistency strength.

        Consistency strength is measured as the minimum gap between the
        chosen state's impact score and all other states' impact scores
        for each descriptor. Higher values indicate stronger consistency.

        Args:
            scenarios: List of scenarios to rank.

        Returns:
            List of (scenario, strength) tuples, sorted by strength
            in descending order.
        """
        from cib.core import ImpactBalance
        import numpy as np

        ranked: List[Tuple[Scenario, float]] = []

        for scenario in scenarios:
            min_gap = float("inf")

            for descriptor in self.matrix.descriptors:
                current_state = scenario.get_state(descriptor)
                balance = ImpactBalance(scenario, self.matrix)
                current_score = balance.get_score(descriptor, current_state)

                for state in self.matrix.descriptors[descriptor]:
                    if state != current_state:
                        score = balance.get_score(descriptor, state)
                        gap = current_score - score
                        if gap < min_gap:
                            min_gap = gap

            ranked.append((scenario, min_gap))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


@dataclass
class MonteCarloResults:
    """
    Results from Monte Carlo consistency probability estimation.

    Attributes:
        scenario_probabilities: Dictionary mapping scenarios to their
            estimated consistency probabilities.
        confidence_intervals: Dictionary mapping scenarios to (lower, upper)
            confidence interval tuples.
        n_samples: Number of Monte Carlo samples used.
    """

    scenario_probabilities: Dict[Scenario, float]
    confidence_intervals: Dict[Scenario, Tuple[float, float]]
    n_samples: int


class MonteCarloAnalyzer:
    """
    Monte Carlo analyzer for probabilistic consistency estimation.

    Estimates P(consistent | z) for scenarios by sampling uncertain CIB
    matrices and checking consistency in each sample.
    """

    def __init__(
        self,
        matrix: object,
        n_samples: int = 10000,
        seed: Optional[int] = None,
    ) -> None:
        """
        The Monte Carlo analyser is initialised.

        Args:
            matrix: UncertainCIBMatrix with confidence codes.
            n_samples: Number of Monte Carlo samples to use.
            seed: Random seed for reproducibility. If None, uses random seed.

        Raises:
            ValueError: If matrix is not an UncertainCIBMatrix or if
                n_samples is non-positive.
        """
        if not hasattr(matrix, "sample_matrix"):
            raise ValueError(
                "Matrix must provide a sample_matrix(seed: int) method"
            )
        # Backwards-compatible type checks are performed for common implementations.
        if UncertainCIBMatrix is not None and isinstance(matrix, UncertainCIBMatrix):
            pass
        elif GaussianCIBMatrix is not None and isinstance(matrix, GaussianCIBMatrix):
            pass
        else:
            # Other sampling-capable matrices are accepted (duck-typing).
            pass
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        self.matrix = matrix  # type: ignore[assignment]
        self.n_samples = n_samples
        self.base_seed = seed if seed is not None else np.random.randint(0, 2**31)

    def estimate_consistency_probability(
        self, scenario: Scenario
    ) -> float:
        """
        Estimate P(consistent | z) for a scenario.

        Args:
            scenario: Scenario to evaluate.

        Returns:
            Estimated probability of consistency (between 0 and 1).
        """
        consistent_count = self._estimate_consistency_count(scenario)
        return consistent_count / self.n_samples

    def _estimate_consistency_count(self, scenario: Scenario) -> int:
        """
        Estimate the number of Monte Carlo draws where the scenario is consistent.
        """
        consistent_count = 0
        for sample_idx in range(self.n_samples):
            if seeds_for_run is not None:
                seeds = seeds_for_run(self.base_seed, sample_idx)
                sample_seed = seeds["judgment_uncertainty_seed"]
            else:
                sample_seed = self.base_seed + sample_idx

            sampled_matrix = self.matrix.sample_matrix(sample_seed)
            if ConsistencyChecker.check_consistency(scenario, sampled_matrix):
                consistent_count += 1
        return consistent_count

    def score_candidates(
        self, candidates: List[Scenario]
    ) -> MonteCarloResults:
        """
        Score multiple candidate scenarios.

        Args:
            candidates: List of scenarios to evaluate.

        Returns:
            MonteCarloResults containing probabilities and confidence intervals.
        """
        scenario_probabilities: Dict[Scenario, float] = {}
        confidence_intervals: Dict[Scenario, Tuple[float, float]] = {}

        for scenario in candidates:
            k = self._estimate_consistency_count(scenario)
            prob = k / self.n_samples
            scenario_probabilities[scenario] = prob
            confidence_intervals[scenario] = self._wilson_interval_from_count(
                k, self.n_samples, level=0.95
            )

        return MonteCarloResults(
            scenario_probabilities=scenario_probabilities,
            confidence_intervals=confidence_intervals,
            n_samples=self.n_samples,
        )

    def get_confidence_intervals(
        self, scenario: Scenario, level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for consistency probability.

        Uses Wilson score interval for binomial proportion.

        Args:
            scenario: Scenario to evaluate.
            level: Confidence level (default 0.95 for 95% interval).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        k = self._estimate_consistency_count(scenario)
        return self._wilson_interval_from_count(k, self.n_samples, level=level)

    @staticmethod
    def _wilson_interval_from_count(
        k: int, n: int, level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Wilson score interval for a binomial proportion given success count k.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if k < 0 or k > n:
            raise ValueError("k must be in [0, n]")
        if not (0.0 < float(level) < 1.0):
            raise ValueError("level must be between 0 and 1")

        try:
            from scipy.stats import norm  # type: ignore
        except (ImportError, ModuleNotFoundError):
            norm = None  # type: ignore[assignment]

        z: float
        if norm is not None:
            z = float(norm.ppf(0.5 + float(level) / 2.0))
        else:
            if np.isclose(level, 0.95):
                z = 1.96
            elif np.isclose(level, 0.99):
                z = 2.576
            elif np.isclose(level, 0.90):
                z = 1.645
            else:
                raise ValueError(
                    "Confidence level requires SciPy unless level is 0.90, 0.95, or 0.99 "
                    f"(got level={level}). Install scipy or use a tabulated level."
                )

        p = k / n
        denominator = 1 + (z**2 / n)
        center = (p + (z**2 / (2 * n))) / denominator
        margin = (z / denominator) * np.sqrt(
            (p * (1 - p) / n) + (z**2 / (4 * n**2))
        )
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        return (float(lower), float(upper))
