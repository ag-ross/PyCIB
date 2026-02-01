#!/usr/bin/env python3
"""
Example: Cycle Detection in CIB Succession

The following is demonstrated:
- How cycles are detected in deterministic succession
- Cycle handling in stochastic succession with dynamic shocks
- Cycle behaviour in multi-period dynamic simulations
- Interpreting cycle results
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib import (
    CIBMatrix,
    Scenario,
    GlobalSuccession,
    DynamicCIB,
    ShockModel,
    UncertainCIBMatrix,
)
from cib.core import ConsistencyChecker
from cib.example_data import DEFAULT_PERIODS


def create_cycle_prone_matrix() -> CIBMatrix:
    """
    Create a CIB matrix that is prone to cycles.

    This matrix is designed so that multiple scenarios have equal or
    near-equal impact scores, which can lead to cycles in succession.
    """
    descriptors = {
        "Policy": ["Low", "Medium", "High"],
        "Economy": ["Weak", "Moderate", "Strong"],
        "Technology": ["Slow", "Medium", "Fast"],
    }

    matrix = CIBMatrix(descriptors)

    # Impacts are set to create scenarios with similar scores.
    # This can lead to cycles when succession alternates between scenarios.

    # Policy -> Economy impacts
    matrix.set_impact("Policy", "Low", "Economy", "Weak", 1.0)
    matrix.set_impact("Policy", "Low", "Economy", "Moderate", 0.0)
    matrix.set_impact("Policy", "Low", "Economy", "Strong", -1.0)
    matrix.set_impact("Policy", "Medium", "Economy", "Weak", 0.0)
    matrix.set_impact("Policy", "Medium", "Economy", "Moderate", 1.0)
    matrix.set_impact("Policy", "Medium", "Economy", "Strong", 0.0)
    matrix.set_impact("Policy", "High", "Economy", "Weak", -1.0)
    matrix.set_impact("Policy", "High", "Economy", "Moderate", 0.0)
    matrix.set_impact("Policy", "High", "Economy", "Strong", 1.0)

    # Economy -> Policy impacts (creates feedback)
    matrix.set_impact("Economy", "Weak", "Policy", "Low", 1.0)
    matrix.set_impact("Economy", "Weak", "Policy", "Medium", 0.0)
    matrix.set_impact("Economy", "Weak", "Policy", "High", -1.0)
    matrix.set_impact("Economy", "Moderate", "Policy", "Low", 0.0)
    matrix.set_impact("Economy", "Moderate", "Policy", "Medium", 1.0)
    matrix.set_impact("Economy", "Moderate", "Policy", "High", 0.0)
    matrix.set_impact("Economy", "Strong", "Policy", "Low", -1.0)
    matrix.set_impact("Economy", "Strong", "Policy", "Medium", 0.0)
    matrix.set_impact("Economy", "Strong", "Policy", "High", 1.0)

    # Technology impacts (weaker, to allow cycles)
    matrix.set_impact("Technology", "Slow", "Economy", "Weak", 0.5)
    matrix.set_impact("Technology", "Fast", "Economy", "Strong", 0.5)
    matrix.set_impact("Economy", "Strong", "Technology", "Fast", 0.5)
    matrix.set_impact("Economy", "Weak", "Technology", "Slow", 0.5)

    return matrix


def example_deterministic_cycle():
    """
    Demonstrate cycle detection in deterministic succession.

    A simple matrix is used where succession can alternate between
    scenarios, creating a cycle.
    """
    print("=" * 60)
    print("Example 1: Deterministic Cycle Detection")
    print("=" * 60)

    matrix = create_cycle_prone_matrix()
    succession = GlobalSuccession()

    # An initial scenario is chosen that may lead to a cycle.
    initial = Scenario(
        {"Policy": "Low", "Economy": "Weak", "Technology": "Slow"}, matrix
    )

    print(f"\nInitial scenario: {initial.to_dict()}")

    # Succession is run to find the attractor.
    result = succession.find_attractor(initial, matrix, max_iterations=100)

    print(f"\nIterations: {result.iterations}")
    print(f"Is cycle: {result.is_cycle}")
    print(f"Path length: {len(result.path)}")

    if result.is_cycle:
        cycle = result.attractor
        assert isinstance(cycle, list)
        print(f"\nCycle detected with {len(cycle)} scenarios:")
        for i, scenario in enumerate(cycle):
            print(f"  {i+1}. {scenario.to_dict()}")
        print("\nThe succession alternates between these scenarios.")
    else:
        attractor = result.attractor
        assert isinstance(attractor, Scenario)
        print(f"\nFixed point attractor: {attractor.to_dict()}")
        print("\nSuccession converged to a single consistent scenario.")


def example_stochastic_cycle():
    """
    Demonstrate cycle handling in stochastic succession with dynamic shocks.

    Dynamic shocks are applied during succession, which can cause
    different cycle behaviors across runs.
    """
    print("\n" + "=" * 60)
    print("Example 2: Cycle Handling in Stochastic Succession")
    print("=" * 60)

    matrix = create_cycle_prone_matrix()

    # An uncertain matrix is created to introduce judgment uncertainty.
    uncertain_matrix = UncertainCIBMatrix(matrix.descriptors, default_confidence=3)
    for src in matrix.descriptors:
        for src_s in matrix.descriptors[src]:
            for tgt in matrix.descriptors:
                if tgt != src:
                    for tgt_s in matrix.descriptors[tgt]:
                        impact_value = matrix.get_impact(src, src_s, tgt, tgt_s)
                        uncertain_matrix.set_impact(
                            src, src_s, tgt, tgt_s, impact_value, confidence=3
                        )

    succession = GlobalSuccession()
    initial = Scenario(
        {"Policy": "Medium", "Economy": "Moderate", "Technology": "Medium"}, matrix
    )

    print(f"\nInitial scenario: {initial.to_dict()}")
    print("\nRunning succession with judgment uncertainty (3 runs):")

    # Multiple runs are performed to show variability.
    for run_idx in range(3):
        seed = 100 + run_idx
        sampled_matrix = uncertain_matrix.sample_matrix(seed)

        result = succession.find_attractor(
            initial, sampled_matrix, max_iterations=100
        )

        print(f"\n  Run {run_idx + 1} (seed={seed}):")
        print(f"    Iterations: {result.iterations}")
        print(f"    Is cycle: {result.is_cycle}")

        if result.is_cycle:
            cycle = result.attractor
            assert isinstance(cycle, list)
            print(f"    Cycle length: {len(cycle)}")
            print(f"    First scenario in cycle: {cycle[0].to_dict()}")
        else:
            attractor = result.attractor
            assert isinstance(attractor, Scenario)
            print(f"    Fixed point: {attractor.to_dict()}")


def example_dynamic_simulation_cycles():
    """
    Demonstrate cycle handling in dynamic multi-period simulations.

    The tie_break parameter controls how cycles are handled when
    they occur during dynamic simulations.
    """
    print("\n" + "=" * 60)
    print("Example 3: Cycle Handling in Dynamic Simulations")
    print("=" * 60)

    matrix = create_cycle_prone_matrix()
    periods = list(DEFAULT_PERIODS)

    dynamic = DynamicCIB(base_matrix=matrix, periods=periods)

    initial = {"Policy": "Low", "Economy": "Weak", "Technology": "Slow"}

    print(f"\nInitial state: {initial}")
    print(f"Periods: {periods}")

    # Two simulations are run with different tie_break strategies.
    for tie_break_mode in ["deterministic_first", "random"]:
        print(f"\n--- Simulation with tie_break='{tie_break_mode}' ---")

        pathway = dynamic.simulate_path(
            initial=initial,
            seed=200,
            tie_break=tie_break_mode,
            max_iterations=100,
        )

        print(f"\nPathway across {len(pathway.scenarios)} periods:")
        for period, scenario in zip(periods, pathway.scenarios):
            print(f"  {period}: {scenario.to_dict()}")

        # Cycle information is checked if available.
        if hasattr(pathway, "cycles_detected"):
            print(f"\nCycles detected: {pathway.cycles_detected}")


def example_cycle_interpretation():
    """
    Demonstrate interpretation of cycle results and best practices.

    Cycles are explained in terms of system dynamics and stability.
    """
    print("\n" + "=" * 60)
    print("Example 4: Interpreting Cycle Results")
    print("=" * 60)

    matrix = create_cycle_prone_matrix()
    succession = GlobalSuccession()

    # Multiple initial scenarios are tested to find cycles.
    test_scenarios = [
        {"Policy": "Low", "Economy": "Weak", "Technology": "Slow"},
        {"Policy": "High", "Economy": "Strong", "Technology": "Fast"},
        {"Policy": "Medium", "Economy": "Moderate", "Technology": "Medium"},
    ]

    print("\nTesting multiple initial scenarios:")
    cycle_count = 0
    fixed_point_count = 0

    for i, state_dict in enumerate(test_scenarios, 1):
        initial = Scenario(state_dict, matrix)
        result = succession.find_attractor(initial, matrix, max_iterations=100)

        if result.is_cycle:
            cycle_count += 1
            cycle = result.attractor
            assert isinstance(cycle, list)
            print(f"\n  Scenario {i}: CYCLE (length {len(cycle)})")
            print(f"    Interpretation: System alternates between scenarios,")
            print(f"    indicating periodic dynamics or instability.")
        else:
            fixed_point_count += 1
            attractor = result.attractor
            assert isinstance(attractor, Scenario)
            print(f"\n  Scenario {i}: FIXED POINT")
            print(f"    Attractor: {attractor.to_dict()}")
            print(f"    Interpretation: System converges to stable state.")

    print(f"\nSummary:")
    print(f"  Cycles found: {cycle_count}")
    print(f"  Fixed points found: {fixed_point_count}")

    print("\nBest practices:")
    print("  - Cycles may indicate system instability or alternating regimes")
    print("  - In stochastic succession, cycles can vary across runs")
    print("  - Use tie_break parameter to control cycle handling in dynamic simulations")
    print("  - Fixed points represent stable system states")


def main():
    """Run all cycle detection examples."""
    example_deterministic_cycle()
    example_stochastic_cycle()
    example_dynamic_simulation_cycles()
    example_cycle_interpretation()

    print("\n" + "=" * 60)
    print("All cycle detection examples completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
