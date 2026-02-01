#!/usr/bin/env python3
"""
Example: Transformation Matrix Analysis

The construction of transformation matrices is demonstrated, where the
perturbations that cause transitions between scenarios are identified.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cib import (
    CIBMatrix,
    Scenario,
    ScenarioAnalyzer,
    TransformationMatrixBuilder,
)
from cib.visualization import ScenarioVisualizer

# Descriptors and states are defined.
descriptors = {
    "Policy": ["Low", "Medium", "High"],
    "Economy": ["Weak", "Moderate", "Strong"],
    "Technology": ["Slow", "Medium", "Fast"],
}

# The matrix is created.
matrix = CIBMatrix(descriptors)

# Impact relationships are set.
matrix.set_impact("Policy", "Low", "Economy", "Weak", 1.5)
matrix.set_impact("Policy", "Low", "Economy", "Moderate", 0.0)
matrix.set_impact("Policy", "Low", "Economy", "Strong", -1.5)
matrix.set_impact("Policy", "Medium", "Economy", "Weak", 0.0)
matrix.set_impact("Policy", "Medium", "Economy", "Moderate", 1.5)
matrix.set_impact("Policy", "Medium", "Economy", "Strong", 0.0)
matrix.set_impact("Policy", "High", "Economy", "Weak", -1.5)
matrix.set_impact("Policy", "High", "Economy", "Moderate", 0.0)
matrix.set_impact("Policy", "High", "Economy", "Strong", 1.5)

matrix.set_impact("Economy", "Weak", "Policy", "Low", 1.5)
matrix.set_impact("Economy", "Weak", "Policy", "Medium", 0.0)
matrix.set_impact("Economy", "Weak", "Policy", "High", -1.5)
matrix.set_impact("Economy", "Moderate", "Policy", "Low", 0.0)
matrix.set_impact("Economy", "Moderate", "Policy", "Medium", 1.5)
matrix.set_impact("Economy", "Moderate", "Policy", "High", 0.0)
matrix.set_impact("Economy", "Strong", "Policy", "Low", -1.5)
matrix.set_impact("Economy", "Strong", "Policy", "Medium", 0.0)
matrix.set_impact("Economy", "Strong", "Policy", "High", 1.5)

matrix.set_impact("Technology", "Slow", "Economy", "Weak", 1.0)
matrix.set_impact("Technology", "Fast", "Economy", "Strong", 1.0)
matrix.set_impact("Economy", "Strong", "Technology", "Fast", 1.0)
matrix.set_impact("Economy", "Weak", "Technology", "Slow", 1.0)

# Consistent scenarios are found.
analyzer = ScenarioAnalyzer(matrix)
consistent_scenarios = analyzer.find_all_consistent(max_scenarios=5)

print(f"\nFound {len(consistent_scenarios)} consistent scenarios:")
for i, scenario in enumerate(consistent_scenarios):
    print(f"  {i}. {scenario.to_dict()}")

if len(consistent_scenarios) >= 2:
    # Transformation matrix is built.
    builder = TransformationMatrixBuilder(base_matrix=matrix)
    transformation_matrix = builder.build_matrix(
        scenarios=consistent_scenarios[:3],
        perturbation_types=["structural", "dynamic"],
        n_trials_per_pair=50,
        structural_sigma_values=[0.1, 0.2, 0.3],
        dynamic_tau_values=[0.2, 0.3],
        seed=123,
    )

    print(f"\nTransformation matrix summary:")
    print(f"  Total scenarios: {transformation_matrix.summary_stats['total_scenarios']}")
    print(
        f"  Pairs with transformations: {transformation_matrix.summary_stats['pairs_with_transformations']}"
    )

    # Transformations are displayed.
    if transformation_matrix.transformations:
        print("\nDetected transformations:")
        for (source, target), perturbations in transformation_matrix.transformations.items():
            print(f"\n  {source.to_dict()} -> {target.to_dict()}:")
            for pert in perturbations:
                print(
                    f"    {pert.perturbation_type} (magnitude={pert.magnitude:.2f}, "
                    f"success_rate={pert.success_rate:.2f})"
                )

        # Transformation graph is plotted.
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "results"
        )
        os.makedirs(results_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))
        ScenarioVisualizer.transformation_graph(
            transformation_matrix, matrix, min_success_rate=0.0
        )
        plt.tight_layout()
        plot_path = os.path.join(results_dir, "transformation_graph.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nTransformation graph saved to {plot_path}")
