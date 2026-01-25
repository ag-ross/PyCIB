#!/usr/bin/env python3
"""
Example 1: Basic Deterministic CIB

This example demonstrates the core CIB functionality:
- Creating a CIB matrix
- Setting impact relationships
- Finding consistent scenarios
- Running basic network analysis on the CIB system
"""

import sys
import os

# The parent directory is added to the import path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib import CIBMatrix, NetworkAnalyzer, ScenarioAnalyzer

print("=" * 60)
print("Example 1: Basic Deterministic CIB")
print("=" * 60)

# Descriptors and states are defined.
descriptors = {
    "Tourism": ["Decrease", "Increase"],
    "Urban_Structure": ["Densification", "Sprawl"],
    "GDP_Growth": ["Weak", "Strong"],
}

# The matrix is created.
matrix = CIBMatrix(descriptors)

# Impact values are set.
matrix.set_impact("Tourism", "Increase", "GDP_Growth", "Strong", 2)
matrix.set_impact("Tourism", "Increase", "GDP_Growth", "Weak", -1)
matrix.set_impact("Tourism", "Decrease", "GDP_Growth", "Strong", -1)
matrix.set_impact("Tourism", "Decrease", "GDP_Growth", "Weak", 1)

matrix.set_impact("Urban_Structure", "Densification", "GDP_Growth", "Strong", 1)
matrix.set_impact("Urban_Structure", "Densification", "GDP_Growth", "Weak", -1)
matrix.set_impact("Urban_Structure", "Sprawl", "GDP_Growth", "Strong", -1)
matrix.set_impact("Urban_Structure", "Sprawl", "GDP_Growth", "Weak", 1)

matrix.set_impact("GDP_Growth", "Strong", "Tourism", "Increase", 1)
matrix.set_impact("GDP_Growth", "Strong", "Tourism", "Decrease", -1)
matrix.set_impact("GDP_Growth", "Weak", "Tourism", "Increase", -1)
matrix.set_impact("GDP_Growth", "Weak", "Tourism", "Decrease", 1)

matrix.set_impact("GDP_Growth", "Strong", "Urban_Structure", "Densification", 1)
matrix.set_impact("GDP_Growth", "Strong", "Urban_Structure", "Sprawl", -1)
matrix.set_impact("GDP_Growth", "Weak", "Urban_Structure", "Densification", -1)
matrix.set_impact("GDP_Growth", "Weak", "Urban_Structure", "Sprawl", 1)

# Consistent scenarios are found.
analyzer = ScenarioAnalyzer(matrix)
consistent_scenarios = analyzer.find_all_consistent(max_scenarios=10)

print(f"\nFound {len(consistent_scenarios)} consistent scenarios:")
for i, scenario in enumerate(consistent_scenarios, 1):
    print(f"  {i}. {scenario.to_dict()}")

if consistent_scenarios:
    print("\nNetwork analysis summary:")
    net = NetworkAnalyzer(matrix)
    centrality = net.compute_centrality_measures(scenario=consistent_scenarios[0])
    top_degree = sorted(
        ((k, v["degree"]) for k, v in centrality.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    for desc, score in top_degree:
        print(f"  Degree centrality: {desc}={score:.3f}")

    pathways = net.find_impact_pathways(
        source="Tourism",
        target="GDP_Growth",
        max_length=3,
        scenario=consistent_scenarios[0],
    )
    if pathways:
        print(f"  Example pathway Tourism -> GDP_Growth: {pathways[0]}")
    else:
        print("  No pathways found from Tourism to GDP_Growth in this scenario network.")

print("\nExample 1 completed successfully.")
