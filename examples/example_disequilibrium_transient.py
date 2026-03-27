"""
Example: Transient disequilibrium (extended pathway API)

The following is demonstrated:
- Use of the extended pathway API with extension_mode="transient"
- Obtaining an ExtendedTransformationPathway with disequilibrium metrics
- Separation of distance to the consistent set from attractor proximity
- Identification of cyclical but still inconsistent behaviour
- Summarising a single pathway via summarize_disequilibrium_path
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cib import CIBMatrix, DynamicCIB, summarize_disequilibrium_path


def build_matrix() -> CIBMatrix:
    descriptors = {"A": ["Low", "High"], "B": ["Low", "High"], "C": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)
    impacts = {
        ("A", "Low", "B", "Low"): -2.0,
        ("A", "Low", "B", "High"): 2.0,
        ("A", "High", "B", "Low"): 1.0,
        ("A", "High", "B", "High"): 2.0,
        ("A", "Low", "C", "High"): 2.0,
        ("A", "High", "C", "Low"): -1.0,
        ("A", "High", "C", "High"): -1.0,
        ("B", "Low", "A", "Low"): 2.0,
        ("B", "Low", "A", "High"): 1.0,
        ("B", "High", "A", "Low"): 2.0,
        ("B", "Low", "C", "Low"): 1.0,
        ("B", "Low", "C", "High"): 1.0,
        ("B", "High", "C", "High"): -2.0,
        ("C", "Low", "A", "High"): 2.0,
        ("C", "High", "A", "Low"): -2.0,
        ("C", "High", "A", "High"): 1.0,
        ("C", "Low", "B", "Low"): 2.0,
        ("C", "High", "B", "Low"): -1.0,
        ("C", "High", "B", "High"): -1.0,
    }
    matrix.set_impacts(impacts)
    return matrix


if __name__ == "__main__":
    matrix = build_matrix()
    dyn = DynamicCIB(matrix, periods=[2025])
    path = dyn.simulate_path_extended(
        initial={"A": "Low", "B": "Low", "C": "Low"},
        extension_mode="transient",
    )
    metric = path.disequilibrium_metrics[0]
    print(path.realised_scenarios[0].to_dict())
    print(
        {
            "distance_to_consistent_set": metric.distance_to_consistent_set,
            "distance_to_nearest_attractor": metric.distance_to_attractor,
            "nearest_attractor_kind": metric.nearest_attractor_kind,
            "time_to_first_consistent_entry": metric.time_to_equilibrium,
            "entered_consistent_set": metric.entered_consistent_set,
        }
    )
    print(summarize_disequilibrium_path(path))
