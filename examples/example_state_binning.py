"""
Example: state binning for high-cardinality descriptors.

The application of state binning is demonstrated, where a reduced matrix is
constructed prior to solver execution.
"""

from __future__ import annotations

from cib.core import CIBMatrix
from cib.reduction import reduce_matrix


def main() -> None:
    descriptors = {
        "A": ["a0", "a1", "a2"],
        "B": ["b0", "b1", "b2"],
    }
    matrix = CIBMatrix(descriptors)
    matrix.set_impact("A", "a0", "B", "b0", 1.0)
    matrix.set_impact("A", "a1", "B", "b0", 3.0)
    matrix.set_impact("A", "a2", "B", "b0", 5.0)

    mapping = {
        "A": {"a0": "low", "a1": "low", "a2": "high"},
        "B": {"b0": "x", "b1": "y", "b2": "y"},
    }
    weights = {"A": {"a0": 1.0, "a1": 1.0, "a2": 1.0}}

    reduced = reduce_matrix(
        matrix,
        mapping=mapping,
        aggregation="weighted_mean",
        weights=weights,
    )

    print("Original descriptors:", matrix.descriptors)
    print("Reduced descriptors:", reduced.descriptors)
    print("Reduced impact A=low -> B=x:", reduced.get_impact("A", "low", "B", "x"))


if __name__ == "__main__":
    main()

