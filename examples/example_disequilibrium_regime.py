"""
Example: Regime-aware disequilibrium

The following is demonstrated:
- Registration of named regimes (RegimeSpec) and threshold-triggered regime switches
- Extended pathway with extension_mode="regime" and equilibrium_mode="relax_unshocked"
- Active regimes, active-matrix provenance, and transition events per period
- Distinction between regime entry and continued regime residence
- Distinction between realised scenarios and equilibrium scenarios
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cib import CIBMatrix, DynamicCIB, RegimeSpec, ThresholdRule


def build_matrix() -> CIBMatrix:
    descriptors = {"A": ["Low", "High"], "B": ["Low", "High"]}
    matrix = CIBMatrix(descriptors)
    matrix.set_impact("B", "Low", "A", "Low", 1.0)
    matrix.set_impact("B", "Low", "A", "High", 0.0)
    matrix.set_impact("B", "High", "A", "Low", 0.0)
    matrix.set_impact("B", "High", "A", "High", 1.0)
    return matrix


def build_boosted_matrix() -> CIBMatrix:
    matrix = build_matrix()
    matrix.set_impact("A", "High", "B", "Low", -2.0)
    matrix.set_impact("A", "High", "B", "High", 2.0)
    return matrix


if __name__ == "__main__":
    dyn = DynamicCIB(build_matrix(), periods=[2025, 2030, 2035])
    dyn.add_regime(RegimeSpec(name="boosted", base_matrix=build_boosted_matrix()))
    dyn.add_threshold_rule(
        ThresholdRule(
            name="HighAEntersBoostedRegime",
            condition=lambda s: s.get_state("A") == "High",
            target_regime="boosted",
        )
    )
    dyn.add_threshold_rule(
        ThresholdRule(
            name="LowBKeepsBoostedPressure",
            condition=lambda s: s.get_state("B") == "Low",
            modifier=lambda base: build_boosted_matrix(),
        )
    )
    path = dyn.simulate_path_extended(
        initial={"A": "High", "B": "Low"},
        extension_mode="regime",
        initial_regime="baseline",
        equilibrium_mode="relax_unshocked",
    )
    print("realised", [scenario.to_dict() for scenario in path.realised_scenarios])
    print("equilibrium", [scenario.to_dict() for scenario in path.equilibrium_scenarios or ()])
    print("regimes", path.active_regimes)
    print("provenance", [state.provenance_labels for state in path.active_matrices])
    print(
        "regime_spells",
        [
            {
                "regime": state.regime_name,
                "entered_regime": state.entered_regime,
                "regime_entry_period": state.regime_entry_period,
                "reaffirmed_by": state.threshold_regime_reaffirmations,
            }
            for state in path.active_matrices
        ],
    )
    print("events", [(event.event_type, event.label) for event in path.transition_events])
    print("diffs", [state.diff_summary for state in path.active_matrices])
