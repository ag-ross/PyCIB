"""
Example: shock robustness completeness workflow.

A workflow is demonstrated with:
  1) confidence-guided calibration helpers,
  2) structural/dynamic descriptor/state shock scaling,
  3) extended robustness metrics.

Outputs are written to the results directory as a concise text report.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import Counter

# The repository root is added to the path so that examples can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cib.core import Scenario  # noqa: E402
from cib.dynamic import DynamicCIB  # noqa: E402
from cib.example_data import (  # noqa: E402
    DATASET_C10_CONFIDENCE,
    DATASET_C10_DESCRIPTORS,
    DATASET_C10_IMPACTS,
    DATASET_C10_INITIAL_SCENARIO,
)
from cib.shocks import (  # noqa: E402
    RobustnessTester,
    ShockModel,
    calibrate_structural_sigma_from_confidence,
    suggest_dynamic_tau_bounds,
)
from cib.uncertainty import UncertainCIBMatrix  # noqa: E402


def _top_states(paths: list, descriptor: str, top_k: int = 5) -> list[tuple[str, int]]:
    counts = Counter(p.scenarios[-1].to_dict()[descriptor] for p in paths)
    return counts.most_common(top_k)


def main() -> None:
    matrix = UncertainCIBMatrix(DATASET_C10_DESCRIPTORS)
    matrix.set_impacts(DATASET_C10_IMPACTS, confidence=DATASET_C10_CONFIDENCE)

    # 1) Calibration helpers: transparent, reproducible defaults (not auto-applied).
    confidence_codes = list(DATASET_C10_CONFIDENCE.values())
    structural_sigma = calibrate_structural_sigma_from_confidence(
        confidence_codes, method="median"
    )
    tau_lo, tau_hi = suggest_dynamic_tau_bounds(
        structural_sigma, low_ratio=0.4, high_ratio=0.8
    )
    dynamic_tau = 0.5 * (tau_lo + tau_hi)

    # 2) Structural robustness with descriptor/state scaling and extended metrics.
    structural_scale_by_descriptor = {
        "Policy_Stringency": 1.20,
        "Fossil_Price_Level": 1.10,
    }
    structural_scale_by_state = {
        ("Policy_Stringency", "High"): 1.25,
    }
    sm = ShockModel(matrix)
    sm.add_structural_shocks(
        sigma=float(structural_sigma),
        scaling_mode="multiplicative_magnitude",
        scaling_alpha=0.50,
        scale_by_descriptor=structural_scale_by_descriptor,
        scale_by_state=structural_scale_by_state,
    )

    tester = RobustnessTester(matrix, sm)
    scenario = Scenario(DATASET_C10_INITIAL_SCENARIO, matrix)
    metrics = tester.evaluate_scenario(
        scenario,
        n_simulations=120,
        seed=123,
        max_iterations=500,
    )

    # Compact sensitivity sweep: structural scaling intensity only.
    sweep = [
        ("low", 0.20),
        ("base", 0.50),
        ("high", 0.90),
    ]
    sweep_results: list[tuple[str, float, float, float, float]] = []
    for label, alpha in sweep:
        sm_sweep = ShockModel(matrix)
        sm_sweep.add_structural_shocks(
            sigma=float(structural_sigma),
            scaling_mode="multiplicative_magnitude",
            scaling_alpha=float(alpha),
            scale_by_descriptor=structural_scale_by_descriptor,
            scale_by_state=structural_scale_by_state,
        )
        tester_sweep = RobustnessTester(matrix, sm_sweep)
        m_sweep = tester_sweep.evaluate_scenario(
            scenario,
            n_simulations=120,
            seed=123,
            max_iterations=500,
        )
        sweep_results.append(
            (
                label,
                float(alpha),
                float(m_sweep.consistency_rate),
                float(m_sweep.attractor_retention_rate),
                float(m_sweep.switch_rate),
            )
        )

    # 3) Dynamic simulation with dynamic scaling maps.
    dyn = DynamicCIB(matrix, periods=[2025, 2030, 2035])
    paths = dyn.simulate_ensemble(
        initial=DATASET_C10_INITIAL_SCENARIO,
        n_runs=100,
        base_seed=321,
        structural_sigma=float(structural_sigma),
        structural_shock_scaling_mode="multiplicative_magnitude",
        structural_shock_scaling_alpha=0.50,
        structural_shock_scale_by_descriptor=structural_scale_by_descriptor,
        structural_shock_scale_by_state=structural_scale_by_state,
        dynamic_tau=float(dynamic_tau),
        dynamic_rho=0.60,
        dynamic_shock_scale_by_descriptor={"Policy_Stringency": 1.10},
        dynamic_shock_scale_by_state={("Policy_Stringency", "High"): 1.20},
    )

    top_final_policy = _top_states(paths, descriptor="Policy_Stringency", top_k=3)

    repo_root = Path(__file__).resolve().parent.parent
    out = repo_root / "results" / "example_shock_robustness_completeness.txt"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("Shock robustness completeness workflow\n\n")
        fh.write("Calibration helpers\n")
        fh.write(f"- Structural sigma (median from confidence): {structural_sigma:.4f}\n")
        fh.write(f"- Suggested dynamic tau bounds: [{tau_lo:.4f}, {tau_hi:.4f}]\n")
        fh.write(f"- Dynamic tau used (midpoint): {dynamic_tau:.4f}\n\n")

        fh.write("Extended robustness metrics (initial scenario)\n")
        fh.write(f"- n_simulations: {metrics.n_simulations}\n")
        fh.write(
            f"- Consistency rate: {metrics.consistency_rate:.4f} "
            f"(CI [{metrics.consistency_interval.lower:.4f}, {metrics.consistency_interval.upper:.4f}])\n"
        )
        fh.write(
            f"- Attractor retention rate: {metrics.attractor_retention_rate:.4f} "
            f"(CI [{metrics.attractor_retention_interval.lower:.4f}, {metrics.attractor_retention_interval.upper:.4f}])\n"
        )
        fh.write(
            f"- Switch rate: {metrics.switch_rate:.4f} "
            f"(CI [{metrics.switch_rate_interval.lower:.4f}, {metrics.switch_rate_interval.upper:.4f}])\n"
        )
        fh.write(
            f"- Mean Hamming distance to base attractor: {metrics.mean_hamming_to_base_attractor:.4f}\n\n"
        )

        fh.write("Dynamic ensemble summary (100 runs, 3 periods)\n")
        fh.write("- Top final Policy_Stringency states:\n")
        for state, count in top_final_policy:
            fh.write(f"  - {state}: {count}\n")
        fh.write("\nSensitivity sweep (structural scaling alpha)\n")
        fh.write("- Same sigma, seeds, and scale maps across settings; only alpha varies.\n")
        fh.write(
            "- Columns: label | alpha | consistency_rate | attractor_retention_rate | switch_rate\n"
        )
        for label, alpha, consistency_rate, retention_rate, switch_rate in sweep_results:
            fh.write(
                f"  - {label:>4s} | {alpha:.2f} | {consistency_rate:.4f} | {retention_rate:.4f} | {switch_rate:.4f}\n"
            )

    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
