"""
Minimal joint-distribution probabilistic CIA example (static).

The `cib.prob` module is demonstrated:
  - factors and marginals are defined,
  - cross-impact multipliers (probability ratios) are defined,
  - a coherent joint distribution over scenarios is fitted,
  - a small number of implied probabilities are inspected.

This example reuses the canonical dataset labels from `cib.example_data`
(Dataset B5 descriptor/state names), but the joint-distribution probabilistic CIA probabilities are a
*separate* model (not derived from CIB impact scores).
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cib.example_data import DATASET_B5_DESCRIPTORS
from cib.prob import DiagnosticsReport, FactorSpec, ProbabilisticCIAModel
from cib.prob.types import ScenarioIndex


def _choose_factors() -> list[FactorSpec]:
    # Keep this example small (5×5×5 = 125 scenarios) while using the real labels.
    names = ["Policy_Stringency", "Grid_Flexibility", "Electrification_Demand"]
    return [FactorSpec(n, DATASET_B5_DESCRIPTORS[n]) for n in names]


def _marginals_from_joint(index: ScenarioIndex, p: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for f in index.factors:
        out[f.name] = {o: 0.0 for o in f.outcomes}
    for k in range(index.size):
        scen = index.scenario_at(k)
        for fname, outcome in zip(index.factor_names, scen.assignment):
            out[fname][outcome] += float(p[k])
    return out


def _multipliers_from_joint(
    index: ScenarioIndex,
    p: np.ndarray,
    *,
    target_factor: str,
    given_factor: str,
) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
    """
    Compute multipliers m_{(i=a)<-(j=b)} from a joint distribution.
    """
    target_factor = str(target_factor)
    given_factor = str(given_factor)
    pos_i = list(index.factor_names).index(target_factor)
    pos_j = list(index.factor_names).index(given_factor)

    marg = _marginals_from_joint(index, p)
    # Pairwise P(i=a, j=b) is computed.
    pair: Dict[Tuple[str, str], float] = {}
    for a in index.factors[pos_i].outcomes:
        for b in index.factors[pos_j].outcomes:
            pair[(a, b)] = 0.0
    for k in range(index.size):
        scen = index.scenario_at(k)
        a = scen.assignment[pos_i]
        b = scen.assignment[pos_j]
        pair[(a, b)] += float(p[k])

    out: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float] = {}
    for a in index.factors[pos_i].outcomes:
        for b in index.factors[pos_j].outcomes:
            pj = float(marg[given_factor][b])
            pi = float(marg[target_factor][a])
            pij = float(pair[(a, b)])
            cond = pij / pj
            out[((target_factor, a), (given_factor, b))] = float(cond / pi)
    return out


def _get_results_dir() -> str:
    """Determine the results directory path."""
    try:
        import cib.example_data as mod
        package_dir = os.path.dirname(os.path.dirname(mod.__file__))
        results_dir = os.path.join(package_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    except Exception:
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        return results_dir


def _plot_top_scenarios(dist, top_n: int = 10):
    """Return a figure with a bar chart of top scenarios by probability."""
    by_idx = [(i, float(dist.p[i])) for i in range(dist.index.size)]
    by_idx.sort(key=lambda x: x[1], reverse=True)
    top_scenarios = by_idx[:top_n]

    scenarios_str = []
    probabilities = []
    for i, p in top_scenarios:
        s = dist.index.scenario_at(i).to_dict()
        scen_str = ', '.join(f"{k}={v}" for k, v in s.items())
        scenarios_str.append(scen_str)
        probabilities.append(p)

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(scenarios_str))
    ax.barh(y_pos, probabilities, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenarios_str, fontsize=8)
    ax.set_xlabel('Probability')
    ax.set_title(f'Top {top_n} Scenarios (Joint-Distribution Probabilistic CIA)')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def _plot_marginal_distributions(dist, factors: list[FactorSpec]):
    """Return a figure with marginal probability distributions for each factor."""
    num_factors = len(factors)
    fig, axes = plt.subplots(1, num_factors, figsize=(5 * num_factors, 4))
    if num_factors == 1:
        axes = [axes]

    for idx, factor in enumerate(factors):
        marginal = dist.marginal(factor.name)
        outcomes = list(marginal.keys())
        probs = list(marginal.values())

        axes[idx].bar(outcomes, probs, alpha=0.85)
        axes[idx].set_xlabel('Outcome')
        axes[idx].set_ylabel('Probability')
        axes[idx].set_title(f'Marginal: {factor.name}')
        axes[idx].set_ylim(0.0, max(probs) * 1.1)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def main() -> None:
    factors = _choose_factors()
    index = ScenarioIndex(factors)

    # A synthetic "true" joint-distribution probabilistic CIA joint distribution is created over the *real*
    # dataset labels (not derived from CIB impacts).
    rng = np.random.default_rng(123)
    p_true = rng.dirichlet(alpha=np.ones(index.size, dtype=float))

    marginals = _marginals_from_joint(index, p_true)
    multipliers = {}
    # Multipliers are provided for two pair relations (enough to constrain useful structure).
    multipliers.update(
        _multipliers_from_joint(
            index, p_true, target_factor="Electrification_Demand", given_factor="Policy_Stringency"
        )
    )
    multipliers.update(
        _multipliers_from_joint(
            index, p_true, target_factor="Grid_Flexibility", given_factor="Policy_Stringency"
        )
    )

    model = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers=multipliers)
    # For this example, target-based reweighting is avoided because some synthetic
    # pairwise targets can be extremely small, which leads to huge weights.
    dist = model.fit_joint(method="direct", kl_weight=0.0, weight_by_target=False, solver_maxiter=5000)

    report = DiagnosticsReport.from_distribution(dist, marginals=marginals, multipliers=multipliers)
    print("Diagnostics:")
    print("  sum_to_one_error =", report.sum_to_one_error)
    print("  marginal_max_abs_error =", report.marginal_max_abs_error)
    print("  pairwise_target_max_abs_error =", report.pairwise_target_max_abs_error)

    print("Top scenarios:")
    by_idx = [(i, float(dist.p[i])) for i in range(dist.index.size)]
    by_idx.sort(key=lambda x: x[1], reverse=True)
    for i, p in by_idx[:5]:
        s = dist.index.scenario_at(i).to_dict()
        print(f"  p={p:.4f}  {s}")

    print()
    print("Implied conditionals (spot check):")
    print(
        "  P(Electrification_Demand=High | Policy_Stringency=High) =",
        dist.conditional(("Electrification_Demand", "High"), ("Policy_Stringency", "High")),
    )

    # Plots are generated and saved.
    results_dir = _get_results_dir()

    fig1 = _plot_top_scenarios(dist, top_n=10)
    plot_path1 = os.path.join(results_dir, "example_probabilistic_cia_static_top_scenarios.png")
    fig1.savefig(plot_path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\nSaved plot to {plot_path1}")

    fig2 = _plot_marginal_distributions(dist, factors)
    plot_path2 = os.path.join(results_dir, "example_probabilistic_cia_static_marginals.png")
    fig2.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved plot to {plot_path2}")


if __name__ == "__main__":
    main()

