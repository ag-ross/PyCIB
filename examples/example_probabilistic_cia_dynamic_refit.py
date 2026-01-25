"""
Minimal joint-distribution probabilistic CIA example (dynamic: per-period refit).

This shows how to compute P(scenario at t) for each period by fitting
independent static models per period.

This example reuses the canonical dataset labels from `cib.example_data`
(Dataset B5 descriptor/state names), but the joint-distribution probabilistic CIA probabilities are a
separate model (not derived from CIB impact scores).
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cib.example_data import DATASET_B5_DESCRIPTORS, DATASET_B5_NUMERIC_MAPPING
from cib.prob import DiagnosticsReport, DynamicProbabilisticCIA, FactorSpec, ProbabilisticCIAModel
from cib.prob.types import ScenarioIndex
from cib.visualization import DynamicVisualizer


def _state_probability_timelines_from_joint(
    *,
    factors: list[FactorSpec],
    periods: list[int],
    dists_by_period,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Return timelines in the same *shape* as cib.pathway helpers:
      timelines[t][factor_name][outcome] = P(X_i(t)=outcome)
    """
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for t in periods:
        dist = dists_by_period[int(t)]
        out[int(t)] = {}
        for f in factors:
            out[int(t)][f.name] = dist.marginal(f.name)
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


def _choose_factors() -> list[FactorSpec]:
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
    target_factor = str(target_factor)
    given_factor = str(given_factor)
    pos_i = list(index.factor_names).index(target_factor)
    pos_j = list(index.factor_names).index(given_factor)

    marg = _marginals_from_joint(index, p)
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


def _compute_numeric_summary_from_prog(
    *,
    timelines: Dict[int, Dict[str, Dict[str, float]]],
    descriptor: str,
    numeric_mapping: Dict[str, float],
    quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95),
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, float]]:
    """
    Compute numeric quantiles and expectations from joint-distribution probabilistic CIA probability timelines.

    Returns quantiles_by_period and numeric_expectation_by_period in the format
    expected by plot_descriptor_stochastic_summary.
    """
    quantiles_by_period: Dict[int, Tuple[float, float, float]] = {}
    numeric_expectation_by_period: Dict[int, float] = {}

    for t, probs_dict in timelines.items():
        if descriptor not in probs_dict:
            continue

        probs = probs_dict[descriptor]
        # A discrete distribution over numeric values is constructed.
        values = []
        weights = []
        for state, prob in probs.items():
            if state in numeric_mapping:
                values.append(float(numeric_mapping[state]))
                weights.append(float(prob))

        if not values:
            continue

        # Quantiles are computed via weighted percentile for discrete distribution.
        values_arr = np.array(values)
        weights_arr = np.array(weights)
        weights_arr = weights_arr / weights_arr.sum()

        sorted_indices = np.argsort(values_arr)
        sorted_values = values_arr[sorted_indices]
        sorted_weights = weights_arr[sorted_indices]
        cumsum_weights = np.cumsum(sorted_weights)

        # Quantiles are computed by finding the value at which cumulative weight reaches the quantile.
        def weighted_quantile(q: float) -> float:
            idx = np.searchsorted(cumsum_weights, q, side='right')
            if idx == 0:
                return float(sorted_values[0])
            if idx >= len(sorted_values):
                return float(sorted_values[-1])
            # Linear interpolation between adjacent values if needed.
            if idx > 0 and cumsum_weights[idx - 1] < q < cumsum_weights[idx]:
                w1 = (q - cumsum_weights[idx - 1]) / (cumsum_weights[idx] - cumsum_weights[idx - 1])
                return float(sorted_values[idx - 1] * (1 - w1) + sorted_values[idx] * w1)
            return float(sorted_values[idx])

        q_low = weighted_quantile(quantiles[0])
        q_mid = weighted_quantile(quantiles[1])
        q_high = weighted_quantile(quantiles[2])

        quantiles_by_period[int(t)] = (q_low, q_mid, q_high)

        # Expected value is computed.
        expectation = float(np.sum(values_arr * weights_arr))
        numeric_expectation_by_period[int(t)] = expectation

    return quantiles_by_period, numeric_expectation_by_period


def _create_ordered_numeric_mapping(outcomes: list[str]) -> Dict[str, float]:
    """
    Create a simple ordered numeric mapping for outcomes when none exists.

    Maps outcomes to evenly spaced values in [0, 1].
    """
    n = len(outcomes)
    if n == 0:
        return {}
    step = 1.0 / (n - 1) if n > 1 else 0.5
    return {outcome: float(i * step) for i, outcome in enumerate(outcomes)}


def main() -> None:
    factors = _choose_factors()
    index = ScenarioIndex(factors)
    periods = [2025, 2030, 2035]

    # Synthetic joint-distribution probabilistic CIA joint distributions per period are created (using real labels).
    # Different Dirichlet draws are used to induce different marginals per period.
    rng = np.random.default_rng(321)
    p_true_by_period = {
        int(t): rng.dirichlet(alpha=np.ones(index.size, dtype=float) * (1.0 + 0.15 * i))
        for i, t in enumerate(periods)
    }

    marginals_by_period = {int(t): _marginals_from_joint(index, p) for t, p in p_true_by_period.items()}

    models_by_period = {}
    for t in periods:
        p_true = p_true_by_period[int(t)]
        marg = marginals_by_period[int(t)]
        multipliers = {}
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
        models_by_period[int(t)] = ProbabilisticCIAModel(
            factors=factors, marginals=marg, multipliers=multipliers
        )

    dyn = DynamicProbabilisticCIA(periods=periods, models_by_period=models_by_period)
    dists = dyn.fit_distributions(
        mode="refit",
        method="direct",
        kl_weight=0.0,
        weight_by_target=False,
        solver_maxiter=5000,
    )

    timelines = _state_probability_timelines_from_joint(
        factors=factors, periods=periods, dists_by_period=dists
    )

    print("State probability timelines:")
    for t in periods:
        report = DiagnosticsReport.from_distribution(
            dists[int(t)],
            marginals=marginals_by_period[int(t)],
            multipliers=models_by_period[int(t)].multipliers,
        )
        print(f"\nPeriod {t}:")
        print(
            f"  diagnostics: marginal_max_abs_error={report.marginal_max_abs_error:.3e}, "
            f"pairwise_target_max_abs_error={report.pairwise_target_max_abs_error:.3e}"
        )
        for f in factors:
            probs = timelines[int(t)][f.name]
            items = ", ".join(f"{k}={v:.3f}" for k, v in probs.items())
            print(f"  {f.name}: {items}")

    print("\nOne conditional check per period:")
    for t in periods:
        c = dists[int(t)].conditional(("Electrification_Demand", "High"), ("Policy_Stringency", "High"))
        print(f"  P(Electrification_Demand=High | Policy_Stringency=High) at {t} = {c:.3f}")

    # Plots are generated and saved using the same comprehensive plotting functions
    # as the dynamic CIB notebook.
    results_dir = _get_results_dir()

    for factor in factors:
        # A numeric mapping is determined (from dataset or created).
        if factor.name in DATASET_B5_NUMERIC_MAPPING:
            numeric_mapping = DATASET_B5_NUMERIC_MAPPING[factor.name]
        else:
            # An ordered mapping is created for factors without explicit mappings.
            numeric_mapping = _create_ordered_numeric_mapping(factor.outcomes)

        # Numeric quantiles and expectations are computed from joint-distribution probabilistic CIA distributions.
        quantiles_by_period, numeric_expectation_by_period = _compute_numeric_summary_from_prog(
            timelines=timelines,
            descriptor=factor.name,
            numeric_mapping=numeric_mapping,
            quantiles=(0.05, 0.5, 0.95),
        )

        # The comprehensive stochastic summary plot is created (same as dynamic CIB).
        DynamicVisualizer.plot_descriptor_stochastic_summary(
            timelines=timelines,
            quantiles_by_period=quantiles_by_period,
            numeric_expectation_by_period=numeric_expectation_by_period,
            descriptor=factor.name,
            title=f"{factor.name} (Joint-Distribution Probabilistic CIA): probability bands + fan",
            y_label="Numeric mapping",
        )

        plt.tight_layout()
        factor_name_safe = factor.name.lower().replace(' ', '_')
        plot_path = os.path.join(
            results_dir, f"example_probabilistic_cia_dynamic_refit_{factor_name_safe}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()

