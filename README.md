# Cross-Impact Balance (CIB) Analysis Package

A Python implementation of Cross-Impact Balance analysis with uncertainty quantification and robustness testing capabilities.

- Andrew G. Ross 24JAN26

## Overview

The methodology extends traditional CIB analysis  with ways to handle uncertainty, test how robust scenarios are, simulate how systems change over time, and model probabilities. The simplest approach uses point estimates from expert workshops to find consistent scenarios by checking all possibilities or using step-by-step methods. When experts are not completely certain about their judgments, confidence levels can be added to create probability distributions. This allows calculating how likely it is that a scenario stays consistent when judgments vary. Uncertainty can also increase over longer time horizons, since experts are usually less certain about the distant future. Scenarios can be tested for robustness by applying structural shocks, which are permanent changes to how factors influence each other. These might represent major policy changes, regime shifts, or fundamental system alterations. Shocks can be independent or correlated, and can use heavy-tailed distributions to capture rare but important (extreme) events. Dynamic analysis simulates how systems evolve over multiple time periods. Some factors change on their own according to transition probabilities. Threshold rules allow the system to switch between different modes when certain conditions are met, creating tipping points. Stochastic shocks add random disturbances that vary over time, with persistence so that large shocks tend to be followed by more disturbances. These stochastic shocks can have heavy tails or jump components to capture rare but 'extreme' events that create very different pathways. Results can be presented as Monte Carlo ensembles showing probability distributions over time, or as branching pathway graphs showing specific transformation routes. A separate probabilistic approach fits joint probability distributions from marginal probabilities and cross-impact multipliers. Multiple expert judgments can be combined with weights, and network analysis reveals system structure. The methodology supports energy transition planning, urban development scenarios, technology assessment, policy analysis, and strategic planning.

This package provides:

- **Phase 1 (MVP)**: Deterministic CIB analysis with consistency checking,
  succession operators, and scenario enumeration
- **Phase 2**: Uncertainty quantification via confidence-coded impacts and
  Monte Carlo estimation, plus robustness testing under structural shocks
- **Optional extensions**: expert aggregation (weighted, partial coverage) and
  simulation-first dynamic (multi-period) CIB with thresholds/cycles.

## Installation

```bash
cd pycib
pip install -e .
```

## Quick Start

### Basic Deterministic CIB

```python
from cib import CIBMatrix, ScenarioAnalyzer

# Define descriptors and states
descriptors = {
    'Tourism': ['Decrease', 'Increase'],
    'Urban_Structure': ['Densification', 'Sprawl'],
    'GDP_Growth': ['Weak', 'Strong']
}

# Create matrix
matrix = CIBMatrix(descriptors)

# Set impact values
matrix.set_impact('Tourism', 'Increase', 'GDP_Growth', 'Strong', 2)
# ... set all impacts

# Find consistent scenarios
analyzer = ScenarioAnalyzer(matrix)
consistent_scenarios = analyzer.find_all_consistent()

print(f"Found {len(consistent_scenarios)} consistent scenarios")
```

### Dynamic CIB (5-state demo: probability bands + fan + spaghetti, fat-tails + jumps)

```python
import matplotlib.pyplot as plt

from cib import (
    DynamicCIB,
    DynamicVisualizer,
    UncertainCIBMatrix,
    numeric_quantile_timelines,
    state_probability_timelines,
)
from cib.example_data import (
    DATASET_B5_CONFIDENCE,
    DATASET_B5_DESCRIPTORS,
    DATASET_B5_IMPACTS,
    DATASET_B5_INITIAL_SCENARIO,
    DATASET_B5_NUMERIC_MAPPING,
    dataset_b5_cyclic_descriptors,
    dataset_b5_threshold_rule_fast_permitting,
)
from cib.shocks import ShockModel

periods = [2025, 2030, 2035, 2040, 2045]
descriptor = "Electrification_Demand"

matrix = UncertainCIBMatrix(DATASET_B5_DESCRIPTORS)
matrix.set_impacts(DATASET_B5_IMPACTS, confidence=DATASET_B5_CONFIDENCE)

dyn = DynamicCIB(matrix, periods=periods)
for cd in dataset_b5_cyclic_descriptors():  # exogenous drift/inertia for selected drivers
    dyn.add_cyclic_descriptor(cd)
dyn.add_threshold_rule(dataset_b5_threshold_rule_fast_permitting())

# Non-Gaussian dynamic shocks: Student-t innovations + rare jumps.
n_runs = 500
base_seed = 123
paths = []
for m in range(n_runs):
    from cib.example_data import seeds_for_run

    seeds = seeds_for_run(base_seed, m)
    sm = ShockModel(matrix)
    sm.add_dynamic_shocks(
        periods=periods,
        tau=0.26,
        rho=0.6,
        innovation_dist="student_t",
        innovation_df=5.0,
        jump_prob=0.02,
        jump_scale=0.70,
    )
    dynamic_shocks = sm.sample_dynamic_shocks(seeds["dynamic_shock_seed"])
    sigma_by_period = {int(t): 1.0 + 0.85 * i for i, t in enumerate(periods)}
    paths.append(
        dyn.simulate_path(
            initial=DATASET_B5_INITIAL_SCENARIO,
            seed=seeds["dynamic_shock_seed"],
            dynamic_shocks_by_period=dynamic_shocks,
            judgment_sigma_scale_by_period=sigma_by_period,
            structural_sigma=0.15,
            structural_seed_base=seeds["structural_shock_seed"],
            equilibrium_mode="relax_unshocked",
        )
    )

timelines = state_probability_timelines(paths, scenario_mode="realized")
timelines_equilibrium = state_probability_timelines(paths, scenario_mode="equilibrium")
quantiles = numeric_quantile_timelines(
    paths,
    descriptor=descriptor,
    numeric_mapping=DATASET_B5_NUMERIC_MAPPING[descriptor],
    quantiles=(0.05, 0.5, 0.95),
    scenario_mode="realized",
)

mapping = DATASET_B5_NUMERIC_MAPPING[descriptor]
expectation = {
    int(t): sum(float(p) * float(mapping[s]) for s, p in timelines[t][descriptor].items())
    for t in timelines
}

DynamicVisualizer.plot_descriptor_stochastic_summary(
    timelines=timelines,
    quantiles_by_period=quantiles,
    numeric_expectation_by_period=expectation,
    descriptor=descriptor,
    title="Electrification_Demand (5-state): probability bands + fan + spaghetti",
    spaghetti_paths=paths,
    spaghetti_numeric_mapping=mapping,
    spaghetti_max_runs=200,
)
plt.tight_layout()
plt.show()
```

### Monte Carlo vs branching (summary)

Detailed guidance on when to use Monte Carlo ensembles versus branching pathway graphs is documented in `docs/Documentation.md`.

## Documentation

See the `examples/` directory for detailed usage examples:

- `dynamic_cib.ipynb`: Canonical 5-state dynamic example (probability bands + fan + spaghetti)

Additional docs:

- `docs/api_reference.md`: high-level API surface
- `docs/Documentation.md`: concise equations, modeling choices, and result interpretation (simulation-first CIB)
- `docs/Probabilistic_CIA.md`: quickstart for the experimental joint-distribution probabilistic CIA extension (`cib.prob`)

## Key Features

### Phase 1: Core Deterministic CIB

- CIBMatrix: Store and manipulate cross-impact relationships
- Scenario: Represent state assignments
- ConsistencyChecker: Validate scenario consistency
- Succession operators: Find consistent scenarios iteratively
- ScenarioAnalyzer: Enumerate and filter scenarios
- Import/Export: CSV and JSON file support

### Phase 2: Uncertainty and Robustness

- UncertainCIBMatrix: Confidence-coded impacts with uncertainty modeling
- MonteCarloAnalyzer: Estimate P(consistent | z) via Monte Carlo
- ShockModel: Apply structural perturbations to impact matrices
- RobustnessTester: Evaluate scenario stability under shocks

## Example Datasets

The package includes an example dataset in `cib.example_data`:

- **Energy transition demo**: 5 descriptors × 5 states (canonical example)

The dataset includes complete impact matrices and confidence codes.

Preferred import path for datasets is `cib.example_data`.

## Results and interpretation

Generated outputs and guidance on interpretation are documented in `docs/Documentation.md` (including `results/` file descriptions, scenario diagnostics, branching trees, and scenario network plots).

## Coding

The author utilised OpenAI as language and code assistant during the preparation of this work.

## Testing

Run the test suite:

```bash
cd pycib
python3 -m pytest tests/
```

## License

See LICENSE file for details.

### Cite as:

Ross, A. G. (2025). A Python implementation of Cross-Impact Balance analysis with uncertainty quantification and robustness testing capabilities. https://github.com/ag-ross/pycib

## References 

Aoki, M., & Yoshikawa, H. (2011). Reconstructing macroeconomics: a perspective from statistical physics and combinatorial stochastic processes. Cambridge University Press.

Baqaee, D. R., & Farhi, E. (2019). The macroeconomic impact of microeconomic shocks: Beyond Hulten's theorem. Econometrica, 87(4), 1155-1203.

Jermann, U., & Quadrini, V. (2012). Macroeconomic effects of financial shocks. American Economic Review, 102(1), 238-271.

Kearney, N. M. (2022). Prophesy: A new tool for dynamic CIB.

Koop, G., & Korobilis, D. (2010). Bayesian multivariate time series methods for empirical macroeconomics. Foundations and Trends in Econometrics, 3(4), 267-358.

Weimer-Jehle, W. (2006). Cross-impact balances: A system-theoretical approach to cross-impact analysis. Technological Forecasting and Social Change, 73(4), 334-361.

Weimer-Jehle, W. (2023). Cross-Impact Balances (CIB) for Scenario Analysis. Cham, Switzerland: Springer.

Roponen, J., & Salo, A. (2024). A probabilistic cross‐impact methodology for explorative scenario analysis. Futures & Foresight Science, 6(1), e165.

Salo, A., Tosoni, E., Roponen, J., & Bunn, D. W. (2022). Using cross‐impact analysis for probabilistic risk assessment. Futures & foresight science, 4(2), e2103.

Vögele, S., Poganietz, W. R., & Mayer, P. (2019). How to deal with non-linear pathways towards energy futures: Concept and application of the cross-impact balance analysis. TATuP–Journal for Technology Assessment in Theory and Practice, 28(3), 20-26.