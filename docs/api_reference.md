# API Reference (high-level)

This project exposes its public API through the `cib` package.

## Core deterministic CIB

- `CIBMatrix`: store cross-impacts and compute impact balances.
- `Scenario`: scenario state assignment (labels at API boundary, 0-based indices internally).
- `ConsistencyChecker`: check CIB consistency of a scenario.
- `ScenarioAnalyzer`: enumerate/filter consistent scenarios (small systems) or find a shortlist via random restarts (large systems).
- `GlobalSuccession`, `LocalSuccession`, `AttractorFinder`: succession operators and attractor discovery.

## Uncertainty (practical)

- `UncertainCIBMatrix`: confidence-coded impacts (1–5) with `sample_matrix(seed)` for Monte Carlo.
- `MonteCarloAnalyzer`: estimate \(P(\mathrm{consistent}\mid z)\) for candidate scenarios and return Wilson intervals.

## Robustness and shocks

- `ShockModel`: structural shocks (independent or correlated) and AR(1) dynamic shocks.
- `RobustnessTester`: robustness score = fraction of simulations where a scenario remains consistent under structural shocks.
- `ShockAwareGlobalSuccession`: global succession with additive dynamic shocks on impact balances.

## Expert aggregation (practical Bayesian-style)

- `ExpertAggregator`: weighted aggregation across multiple experts with partial coverage.
- `GaussianCIBMatrix`: per-cell \((\mu,\sigma)\) uncertainty with `sample_matrix(seed)` for Monte Carlo.

## Dynamic CIB (simulation-first)

- `DynamicCIB`: simulate multi-period pathways with optional cyclic descriptors and threshold-triggered CIM modifiers. Optional equilibrium outputs can be requested via `equilibrium_mode` to obtain a per-period unshocked relaxation (matrix-consistent attractor) alongside the shock-realized trajectory.
- `ThresholdRule`: conditionally modify the active CIM.
- `CyclicDescriptor`: per-period state transitions via a transition probability matrix.
- `TransformationPathway`: pathway representation. When `equilibrium_mode` is enabled, `equilibrium_scenarios` is populated to provide an equilibrium analogue of `scenarios`.
- `state_probability_timelines`, `numeric_quantile_timelines`, `pathway_frequencies`: summarize ensembles. `state_probability_timelines` and `numeric_quantile_timelines` support `scenario_mode="realized"` (default) or `scenario_mode="equilibrium"` when equilibrium scenarios are present.

## Hybrid branching pathways (enumerate-or-sample)

- `BranchingPathwayBuilder`: build a branching pathway graph across periods by enumerating consistent scenarios when feasible and falling back to Monte Carlo sampling when not. Nodes can be requested in `node_mode="equilibrium"` (default) to report matrix-consistent attractors even when dynamic shocks are used during sampling; `node_mode="realized"` preserves forcing-realized nodes.
- `BranchingResult`: branching output container (period layers, weighted edges, top paths).

## Scenario diagnostics / scoring

- `scenario_diagnostics`: compute consistency, inconsistency details, margins (“brink”), and total impact score for a scenario.
- `ScenarioDiagnostics`: diagnostics dataclass returned by `scenario_diagnostics`.
- `impact_label`, `judgment_section_labels`: qualitative labels for numeric impacts (hindering/promoting).

## Visualization

- `MatrixVisualizer`, `ScenarioVisualizer`: basic plots.
- `UncertaintyVisualizer`: probability intervals for scenario scores.
- `ShockVisualizer`: robustness plots.
- `DynamicVisualizer`: stacked state-probability bands over time.
- `DynamicVisualizer.plot_pathway_tree(...)`: visualize a branching pathway graph with weighted edges.
- `DynamicVisualizer.plot_descriptor_branching_summary(...)`: branching-derived analogue of the combined descriptor plot (bands + fan + top-path traces).

## Probabilistic CIA (Joint-Distribution Extension)

This API lives under `cib.prob` and is intentionally not re-exported at top-level `cib`.
It is currently **experimental** (API and semantics may evolve).

- `cib.prob.FactorSpec`: factor name + outcomes
- `cib.prob.ProbabilisticCIAModel`: fit a joint distribution from marginals + multipliers
- `cib.prob.JointDistribution`: dense joint distribution for small scenario spaces
- `cib.prob.DiagnosticsReport`: consistency checks and fit diagnostics
- `cib.prob.DynamicProbabilisticCIA`: per-period refit wrapper (Phase 1)

