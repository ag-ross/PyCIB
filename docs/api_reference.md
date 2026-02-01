# API Reference (high-level)

This project exposes its public API through the `cib` package.

## Core deterministic CIB

- `CIBMatrix`: store cross-impacts and compute impact balances.
- `Scenario`: scenario state assignment (labels at API boundary, 0-based indices internally).
- `ConsistencyChecker`: check CIB consistency of a scenario.
- `ScenarioAnalyzer`: enumerate/filter consistent scenarios (small systems) or find a shortlist via random restarts (large systems).
- `GlobalSuccession`, `LocalSuccession`, `AttractorFinder`: succession operators and attractor discovery.

## Scaling solvers (opt-in)

The following APIs are intended for scaling workflows and are exposed as explicit methods on `ScenarioAnalyzer`.

- `ScenarioAnalyzer.find_attractors_monte_carlo(...)`: Monte Carlo attractor discovery with estimated weights (approximate).
- `ScenarioAnalyzer.find_all_consistent_exact(...)`: exact consistency enumeration via a pruned search (exact).

Configuration dataclasses live under `cib.solvers.config`:

- `MonteCarloAttractorConfig`
- `ExactSolverConfig`

Sparse compute backends may be selected for Monte Carlo succession workflows:

- `MonteCarloAttractorConfig(fast_backend="dense" | "sparse")`
- `cib.sparse_scoring.SparseCIBScorer` is used when `fast_backend="sparse"` is selected.

## Scenario feasibility constraints (opt-in)

Feasibility constraints may be specified when scenarios are required to satisfy domain rules in addition to CIB consistency.

The constraint specifications are provided in `cib.constraints`:

- `ForbiddenPair`: a pair of descriptor-state assignments is forbidden.
- `Implies`: an implication constraint is enforced (if A is a given state, B is required to be a given state).
- `AllowedStates`: allowed state labels are restricted for a descriptor.

Integration points:

- Exact enumeration: constraints may be passed via `ExactSolverConfig(constraints=[...])` and are applied during the search (branches are pruned when violations are detected).
- Deterministic analysis helpers: constraints may be passed via `ScenarioAnalyzer.find_all_consistent(..., constraints=[...])` and are applied as feasibility filters.

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

- `DynamicCIB`: simulate multi-period pathways with optional cyclic descriptors and threshold-triggered CIM modifiers. Optional equilibrium outputs can be requested via `equilibrium_mode` to obtain a per-period unshocked relaxation (matrix-consistent attractor) alongside the shock-realised trajectory.
- `ThresholdRule`: conditionally modify the active CIM.
- `CyclicDescriptor`: per-period state transitions via a transition probability matrix.
- `TransformationPathway`: pathway representation. When `equilibrium_mode` is enabled, `equilibrium_scenarios` is populated to provide an equilibrium analogue of `scenarios`.
- `state_probability_timelines`, `numeric_quantile_timelines`, `pathway_frequencies`: summarise ensembles. `state_probability_timelines` and `numeric_quantile_timelines` support `scenario_mode="realized"` (default) or `scenario_mode="equilibrium"` when equilibrium scenarios are present.

## Hybrid branching pathways (enumerate-or-sample)

- `BranchingPathwayBuilder`: build a branching pathway graph across periods by enumerating consistent scenarios when feasible and falling back to Monte Carlo sampling when not. Nodes can be requested in `node_mode="equilibrium"` (default) to report matrix-consistent attractors even when dynamic shocks are used during sampling; `node_mode="realized"` preserves forcing-realised nodes.
- `BranchingResult`: branching output container (period layers, weighted edges, top paths).

## Scenario diagnostics / scoring

- `scenario_diagnostics`: compute consistency, inconsistency details, margins (“brink”), and total impact score for a scenario.
- `ScenarioDiagnostics`: diagnostics dataclass returned by `scenario_diagnostics`.
- `impact_label`, `judgment_section_labels`: qualitative labels for numeric impacts (hindering/promoting).

## Attribution and sensitivity (local)

The following utilities are provided for local explanation of scenarios under a fixed CIM:

- `attribute_scenario`: a per-descriptor margin-to-switching decomposition is returned, including per-source contributions.
- `ScenarioAttribution`, `DescriptorAttribution`, `Contribution`: attribution dataclasses.
- `flip_candidates_for_descriptor`: bounded single-cell flip candidates are suggested (heuristic).

## Rare-event diagnostics

The following utilities are provided for basic rare-event reliability diagnostics:

- `event_rate_diagnostics`: event rate estimates and Wilson confidence intervals are provided, with an undersampling flag.
- `near_miss_rate`: a near-miss rate is computed using the minimum margin-to-switching as a proxy.

## Global sensitivity

The following utilities are provided for ensemble-level sensitivity reporting:

- `compute_global_sensitivity_dynamic`: a driver–outcome sensitivity report is computed for dynamic pathway ensembles.
- `compute_global_sensitivity_attractors`: a diagnostics summary is computed for a Monte Carlo attractor discovery result (rare-attractor warnings included).
- `DriverSpec`, `OutcomeSpec`: user-defined drivers/outcomes may be provided.
- `GlobalSensitivityReport`: the report dataclass is returned.

## Visualisation

- `MatrixVisualizer`, `ScenarioVisualizer`: basic plots.
- `UncertaintyVisualizer`: probability intervals for scenario scores.
- `ShockVisualizer`: robustness plots.
- `DynamicVisualizer`: stacked state-probability bands over time.
- `DynamicVisualizer.plot_pathway_tree(...)`: visualise a branching pathway graph with weighted edges.
- `DynamicVisualizer.plot_descriptor_branching_summary(...)`: branching-derived analogue of the combined descriptor plot (bands + fan + top-path traces).

## Probabilistic CIA (Joint-Distribution Extension)

This API lives under `cib.prob` and is intentionally not re-exported at top-level `cib`.
It is intended for research use; documented limitations are to be assumed.

- `cib.prob.FactorSpec`: factor name + outcomes
- `cib.prob.ProbabilisticCIAModel`: fit a joint distribution from marginals + multipliers
- `cib.prob.JointDistribution`: dense joint distribution for small scenario spaces
- `cib.prob.ApproxJointDistribution`: approximate joint distribution represented by sampled support
- `cib.prob.FitReport`: solver metadata and objective decomposition for a fit
- `cib.prob.DiagnosticsReport`: distribution consistency checks and diagnostics (including an optional `fit_report` field)
- `cib.prob.DynamicProbabilisticCIA`: dynamic wrapper (refit and predict–update modes)
- `cib.prob.event_probability_bounds(...)` / `ProbabilisticCIAModel.event_probability_bounds(...)`: linear-programming identification bounds for small scenario spaces

