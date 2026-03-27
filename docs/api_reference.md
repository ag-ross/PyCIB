# API Reference (high-level)

This project exposes its public API through the `cib` package.

## Core deterministic CIB

- `CIBMatrix`: cross-impacts are stored and impact balances are computed.
- `Scenario`: scenario state assignment (labels at API boundary, 0-based indices internally).
- `ConsistencyChecker`: CIB consistency of a scenario is checked.
- `ScenarioAnalyzer`: consistent-scenario discovery supports explicit contracts via `find_all_consistent(mode=...)` (`"exhaustive"` default, `"shortlist"` opt-in, `"auto"` legacy behavior) plus exact and Monte Carlo solver entrypoints.
- `GlobalSuccession`, `ConstrainedGlobalSuccession`, `LocalSuccession`, `AttractorFinder`: succession operators and attractor discovery.

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
- `MonteCarloAnalyzer`: \(P(\mathrm{consistent}\mid z)\) is estimated for candidate scenarios and Wilson intervals are returned.

## Robustness and shocks

- `ShockModel`: structural shocks (independent or correlated), optional structural scaling (`additive` or `multiplicative_magnitude`) plus descriptor/state multipliers, and AR(1) dynamic shocks with optional descriptor/state multipliers.
- `RobustnessTester`: consistency robustness and extended robustness metrics (attractor retention, switch rate, mean Hamming distance to base attractor, Wilson intervals) under structural shocks.
- `calibrate_structural_sigma_from_confidence(...)`: helper to derive a structural sigma candidate from confidence codes (`mean`, `median`, or `p75`).
- `suggest_dynamic_tau_bounds(...)`: helper to suggest a dynamic tau range as proportions of structural sigma.

## Expert aggregation (practical Bayesian-style)

- `ExpertAggregator`: weighted aggregation across multiple experts with partial coverage.
- `GaussianCIBMatrix`: per-cell \((\mu,\sigma)\) uncertainty with `sample_matrix(seed)` for Monte Carlo.

## Dynamic CIB (simulation-first)

- `DynamicCIB`: multi-period pathways are simulated with optional cyclic descriptors and threshold-triggered CIM modifiers. Optional equilibrium outputs can be requested via `equilibrium_mode` so that a per-period unshocked relaxation (matrix-consistent attractor) is obtained alongside the shock-realised trajectory. Optional dynamic feasibility controls are available via `constraints`/`constraint_index` with `constraint_mode="strict" | "repair"`, bounded repair (`constrained_top_k`, `constrained_backtracking_depth`), cyclic retry control (`cyclic_infeasible_retries`), and first-period output selection (`first_period_output_mode`). In repair mode, period-locked cyclic descriptors remain fixed and repair is currently supported with the built-in `GlobalSuccession` operator. Dynamic shock forcing via `dynamic_shocks_by_period` and sampled `dynamic_tau` shocks is also currently supported with the built-in `GlobalSuccession` operator.
- `DynamicCIB.simulate_path_extended(...)`: additive disequilibrium-aware dynamic entrypoint; an `ExtendedTransformationPathway` is returned. Supported extension modes are `"transient"`, `"regime"`, and `"path_dependent"`. Important configuration points include `initial_regime`, `equilibrium_mode`, `memory_state`, `regime_transition_rule`, `transition_kernel`, `adaptive_matrix_updater`, and the `return_*` switches for optional output surfaces.
- `DynamicCIB.trace_to_equilibrium(...)`: an arbitrary initial scenario is traced until the consistent set is entered or the iteration budget is exhausted.
- `DynamicCIB.add_regime(...)`, `DynamicCIB.set_regime_transition_rule(...)`: named regimes and regime transitions are configured for extended dynamic runs.
- `ThresholdRule`: threshold rule for either a temporary in-regime matrix modifier (`modifier=...`) or an explicit regime switch (`target_regime="..."`). Modifier callbacks are invoked copy-on-write against a defensive matrix clone; threshold-activation event metadata includes `modifier_returned_distinct_object` for object-identity diagnostics.
- `CyclicDescriptor`: per-period state transitions via a transition probability matrix.
- `TransformationPathway`: pathway representation. When `equilibrium_mode` is enabled, `equilibrium_scenarios` is populated so that an equilibrium analogue of `scenarios` is provided.
- `ExtendedTransformationPathway`: rich pathway representation with realised scenarios, optional `equilibrium_scenarios`, disequilibrium metrics, active regimes, active matrices, transition events, optional memory states, and optional structural-consistency diagnostics.
- `PerPeriodDisequilibriumMetrics`: per-period consistency, margin, brink, distance-to-equilibrium, distance-to-consistent-set, attractor kind, attractor size, and time-to-equilibrium diagnostics. `distance_to_equilibrium` and `distance_to_attractor` both report attractor proximity; `distance_to_consistent_set` reports distance to the local consistent set.
- `ActiveMatrixState`: active-matrix lineage/provenance summary for one period. `provenance_labels` distinguish named regimes (for example `regime:boosted`) from temporary within-regime modifiers (for example `threshold_modifier:RuleName`), actual threshold-triggered regime switches (`threshold_regime_transition:RuleName`), and same-regime reaffirmations (`threshold_regime_reaffirmation:RuleName`). Regime-spell metadata (`entered_regime`, `regime_entry_period`) is also exposed.
- `TransitionEvent`: structured event log item (`threshold_activation`, `regime_transition`, `structural_shock`, `judgment_sampling`, `adaptive_matrix_update`, `memory_update`, `lock_in`, `irreversible_transition`). `regime_transition` denotes an actual regime change only; threshold rules that merely re-validate the current regime are logged as `threshold_activation` with `activation_kind="regime_reaffirmation"`.
- `MemoryState`, `StructuralConsistencyState`, `PathDependentState`: path-dependent state/diagnostic containers.
- `state_probability_timelines`, `numeric_quantile_timelines`, `pathway_frequencies`: summarise ensembles. `state_probability_timelines` and `numeric_quantile_timelines` support `scenario_mode="realized"` (default) or `scenario_mode="equilibrium"` when equilibrium scenarios are present.
- `summarize_disequilibrium_path(...)`, `summarize_disequilibrium_ensemble(...)`: non-mutating summaries for extended pathways.

## Disequilibrium scoring / attribution

- `ScenarioDiagnostics`: diagnostics dataclass returned by `scenario_diagnostics(...)`.
- `AttractorDiagnostics`: nearest-attractor diagnostics returned by `attractor_distance(...)`.
- `equilibrium_distance(...)`: Hamming distance to the nearest exact or fallback attractor under the active matrix.
- `consistent_set_distance(...)`: Hamming distance to the nearest exact local consistent state when the full state space is searched, or to the first consistent state reached by the configured fallback trace.
- `attractor_distance(...)`: nearest-attractor proximity, attractor kind, and attractor size under the active matrix.
- `descriptor_disequilibrium_contributions(...)`: serialisable per-descriptor contribution summary built on the attribution layer.
- `cumulative_disequilibrium_burden(...)`: aggregate negative consistency margins across a path.

## Hybrid branching pathways (enumerate-or-sample)

- `BranchingPathwayBuilder`: a branching pathway graph across periods is built by enumerating consistent scenarios when feasible and falling back to Monte Carlo sampling when not. Nodes can be requested in `node_mode="equilibrium"` (default) so that matrix-consistent attractors are reported even when dynamic shocks are used during sampling; `node_mode="realized"` preserves forcing-realised nodes. When a `memory_state` and/or `transition_kernel` is supplied, branching is switched to an explicit memory-aware sampling contract keyed by scenario, regime residence, memory state, and retained realised-history signature. `history_horizon` can be used so that full or trailing realised-path context is retained in node identity and downstream regime/kernel evaluation.
- `BranchRegimeState`: node-level regime-residence record used by branching outputs.
- `BranchingResult`: branching output container (period layers, weighted edges, top paths). In regime-aware branching, `active_regimes` and `regime_states_by_period` are stored by period layer; in memory-aware branching, `memory_states_by_period`, `history_signatures_by_period`, and an explicit `approximation_contract` describing the sampled approximation used when node identity depends on memory and retained path history are also stored. `node_records(...)` and `top_path_details()` return layer nodes and top paths with regime context attached so that identical scenarios under different regimes remain distinguishable in summaries.
- `branching_regime_residence_timelines(...)`: weighted summaries of active regime, fresh regime entry, and threshold-driven regime reaffirmation by period.
- `RegimeSpec`, `RegimeTransitionRule`: named regime definitions and regime-transition interfaces for extended dynamic/branching workflows.

## Scenario diagnostics / scoring

- `scenario_diagnostics`: consistency, inconsistency details, margins (“brink”), and total impact score for a scenario are computed.
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

## Transformation analysis

- `TransformationMatrixBuilder`: scenario-to-scenario perturbation testing; `build_matrix(...)` accepts optional `extension_mode`, `regime_name`, `active_matrix`, and `max_hamming_match` so that perturbation analysis can be run against a specific regime/active matrix with an explicit matching contract.
- `TransformationMatrixBuilder.analyze_path_to_path_transformations(...)`: scenario changes and memory-only divergences are analysed under their active matrix, regime, and optional memory context. When `initial_scenario` and a `transition_kernel` are supplied, full-path replay, source-state replay, and perturbation support are reported separately. For memory-only changed periods (no scenario-state change), perturbation testing is skipped and `supported_by_perturbations` is reported as `False`.
- `summarize_path_to_path_transformations(...)`: sequence-level summary for path-to-path comparisons. Optional memory-state and regime context can be supplied so that changed periods and memory-changed periods are reported together. Memory-change detection includes `MemoryState.period` in addition to `values`, `flags`, and `export_label`.
- `explain_regime_transformation(...)`: lightweight regime-aware explanation payload for a transition.

## Path dependence

- `TransitionKernel`, `DefaultTransitionKernel`: transition-law interfaces for path-dependent dynamic runs. The default kernel can use previous-path context so that cycles are moved through, and lock-in effects can be surfaced via transition events when memory constrains realised states.
- `TransitionKernel.replay_path(...)`, `TransitionReplayRecord`, `TransitionReplayResult`: replay utilities for checking whether a target path is reproduced by the actual transition law, including explicit period labels, optional initial snapshots, and optional memory-state matching.
- `AdaptiveCIMUpdater`: interface for history-dependent active-matrix updates.
- `HysteresisRule`, `IrreversibilityRule`: minimal helper rules for path-dependent memory flags.
- `check_structural_consistency(...)`: structural consistency is evaluated independently from local impact-balance consistency, including history-aware checks for persistent lock-in semantics and cycle-phase continuity.

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
- RNG contract: in both direct and iterative fit modes, `random_seed=None` is non-deterministic and explicit integer seeds are reproducible.

