# PyCIB Documentation and Mathematical foundations (concise)

## Practical introduction (workshop to analysis, in plain terms)

This section is provided as a thematic guide for researchers and practitioners who are not yet familiar with the options provided by this package. Guidance is provided on when each option is to be selected, what is required as inputs, and what is obtained as outputs. All detailed definitions, mathematics, and configuration parameters are provided later in this document (see also the README in this project for references).

### 1) What is produced by a workshop (inputs in plain terms)

The following items are typically produced by a CIB workshop and are used as inputs.

- **A set of descriptors (factors)**: a small list of factors is selected (for example, policy, infrastructure, behaviour). Each factor is represented as a named descriptor.
- **Discrete states per descriptor**: each descriptor is assigned a small set of named states (for example, Low, Medium, High). These names are the primary labels used in reporting.
- **Influence judgements between descriptors**: statements about how one descriptor affects another are recorded and converted into a bounded score (commonly \(-3\) to \(+3\)). Positive scores are used to represent promotion/support, and negative scores are used to represent hindrance.
- **Confidence (optional but should be common)**: if confidence is recorded for a judgement (for example, on a 1–5 scale), it is treated as uncertainty about that particular score. Where uncertainty workflows are selected, lower confidence is represented as wider variation around the recorded mean score.
- **A baseline scenario (optional)**: when time-based pathways are required, an initial scenario is selected by choosing one state for each descriptor.
- **Domain rules (optional)**: if some combinations are known to be infeasible or are required to be enforced, those rules are specified explicitly (for example, an implication or a forbidden combination).
- **Time periods (optional)**: when a time horizon is required, a small set of periods is selected (for example, 2025–2050 in steps).

At this stage, code-level inputs are not required to be defined. The workshop artefacts above are sufficient for selection of an appropriate analysis approach.

### 2) How an approach is selected (decision guide)

Selection is typically driven by the question being asked and by the type of output required for reporting.

- **A small set of “workshop-consistent” futures is required**: deterministic CIB is typically selected first.
- **Robustness to uncertain judgements is required**: uncertainty workflows are typically selected (confidence-coded uncertainty and, where required, additional stress testing).
- **A complete set of consistent scenarios is required (within feasibility limits)**: exact or pruned enumeration is typically selected.
- **The relative importance of alternative attractors is required (as frequencies)**: Monte Carlo attractor analysis is typically selected.
- **A storyline across time is required**: dynamic multi-period simulation is typically selected.
- **A compact pathway map is required for communication**: branching pathway summaries are typically selected.
- **Known infeasible combinations are required to be enforced**: domain feasibility constraints are typically selected.
- **High-cardinality descriptors are present**: model reduction (state binning) is typically selected before solver execution.
- **Structural interpretation is required**: network analysis is typically selected (impact networks and scenario similarity networks are provided for different purposes).
- **The causes of transitions are to be studied explicitly**: transformation matrix analysis is typically selected.
- **An explicit probability model over a reduced factor set is required**: `cib.prob` is typically selected as a separate probabilistic CIA model.

### 3) One running workshop example (used across options)

An illustrative workshop system is described here so that each option may be explained by reference to the same material. The system is intentionally small and is not to be interpreted as a recommended model.

Five descriptors may be considered, each with states Low, Medium, and High:

- Policy stringency
- Grid flexibility
- Electrification demand
- Public acceptance
- Permitting speed

Qualitative workshop statements such as the following are commonly elicited and then converted into influence scores:

- When policy stringency is high, electrification demand is supported.
- When grid flexibility is low, high electrification demand is hindered.
- When public acceptance is high, high policy stringency is supported.
- When permitting speed is high, high grid flexibility is supported.

An initial baseline scenario may be selected for time-based work (for example, all descriptors at Medium), and a period grid may be selected if required (for example, 2025, 2030, 2035, 2040, 2045, 2050).

### 4) Option-by-option thematic guide (when, inputs, outputs, and examples)

This section summarises the main options provided by the codebase. Each option is described by (i) when it is to be selected, (ii) what is required, and (iii) what is obtained.

#### 4.1 Deterministic CIB (central estimate: consistent scenarios and attractors)

- **When it is to be selected**: when workshop mean judgements are intended to be treated as fixed and a small set of internally consistent scenarios is required as the primary workshop output.
- **What is required (inputs)**:
  - descriptors and states;
  - influence scores (workshop mean judgements);
  - an optional cap on how many scenarios are to be returned when the space is large.
- **What is obtained (outputs)**:
  - consistent scenarios (scenarios that fit the influence structure);
  - an attractor per initial scenario when succession is applied (a fixed-point scenario or a cycle, depending on the system).
- **Example (workshop framing)**: the five-descriptor example above may be used to obtain a small set of scenario archetypes, such as a high-stringency / high-demand / high-flexibility combination and alternative consistent outcomes implied by the same judgements.

Worked reference workflows are provided in `examples/example_1_basic_cib.py` and `examples/example_cycle_detection.py`.

#### 4.2 Exact and pruned enumeration (complete consistent-scenario set when feasible)

- **When it is to be selected**: when a complete set of consistent scenarios is required for a small-to-moderate scenario space, or when an exact search with pruning is required.
- **What is required (inputs)**:
  - descriptors, states, and influence scores;
  - optional domain rules if admissibility beyond consistency is required;
  - an understanding of feasibility limits (enumeration scales with the number of descriptors and states).
- **What is obtained (outputs)**:
  - a complete list of consistent scenarios when the solver completes;
  - solver diagnostics indicating completeness and any time-limit truncation when configured.
- **Example (workshop framing)**: if the five-descriptor system is reduced (or if fewer descriptors are used), exact enumeration may be used to list all consistent scenario archetypes explicitly so that none are omitted.

Worked reference workflows are provided in `examples/example_enumeration_c10.py` and `examples/example_solver_modes.py`.
Relevant technical reference is provided in the section “Scaling workflows (solver modes and benchmarks)” below.

#### 4.3 Monte Carlo attractor analysis (empirical frequencies of attractors)

- **When it is to be selected**: when the number of possible initial scenarios is large and relative attractor importance is required as empirical frequencies, rather than as an exhaustive list.
- **What is required (inputs)**:
  - descriptors, states, and influence scores;
  - a run count and a random seed for reproducibility;
  - an optional choice of succession mode and cycle handling policy.
- **What is obtained (outputs)**:
  - a set of discovered attractors (fixed points and/or cycles) with counts or weights;
  - diagnostics of convergence and run completion.
- **Example (workshop framing)**: the five-descriptor example may yield several attractors; Monte Carlo analysis may be used to estimate which attractors are common and which are rare under the assumed sampling design.

Worked reference workflows are provided in `examples/example_solver_modes.py` and `examples/example_solver_modes_c10.py`.
Relevant technical reference is provided in the section “Scaling workflows (solver modes and benchmarks)” below.

#### 4.4 Dynamic multi-period pathways (simulation-first time evolution)

- **When it is to be selected**: when a storyline across time is required and the system is intended to be studied across a sequence of periods starting from an initial scenario.
- **What is required (inputs)**:
  - descriptors, states, and influence scores;
  - an initial scenario;
  - a period grid;
  - optional uncertainty and shock assumptions when variation across runs is required.
- **What is obtained (outputs)**:
  - realised pathways across periods for each run;
  - per-period summaries when many runs are aggregated (for example, how often each state occurs at each period);
  - optional equilibrium relaxations where the distinction between realised and equilibrium reporting is required.
- **Example (workshop framing)**: the five-descriptor example may be run from a Medium baseline; under uncertainty and shocks, switching may be observed between alternative future states (for example, a shift to high policy stringency with delayed grid flexibility under one set of assumptions).

Worked reference workflows are provided in `examples/example_dynamic_cib_c10.py`, `examples/example_dynamic_cib_c15_rare_events.py`, and `examples/dynamic_cib.ipynb`.
Relevant technical reference is provided in the section “Simulation-first probabilistic CIB (stochastic simulation workflow)” below.

#### 4.5 Branching pathway summaries (compact pathway maps)

- **When it is to be selected**: when a compact representation of a small number of dominant pathway archetypes is required for communication, rather than a large ensemble of individual trajectories.
- **What is required (inputs)**:
  - descriptors, states, and influence scores;
  - an initial scenario and a period grid;
  - a decision on whether enumeration is feasible for a period (exact) or whether transitions are to be estimated by sampling.
- **What is obtained (outputs)**:
  - a per-period pathway graph or tree with weighted transitions;
  - a ranked set of top pathways when configured.
- **Example (workshop framing)**: a small number of high-weight futures may be presented as a pathway tree, showing common transitions (for example, an early move to high policy stringency followed by later shifts in demand and flexibility).

Worked reference workflows are provided in `examples/dynamic_cib.ipynb`.
Relevant technical reference is provided in the section “Simulation-first probabilistic CIB (stochastic simulation workflow)” below.

#### 4.6 Domain feasibility constraints (inadmissible combinations and implications)

- **When it is to be selected**: when some combinations are to be excluded or enforced by domain reasoning, separately from the internal consistency principle.
- **What is required (inputs)**:
  - a list of rules, commonly expressed as forbidden pairs, implications, and allowed-state restrictions;
  - agreement on whether a rule is to be treated as hard admissibility (excluded) rather than as a workshop influence judgement.
- **What is obtained (outputs)**:
  - consistent scenarios and pathways restricted to the admissible space;
  - solver pruning and performance improvements when constraints exclude large parts of the space.
- **Example (workshop framing)**: a rule may be specified such that high policy stringency is not admissible with low public acceptance; under this rule, otherwise-consistent scenarios are excluded from reporting.

Worked reference workflows are provided in `examples/example_enumeration_c10.py` and `examples/example_solver_modes_c10.py`.
Relevant technical reference is provided in the section “Scaling workflows (solver modes and benchmarks)” below.

#### 4.7 Model reduction (state binning for high-cardinality descriptors)

- **When it is to be selected**: when descriptors have many states and direct analysis becomes infeasible or difficult to interpret; reduction is commonly selected before exact enumeration and before large Monte Carlo runs.
- **What is required (inputs)**:
  - a mapping from many original states into a smaller number of bins (for example, merging very similar categories);
  - a procedure for mapping scenarios into the reduced state space for reporting.
- **What is obtained (outputs)**:
  - a reduced descriptor system and reduced influence structure suitable for solver execution;
  - mapped scenarios and outputs expressed in reduced categories.
- **Example (workshop framing)**: if electrification demand has many categories, a reduced Low/Medium/High scheme may be selected for analysis, and results may be reported in those reduced categories.

Worked reference workflows are provided in `examples/example_state_binning.py`.
Relevant technical reference is provided in the section “Scaling workflows (solver modes and benchmarks)” below.

#### 4.8 Network analysis (structural interpretation and communication)

- **When it is to be selected**: when a structural view is required, either to communicate influences between descriptors or to summarise an ensemble of scenarios at a fixed period.
- **What is required (inputs)**:
  - the influence structure for impact-network summaries;
  - a selected scenario (for scenario-dependent weights) when required;
  - an ensemble of scenarios at a fixed period when scenario-similarity networks are required.
- **What is obtained (outputs)**:
  - an impact network over descriptors (showing where influences are strong and in which direction);
  - a scenario similarity network (showing which scenarios are similar within an ensemble, without implying temporal transitions).
- **Example (workshop framing)**: an impact network may be used to show that policy stringency is a central driver in the influence structure, while a scenario similarity network may be used to show clusters of common final-period outcomes in a dynamic run.

Worked reference workflows are provided in `examples/example_1_basic_cib.py` and `examples/dynamic_cib.ipynb`.

#### 4.9 Transformation matrices (what perturbations induce transitions)

- **When it is to be selected**: when the mechanisms that cause transitions between scenario archetypes are to be studied explicitly (for example, which types of perturbations can move the system from one attractor to another).
- **What is required (inputs)**:
  - a set of reference scenarios or attractors between which transitions are to be tested;
  - a specification of the perturbation types to be tested (for example, uncertainty, structural perturbations, dynamic shocks).
- **What is obtained (outputs)**:
  - a transition summary indicating which perturbations can produce which transitions;
  - an auditable basis for claims about sensitivity of the system to particular disturbance classes.
- **Example (workshop framing)**: a transition from a high-demand attractor to a medium-demand attractor may be shown to require strong perturbations to particular influence judgements or to require particular dynamic disturbances.

Worked reference workflows are provided in `examples/example_transformation_matrix.py`.
Relevant technical reference is provided in the section “Simulation-first probabilistic CIB (stochastic simulation workflow)” below.

#### 4.10 Joint-distribution probabilistic CIA (`cib.prob`) (explicit probability model, separate from simulation-first CIB)

- **When it is to be selected**: when an explicit joint probability distribution over a reduced factor set is required for downstream probabilistic reasoning, and inputs are available in the form of marginal probabilities and dependency multipliers.
- **What is required (inputs)**:
  - a factor set and discrete outcomes for each factor (typically a reduced set);
  - marginal probabilities for each factor outcome;
  - optional dependency information as probability-ratio multipliers, treated as targets to be matched under fixed marginals.
- **What is obtained (outputs)**:
  - a fitted joint distribution with diagnostics;
  - feasibility handling behaviour (strict rejection or repair projection) when incoherent inputs are provided;
  - dynamic variants when per-period refits or predict–update regularisation are selected.
- **Example (workshop framing)**: if probabilities are required for a reduced model of policy stringency, grid flexibility, and electrification demand, and if marginal probabilities are available from external sources, `cib.prob` may be used to construct an explicit \(P(x)\) that respects those marginals and the selected dependency constraints.

Worked reference workflows are provided in `examples/example_probabilistic_cia_static.py`, `examples/example_probabilistic_cia_dynamic_refit.py`, `examples/example_probabilistic_cia_dynamic_predict_update.py`, `examples/example_probabilistic_cia_strict_vs_repair.py`, `examples/example_probabilistic_cia_sparse_kl.py`, and `examples/example_probabilistic_cia_scaling_iterative.py`.
Relevant technical reference is provided in the section “Joint-distribution probabilistic CIA (`cib.prob`)” below.

### 5) Common pitfalls and interpretation notes (non-technical)

The following interpretation issues can be encountered and are highlighted to reduce misuse.

- **Two meanings of probability are implemented**: simulation-first probabilistic CIB uses empirical frequencies from repeated simulation, while `cib.prob` fits an explicit joint distribution from marginals and multipliers. These outputs are not interchangeable and are to be reported distinctly.
- **Sampling and pruning effects are to be distinguished**: branching pathway summaries may omit low-weight branches for readability, while Monte Carlo ensembles represent full trajectory sampling; differences between the two are expected under finite sampling and pruning.
- **Domain rules are not workshop judgements**: feasibility constraints are external admissibility rules and are applied as exclusions or implications; they are not to be interpreted as influence strengths.
- **Reduction changes the reporting vocabulary**: state binning changes what the labels mean; reduced outputs are to be reported as reduced categories and are not to be reinterpreted as the full-resolution original states.

The detailed mathematical foundations and technical definitions are provided below. The section “Scope note: two different probabilistic features” is to be consulted before probability language is used in reporting.

## Terminology (abbreviations)

The following abbreviations are used:

- Cross-Impact Balance (CIB)
- Cross-Impact Analysis (CIA)
- Cross-Impact Matrix (CIM)

## Thematic overview: sequence of events and where uncertainty enters

This document describes, in practical terms, how a CIB workflow is made probabilistic, where uncertainty and stochasticity are introduced, and how they are manifested in plots (including median trajectories). The aim is to make explicit the sequence of computations that sit behind the deterministic CIB principles described below.

The material is organised as follows:

- Joint-distribution probabilistic CIA (`cib.prob`) is described first, for cases where an explicit fitted joint distribution is required.
- Simulation-first probabilistic CIB is described next, covering uncertainty modelling, dynamic shocks, threshold rules, cyclic descriptors, and cycle handling.
- Scaling workflows (solver modes and benchmarks) are described afterwards, as a practical guide for solver entrypoints and test execution.

## Scope note: two different “probabilistic” features

Two distinct uses of “probabilistic” are implemented in this repository, and conflation is to be avoided:

- **Simulation-first probabilistic CIB**: uncertainty is introduced into the CIB simulation workflow (e.g. confidence-coded sampling of the CIM, structural shocks, AR(1) dynamic shocks, thresholds, cyclic descriptors). Outputs are empirical frequencies obtained from Monte Carlo ensembles and/or branching pathway graphs.
- **Joint-distribution probabilistic CIA (`cib.prob`)**: “probability” means an explicit fitted joint distribution \(P(x)\) over discrete factors, constrained by user-specified marginals and cross-impact multipliers (probability ratios). This is not derived from CIB impact scores, and it does not use the shocks/threshold/cycle simulation machinery described below.

For usage, theory, and limitations of joint-distribution probabilistic CIA, the section “Joint-distribution probabilistic CIA (`cib.prob`)” below is to be consulted. Example scripts are provided under `examples/`.

## Joint-distribution probabilistic CIA (`cib.prob`)

An explicit joint distribution \(P(x)\) over discrete factors is fitted from (i) user-specified marginals and (ii) cross-impact multipliers interpreted as conditional probability ratios.
The fitted model is intended for research use where an explicit probabilistic scenario space representation is required.

### Recommended usage and scope (what is and is not provided)

The following scope boundaries are intended to prevent conflation with simulation-first probabilistic CIB outputs.

In summary, probabilities are obtained in this repository in two different ways:
(i) empirical frequencies are obtained by simulating the CIB dynamics model under configured uncertainty and shocks, and
(ii) an explicit joint distribution is fitted from user-specified marginals and multipliers in `cib.prob`.
The second approach is to be interpreted as a separate probabilistic CIA model and is not to be interpreted as “probabilities of CIB attractors” unless it has been parameterised explicitly as such.

- **What is provided**:
  - an explicit fitted joint distribution over a specified factor set, constrained by provided marginals and multiplier-derived targets
  - auditable feasibility handling (strict rejection or repair projection) and fit diagnostics
  - a dense reference solver intended for small scenario spaces
  - an approximate scaling back-end intended for large scenario spaces when dense enumeration is not desired

- **What is not provided**:
  - probabilities derived from CIB impact balances or from succession dynamics (these are to be obtained from the simulation-first workflow)
  - a claim of causal dynamics in time (predict–update is provided as regularisation via a KL baseline, rather than as a mechanistic transition model)
  - exact inference for large scenario spaces when the approximate back-end is selected

- **When the simulation-first workflow is to be preferred**:
  - when uncertainty is to be expressed through confidence-coded sampling of the CIM, structural shocks, dynamic shocks, cyclic descriptors, and threshold rules
  - when large descriptor systems are being studied, where model reduction and Monte Carlo / branching approaches are the standard scaling mechanisms

- **When `cib.prob` is to be preferred**:
  - when a reduced factor set is being analysed and an explicit joint distribution is required for downstream probabilistic reasoning
  - when a small sub-model is being used for sensitivity analysis or for interpretation of multiplier-implied dependencies
  - when identification bounds are required under a specified constraint set (small scenario spaces only)

### Concepts and notation

- **Factors** \(X_1,\dots,X_n\) are discrete variables with named outcomes.
- A **scenario** is a complete assignment \(x=(x_1,\dots,x_n)\).
- **Marginals** \(\hat P(X_i=a)\) are provided for each factor/outcome pair.
- **Cross-impact multipliers** (probability ratios) are provided as:
  \[
  m_{(i=a)\leftarrow(j=b)}=\frac{P(X_i=a\mid X_j=b)}{P(X_i=a)}.
  \]

### Inputs

The following inputs are required:

- a factor specification (`cib.prob.FactorSpec`) giving factor names and outcome labels
- marginal probabilities for every factor outcome

The following inputs are optional:

- multipliers for selected ordered factor-outcome pairs
- relevance specifications used to down-weight constraints (`cib.prob.RelevanceSpec`)
- dynamic priors used in predict–update mode (Section “Dynamic modelling” below)

### Feasibility and coherence conditions

#### Pairwise target construction

When multipliers are provided, implied pairwise targets are constructed as:

\[
P(X_i=a, X_j=b) \equiv m_{(i=a)\leftarrow(j=b)} \, P(X_i=a)\, P(X_j=b).
\]

This construction is used as a stable moment target under fixed marginals.

#### Fréchet bounds (pairwise feasibility)

For any target \(P(X_i=a, X_j=b)\) and given marginals \(P(X_i=a)\) and \(P(X_j=b)\), feasibility requires:

\[
\max(0, P(X_i=a)+P(X_j=b)-1) \le P(X_i=a, X_j=b) \le \min(P(X_i=a), P(X_j=b)).
\]

Violations are treated as evidence of incoherent inputs. Two feasibility strategies are supported:

- **Strict**: an exception is raised when a target violates Fréchet bounds beyond a tolerance.
- **Repair**: violating targets are projected into the feasible interval and the applied adjustments are reported.

#### Multiplier normalisation (conditional coherence)

For a complete multiplier set for fixed \((i, j=b)\), the following constraint is implied by conditional normalisation:

\[
\sum_a m_{(i=a)\leftarrow(j=b)}\,P(X_i=a)=1.
\]

When complete multiplier contexts are provided, normalisation issues are detected and are surfaced in diagnostics. Optional enforcement is supported.

### Fitting objectives and methods

#### Dense reference solver (small scenario spaces)

For small scenario spaces, a dense joint probability vector \(p\) is fitted under hard constraints:

- \(p \ge 0\)
- \(\sum_x p(x)=1\)
- exact marginal matching for all provided marginals

Pairwise targets are treated as soft constraints and are matched in a weighted least-squares sense.
An optional KL regulariser is provided as a minimum-information tie-breaker.

The objective is:

\[
\min_{p} \sum_k w_k\,(A_k p - t_k)^2 + \lambda\,\mathrm{KL}(p \,\Vert\, q),
\]

where \(t_k\) are the implied pairwise targets, \(A_k p\) are the implied pairwise marginals under \(p\), and \(q\) is a baseline distribution (independence or a dynamic prior).

When numerical failure is encountered in the default optimiser, a solver fallback is applied to improve robustness.

#### Scalable approximate solver (large scenario spaces)

For large scenario spaces, explicit enumeration is not required. An approximate solver is provided which returns an approximate distribution representation suitable for:

- marginal and conditional estimation
- diagnostic reporting
- comparison to the dense solver on small systems

The approximation back-end is implemented as a sampled-support representation (`cib.prob.ApproxJointDistribution`) produced by a stochastic fitting procedure.
The returned distribution is approximate and does not provide exact scenario probabilities for arbitrary assignments unless they are present in the sampled support.

### Dynamic modelling

Two dynamic modes are supported:

- **Refit**: each period is fitted independently from the period’s marginals and multipliers.
- **Predict–update**: a prior distribution from the previous period is used as the KL baseline \(q\), so that temporal coupling is induced through regularisation.

The predict–update objective is treated as a regularised fit, rather than as a claim of causal temporal dynamics.

### Diagnostics and reporting

Diagnostics are intended to enable a determination of whether a fit is reliable for downstream use.
The following reporting objects are provided:

- `cib.prob.FitReport`: solver metadata and objective decomposition (including residual summaries)
- `cib.prob.DiagnosticsReport`: marginal and pairwise target agreement checks (and feasibility warnings)

### Risk bounds (identification bounds; small spaces)

When the constraint set is incomplete, event probabilities may be not uniquely identified.
In the dense regime, conservative bounds may be computed as linear programmes:

\[
\underline{P}(E) = \min_{p \in \mathcal{C}} \sum_{x\in E} p(x), \qquad
\overline{P}(E) = \max_{p \in \mathcal{C}} \sum_{x\in E} p(x),
\]

where \(\mathcal{C}\) denotes the feasible set induced by the configured constraints.
These bounds are returned as identification bounds, rather than as sampling uncertainty intervals.

### Examples

The following examples are provided under `examples/`:

- `example_probabilistic_cia_static.py`: dense static fit and implied conditionals
- `example_probabilistic_cia_dynamic_refit.py`: per-period refit and probability timelines
- `example_probabilistic_cia_dynamic_predict_update.py`: predict–update regularisation
- `example_probabilistic_cia_strict_vs_repair.py`: strict versus repair feasibility behaviour
- `example_probabilistic_cia_sparse_kl.py`: sparse multipliers with KL regularisation and risk bounds
- `example_probabilistic_cia_scaling_iterative.py`: scaling with the iterative approximate back-end

### Method specification (formal)

The following summary is provided so that the method can be re-implemented independently from the description.

Let \(X_1,\dots,X_n\) be discrete random variables (factors), each taking values in a finite outcome set \(\mathcal{X}_i\).
A scenario is denoted by \(x=(x_1,\dots,x_n)\in\mathcal{X}_1\times\cdots\times\mathcal{X}_n\).

Marginal probabilities are provided as \(\hat P_i(a) \equiv \hat P(X_i=a)\), with \(\sum_{a\in\mathcal{X}_i} \hat P_i(a)=1\).
Cross-impact multipliers are provided as ratios:

\[
m_{(i=a)\leftarrow(j=b)} \equiv \frac{P(X_i=a \mid X_j=b)}{P(X_i=a)}.
\]

Given fixed marginals, each multiplier is converted into a pairwise probability target:

\[
t_{i,a,j,b} \equiv \hat P(X_i=a,X_j=b) := m_{(i=a)\leftarrow(j=b)}\,\hat P_i(a)\,\hat P_j(b).
\]

For each candidate target \(t\), feasibility is required under Fréchet bounds:

\[
\max(0, \hat P_i(a)+\hat P_j(b)-1) \le t \le \min(\hat P_i(a),\hat P_j(b)).
\]

For small scenario spaces, the fitted distribution \(p\) is selected by minimising:

\[
\min_{p} \sum_{k} w_k (A_k p - t_k)^2 + \lambda\,\mathrm{KL}(p \,\Vert\, q),
\]

under the hard constraints \(p\ge 0\), \(\sum_x p(x)=1\), and exact marginal matching.
In predict–update mode, the baseline \(q\) is set to the previous period distribution \(p_{t-1}\).

## Simulation-first probabilistic CIB (stochastic simulation workflow)

This section describes the simulation-first workflow in which probabilities are obtained as empirical frequencies from repeated simulation under configured uncertainty and shocks. It is to be interpreted as distinct from `cib.prob`, which fits an explicit joint distribution from marginals and multipliers.

### 1) Deterministic core (workshop “central estimate”)

At the core of CIB is a deterministic cross-impact matrix (CIM) \(C\) populated by workshop judgements. Given a scenario \(z\) (one state chosen per descriptor), impact scores \(\theta_{j,l}(z)\) are computed and the consistency principle (maximum score condition) is applied to determine whether \(z\) is consistent. Succession operators iterate a scenario towards an attractor (typically a fixed point, sometimes a cycle), which provides a canonical “consistent scenario” implied by the workshop’s central judgements.

### 2) Judgement uncertainty (confidence-coded sampling)

Workshop participants are typically more confident about some judgements than others. In this implementation, a confidence code \(c\in\{1,\dots,5\}\) is converted into a standard deviation \(\sigma(c)\). A sampled matrix \(C^{(m)}\) is generated by drawing each cell around its mean judgement and clipping to \([-3,+3]\). This is the first place where runs diverge: different samples imply different impact balances and may yield different attractors.

Optionally, uncertainty can be made time-dependent (a common elicitation assumption): \(\sigma\) may be scaled up for longer horizons, reflecting that respondents are less certain about the longer-term future.

### 3) Structural shocks (stress-testing, regime perturbations)

Structural shocks are represented as perturbations to the CIM entries, distinct from judgement uncertainty. Conceptually, the following question is addressed: “If the world deviates from the elicited system, how robust are the consistent scenarios?” Structural shocks are applied by perturbing the CIM and repeating the consistency and succession computations under the perturbed matrix. In robustness testing and other static stress-testing uses, a single perturbed CIM is sampled per shock draw. In `DynamicCIB.simulate_path()`, structural shocks are re-sampled per period when `structural_sigma` is provided. In `DynamicCIB.simulate_ensemble()`, structural shocks are applied once per run when `judgment_sigma_growth` is zero, and are applied per period when `judgment_sigma_growth` is positive.

### 4) Dynamic shocks (AR(1) stochastic forcing during succession)

Dynamic shocks are applied at the impact-balance level during within-period succession. They are designed to model time-varying disturbances that can (a) nudge balances between close competing states and (b) occasionally induce switching or activate threshold-triggered regime shifts. Persistence is introduced by the AR(1) structure: shocks are not independent across periods, so sequences can exhibit momentum.

Fat-tailed innovations (Student-\(t\)) and jump components provide “rare event” behaviour: a small number of runs can experience large disturbances, producing visibly different trajectories.

### 4b) Realised versus equilibrium trajectories (recommended for research reporting)

Two complementary outputs are useful when dynamic shocks are present:

- **Realised trajectory**: the per-period scenario produced by shock-aware succession. This is the natural object for studying perturbation-driven switching and threshold activation. A realised scenario is not required to be CIB-consistent with respect to an unshocked matrix because selection is performed under perturbed scores \(\theta'=\theta+\eta\).
- **Equilibrium trajectory (optional)**: a per-period unshocked relaxation that starts from the realised scenario and applies standard succession on the same active period matrix. This produces a scenario that is CIB-consistent with respect to the period matrix used for that period.

This separation is recommended because it preserves the classical CIB definition of consistency for equilibrium reporting while retaining stochastic forcing as a mechanism for transitions. The gap between realised and equilibrium scenarios (for example, Hamming distance or consistency-margin differences) can be used as a diagnostic of proximity to switching thresholds or of “shock pressure”.

### 5) Threshold rules and cyclic (exogenous/inertial) descriptors

Two additional mechanisms govern when and how the active CIM changes and how certain descriptors evolve:

- **Threshold rules** conditionally modify the active CIM based on the current scenario (e.g. tipping points or policy regimes). A small disturbance can move the system across a threshold, which then changes the CIM and alters subsequent attractors. When multiple rules match, the applied behaviour is controlled by a threshold matching policy. The default behaviour is sequential application of all matching modifiers in the configured order. An alternative first-match-only policy is supported.
- **Cyclic descriptors** evolve via an explicit transition matrix between periods (often with strong persistence), representing exogenous drift or inertia that is not endogenously resolved by CIB within a period. These descriptors are “locked” during within-period succession so that their exogenous evolution is not overwritten.

Threshold rule timing:

- In both `DynamicCIB.simulate_path()` and `BranchingPathwayBuilder.build()`, threshold rules for period t+1 are evaluated on the scenario at the start of period t+1 (after cyclic transitions have been applied). The active CIM used for the transition into period t+1 is therefore determined by that post-cyclic scenario in both entry points.
- It is important to be explicit about which scenario state a threshold condition is evaluated against when threshold conditions depend on cyclic descriptors; the implementation uses the post-cyclic state consistently.

### 6) What a “Monte Carlo run” is (and why it is a single path)

A single Monte Carlo run produces one realised pathway \(z_{t_0}, z_{t_1}, \dots\) because one draw of the stochastic elements (judgement sampling, shocks, cyclic transitions) is made and then the pathway is advanced forward in time. The branching behaviour is not within a single run; it emerges across many runs. The distributional plots (probability bands, fan charts) are summaries of the ensemble.

### 7) Branching pathway graphs versus Monte Carlo ensembles

The ensemble can be represented in two complementary ways:

- **Monte Carlo ensemble (simulation-first)**: many complete pathways are simulated and summarised. This is the most direct estimator for per-period marginals such as \(P(z_j(t)=s)\), quantiles, and expectations.
- **Branching pathway graph (enumerate-or-sample)**: an explicit per-period graph is built whose nodes are scenarios (typically attractors) and whose edges carry transition weights. In sampling mode, those weights are estimated from repeated stochastic transitions out of each node. The resulting graph can be propagated forward to obtain the same per-period marginals as the Monte Carlo ensemble, subject to sampling noise and any pruning/caps applied for readability.

### 8) Network analysis layer (impact networks and scenario similarity networks)

In addition to the deterministic/probabilistic CIB computations above, a network-analysis layer is provided by this implementation for two distinct graph objects:

1. **Impact network (descriptor graph)**: a directed graph over descriptors that captures aggregated or state-specific influences.
   - Nodes are descriptors.
   - A directed edge \(i \rightarrow j\) exists when the corresponding CIB impacts are non-zero under the chosen aggregation rule.
   - For each ordered pair \((i,j)\), impacts are aggregated across all state pairs \((k,l)\) using a specified rule (for example, mean absolute or mean signed), producing:
     - A signed representative weight \(w_{i \rightarrow j}\) (promoting vs. hindering).
    - An absolute weight \(|w_{i \rightarrow j}|\) used for magnitude-based analysis and visualisation.
   - Centrality measures and pathway enumeration are then applied to this descriptor graph.

2. **Scenario similarity network (scenario graph)**: a graph over scenarios that is used as a visual summary of an ensemble at a fixed period.
   - Nodes are scenarios (state assignment vectors) at a single period \(t\), typically the final period in a dynamic run.
   - Node weights are the empirical frequencies from the Monte Carlo ensemble, \(n(z,t)\), where \(n\) counts how many runs produced scenario \(z\) at time \(t\).
   - Edges represent similarity, not temporal transition. A common rule is Hamming distance:
     \[
     d(z,z') = \sum_{j=1}^{N}\mathbf{1}\{z_j \ne z'_j\}.
     \]
     Scenarios are connected when \(d(z,z')\) is small, and an edge weight can be defined as \(1/(1+d)\).

This separation matters: the impact network is a structural summary of the CIM, while the scenario similarity network is an empirical summary of the simulated distribution over scenarios at a specific time.

Implementation notes:

- The network analysis layer is implemented in `cib/network_analysis.py`.
- Community detection uses Louvain clustering via `networkx.community.louvain_communities()`, which requires NetworkX 3.0 or newer.
- The example notebook generates a final-period scenario similarity plot and writes a text file with the plotted scenario definitions and their Monte Carlo counts.

### 9) How median trajectories arise (and why they can appear constant)

Numeric summaries for an ordered descriptor are included in the plots by applying a mapping \(\mathrm{map}(s)\) from each discrete state \(s\) to a numeric value. Given per-period state probabilities \(P_t(s)\), the expected value is:

\[
\mathbb{E}[X_t] = \sum_s P_t(s)\,\mathrm{map}(s)
\]

Quantiles (including the median) are defined with respect to the induced discrete distribution over \(\mathrm{map}(s)\). A median can appear constant over time when the same state (often the middle state) retains at least 50% probability across periods; in that case the 0.5-quantile does not move even though uncertainty bands widen and the expected value can drift.

### 10) Cycle detection in succession and dynamic simulations

Cycles occur when succession operators revisit previously encountered scenarios, creating a periodic attractor rather than a fixed point. This section explains when cycles occur, how they are detected, and how they are handled in different contexts.

#### When cycles occur

Cycles can emerge in CIB succession when:
- Multiple scenarios have equal or near-equal impact scores, causing succession to alternate between them
- Stochastic succession introduces randomness that prevents convergence to a single fixed point
- System dynamics create periodic patterns (e.g., seasonal effects, alternating policy regimes)

#### Cycle detection mechanism

The `SuccessionOperator.find_attractor()` method detects cycles by maintaining a visited set of scenarios. When a successor scenario has been encountered before, a cycle is identified:

1. Succession proceeds from an initial scenario
2. Each successor is checked against previously visited scenarios
3. If a duplicate is found, the cycle is extracted (from first occurrence to current)
4. `AttractorResult.is_cycle` is set to `True` and `attractor` contains the cycle list

Implementation note:

- The cycle list contains the distinct periodic states only. The repeated closing state
  (the first state repeated at the end when a loop is detected) is not included. This
  avoids double-counting and prevents bias when a cycle state is selected at random.

#### Handling cycles in deterministic succession

For deterministic succession (no stochastic elements):
- Cycles are detected deterministically
- The cycle list contains all scenarios in the periodic sequence
- The first scenario in the cycle is typically used as the representative attractor

#### Handling cycles in stochastic succession

When dynamic shocks or judgement uncertainty are present:
- Cycles may be detected differently across runs due to stochasticity
- The `tie_break` parameter in `DynamicCIB.simulate_path()` controls cycle handling:
  - `"deterministic_first"`: Always selects the first scenario in the cycle
  - `"random"`: Randomly selects a scenario from the cycle using the random number generator
- This ensures reproducibility while allowing stochastic variation

#### Cycle detection in branching pathways

In `BranchingPathwayBuilder`, cycles are handled by selecting the first scenario in the cycle as the representative node. This prevents infinite loops while preserving the cycle information in the succession path.

#### Practical implications

- **Fixed points vs cycles**: Stable system states are represented by fixed points; periodic dynamics are represented by cycles.
- **Interpretation**: System instability or alternating regimes may be indicated by cycles.
- **Stability**: Sensitivity to perturbations may be greater for systems with cycles than for fixed-point attractors.

### Impact scores and balances

For a scenario \(z = [z_1,\dots,z_N]\), the impact score of target descriptor \(j\) being in state \(l\) is:

\[
\theta_{j,l}(z) = \sum_{i \ne j} C_{i\to j}(z_i, l)
\]

where \(C_{i\to j}(k,l)\) is the impact of source descriptor \(i\) being in state \(k\) on target descriptor \(j\) being in state \(l\).

The impact balance for descriptor \(j\) is the vector \((\theta_{j,1},\dots,\theta_{j,s_j})\).

### Consistency principle

A scenario is consistent if, for every descriptor \(j\), the chosen state \(z_j\) achieves the maximum (ties allowed):

\[
\theta_{j,z_j}(z) \ge \theta_{j,l}(z)\quad \forall l
\]

### Practical uncertainty model (confidence-coded)

Each judgement cell is modelled as a bounded “noise around the workshop mean”:

\[
C_{i\to j}(k,l) \sim \mathrm{Normal}(\mu_{i,j,k,l}, \sigma^2_{i,j,k,l})\ \text{clipped to }[-3,+3]
\]

with \(\mu\) from the point judgement table and \(\sigma\) from a confidence code \(c\in\{1,\dots,5\}\).

Default mapping (must match `cib.example_data.CONFIDENCE_TO_SIGMA`):

- \(c=5\): \(\sigma=0.2\)
- \(c=4\): \(\sigma=0.5\)
- \(c=3\): \(\sigma=0.8\)
- \(c=2\): \(\sigma=1.2\)
- \(c=1\): \(\sigma=1.5\)

### Probabilistic consistency

For a fixed candidate scenario \(z\), estimate:

\[
P(\mathrm{consistent}\mid z) \approx \frac{1}{M}\sum_{m=1}^M \mathbf{1}\{z\ \text{is consistent in sampled }C^{(m)}\}
\]

### Consistency margin (margin-to-switching)

In this implementation, a “consistency margin” is defined as follows.

For a descriptor \(j\) with chosen state \(z_j\), the best alternative state is:
\[
l^* = \arg\max_{l\ne z_j}\theta_{j,l}(z)
\]

The per-descriptor margin-to-switching is:
\[
m_j(z) = \theta_{j,z_j}(z) - \theta_{j,l^*}(z)
\]

The scenario-level consistency margin is:
\[
m(z) = \min_j m_j(z)
\]

Positive values indicate that each chosen state is strictly preferred over its best alternative. A value of 0 indicates a tie at the maximum for at least one descriptor. Negative values indicate inconsistency.

### Near-miss diagnostics (operational)

A “near miss” is operationally defined as a scenario with a small margin-to-switching:
\[
m(z) \le \varepsilon
\]

In practice, \(\varepsilon\) is selected as a small threshold (for example, 0.25) and the near-miss rate is reported as the fraction of scenarios in a set (for example, final-period scenarios across a Monte Carlo ensemble) that satisfy the condition.

### Global sensitivity (association, not causality)

Ensemble-level sensitivity reporting may be performed by treating selected features (for example, initial states, cyclic realised states, or rule-activation indicators) as “drivers” and selected summaries (for example, final states or transition counts) as “outcomes”.

Association-based sensitivity metrics are reported. Causal interpretation is not implied by these summaries, and the reported relationships are to be interpreted as conditional on the configured modelling assumptions (CIM, uncertainty model, shocks, cyclic transitions, and threshold rules).

### Structural shocks

The CIM is perturbed by structural shock stress testing:

\[
C \leftarrow C + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, \Sigma)
\]

The default is independent shocks (diagonal \(\Sigma\)); correlated shocks are supported via an explicit correlation matrix.

### Optional fat tails / rare events

For “rare event” realism, disturbances can be sampled from heavy-tailed or jump-mixture distributions:

- **Student-t shocks** (fat tails): \(\varepsilon \sim t_{\nu}\) scaled to match a target variance.
- **Normal + jumps**: \(\varepsilon = \xi + J\), where \(\xi\sim\mathcal{N}(0,\sigma^2)\) and \(J\) is an independent jump term that is applied with probability \(p\).

In the current implementation, correlated structural shocks remain Gaussian by construction; non-Gaussian structural shocks are supported for independent (per-cell) shocks.

### Dynamic shocks (AR(1))

Dynamic shocks are applied at the impact-balance level during within-period succession:

\[
\theta'_{j,l}(t) = \theta_{j,l}(t) + \eta_{j,l}(t)
\]

\[
\eta_{j,l}(t) = \rho\,\eta_{j,l}(t-1) + u_{j,l}(t)
\]

The innovation \(u_{j,l}(t)\) is mean-zero with long-run scale \(\tau\). Common choices are:

- Gaussian innovations: \(u_{j,l}(t)\sim \mathcal{N}(0,(1-\rho^2)\tau^2)\)
- Student-t innovations (fat tails): \(u_{j,l}(t)\sim t_{\nu}\) scaled to match \((1-\rho^2)\tau^2\)
- Jump innovations: with probability \(p\), a jump term is added to the innovation (rare events)

### Branching pathways (hybrid enumeration + sampling)

When transformation pathways are constructed across sub-periods, two complementary modes are useful:

- **Enumeration (enumeration-based branching)**: if the scenario space is small, all consistent scenarios for the active CIM in a sub-period are enumerated and treated as the scenario set for that period.
  - Uses a deterministic base matrix for transitions (no judgement-uncertainty sampling, no structural shocks).
  - Transition probabilities are exact (uniform over enumerated consistent scenarios, per parent).

- **Sampling (Monte Carlo)**: if the scenario space is large, the reachable scenario set and transition probabilities are approximated by repeated random restarts, uncertainty/shock sampling, and succession.
  - Respects judgement uncertainty (`judgment_sigma_scale_by_period`) and structural shocks (`structural_sigma`) when configured.
  - Transition probabilities are estimated from counts and converge with more samples.

Dynamic-shock note for branching:

- In `BranchingPathwayBuilder`, dynamic shocks are sampled independently per transition layer. AR(1) persistence across periods is not represented in the branching graph, and `dynamic_rho` is not used to induce cross-period momentum.

Important:

- Enumeration is used when scenario space size is <= `max_states_to_enumerate`; sampling is used otherwise.
- If `structural_sigma` and/or `judgment_sigma_scale_by_period` are set but enumeration mode is selected, those parameters are ignored by design. When uncertainty is required to be applied, `max_states_to_enumerate` is to be decreased so that sampling mode is forced.

The hybrid “enumerate-or-sample” approach uses enumeration when feasible and otherwise sampling, yielding a branching pathway graph with weighted edges between consecutive periods.

### Transformation matrices

Transformation matrices identify which perturbations cause transitions between scenarios. For each scenario pair (i, j), different perturbation types (structural shocks, dynamic shocks, judgement uncertainty) are tested to determine which perturbations cause transitions from scenario i to scenario j. A transformation is considered successful if the attractor reached from scenario i equals scenario j or is within Hamming distance 1.

Notes:

- For dynamic shocks, a single AR(1) shock field \(\eta\) is sampled at the (descriptor, candidate_state) level using the same generator as the dynamic simulation layer. A single pseudo-period is used because the transformation-matrix test is a static (one-step) perturbation experiment rather than a multi-period time series.
- If succession yields a cycle, the transformation condition is evaluated against all states in the cycle (not only a representative element).

The `TransformationMatrixBuilder` class builds transformation matrices by testing perturbations across scenario pairs.

### Expert aggregation (weighted)

For a cell with expert means \(\mu_i\), standard deviations \(\sigma_i\), and weights \(w_i\) (renormalised over experts that provided the cell):

\[
\mu = \sum_i w_i \mu_i
\]

\[
\sigma^2 = \sum_i w_i^2\sigma_i^2 + \sum_i w_i(\mu_i-\mu)^2
\]


## Scaling workflows (solver modes and benchmarks)

This section is provided as a practical guide to the scaling-related solver entrypoints and the benchmark fixtures used for regression testing.

### Solver entrypoints (large systems)

Two scaling-oriented solver entrypoints are exposed on `ScenarioAnalyzer`:

- `ScenarioAnalyzer.find_attractors_monte_carlo(...)`:
  - An approximate method, where initial scenarios are sampled and succession is run to a fixed point or a cycle.
  - Estimated attractor weights are returned as Monte Carlo frequencies.
- `ScenarioAnalyzer.find_all_consistent_exact(...)`:
  - An exact method, where a depth-first partial-assignment search is performed with an optional sound pruning bound.
  - The complete consistent-scenario set is returned when the search completes.

Configuration dataclasses are defined in `cib.solvers.config` (`MonteCarloAttractorConfig`, `ExactSolverConfig`).

An optional sparse scoring backend may be selected for Monte Carlo succession workflows by setting `MonteCarloAttractorConfig(fast_backend="sparse")`.
This backend is provided in `cib.sparse_scoring` and is intended for cases where the impact structure is sparse and memory pressure from dense tensors is not desired.

An exact enumeration sparse rewrite is not provided. If exact enumeration is required at sizes where dense `delta_max` precomputation is the limiting factor, the exact solver would need to be rewritten with a different pruning bound that is compatible with sparse structures.

### Feasibility constraints (domain rules)

In addition to the CIB consistency principle, feasibility constraints may be applied when domain rules are required to be respected.
These rules are treated as external to CIB consistency and are applied as admissibility conditions on scenarios.

Constraint specifications are defined in `cib.constraints` and include:

- `ForbiddenPair`: a descriptor-state pair combination is forbidden.
- `Implies`: an implication is enforced (if A is a given state, B is required to be a given state).
- `AllowedStates`: a descriptor is restricted to an allowed subset of state labels.

Application points:

- In exact enumeration (`ScenarioAnalyzer.find_all_consistent_exact(...)`), constraints may be provided via `ExactSolverConfig(constraints=[...])`.
  Violations are detected during the depth-first search and branches are pruned accordingly.
- In deterministic analysis helper workflows (`ScenarioAnalyzer.find_all_consistent(...)`), constraints may be provided via the `constraints=` argument.
  Candidates that violate constraints are excluded.

### Model reduction (high-cardinality descriptors)

When descriptor cardinalities are high, model reduction is typically required before solver execution.
State binning utilities are provided in `cib.reduction` (`reduce_matrix`, `bin_states`, `map_scenario_to_reduced`).

The usage of these utilities is demonstrated in `examples/example_state_binning.py`.

### Benchmarks and test execution

Benchmark fixtures are defined in `cib.benchmark_data`. They are intended to be exercised through tests so that solver wiring and correctness parity are checked in CI.

- Smoke benchmark tests:

```bash
python3 -m pytest -q tests/test_benchmarks_smoke.py
```

- Full test suite:

```bash
python3 -m pytest -q
```

Performance targets are not asserted in CI. Local performance measurement is expected to be performed on representative workloads.

Multi-period example outputs are aligned to a shared period grid, `DEFAULT_PERIODS`, defined in `cib.example_data`. This is used by the example suite so that plots and per-period summaries can be compared directly.

An end-to-end example of the scaling solver entrypoints is provided in `examples/example_solver_modes.py`.

For a workshop-scale dataset (10 descriptors × 3 states), `DATASET_C10` is provided in `cib.example_data` and is exercised in `examples/example_solver_modes_c10.py`.

A dynamic Monte Carlo ensemble plot for `DATASET_C10` is generated by `examples/example_dynamic_cib_c10.py` and is written to `results/example_dynamic_c10_plot_1.png`.

When full enumeration is feasible, `examples/example_enumeration_c10.py` enumerates the complete `DATASET_C10` configuration space and filters the full consistent-scenario set, with a parity cross-check against the exact pruned solver. A diagnostic plot is written to `results/example_enumeration_c10_plot_1.png`.

Exact attractor basin weights for `DATASET_C10` are computed in `examples/example_attractor_basin_validation_c10.py` by running succession from all initial scenarios, and are compared to Monte Carlo basin estimates. A comparison plot is written to `results/example_attractor_basin_validation_c10_plot_1.png`.

For a heavier illustration of rare events and regime switching, `DATASET_C15` is provided in `cib.example_data` and is exercised in `examples/example_dynamic_cib_c15_rare_events.py`. The default run is kept small; a larger run can be requested via the `CIB_C15_RUNS` environment variable.

## Output files and result interpretation

When examples are run or the package is imported, several output files are generated in the `results/` directory.

## Monte Carlo ensembles versus branching pathway graphs

Both approaches are probabilistic and can use the same uncertainty and shock settings. The difference is the output representation:

- **Monte Carlo ensemble (individual-path plots, probability bands, fan chart)**:
  - Many full trajectories \(z_{2025}, z_{2030}, \dots\) are run and summarised.
  - Best when time-series uncertainty (state probabilities, quantiles, expected value) is desired and when the state space is too large to map explicitly.
  - A distribution over outcomes is produced without explicitly constructing a scenario graph.

- **Branching pathway graph (scenario graph with edge weights)**:
  - A per-period graph is built where nodes are unique scenarios (typically attractors) and edges carry transition probabilities.
  - Transition probabilities are exact when enumerating and estimated when sampling.
  - Best when a readable set of distinct pathway archetypes, top-k most likely paths, and a compact scenario map for communication are desired.
  - In sampling mode, the result is an approximation of transition frequencies. With pruning (`top_paths`, `max_nodes_per_period`), rare branches are intentionally dropped for readability.

### Should outcomes match

- In principle, convergence is expected when both methods represent the same stochastic model and sufficient samples are run. Under these conditions, convergence between branching-derived per-period summaries and Monte Carlo ensemble summaries is expected.
- In practice, they can differ due to finite sampling noise, pruning and caps, and different sampling design. Full trajectories are sampled by Monte Carlo; local transition probabilities per node are estimated by branching and propagated forward.

### Monte Carlo run counts (testing and practice)

The number of Monte Carlo runs required depends on the method, the system size, and the intended use:

**Monte Carlo ensemble (full pathway simulation):**
- **Rapid overview and testing:** \(M = 200\)–\(250\) runs can be acceptable if Monte Carlo error bars are reported. Approximate trends are provided and major pathways are identified.
- **Stable bands for reporting:** \(M \ge 2{,}000\) runs are preferred (and increased further for larger \(N\), more states, or many periods). Sampling noise in probability bands and quantile estimates is reduced, especially for rare events and tail probabilities.
- **Computational trade-off:** a complete trajectory across all periods is simulated by each run, so total cost scales linearly with \(M\) and the number of periods.

**Hybrid branching (transition sampling):**
- **Per-transition samples:** how many Monte Carlo draws are used to estimate transition probabilities from each parent node to its children is controlled by the `n_transition_samples` parameter. Default is 200; the example uses 120.
- **Total computational cost:** depends on the number of nodes per period and the number of periods. If a period has \(K\) nodes and each requires `n_transition_samples` draws, the cost for that transition layer is \(K \times\) `n_transition_samples`.
- **Testing vs production:** for rapid testing, `n_transition_samples = 50`–\(120\) may suffice to identify major branches. For stable transition probability estimates (especially for rare but important transitions), 200–500 samples per transition are preferred.
- **Enumeration mode:** when the scenario space is small enough to enumerate (below `max_states_to_enumerate`), transition probabilities are exact and no sampling is needed.

**Node pruning (`max_nodes_per_period`):**
- **Purpose:** node explosion is limited by pruning low-probability scenarios at each period, keeping the branching graph readable and computationally manageable. Without pruning, the number of nodes can grow exponentially across periods.
- **How it works:** when the number of nodes in a period exceeds `max_nodes_per_period`, the top-\(K\) nodes ranked by incoming probability mass (sum of transition weights from all parent nodes) are kept by the builder. Transition probabilities are renormalised after pruning, and the graph remains connected (if a parent's outgoing edges are all pruned, it falls back to the most likely kept node).
- **Testing:** `max_nodes_per_period = 20`–\(40\) is set for rapid exploration and visualisation. Enough diversity is provided to see major pathway archetypes while keeping plots readable.
- **Production:** for comprehensive analysis, `max_nodes_per_period = 50`–\(100\) or `None` (no pruning) is considered. Rare but potentially important scenarios are preserved by higher values, at the cost of increased computation and visual complexity. The choice depends on whether completeness (capturing all significant pathways) or readability (focussing on high-probability paths) is prioritised.
- **Trade-off:** rare branches are intentionally dropped by pruning for readability, which can cause differences between branching-derived summaries and full Monte Carlo ensembles. If rare events need to be captured, either `max_nodes_per_period` is increased or Monte Carlo ensembles are used without explicit graph construction.

**Comparison:**
- Fewer total runs (200–2,000) are required by Monte Carlo ensembles but each run is a full trajectory. Best for estimating time-series marginals and overall distributional properties.
- More total samples (summed across all transitions) are required by hybrid branching but it can be more efficient when the branching factor is low or when enumeration is feasible. Best for explicit pathway structures and when distinct scenario archetypes need to be identified.
- Both methods benefit from more samples for stable estimates, but the computational geometry differs: Monte Carlo is trajectory-based; branching is node-based with local transition estimation.

### Example (branching pathway tree)

```python
import matplotlib.pyplot as plt

from cib import BranchingPathwayBuilder, DynamicVisualizer

builder = BranchingPathwayBuilder(
    base_matrix=matrix,
    periods=periods,
    initial=DATASET_B5_INITIAL_SCENARIO,
    cyclic_descriptors=dataset_b5_cyclic_descriptors(),
    threshold_rules=[dataset_b5_threshold_rule_fast_permitting()],
    max_states_to_enumerate=2000,
    n_transition_samples=120,
    max_nodes_per_period=40,
    base_seed=123,
    structural_sigma=0.15,
    judgment_sigma_scale_by_period=sigma_by_period,
    dynamic_tau=0.26,
    dynamic_rho=0.6,
    dynamic_innovation_dist="student_t",
    dynamic_innovation_df=5.0,
    dynamic_jump_prob=0.02,
    dynamic_jump_scale=0.70,
)

branching = builder.build(top_k=10)

plt.figure(figsize=(12, 4))
DynamicVisualizer.plot_pathway_tree(
    periods=branching.periods,
    scenarios_by_period=branching.scenarios_by_period,
    edges=branching.edges,
    top_paths=branching.top_paths,
    key_descriptors=["Policy_Stringency", "Permitting_Speed", "Electrification_Demand"],
    title="Branching pathway tree (top paths only)",
    min_edge_weight=0.03,
)
plt.tight_layout()
plt.show()
```

### Automatically generated files (on import)

- **`results/dataset_b5_cim.txt`**: Cross-impact matrix (CIM) in standard CIB text format.
  - All impact relationships between descriptors are contained.
  - Impact values with confidence codes (1–5) are shown for each relationship.
  - Format: Source[State] → Target[State] = Impact(Confidence).
  - Useful for understanding the complete impact structure and verifying workshop inputs.

- **`results/scenario_scoring_output.txt`**: Initial scenario diagnostics.
  - Consistency status (True/False).
  - Consistency margin (difference between the chosen state and the best alternative state).
  - Total impact score (sum of chosen-state scores).
  - Brink descriptors (descriptors near switching thresholds).
  - Impact balances for all descriptors that show scores for each state.
  - Useful for identifying unstable descriptors and diagnosing inconsistencies.

### Example output files (when running examples)

- **`results/example_dynamic_cib_notebook_plot_*.png`**: Visualisations that are generated from `examples/dynamic_cib.ipynb`.
  - Probability bands and numeric summaries for selected descriptors.
  - Branching pathway tree visualisations.
  - Scenario network plots, where enabled.

### Interpreting results

**CIM file (`dataset_b5_cim.txt`)**:
- Impact values range from -3 (strongly hindering) to +3 (strongly promoting).
- Confidence codes: 5 = very high (σ=0.2), 4 = high (σ=0.5), 3 = medium (σ=0.8), 2 = low (σ=1.2), 1 = very low (σ=1.5).
- Higher epistemic uncertainty in workshop judgements is indicated by lower confidence.

**Scenario scoring (`scenario_scoring_output.txt`)**:
- Consistency: True means all descriptors are in their optimal states given the scenario.
- Consistency margin: inconsistency is indicated by negative values; exact consistency is indicated by 0.
- Brink descriptors: descriptors where a different state could be flipped to by small changes.
- Impact balances: stronger support for that state is indicated by higher scores.

**Dynamic visualisations**:
- Probability bands: higher uncertainty about state outcomes is indicated by wider bands.
- Quantile bands: the 90% credible interval (5th to 95th percentile) of numeric outcomes is shown.
- Individual-path plots: variability is shown by individual paths; stable pathways are indicated by clustering.
- Branching trees: more likely transitions are indicated by thicker edges; alternative futures are shown by multiple branches.

**Scenario network plots (final-period Monte Carlo summary)**:
- What nodes represent: each node is a distinct final-period scenario (a complete categorical state assignment across all descriptors) that is selected from the Monte Carlo ensemble.
- Node labels: `S11` is an index within the plotted subset. If a second line is shown (for example, `2`), it is the Monte Carlo count for that scenario.
- Node size: node size is proportional to scenario frequency when node weights are provided. This is a distribution visualisation; there is no well-defined mean scenario for categorical state vectors.
- What edges represent: similarity between scenarios is represented by edges, not time evolution. With `edge_metric="hamming"`, scenarios are connected when they differ in a small number of descriptor-state choices, and edge weights reflect that distance.
- Why isolated nodes occur: a node can be isolated if no sufficiently similar neighbours exist within the plotted subset, or if connections are pruned by edge limits for readability.
