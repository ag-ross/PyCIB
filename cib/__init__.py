"""
Core CIB analysis modules.

This package provides the fundamental components for Cross-Impact Balance
analysis, including matrix structures, scenario representation, consistency
checking, and succession operators. It also provides disequilibrium extensions
(transient, regime-aware, and path-dependent dynamics), regime definitions and
transition rules, path-dependent state and adaptive matrix machinery, transition
kernels, structural consistency diagnostics, and extended pathway types and
summaries.
"""

from cib.analysis import FindAllConsistentResult, MonteCarloAnalyzer, ScenarioAnalyzer
from cib.bayesian import ExpertAggregator, GaussianCIBMatrix
from cib.cyclic import CyclicDescriptor
from cib.core import (
    CIBMatrix,
    ConsistencyChecker,
    ImpactBalance,
    Scenario,
)
from cib.dynamic import DynamicCIB
from cib.branching import BranchRegimeState, BranchingPathwayBuilder, BranchingResult
from cib.scoring import (
    AttractorDiagnostics,
    ScenarioDiagnostics,
    attractor_distance,
    consistent_set_distance,
    cumulative_disequilibrium_burden,
    descriptor_disequilibrium_contributions,
    equilibrium_distance,
    impact_label,
    judgment_section_labels,
    scenario_diagnostics,
)
from cib.pathway import (
    ActiveMatrixState,
    ExtendedTransformationPathway,
    MemoryState,
    PathDependentState,
    PerPeriodDisequilibriumMetrics,
    StructuralConsistencyState,
    TransitionEvent,
    TransformationPathway,
    branching_regime_residence_timelines,
    numeric_quantile_timelines,
    pathway_frequencies,
    state_probability_timelines,
    summarize_disequilibrium_ensemble,
    summarize_disequilibrium_path,
)
from cib.regimes import CallableRegimeTransitionRule, RegimeSpec, RegimeTransitionRule
from cib.path_dependence import (
    AdaptiveCIMUpdater,
    CallableAdaptiveCIMUpdater,
    HysteresisRule,
    IrreversibilityRule,
)
from cib.transition_kernel import (
    CallableTransitionKernel,
    DefaultTransitionKernel,
    TransitionReplayRecord,
    TransitionReplayResult,
    TransitionKernel,
)
from cib.structural_consistency import check_structural_consistency
from cib.shocks import (
    RobustnessMetrics,
    RobustnessTester,
    ShockModel,
    calibrate_structural_sigma_from_confidence,
    suggest_dynamic_tau_bounds,
)
from cib.succession import (
    AttractorFinder,
    ConstrainedGlobalSuccession,
    GlobalSuccession,
    LocalSuccession,
    SuccessionOperator,
)
from cib.uncertainty import ConfidenceMapper, UncertainCIBMatrix
from cib.reduction import reduce_matrix, bin_states, map_scenario_to_reduced
from cib.utils import (
    load_from_csv,
    load_from_json,
    save_to_csv,
    save_to_json,
)
from cib.threshold import ThresholdRule
from cib.transformation_matrix import (
    PerturbationInfo,
    TransformationMatrix,
    TransformationMatrixBuilder,
    explain_regime_transformation,
    summarize_path_to_path_transformations,
)
from cib.attribution import (
    Contribution,
    DescriptorAttribution,
    FlipCandidate,
    ScenarioAttribution,
    attribute_scenario,
    flip_candidates_for_descriptor,
)
from cib.rare_events import (
    BinomialInterval,
    EventRateDiagnostics,
    event_rate_diagnostics,
    min_switch_margin,
    near_miss_rate,
    wilson_interval_from_count,
)
from cib.sensitivity import (
    DriverSpec,
    GlobalSensitivityReport,
    ImportanceSummary,
    OutcomeSensitivity,
    OutcomeSpec,
    compute_global_sensitivity_attractors,
    compute_global_sensitivity_dynamic,
)

_OPTIONAL_EXPORTS = {
    "MatrixVisualizer": (
        "cib.visualization",
        "MatrixVisualizer",
        "Optional visualization dependencies are missing. Install with: pip install pycib[viz]",
    ),
    "ScenarioVisualizer": (
        "cib.visualization",
        "ScenarioVisualizer",
        "Optional visualization dependencies are missing. Install with: pip install pycib[viz]",
    ),
    "UncertaintyVisualizer": (
        "cib.visualization",
        "UncertaintyVisualizer",
        "Optional visualization dependencies are missing. Install with: pip install pycib[viz]",
    ),
    "ShockVisualizer": (
        "cib.visualization",
        "ShockVisualizer",
        "Optional visualization dependencies are missing. Install with: pip install pycib[viz]",
    ),
    "DynamicVisualizer": (
        "cib.visualization",
        "DynamicVisualizer",
        "Optional visualization dependencies are missing. Install with: pip install pycib[viz]",
    ),
    "NetworkGraphBuilder": (
        "cib.network_analysis",
        "NetworkGraphBuilder",
        "Optional network dependencies are missing. Install with: pip install pycib[network]",
    ),
    "NetworkAnalyzer": (
        "cib.network_analysis",
        "NetworkAnalyzer",
        "Optional network dependencies are missing. Install with: pip install pycib[network]",
    ),
    "ImpactPathwayAnalyzer": (
        "cib.network_analysis",
        "ImpactPathwayAnalyzer",
        "Optional network dependencies are missing. Install with: pip install pycib[network]",
    ),
}


def __getattr__(name: str):
    import importlib

    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(f"module 'cib' has no attribute {name!r}")
    module_name, attr_name, install_hint = _OPTIONAL_EXPORTS[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(install_hint) from exc
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "CIBMatrix",
    "GaussianCIBMatrix",
    "Scenario",
    "ConsistencyChecker",
    "ImpactBalance",
    "SuccessionOperator",
    "GlobalSuccession",
    "ConstrainedGlobalSuccession",
    "LocalSuccession",
    "AttractorFinder",
    "ScenarioAnalyzer",
    "FindAllConsistentResult",
    "DynamicCIB",
    "BranchingPathwayBuilder",
    "BranchingResult",
    "BranchRegimeState",
    "ScenarioDiagnostics",
    "AttractorDiagnostics",
    "scenario_diagnostics",
    "equilibrium_distance",
    "attractor_distance",
    "consistent_set_distance",
    "descriptor_disequilibrium_contributions",
    "cumulative_disequilibrium_burden",
    "impact_label",
    "judgment_section_labels",
    "TransformationPathway",
    "ExtendedTransformationPathway",
    "PerPeriodDisequilibriumMetrics",
    "ActiveMatrixState",
    "TransitionEvent",
    "MemoryState",
    "PathDependentState",
    "StructuralConsistencyState",
    "pathway_frequencies",
    "state_probability_timelines",
    "branching_regime_residence_timelines",
    "numeric_quantile_timelines",
    "summarize_disequilibrium_path",
    "summarize_disequilibrium_ensemble",
    "ThresholdRule",
    "CyclicDescriptor",
    "RegimeSpec",
    "RegimeTransitionRule",
    "CallableRegimeTransitionRule",
    "UncertainCIBMatrix",
    "ConfidenceMapper",
    "MonteCarloAnalyzer",
    "ExpertAggregator",
    "ShockModel",
    "RobustnessTester",
    "RobustnessMetrics",
    "calibrate_structural_sigma_from_confidence",
    "suggest_dynamic_tau_bounds",
    "load_from_csv",
    "save_to_csv",
    "load_from_json",
    "save_to_json",
    "TransformationMatrix",
    "TransformationMatrixBuilder",
    "PerturbationInfo",
    "summarize_path_to_path_transformations",
    "explain_regime_transformation",
    "reduce_matrix",
    "bin_states",
    "map_scenario_to_reduced",
    "TransitionKernel",
    "DefaultTransitionKernel",
    "CallableTransitionKernel",
    "TransitionReplayRecord",
    "TransitionReplayResult",
    "AdaptiveCIMUpdater",
    "CallableAdaptiveCIMUpdater",
    "HysteresisRule",
    "IrreversibilityRule",
    "check_structural_consistency",
    "Contribution",
    "DescriptorAttribution",
    "FlipCandidate",
    "ScenarioAttribution",
    "attribute_scenario",
    "flip_candidates_for_descriptor",
    "BinomialInterval",
    "EventRateDiagnostics",
    "wilson_interval_from_count",
    "event_rate_diagnostics",
    "min_switch_margin",
    "near_miss_rate",
    "DriverSpec",
    "OutcomeSpec",
    "ImportanceSummary",
    "OutcomeSensitivity",
    "GlobalSensitivityReport",
    "compute_global_sensitivity_dynamic",
    "compute_global_sensitivity_attractors",
]
