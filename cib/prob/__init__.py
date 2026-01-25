"""
Joint-distribution probabilistic cross-impact analysis (CIA).

This subpackage is intentionally separate from the existing `cib` probabilistic
outputs (which are empirical Monte Carlo / branching frequencies from the CIB
simulation model). Here, "probability" means an explicit joint distribution
over factor outcomes constrained by marginals and probabilistic cross-impact
multipliers.
"""

from cib.prob.types import FactorSpec, ProbScenario, ScenarioIndex
from cib.prob.model import ProbabilisticCIAModel, JointDistribution
from cib.prob.diagnostics import DiagnosticsReport
from cib.prob.dynamic import DynamicProbabilisticCIA

__all__ = [
    "FactorSpec",
    "ProbScenario",
    "ScenarioIndex",
    "ProbabilisticCIAModel",
    "JointDistribution",
    "DiagnosticsReport",
    "DynamicProbabilisticCIA",
]

