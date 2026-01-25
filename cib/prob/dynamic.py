from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from cib.prob.model import JointDistribution, ProbabilisticCIAModel


@dataclass
class DynamicProbabilisticCIA:
    """
    Dynamic wrapper for per-period probabilistic CIA models.

    In Phase-1 implementation, "refit" mode is supported: each period is fitted independently.
    """

    periods: Sequence[int]
    models_by_period: Mapping[int, ProbabilisticCIAModel]

    def fit_distributions(
        self,
        *,
        mode: str = "refit",
        **fit_opts: object,
    ) -> Dict[int, JointDistribution]:
        mode = str(mode).strip().lower()
        if mode != "refit":
            raise NotImplementedError(
                "Only mode='refit' is implemented in Phase 1. "
                "predict-update is planned next."
            )
        out: Dict[int, JointDistribution] = {}
        for t in self.periods:
            if int(t) not in self.models_by_period:
                raise ValueError(f"Missing model for period {int(t)}")
            out[int(t)] = self.models_by_period[int(t)].fit_joint(**fit_opts)
        return out

