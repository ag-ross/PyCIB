from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from cib.prob.model import JointDistribution, ProbabilisticCIAModel


@dataclass
class DynamicProbabilisticCIA:
    """
    Dynamic wrapper for per-period probabilistic CIA models.

    Two dynamic modes are supported:

    - mode="refit": each period is fitted independently.
    - mode="predict-update": the fitted distribution from period t-1 is used as a KL baseline
      for period t (regularised predict–update).
    """

    periods: Sequence[int]
    models_by_period: Mapping[int, ProbabilisticCIAModel]

    @staticmethod
    def _index_signature(dist: JointDistribution) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (str(factor.name), tuple(str(outcome) for outcome in factor.outcomes))
            for factor in dist.index.factors
        )

    def fit_distributions(
        self,
        *,
        mode: str = "refit",
        **fit_opts: object,
    ) -> Dict[int, JointDistribution]:
        mode = str(mode).strip().lower()
        if mode not in {"refit", "predict-update", "predict_update"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        if mode in {"predict-update", "predict_update"}:
            method = str(fit_opts.get("method", "direct")).strip().lower()
            if method != "direct":
                raise ValueError(
                    "Predict–update mode is supported only for method='direct' "
                    "(a dense JointDistribution is required for the KL baseline)."
                )
        out: Dict[int, JointDistribution] = {}
        prev: Optional[JointDistribution] = None
        for t in self.periods:
            if int(t) not in self.models_by_period:
                raise ValueError(f"Missing model for period {int(t)}")
            model = self.models_by_period[int(t)]
            if mode == "refit":
                out[int(t)] = model.fit_joint(**fit_opts)
                prev = out[int(t)]
                continue

            # Predict–update: previous period distribution is used as KL baseline.
            if prev is None:
                out[int(t)] = model.fit_joint(**fit_opts)
                prev = out[int(t)]
                continue

            current_signature = tuple(
                (str(factor.name), tuple(str(outcome) for outcome in factor.outcomes))
                for factor in model.index.factors
            )
            previous_signature = self._index_signature(prev)
            if current_signature != previous_signature:
                raise ValueError(
                    "Predict-update KL baseline is incompatible with the current period "
                    "factor space (factor names/order/outcomes differ)"
                )

            opts = dict(fit_opts)
            opts["kl_baseline"] = prev.p
            # A small epsilon is used by default to avoid zeros in the baseline distribution.
            opts.setdefault("kl_baseline_eps", 1e-12)
            out[int(t)] = model.fit_joint(**opts)
            prev = out[int(t)]
        return out

