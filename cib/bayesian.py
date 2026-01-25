"""
Practical Bayesian-style extensions for CIB.

This module intentionally keeps uncertainty handling lightweight 
rather than full Bayesian updating.

It provides:
  - GaussianCIBMatrix: per-cell (mu, sigma) uncertainty without discrete confidence codes.
  - ExpertAggregator: weighted aggregation of multiple experts with partial coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix
from cib.example_data import clip_impact, sigma_from_confidence

ImpactKey = Tuple[str, str, str, str]


class GaussianCIBMatrix(CIBMatrix):
    """
    CIB matrix with per-cell Gaussian uncertainty parameters.

    Each impact cell stores:
      - mean μ (impact point estimate),
      - standard deviation σ (epistemic uncertainty around μ).

    Sampling uses Normal(μ, σ) per cell with clipping to [-3, +3] by default.
    """

    def __init__(
        self,
        descriptors: Dict[str, List[str]],
        default_sigma: float = 0.8,
        clip_lo: float = -3.0,
        clip_hi: float = +3.0,
    ) -> None:
        super().__init__(descriptors)
        if default_sigma < 0:
            raise ValueError("default_sigma must be non-negative")
        self._sigma: Dict[ImpactKey, float] = {}
        self.default_sigma = float(default_sigma)
        self.clip_lo = float(clip_lo)
        self.clip_hi = float(clip_hi)

    def set_impact(
        self,
        src_desc: str,
        src_state: str,
        tgt_desc: str,
        tgt_state: str,
        value: float,
        sigma: Optional[float] = None,
    ) -> None:
        """
        Set impact mean and optional sigma.

        Args:
            src_desc: Source descriptor name.
            src_state: Source state label.
            tgt_desc: Target descriptor name.
            tgt_state: Target state label.
            value: Mean impact value.
            sigma: Standard deviation for this cell. If None, uses default_sigma.
        """
        super().set_impact(src_desc, src_state, tgt_desc, tgt_state, value)
        if sigma is None:
            sigma = self.default_sigma
        sigma = float(sigma)
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self._sigma[(src_desc, src_state, tgt_desc, tgt_state)] = sigma

    def set_impacts(
        self,
        impacts: Mapping[ImpactKey, float],
        sigmas: Optional[Mapping[ImpactKey, float]] = None,
        default_sigma: Optional[float] = None,
    ) -> None:
        """
        Set impacts in bulk with optional per-cell sigmas.
        """
        if default_sigma is None:
            default_sigma = self.default_sigma
        for key, value in impacts.items():
            sigma = (
                float(sigmas[key]) if sigmas is not None and key in sigmas else default_sigma
            )
            self.set_impact(key[0], key[1], key[2], key[3], float(value), sigma=sigma)

    def get_sigma(self, key: ImpactKey) -> float:
        """Get sigma for an impact cell key."""
        return float(self._sigma.get(key, self.default_sigma))

    def sample_matrix(self, seed: int) -> CIBMatrix:
        """
        Sample a noisy CIM using per-cell Normal(μ, σ) with clipping.
        """
        rng = np.random.default_rng(int(seed))
        impacts_dict: Dict[ImpactKey, float] = {}

        for src_desc in self.descriptors:
            for src_state in self.descriptors[src_desc]:
                for tgt_desc in self.descriptors:
                    if src_desc == tgt_desc:
                        continue
                    for tgt_state in self.descriptors[tgt_desc]:
                        key = (src_desc, src_state, tgt_desc, tgt_state)
                        mu = self.get_impact(src_desc, src_state, tgt_desc, tgt_state)
                        sigma = self.get_sigma(key)
                        impacts_dict[key] = float(
                            np.clip(
                                rng.normal(loc=float(mu), scale=float(sigma)),
                                self.clip_lo,
                                self.clip_hi,
                            )
                        )

        sampled = CIBMatrix(self.descriptors)
        sampled.set_impacts(impacts_dict)
        return sampled


@dataclass(frozen=True)
class ExpertJudgments:
    """
    One expert's judgments for a (possibly partial) set of impact cells.
    """

    impacts: Mapping[ImpactKey, float]
    sigmas: Mapping[ImpactKey, float]
    weight: float


class ExpertAggregator:
    """
    Aggregate multiple experts' impact judgments with weights and uncertainty.

    Supports partial coverage: if an expert does not provide a cell, they are
    excluded from that cell's aggregation and weights are renormalized.

    Aggregation per cell:
      - μ = Σ w_i μ_i
      - σ² = Σ w_i² σ_i² + Σ w_i (μ_i - μ)²
    """

    def __init__(self, descriptors: Dict[str, List[str]]) -> None:
        self.descriptors = descriptors.copy()
        self._experts: List[ExpertJudgments] = []

    def add_expert(
        self,
        impacts: Mapping[ImpactKey, float],
        *,
        weight: float = 1.0,
        confidence: Optional[Mapping[ImpactKey, int]] = None,
        default_confidence: int = 3,
        sigmas: Optional[Mapping[ImpactKey, float]] = None,
        default_sigma: Optional[float] = None,
    ) -> None:
        """
        Add an expert's judgments.

        You can specify uncertainty via either:
          - confidence codes (1..5), or
          - explicit per-cell sigmas.

        Args:
            impacts: Mapping of impact keys to means.
            weight: Non-negative weight for this expert (relative; renormalized per cell).
            confidence: Optional per-cell confidence codes.
            default_confidence: Confidence code used when confidence is omitted for a provided key.
            sigmas: Optional per-cell sigmas.
            default_sigma: Sigma used when neither confidence nor sigmas provide a value.
        """
        weight = float(weight)
        if weight < 0:
            raise ValueError("weight must be non-negative")
        if not (1 <= int(default_confidence) <= 5):
            raise ValueError("default_confidence must be in [1, 5]")

        if default_sigma is None:
            default_sigma = sigma_from_confidence(int(default_confidence))
        default_sigma = float(default_sigma)
        if default_sigma < 0:
            raise ValueError("default_sigma must be non-negative")

        sigma_map: Dict[ImpactKey, float] = {}
        for key in impacts.keys():
            if sigmas is not None and key in sigmas:
                sigma_map[key] = float(sigmas[key])
            elif confidence is not None and key in confidence:
                sigma_map[key] = float(sigma_from_confidence(int(confidence[key])))
            elif confidence is not None:
                sigma_map[key] = float(sigma_from_confidence(int(default_confidence)))
            else:
                sigma_map[key] = default_sigma

        self._experts.append(
            ExpertJudgments(impacts=impacts, sigmas=sigma_map, weight=weight)
        )

    def aggregate(self) -> GaussianCIBMatrix:
        """
        Aggregate to a GaussianCIBMatrix with per-cell (μ, σ).
        """
        impacts, sigmas = self.aggregate_impacts()
        matrix = GaussianCIBMatrix(self.descriptors)
        matrix.set_impacts(impacts, sigmas=sigmas)
        return matrix

    def aggregate_impacts(self) -> Tuple[Dict[ImpactKey, float], Dict[ImpactKey, float]]:
        """
        Aggregate and return (mean_impacts, sigma_impacts) dictionaries.
        """
        if not self._experts:
            raise ValueError("No experts added")

        keys: set[ImpactKey] = set()
        for e in self._experts:
            keys.update(e.impacts.keys())

        mean_out: Dict[ImpactKey, float] = {}
        sigma_out: Dict[ImpactKey, float] = {}

        for key in keys:
            rows: List[Tuple[float, float, float]] = []
            for e in self._experts:
                if key in e.impacts:
                    mu_i = float(e.impacts[key])
                    sigma_i = float(e.sigmas[key])
                    w_i = float(e.weight)
                    rows.append((mu_i, sigma_i, w_i))

            if not rows:
                continue

            sum_w = sum(w for _, _, w in rows)
            if sum_w <= 0:
                raise ValueError("At least one expert weight must be positive")
            weights = [w / sum_w for _, _, w in rows]

            mu = sum(w * mu_i for (mu_i, _, _), w in zip(rows, weights))
            mu = clip_impact(mu)

            var_within = sum((w**2) * (sigma_i**2) for (_, sigma_i, _), w in zip(rows, weights))
            var_between = sum(w * ((mu_i - mu) ** 2) for (mu_i, _, _), w in zip(rows, weights))
            sigma = sqrt(max(0.0, var_within + var_between))

            mean_out[key] = float(mu)
            sigma_out[key] = float(sigma)

        return mean_out, sigma_out

