"""
Uncertainty modeling for CIB analysis.

This module provides classes for handling confidence-coded impacts and
sampling uncertain CIB matrices for Monte Carlo analysis.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from cib.core import CIBMatrix

from cib.example_data import sample_uncertain_cim, sigma_from_confidence


class ConfidenceMapper:
    """
    Maps confidence codes to uncertainty parameters.

    Provides conversion between expert confidence levels (1-5) and
    standard deviation values for uncertainty modeling.
    """

    @staticmethod
    def confidence_to_sigma(confidence: int) -> float:
        """
        Map confidence code (1-5) to standard deviation.

        Args:
            confidence: Confidence level from 1 (low) to 5 (high).

        Returns:
            Standard deviation value for uncertainty modeling.

        Raises:
            ValueError: If confidence is outside valid range [1, 5].
        """
        return sigma_from_confidence(confidence)

    @staticmethod
    def sigma_from_confidence(confidence: int) -> float:
        """
        Alias for confidence_to_sigma.

        Args:
            confidence: Confidence level from 1 (low) to 5 (high).

        Returns:
            Standard deviation value for uncertainty modeling.
        """
        return ConfidenceMapper.confidence_to_sigma(confidence)


class UncertainCIBMatrix(CIBMatrix):
    """
    CIB matrix with confidence-coded uncertainty.

    Extends CIBMatrix to include confidence codes for each impact value,
    enabling Monte Carlo sampling of uncertain impact matrices.
    """

    def __init__(
        self,
        descriptors: Dict[str, list[str]],
        default_confidence: int = 3,
    ) -> None:
        """
        Initialize uncertain CIB matrix.

        Args:
            descriptors: Dictionary mapping descriptor names to state lists.
            default_confidence: Default confidence level (1-5) for impacts
                not explicitly assigned. Defaults to 3 (medium confidence).

        Raises:
            ValueError: If descriptors are invalid or default_confidence
                is outside range [1, 5].
        """
        super().__init__(descriptors)
        self._confidence: Dict[Tuple[str, str, str, str], int] = {}
        self.default_confidence = default_confidence

        if not (1 <= default_confidence <= 5):
            raise ValueError(
                f"Default confidence must be in range [1, 5], "
                f"got {default_confidence}"
            )

    def set_impact(
        self,
        src_desc: str,
        src_state: str,
        tgt_desc: str,
        tgt_state: str,
        value: float,
        confidence: Optional[int] = None,
    ) -> None:
        """
        Set impact value with confidence code.

        Args:
            src_desc: Source descriptor name.
            src_state: Source state label.
            tgt_desc: Target descriptor name.
            tgt_state: Target state label.
            value: Impact value.
            confidence: Confidence level (1-5). If None, uses this matrix's
                default_confidence.

        Raises:
            ValueError: If confidence is outside range [1, 5] or if
                descriptor/state names are invalid.
        """
        super().set_impact(src_desc, src_state, tgt_desc, tgt_state, value)

        if confidence is None:
            confidence = self.default_confidence

        if not (1 <= confidence <= 5):
            raise ValueError(
                f"Confidence must be in range [1, 5], got {confidence}"
            )

        key = (src_desc, src_state, tgt_desc, tgt_state)
        self._confidence[key] = confidence

    def set_impacts(
        self,
        impacts: Dict[Tuple[str, str, str, str], float],
        confidence: Optional[Dict[Tuple[str, str, str, str], int]] = None,
        default_confidence: Optional[int] = None,
    ) -> None:
        """
        Set multiple impacts with optional confidence codes.

        Args:
            impacts: Dictionary mapping (src_desc, src_state, tgt_desc,
                tgt_state) to impact values.
            confidence: Optional dictionary mapping same keys to confidence
                codes. If None, uses default_confidence for all.
            default_confidence: Default confidence for impacts not in
                confidence dict. If None, uses instance default_confidence.

        Raises:
            ValueError: If confidence values are invalid.
        """
        if default_confidence is None:
            default_confidence = self.default_confidence

        for key, value in impacts.items():
            src_desc, src_state, tgt_desc, tgt_state = key
            conf = (
                confidence.get(key, default_confidence)
                if confidence is not None
                else default_confidence
            )
            self.set_impact(src_desc, src_state, tgt_desc, tgt_state, value, conf)

    def get_confidence(
        self, src_desc: str, src_state: str, tgt_desc: str, tgt_state: str
    ) -> int:
        """
        Get confidence code for an impact.

        Args:
            src_desc: Source descriptor name.
            src_state: Source state label.
            tgt_desc: Target descriptor name.
            tgt_state: Target state label.

        Returns:
            Confidence code (1-5), or default_confidence if not set.

        Raises:
            ValueError: If descriptor or state names are invalid.
        """
        if src_desc not in self.descriptors:
            raise ValueError(f"Source descriptor '{src_desc}' not found")
        if tgt_desc not in self.descriptors:
            raise ValueError(f"Target descriptor '{tgt_desc}' not found")
        if src_state not in self.descriptors[src_desc]:
            raise ValueError(
                f"Source state '{src_state}' not found for "
                f"descriptor '{src_desc}'"
            )
        if tgt_state not in self.descriptors[tgt_desc]:
            raise ValueError(
                f"Target state '{tgt_state}' not found for "
                f"descriptor '{tgt_desc}'"
            )

        key = (src_desc, src_state, tgt_desc, tgt_state)
        return self._confidence.get(key, self.default_confidence)

    def sample_matrix(self, seed: int, sigma_scale: float = 1.0) -> CIBMatrix:
        """
        Sample a noisy CIM from the uncertainty model.

        Uses confidence codes to sample impact values from truncated normal
        distributions around the point estimates.

        Args:
            seed: Random seed for reproducibility.
            sigma_scale: Multiplicative scale applied to confidence-derived sigmas.
                This is useful for time-increasing uncertainty assumptions.

        Returns:
            A new CIBMatrix with sampled impact values.
        """
        impacts_dict: Dict[Tuple[str, str, str, str], float] = {}
        confidence_dict: Dict[Tuple[str, str, str, str], int] = {}

        for src_desc in self.descriptors:
            for src_state in self.descriptors[src_desc]:
                for tgt_desc in self.descriptors:
                    if src_desc == tgt_desc:
                        continue
                    for tgt_state in self.descriptors[tgt_desc]:
                        key = (src_desc, src_state, tgt_desc, tgt_state)
                        impacts_dict[key] = self.get_impact(
                            src_desc, src_state, tgt_desc, tgt_state
                        )
                        confidence_dict[key] = self.get_confidence(
                            src_desc, src_state, tgt_desc, tgt_state
                        )

        sampled_impacts = sample_uncertain_cim(
            impacts=impacts_dict,
            confidence=confidence_dict,
            seed=seed,
            sigma_scale=float(sigma_scale),
        )

        sampled_matrix = CIBMatrix(self.descriptors)
        sampled_matrix.set_impacts(sampled_impacts)

        return sampled_matrix
