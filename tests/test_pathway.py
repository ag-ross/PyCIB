from __future__ import annotations

import numpy as np
import pytest

from cib.pathway import _weighted_quantiles


def test_weighted_quantiles_rejects_negative_weights() -> None:
    values = np.array([0.0, 1.0, 2.0], dtype=float)
    weights = np.array([0.5, -0.1, 0.6], dtype=float)

    with pytest.raises(ValueError, match="weights must be non-negative"):
        _ = _weighted_quantiles(values, weights, (0.1, 0.5, 0.9))


def test_weighted_quantiles_rejects_non_finite_weights() -> None:
    values = np.array([0.0, 1.0, 2.0], dtype=float)
    weights = np.array([0.5, np.nan, 0.5], dtype=float)

    with pytest.raises(ValueError, match="weights must be finite"):
        _ = _weighted_quantiles(values, weights, (0.1, 0.5, 0.9))
