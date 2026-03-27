"""
Unit tests for iterative probabilistic RNG seed semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from cib.prob import fit_iterative as fit_iterative_mod
from cib.prob.model import ProbabilisticCIAModel
from cib.prob.types import FactorSpec


def _small_iterative_model() -> ProbabilisticCIAModel:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}
    multipliers = {
        (("A", "a1"), ("B", "b1")): 1.4,
        (("A", "a0"), ("B", "b1")): 0.7,
        (("A", "a1"), ("B", "b0")): 0.8,
        (("A", "a0"), ("B", "b0")): 1.1,
    }
    return ProbabilisticCIAModel(
        factors=factors,
        marginals=marginals,
        multipliers=multipliers,
    )


def _signature(dist) -> tuple[tuple[int, ...], tuple[float, ...]]:
    idx = tuple(int(x) for x in np.asarray(dist.support_indices))
    probs = tuple(float(x) for x in np.round(np.asarray(dist.support_probabilities), 6))
    return idx, probs


def test_iterative_seed_none_forwards_nondeterministic_rng_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`random_seed=None` should be forwarded as None to default_rng."""
    calls: list[object] = []
    original_default_rng = np.random.default_rng

    def _spy_default_rng(seed=None):  # type: ignore[no-untyped-def]
        calls.append(seed)
        # Return a deterministic generator so this test is stable.
        return original_default_rng(0)

    monkeypatch.setattr(fit_iterative_mod.np.random, "default_rng", _spy_default_rng)

    model = _small_iterative_model()
    model.fit_joint(
        method="iterative",
        random_seed=None,
        iterative_burn_in_sweeps=20,
        iterative_n_samples=120,
        iterative_thinning=1,
    )

    assert calls
    assert calls[0] is None


def test_iterative_seed_none_produces_variability_across_runs() -> None:
    """None seed should produce non-identical outputs across repeated runs."""
    model = _small_iterative_model()
    signatures = []
    for _ in range(6):
        signatures.append(
            _signature(
                model.fit_joint(
                    method="iterative",
                    random_seed=None,
                    iterative_burn_in_sweeps=200,
                    iterative_n_samples=1200,
                    iterative_thinning=1,
                )
            )
        )
    assert len(set(signatures)) > 1


def test_iterative_explicit_seed_remains_reproducible() -> None:
    """Explicit integer seed should preserve deterministic reproducibility."""
    model = _small_iterative_model()
    a = _signature(
        model.fit_joint(
            method="iterative",
            random_seed=123,
            iterative_burn_in_sweeps=200,
            iterative_n_samples=1200,
            iterative_thinning=1,
        )
    )
    b = _signature(
        model.fit_joint(
            method="iterative",
            random_seed=123,
            iterative_burn_in_sweeps=200,
            iterative_n_samples=1200,
            iterative_thinning=1,
        )
    )
    assert a == b
