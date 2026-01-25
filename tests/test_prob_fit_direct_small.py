import numpy as np

from cib.prob.model import ProbabilisticCIAModel
from cib.prob.types import FactorSpec


def test_fit_direct_two_by_two_recovers_conditionals() -> None:
    factors = [
        FactorSpec("A", ["a0", "a1"]),
        FactorSpec("B", ["b0", "b1"]),
    ]

    # Fixed marginals.
    marginals = {
        "A": {"a0": 0.6, "a1": 0.4},
        "B": {"b0": 0.7, "b1": 0.3},
    }

    # Conditionals are chosen consistent with P(A=a1)=0.4:
    # 0.7 * P(a1|b0) + 0.3 * P(a1|b1) = 0.4.
    p_a1_b1 = 0.6
    p_a1_b0 = (0.4 - 0.3 * p_a1_b1) / 0.7  # 0.314285714...

    # Convert to multipliers m = P(A=a|B=b)/P(A=a) for both outcomes a0,a1.
    multipliers = {
        (("A", "a1"), ("B", "b1")): p_a1_b1 / marginals["A"]["a1"],  # 1.5
        (("A", "a0"), ("B", "b1")): (1.0 - p_a1_b1) / marginals["A"]["a0"],
        (("A", "a1"), ("B", "b0")): p_a1_b0 / marginals["A"]["a1"],
        (("A", "a0"), ("B", "b0")): (1.0 - p_a1_b0) / marginals["A"]["a0"],
    }

    model = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers=multipliers)
    dist = model.fit_joint(method="direct", kl_weight=1e-8, solver_maxiter=3000)

    assert np.isclose(float(np.sum(dist.p)), 1.0, atol=1e-9)
    assert float(np.min(dist.p)) >= -1e-12

    implied_A = dist.marginal("A")
    implied_B = dist.marginal("B")
    assert abs(implied_A["a0"] - 0.6) < 1e-6
    assert abs(implied_A["a1"] - 0.4) < 1e-6
    assert abs(implied_B["b0"] - 0.7) < 1e-6
    assert abs(implied_B["b1"] - 0.3) < 1e-6

    # Check conditional recovery.
    implied = dist.conditional(("A", "a1"), ("B", "b1"))
    assert abs(float(implied) - float(p_a1_b1)) < 5e-4

