from cib.prob.dynamic import DynamicProbabilisticCIA
from cib.prob.model import ProbabilisticCIAModel
from cib.prob.types import FactorSpec


def test_dynamic_refit_returns_distribution_per_period() -> None:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]

    m1 = {"A": {"a0": 0.5, "a1": 0.5}, "B": {"b0": 0.6, "b1": 0.4}}
    m2 = {"A": {"a0": 0.7, "a1": 0.3}, "B": {"b0": 0.2, "b1": 0.8}}

    model1 = ProbabilisticCIAModel(factors=factors, marginals=m1, multipliers={})
    model2 = ProbabilisticCIAModel(factors=factors, marginals=m2, multipliers={})

    dyn = DynamicProbabilisticCIA(periods=[2025, 2030], models_by_period={2025: model1, 2030: model2})
    dists = dyn.fit_distributions(mode="refit", method="direct", kl_weight=0.0)

    assert set(dists.keys()) == {2025, 2030}
    assert abs(sum(dists[2025].p) - 1.0) < 1e-9
    assert abs(sum(dists[2030].p) - 1.0) < 1e-9

