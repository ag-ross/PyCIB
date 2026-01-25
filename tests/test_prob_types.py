import pytest

from cib.prob.types import FactorSpec, ProbScenario, ScenarioIndex


def test_scenario_index_roundtrip() -> None:
    factors = [FactorSpec("A", ["Low", "High"]), FactorSpec("B", ["No", "Yes"])]
    idx = ScenarioIndex(factors)
    assert idx.size == 4

    s0 = idx.scenario_at(0)
    assert isinstance(s0, ProbScenario)
    assert s0.to_dict() == {"A": "Low", "B": "No"}

    i = idx.index_of({"A": "High", "B": "Yes"})
    assert i == 3
    s = idx.scenario_at(i)
    assert s.to_dict() == {"A": "High", "B": "Yes"}


def test_factor_validation() -> None:
    with pytest.raises(ValueError):
        FactorSpec("", ["a"])
    with pytest.raises(ValueError):
        FactorSpec("X", [])
    with pytest.raises(ValueError):
        FactorSpec("X", ["a", "a"])

