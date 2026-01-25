from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union


@dataclass(frozen=True)
class FactorSpec:
    """
    One discrete factor with named outcomes.

    In `cib` terminology this is analogous to a descriptor with categorical
    states, but we keep separate naming to avoid semantic confusion.
    """

    name: str
    outcomes: Tuple[str, ...]

    def __init__(self, name: str, outcomes: Sequence[str]) -> None:
        name = str(name).strip()
        if not name:
            raise ValueError("Factor name cannot be empty")
        outs = tuple(str(o) for o in outcomes)
        if not outs:
            raise ValueError(f"Factor {name!r} must have at least one outcome")
        if len(set(outs)) != len(outs):
            raise ValueError(f"Factor {name!r} has duplicate outcomes")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "outcomes", outs)


@dataclass(frozen=True)
class ProbScenario:
    """
    Immutable complete assignment of outcomes to factors.

    `assignment` is ordered to match `factor_names`.
    """

    factor_names: Tuple[str, ...]
    assignment: Tuple[str, ...]

    def __init__(self, *, factor_names: Sequence[str], assignment: Sequence[str]) -> None:
        fn = tuple(str(x) for x in factor_names)
        if not fn:
            raise ValueError("factor_names cannot be empty")
        if len(set(fn)) != len(fn):
            raise ValueError("factor_names must be unique")
        a = tuple(str(x) for x in assignment)
        if len(a) != len(fn):
            raise ValueError("assignment length must match factor_names length")
        object.__setattr__(self, "factor_names", fn)
        object.__setattr__(self, "assignment", a)

    def to_dict(self) -> Dict[str, str]:
        return {k: v for k, v in zip(self.factor_names, self.assignment)}


AssignmentLike = Union[
    ProbScenario,
    Mapping[str, str],
    Sequence[str],
]


class ScenarioIndex:
    """
    Dense scenario indexing for small factor spaces.

    Provides:
      - enumeration of all scenarios
      - mapping assignment <-> integer index
    """

    def __init__(self, factors: Sequence[FactorSpec]) -> None:
        if not factors:
            raise ValueError("factors cannot be empty")
        self.factors: Tuple[FactorSpec, ...] = tuple(factors)
        names = [f.name for f in self.factors]
        if len(set(names)) != len(names):
            raise ValueError("Factor names must be unique")
        self.factor_names: Tuple[str, ...] = tuple(names)
        self._outcomes_by_factor: Dict[str, Tuple[str, ...]] = {
            f.name: f.outcomes for f in self.factors
        }
        self._radixes: Tuple[int, ...] = tuple(len(f.outcomes) for f in self.factors)
        self.size: int = int(reduce(mul, self._radixes, 1))

        # Multipliers are precomputed for mixed-radix conversion.
        mults: List[int] = []
        running = 1
        for r in reversed(self._radixes[1:]):
            running *= int(r)
            mults.append(running)
        # For N factors, multipliers are aligned with factor order.
        # Example: radixes [2,3,2] yield multipliers [6,2,1].
        mults = list(reversed(mults))
        mults.append(1)
        self._multipliers: Tuple[int, ...] = tuple(mults)

    def outcome_index(self, factor: str, outcome: str) -> int:
        outs = self._outcomes_by_factor.get(str(factor))
        if outs is None:
            raise ValueError(f"Unknown factor: {factor!r}")
        try:
            return int(outs.index(str(outcome)))
        except ValueError as exc:
            raise ValueError(f"Unknown outcome {outcome!r} for factor {factor!r}") from exc

    def _assignment_tuple(self, assignment: AssignmentLike) -> Tuple[str, ...]:
        if isinstance(assignment, ProbScenario):
            if assignment.factor_names != self.factor_names:
                raise ValueError("ProbScenario factor_names do not match ScenarioIndex")
            return tuple(assignment.assignment)
        if isinstance(assignment, Mapping):
            return tuple(str(assignment[name]) for name in self.factor_names)
        return tuple(str(x) for x in assignment)

    def index_of(self, assignment: AssignmentLike) -> int:
        a = self._assignment_tuple(assignment)
        if len(a) != len(self.factors):
            raise ValueError("assignment has wrong length")
        idx = 0
        for factor_name, outcome, mult in zip(self.factor_names, a, self._multipliers):
            oidx = self.outcome_index(factor_name, outcome)
            idx += int(oidx) * int(mult)
        return int(idx)

    def scenario_at(self, idx: int) -> ProbScenario:
        idx = int(idx)
        if idx < 0 or idx >= self.size:
            raise ValueError("idx out of range")
        rem = idx
        outs: List[str] = []
        for factor, mult, radix in zip(self.factors, self._multipliers, self._radixes):
            digit = rem // int(mult)
            rem = rem % int(mult)
            if digit < 0 or digit >= int(radix):
                raise RuntimeError("internal mixed-radix conversion error")
            outs.append(factor.outcomes[int(digit)])
        return ProbScenario(factor_names=self.factor_names, assignment=tuple(outs))

    def enumerate_all(self) -> List[ProbScenario]:
        return [self.scenario_at(i) for i in range(self.size)]

