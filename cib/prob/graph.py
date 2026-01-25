from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set


@dataclass(frozen=True)
class RelevanceSpec:
    """
    Minimal DAG parent specification: parents[child] = {parent1, parent2, ...}.

    This is a placeholder for Phase 1; dense joint fitting is currently used for small scenario spaces.
    """

    parents: Mapping[str, Set[str]]


def topological_order(nodes: Sequence[str], parents: Mapping[str, Set[str]]) -> List[str]:
    """
    Kahn topological sort.
    """
    nodes = [str(n) for n in nodes]
    parent_map: Dict[str, Set[str]] = {str(n): set(parents.get(str(n), set())) for n in nodes}

    out: List[str] = []
    ready = [n for n in nodes if not parent_map.get(n)]
    ready.sort()
    while ready:
        n = ready.pop(0)
        out.append(n)
        for m in nodes:
            if n in parent_map.get(m, set()):
                parent_map[m].remove(n)
                if not parent_map[m] and m not in out and m not in ready:
                    ready.append(m)
                    ready.sort()

    if len(out) != len(nodes):
        raise ValueError("Graph has at least one cycle")
    return out

