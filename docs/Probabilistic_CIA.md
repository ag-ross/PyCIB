# Probabilistic CIA (Joint-Distribution) — Quickstart

This document describes the joint-distribution probabilistic cross-impact analysis extension under `cib.prob`.

It is intentionally separate from the existing “probabilistic” outputs in `cib`, which are empirical frequencies from Monte Carlo / branching simulations of the CIB dynamics model.

## Status: experimental

`cib.prob` is currently experimental:

- the implementation is currently focused on small scenario spaces (dense joint fitting) and refit-style dynamics
- some planned features (graph/relevance scaling, predict–update dynamics, risk bounds) are not yet implemented

## Concepts (minimal)

- **Factors** \(X_1,\dots,X_n\) are discrete variables with named outcomes.
- A **scenario** is a complete assignment \(x=(x_1,\dots,x_n)\).
- You provide:
  - **marginals** \(\hat P(X_i=a)\)
  - **cross-impact multipliers** (probability ratios)
    \[
    m_{(i=a)\leftarrow(j=b)}=\frac{P(X_i=a\mid X_j=b)}{P(X_i=a)}
    \]
- The solver fits a coherent joint distribution \(P(x)\) that matches the marginals exactly and best matches the multiplier constraints (via stable pairwise-moment targets).

## Quickstart example

Run:

```bash
cd pycib
python3 examples/example_probabilistic_cia_static.py
```

The example:

- reuses the canonical Dataset B5 descriptor/state labels from `cib.example_data`
- generates a synthetic joint-distribution probabilistic CIA joint distribution over a small subset of those labels
- computes marginals + multipliers from that joint (so the constraint set is coherent)
- fits a dense joint distribution
- prints top scenarios and implied conditionals

## Practical notes

- **Feasibility matters**: multipliers are not arbitrary. If multipliers imply pairwise probabilities that violate Fréchet bounds given the marginals, `cib.prob` currently raises a `ValueError` (fail-fast).
- **Small scenario spaces only (for now)**: the implemented `direct` fitter enumerates all scenarios and is intended for \(\prod_i |X_i|\) up to the low tens-of-thousands (configurable).
- **KL regularization**: `kl_weight>0` adds a minimum-information tie-breaker toward the independent baseline implied by the marginals (helps when constraints are sparse).

## Dynamic (per-period refit) example

This Phase-1 dynamic mode fits each period independently.

```bash
cd pycib
python3 examples/example_probabilistic_cia_dynamic_refit.py
```

