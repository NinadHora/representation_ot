# Case study 2: score repair via Wasserstein barycenters (tabular data)

Implementation of fairness post-processing by optimal transport: the score
distribution of each sensitive group is transported to the Wasserstein
barycenter of the group distributions, enforcing demographic parity with
minimal information loss (Gordaliza et al., 2019; Chzhen et al., 2020).

## Contents

- `sinkhorn_fairness.py`: JAX pipeline. Computes entropic Wasserstein
  barycenters via Sinkhorn iterations in the log domain and extracts monotone
  transport maps through barycentric projection followed by isotonic
  regression. Supports intersectional sensitive attributes via marginal masks.
- `fairness_fairface_ot.ipynb`: end-to-end notebook. Trains a LightGBM base
  model, applies the repair maps to test scores, and benchmarks against the
  sequential Wasserstein post-processing of EquiPy.

## Requirements

```
pip install "jax[cpu]>=0.4.20" pot lightgbm equipy pandas scikit-learn
```

The lower bound on the JAX version is required: earlier versions do not accept
`jit(static_argnames=...)` as a decorator, which breaks the module at import
time.

## Role within the repository

In dimension one the optimal transport map is unique, monotone, and given in
closed form by the composition of a cumulative distribution function with a
quantile function. This case study exercises that regime. The limitations of
the one-dimensional construction, once the objects of interest become
high-dimensional embedding distributions, motivate the research questions of
the main README.
