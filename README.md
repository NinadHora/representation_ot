# ot-faces

Optimal transport for fairness auditing of facial embeddings.

Uses entropy-regularized Wasserstein distances (Sinkhorn algorithm) to measure whether national and linguistic categories form geometrically coherent structures in face embedding space. Applied to 190 subjects from [Casual Conversations v2](https://ai.meta.com/datasets/casual-conversations-v2-dataset/) (Meta).

## Results

The embedding model does not classify nationality. It classifies phenotype and acquisition conditions, and this functions as a proxy for nationality only when the country is phenotypically homogeneous. When it is not (Brazil), the category dissolves.

| Country   |  n  |  ρ    |  p      | Accuracy |
|-----------|-----|-------|---------|----------|
| Indonesia |  25 | 0.767 | < 0.001 | 92.0%    |
| India     | 110 | 0.869 | < 0.001 | 75.5%    |
| Mexico    |  14 | 0.890 | 0.004   | 42.9%    |
| Brazil    |  41 | 0.963 | 0.009   | 34.1%    |

DP gap: **57.9pp**. Conditional DP gap (controlling for skin tone): **61.1pp** — phenotype does not mediate the disparity.

## Pipeline

```
Phase 1: run_ot_ccv2.py        Exploratory (sampled distances, prototype classification)
Phase 2: permutation_test.py   Full distance matrix (17,955 pairs), permutation tests, bootstrap CI
Phase 3: cross_cutting.py      Four-quadrant analysis (country x Monk Scale), Brazil deep dive
Phase 4: fairness_metrics.py   DP, conditional DP, counterfactual decomposition
         visualize.py          Plots from CSV outputs
```

All phases after Phase 2 reuse the pre-computed distance matrix — no Sinkhorn recomputation.

## Structure

```
sinkhorn.py            Sinkhorn algorithm, Wasserstein distance, barycenter (from scratch, NumPy only)
data.py                Shared data loading (CCv2 embeddings + annotations)
run_ot_ccv2.py         Phase 1
permutation_test.py    Phase 2
cross_cutting.py       Phase 3
fairness_metrics.py    Phase 4
visualize.py           Plot generation
jobs/                  SLURM batch scripts for RECOD.AI cluster
```

## Requirements

NumPy, matplotlib. No OT library dependencies — Sinkhorn is implemented from scratch.

Ran on the RECOD.AI cluster at UNICAMP (NVIDIA Quadro P5000). Full distance matrix: 8 seconds. Permutation tests (1000×): ~3 minutes.

## Data

CCv2 embeddings (512-d, from `ccv2-audit-kit`) and annotations are not included. Paths are configured for the RECOD cluster in `data.py`.

## Reference

> Nina da Hora (2026). "Measuring Categorical Coherence in Facial Embedding Space via Entropy-Regularized Optimal Transport." Instituto da Hora / IC-UNICAMP / DPCT-UNICAMP. Verão IMPA 2026.
