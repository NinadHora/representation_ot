# Case study 1: geometric coherence of demographic categories (CCv2)

Pipeline for testing whether demographic categories (country, language, skin
tone) form geometrically coherent structures in facial embedding space, using
entropy-regularized Wasserstein distances.

## Execution order

1. `phase1_exploratory.py`: sampled pairwise distances, intra/inter variance
   ratios, prototype-based classification by country and language.
2. `phase2_coherence.py`: full pairwise distance matrix (17,955 pairs),
   permutation tests of the null hypothesis rho = 1, bootstrap confidence
   intervals. Saves the matrix to disk.
3. `phase3_cross_cutting.py`: four-quadrant analysis (country x Monk Scale)
   over the pre-computed matrix; deep dive into the Brazilian subsample.
4. `phase4_downstream_fairness.py`: demographic parity, conditional DP, and
   counterfactual decomposition, connecting geometric incoherence to standard
   group fairness metrics.
5. `visualize.py`: plots from the CSV outputs of the previous phases.

Phases 3 and 4 reuse the distance matrix computed in Phase 2; no Sinkhorn
recomputation is performed.

## Data

Paths in `data.py` point to the RECOD.AI cluster layout:

- embeddings: `ccv2-audit-kit/outputs/embeddings/all_embeddings.npz`
- annotations: `casual_conversations_v2/annotations/CasualConversationsV2.json`

Adjust `EMBEDDINGS_PATH` and `ANNOTATIONS_PATH` for other environments. The
dataset is distributed by Meta under a data use agreement and is not included
in this repository.

## Execution

From the repository root:

```
python case_studies/facial_embeddings/phase1_exploratory.py
python case_studies/facial_embeddings/phase2_coherence.py
python case_studies/facial_embeddings/phase3_cross_cutting.py
python case_studies/facial_embeddings/phase4_downstream_fairness.py
python case_studies/facial_embeddings/visualize.py
```
