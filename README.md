# Representation OT

## Optimal Transport as a Language for the Geometry of Computational Representation

**Research question:** who gets to be computationally represented, and can representational erasure be detected geometrically, in the structure of the data itself, before any classifier is trained?

This repository investigates the geometry of representation spaces through the theory of Optimal Transport (OT). Rather than treating OT as a device for correcting fairness violations, we employ it as a mathematical language for comparing distributions of representations: for measuring when two populations occupy geometrically distinct regions of an embedding space, when a demographic category forms a geometrically coherent structure, and when such a category dissolves under closer analysis.

Fairness is therefore not the premise of this work. It is one application of the framework, addressed as a downstream consequence of representational geometry.

## Research questions

1. When do two populations possess geometrically different representations in an embedding space?
2. How can displacement between representation spaces be measured?
3. Can Optimal Transport detect erasure prior to classification, that is, directly in the geometry of the data?
4. How can distributions of representations produced by different models be compared?
5. How can information loss along a computer vision pipeline be quantified?

None of these questions depends on a particular definition of fairness. They are mathematical questions about representation, subsequently applicable to facial recognition, medical imaging, multimodal models, and representation learning in general.

## Repository structure

```
representation-ot/
    core/                                Mathematical core, independent of any application
        sinkhorn.py                      Sinkhorn algorithm, entropy-regularized Wasserstein
                                         distance, and Wasserstein barycenters, implemented
                                         from scratch in NumPy

    case_studies/
        facial_embeddings/               Case study 1: Casual Conversations v2
            data.py                      Data loading (CCv2 embeddings and annotations)
            phase1_exploratory.py        Sampled pairwise distances, intra/inter ratios,
                                         prototype-based classification
            phase2_coherence.py          Full pairwise distance matrix (17,955 pairs),
                                         permutation tests, bootstrap confidence intervals
            phase3_cross_cutting.py      Four-quadrant analysis (country x Monk Scale),
                                         Brazil deep dive
            phase4_downstream_fairness.py  Demographic parity, conditional DP,
                                           counterfactual decomposition
            visualize.py                 Plot generation from CSV outputs

        tabular_repair/                  Case study 2: score repair via Wasserstein barycenters
            sinkhorn_fairness.py         JAX pipeline: entropic barycenters, monotone Monge
                                         maps via barycentric projection and isotonic regression
            fairness_fairface_ot.ipynb   End-to-end notebook

    demos/                               Synthetic examples, runnable without dataset access (planned)
    tests/                               Validation of core/ against the POT library (planned)
    requirements.txt
```

## Case study 1: geometric coherence of demographic categories

Face embedding models map faces into a 512-dimensional space. Demographic annotations (country, skin tone) partition the subjects, but the central question is whether the geometry of the embedding space sustains those partitions.

Applied to 190 subjects from [Casual Conversations v2](https://ai.meta.com/datasets/casual-conversations-v2-dataset/) (Meta), the pipeline computes entropy-regularized Wasserstein distances between subject-level embedding distributions and tests, via permutation, whether national and linguistic categories form geometrically coherent structures.

### Results

The embedding model does not classify nationality. It classifies phenotype and acquisition conditions, and this functions as a proxy for nationality only when the country is phenotypically homogeneous. When it is not (Brazil), the category dissolves.

| Country   |  n  |  rho  |  p      | Accuracy |
|-----------|-----|-------|---------|----------|
| Indonesia |  25 | 0.767 | < 0.001 | 92.0%    |
| India     | 110 | 0.869 | < 0.001 | 75.5%    |
| Mexico    |  14 | 0.890 | 0.004   | 42.9%    |
| Brazil    |  41 | 0.963 | 0.009   | 34.1%    |

Demographic parity gap: 57.9pp. Conditional DP gap, controlling for skin tone: 61.1pp, indicating that phenotype does not mediate the disparity.

In the terms of this repository, the dissolution of the Brazilian category is an instance of representational erasure detected in the geometry of the embeddings, prior to the training of any classifier. This reframes algorithmic auditing: instead of asking whether the output of a model is fair, one asks whether the representation space sustains the category at all.

### Pipeline

All phases after Phase 2 reuse the pre-computed distance matrix, with no Sinkhorn recomputation. Full distance matrix: 8 seconds. Permutation tests (1000 iterations): approximately 3 minutes (RECOD.AI cluster, NVIDIA Quadro P5000).

To run a phase from the repository root:

```
python case_studies/facial_embeddings/phase2_coherence.py
```

## Case study 2: score repair via Wasserstein barycenters

The classical fairness-via-OT construction (Gordaliza et al., 2019; Chzhen et al., 2020) post-processes the scores of a trained model by transporting the score distribution of each group to the Wasserstein barycenter, achieving demographic parity with minimal information loss. The implementation, in JAX, computes entropic barycenters via Sinkhorn iterations and extracts monotone Monge maps through barycentric projection followed by isotonic regression, benchmarked against [EquiPy](https://github.com/EquiPy/EquiPy).

Within the framework of this repository, the case study makes explicit the contrast between dimension one, where the optimal transport map admits a closed form given by composition of quantile functions, and high dimension, where no ordering, no closed form, and no unique tractable map exist. The open research questions of this project live precisely in that gap.

## Mathematical core

`core/sinkhorn.py` implements, from scratch and with NumPy as its only dependency:

- entropy-regularized optimal transport (Sinkhorn iterations);
- the regularized Wasserstein distance;
- Wasserstein barycenters (Agueh and Carlier, 2011, in the entropic version).

The core carries no dependence on any dataset or fairness definition.

## Roadmap

- Synthetic demonstration in `demos/`, reproducible without access to CCv2: a Gaussian mixture with a deliberately incoherent category that the pipeline must detect.
- Tests validating `core/` against the [POT](https://pythonot.github.io/) library.
- Multi-architecture comparison (FaceNet, ArcFace, VGGFace2): does geometric incoherence persist across embedding models? (research question 4)
- Transport beyond dimension one: sliced Wasserstein and neural OT maps directly in the 512-dimensional embedding space (research question 2).

## Requirements

Case study 1 requires only NumPy and matplotlib; the Sinkhorn algorithm is implemented from scratch, with no OT library dependencies. Case study 2 additionally requires JAX, POT, LightGBM, and EquiPy. See `requirements.txt`.

## Data

CCv2 embeddings (512-dimensional, produced by `ccv2-audit-kit`) and annotations are not included, in accordance with Meta's data use agreement. Paths are configured for the RECOD.AI cluster in `case_studies/facial_embeddings/data.py`. The planned `demos/` folder will allow the full pipeline to be exercised on synthetic data without dataset access.

## Reference

Nina da Hora (2026). "Measuring Categorical Coherence in Facial Embedding Space via Entropy-Regularized Optimal Transport." Instituto da Hora / IC-UNICAMP / DPCT-UNICAMP. Verão IMPA 2026.

Additional references:

- Peyre, G. and Cuturi, M. (2019). Computational Optimal Transport. Foundations and Trends in Machine Learning.
- Agueh, M. and Carlier, G. (2011). Barycenters in the Wasserstein space. SIAM Journal on Mathematical Analysis.
- Gordaliza, P., del Barrio, E., Gamboa, F. and Loubes, J.-M. (2019). Obtaining fairness using optimal transport theory. ICML.
- Chzhen, E., Denis, C., Hebiri, M., Oneto, L. and Pontil, M. (2020). Fair regression with Wasserstein barycenters. NeurIPS.
- Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal transport. NeurIPS.

This repository is part of a broader research program on how computational systems construct, transform, deform, and erase representations, extending from facial recognition (MSc research) toward the differential geometry of representation spaces (PhD research).
