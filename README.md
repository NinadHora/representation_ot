# representation_ot

**Optimal transport as a language for the geometry of learned representations.**

Most work at the intersection of optimal transport (OT) and machine learning fairness treats OT as a repair tool: a way to move distributions so that a downstream metric improves. This repository inverts that framing. OT is used here as a *measurement language* for the geometry of representation spaces — fairness outcomes are treated as downstream consequences of that geometry, not as the starting point.

The guiding thesis: **representational erasure can be detected geometrically, in the embedding space itself, before any classifier is trained.**

> ⚠️ This is an active research repository. Preliminary findings are described qualitatively below; complete results, statistical analyses, and experimental details will appear in a forthcoming preprint. If you build on the ideas or code here, please cite (see [Citation](#citation)).

## Research questions

1. Can category coherence (or its absence) be quantified directly from the geometry of an embedding distribution, using OT distances between subgroup distributions?
2. How should transport-based comparisons be formulated in high-dimensional embedding spaces, where closed-form solutions and monotone rearrangements are no longer available?
3. What do face-embedding models actually encode when trained representations are probed with demographic and geographic categories — and when do those categories dissolve?
4. Under what conditions do 1-D score-repair results (barycenter-based post-processing) extend, fail, or require reformulation in higher dimensions?
5. What does a geometry-first account of representational harm change about how algorithmic audits are designed?

## Repository structure

```
core/           From-scratch NumPy implementation of entropic OT (Sinkhorn),
                written for transparency and pedagogy rather than speed.
case_studies/   Applied studies using the core machinery (see below).
demos/          Self-contained synthetic examples (in progress).
```

## Case studies

### 1 · Geometric coherence of demographic categories in face embeddings

Face embeddings from Casual Conversations v2 (CCv2) are analyzed at the subject level: each subject's embeddings are treated as a distribution, and entropic Wasserstein distances between subject-level distributions are tested (via permutation tests) for alignment with national and linguistic category structure.

**Preliminary finding (qualitative):** category coherence in the embedding geometry tracks phenotypic homogeneity rather than the categories themselves. Nationality behaves as a geometric category only where it is phenotypically homogeneous; phenotypically heterogeneous categories dissolve. Full quantitative results are reserved for the forthcoming preprint.

### 2 · Score repair via Wasserstein barycenters

A from-scratch reimplementation of barycenter-based fair score post-processing (Gordaliza et al.; Chzhen et al.) in JAX, using monotone (Monge) transport maps and isotonic regression, validated against the EquiPy reference implementation. This case study is included as a working demonstration of the 1-D theory — and of the gap between 1-D closed-form results and the high-dimensional setting, where the open questions of this project live.

## Data

CCv2 embeddings, annotations, and any derived artifacts are **not** included in this repository, in accordance with the dataset's Data Use Agreement. Access to CCv2 must be requested directly from Meta AI. All code here operates on locally provided paths and runs end-to-end on synthetic data (see `demos/`).

## Ongoing work

Extensions toward cross-model comparison and high-dimensional transport are in progress. Details will be released together with the preprint.

## References

- Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.* NeurIPS.
- Peyré, G., & Cuturi, M. (2019). *Computational Optimal Transport.* Foundations and Trends in Machine Learning.
- Gordaliza, P., del Barrio, E., Fabrice, G., & Loubes, J.-M. (2019). *Obtaining Fairness using Optimal Transport Theory.* ICML.
- Chzhen, E., Denis, C., Hebiri, M., Oneto, L., & Pontil, M. (2020). *Fair Regression with Wasserstein Barycenters.* NeurIPS.
- Hazirbas, C., et al. (2022). *Casual Conversations v2.* Meta AI.

## Citation

If you use this code or build on this framing, please cite via the `CITATION.cff` file in this repository (GitHub's "Cite this repository" button), or the Zenodo DOI of the latest release.

## License

MIT — see `LICENSE`.
