# Tests (planned)

Validation of the mathematical core against the POT library
(https://pythonot.github.io/) on small problems:

- regularized Wasserstein distance: agreement with `ot.sinkhorn2` within 1e-4;
- Wasserstein barycenter: agreement with `ot.bregman.barycenter`;
- convergence and marginal constraint satisfaction of the transport plan.
