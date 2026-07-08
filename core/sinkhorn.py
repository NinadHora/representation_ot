"""
Entropy-regularized optimal transport via the Sinkhorn algorithm.

References:
    Cuturi (2013), "Sinkhorn Distances" (NeurIPS)
    Peyré & Cuturi (2019), "Computational Optimal Transport" (FnTML)
"""

import numpy as np
from typing import Optional, Tuple


def cost_matrix(X: np.ndarray, Y: np.ndarray, metric="sqeuclidean") -> np.ndarray:
    """Squared Euclidean cost matrix between point clouds X (n,d) and Y (m,d)."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    M = XX + YY.T - 2 * (X @ Y.T)
    M = np.maximum(M, 0)
    if metric == "euclidean":
        M = np.sqrt(M)
    return M


def sinkhorn(a, b, M, eps, max_iter=1000, tol=1e-8, log_domain=False):
    """
    Solve the regularized OT problem: min_P <P,M> + eps * KL(P || a⊗b).

    Returns (P, cost, info) where P is the transport plan, cost = <P,M>,
    and info contains convergence diagnostics.
    """
    if log_domain:
        return _sinkhorn_log(a, b, M, eps, max_iter, tol)

    K = np.exp(-M / eps)
    u, v = np.ones(len(a)), np.ones(len(b))

    for i in range(max_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)
        err = np.abs(u * (K @ v) - a).sum()
        if err < tol:
            break
        if not np.isfinite(u).all():
            raise RuntimeError(f"Sinkhorn diverged at iteration {i}. Try larger eps or log_domain=True.")

    P = u[:, None] * K * v[None, :]
    return P, np.sum(P * M), {"converged": err < tol, "iterations": i + 1, "error": err}


def _sinkhorn_log(a, b, M, eps, max_iter, tol):
    """Log-domain Sinkhorn for numerical stability."""
    n, m = M.shape
    f, g = np.zeros(n), np.zeros(m)
    log_a, log_b = np.log(a + 1e-300), np.log(b + 1e-300)

    def lse(X, axis):
        mx = X.max(axis=axis, keepdims=True)
        return mx.squeeze(axis=axis) + np.log(np.exp(X - mx).sum(axis=axis))

    for i in range(max_iter):
        f = eps * log_a - eps * lse((-M + g[None, :]) / eps, axis=1)
        g = eps * log_b - eps * lse((-M + f[:, None]) / eps, axis=0)
        log_P = (f[:, None] + g[None, :] - M) / eps
        err = np.abs(np.exp(lse(log_P, axis=1)) - a).sum()
        if err < tol:
            break

    P = np.exp((f[:, None] + g[None, :] - M) / eps)
    return P, np.sum(P * M), {"converged": err < tol, "iterations": i + 1, "error": err}


def wasserstein_distance(X, Y, a=None, b=None, eps=0.01, **kw) -> float:
    """W_eps(mu, nu) between point clouds X and Y with weights a, b."""
    if a is None: a = np.ones(len(X)) / len(X)
    if b is None: b = np.ones(len(Y)) / len(Y)
    _, d, _ = sinkhorn(a, b, cost_matrix(X, Y), eps, **kw)
    return d


def sinkhorn_divergence(X, Y, a=None, b=None, eps=0.01, **kw) -> float:
    """S_eps(mu,nu) = W_eps(mu,nu) - W_eps(mu,mu)/2 - W_eps(nu,nu)/2."""
    return (wasserstein_distance(X, Y, a, b, eps, **kw)
            - 0.5 * wasserstein_distance(X, X, a, a, eps, **kw)
            - 0.5 * wasserstein_distance(Y, Y, b, b, eps, **kw))


def wasserstein_barycenter(measures, weights=None, support=None, eps=0.01, n_iter=100):
    """
    Fixed-support Wasserstein barycenter of empirical measures.

    Args:
        measures: list of (support_k, weights_k) tuples
        weights: barycentric weights (default: uniform)
        support: fixed support points (default: first measure's support)
    """
    s = len(measures)
    if weights is None: weights = np.ones(s) / s
    if support is None: support = measures[0][0].copy()
    p = len(support)

    kernels = [np.exp(-cost_matrix(support, mk[0]) / eps) for mk in measures]
    v_list = [np.ones(len(mk[0])) for mk in measures]

    for _ in range(n_iter):
        log_bary = np.zeros(p)
        for k in range(s):
            u_k = 1.0 / (kernels[k] @ v_list[k])
            v_list[k] = measures[k][1] / (kernels[k].T @ u_k)
            log_bary += weights[k] * np.log(u_k * (kernels[k] @ v_list[k]) + 1e-300)
        bary = np.exp(log_bary)
        bary /= bary.sum()
        for k in range(s):
            v_list[k] = measures[k][1] / (kernels[k].T @ (bary / (kernels[k] @ v_list[k])))

    return bary
