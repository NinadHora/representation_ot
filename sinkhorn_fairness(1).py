import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import ot

import jax
import jax.numpy as jnp
from jax import lax
from jax import jit
from jax.scipy.special import logsumexp
jax.config.update("jax_enable_x64", True)


def fast_bins_choice(S_calib: pd.DataFrame, target_per_bin=30, min_bins=30, max_bins=200):
    """
    Approximate a good n_bins from group sizes only.
    target_per_bin: desired avg samples per bin in the smallest group (20-50 is typical).
    """
    g = S_calib.astype(str).agg("_".join, axis=1)
    counts = g.value_counts()
    n_min = int(counts.min())
    n_bins = max(min_bins, min(max_bins, n_min // target_per_bin))
    return n_bins, n_min, counts


def make_grid(scores, n_bins=200, pad=1e-6):
    s = np.asarray(scores, dtype=float)
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        hi = lo + 1.0
    lo -= pad * (hi - lo)
    hi += pad * (hi - lo)
    return np.linspace(lo, hi, n_bins)


def hist_on_grid(scores, grid):
    """
    Returns a probability vector mu over grid bins using counts (simple, stable).
    Each score is assigned to nearest bin (via digitize).
    """
    scores = np.asarray(scores, dtype=float)
    grid = np.asarray(grid, dtype=float)

    # bin edges midpoints
    edges = (grid[:-1] + grid[1:]) / 2.0
    idx = np.digitize(scores, edges)  # in [0, len(grid)-1]
    counts = np.bincount(idx, minlength=len(grid)).astype(float)

    total = counts.sum()
    if total <= 0:
        # degenerate group: return uniform to avoid nan
        return np.ones(len(grid), dtype=float) / len(grid)
    return counts / total


def create_marginal_masks(S_calib: pd.DataFrame):
    masks = {}
    sensitive_cols = list(S_calib)

    # groupby gives you row indices per unique sensitive-combo
    for key, idx in S_calib.groupby(sensitive_cols, observed=True).groups.items():
        if len(sensitive_cols) == 1:
            key = (key,)  # make keys uniform
        mask = S_calib.index.isin(idx)  # numpy bool array aligned with S_calib.index
        masks[tuple(key)] = mask

    return masks

def create_marginals(S_calib: pd.DataFrame, p_calib: np.array, grid: np.array):
    masks = create_marginal_masks(S_calib)
    marginals = []
    weights = []

    for key, mask in masks.items():
        hist = hist_on_grid(p_calib[mask], grid=grid)
        marginals.append(hist)
        weights.append(p_calib[mask].shape[0])

    return np.array(marginals), np.array(weights)

@jit(static_argnames=[
    'reg',
    'maxiter',
    'return_diagnostics',
])
def barycenter_sinkhorn(
        measures: jnp.ndarray,
        cost: jnp.ndarray,
        lambdas: jnp.ndarray,
        reg: float = 1e-3,
        tol: float = 1e-4,
        maxiter: int = 10000,
        return_diagnostics: bool = True,
        error_check_every: int = 20,
):
    lam = (lambdas / lambdas.sum())[:, None]   # (J,1)
    # clip measures for log computations to be a little bit safer in practice
    log_measures = jnp.log(jnp.maximum(measures, 1e-30))
    lnb = jnp.zeros_like(measures[0])
    lnK = -cost / reg

    def cond_fn(state):
        i, lnus, lnvs, lnb, err, errors = state
        return jnp.logical_and(i < maxiter, err > tol)

    def body_fn(state):
        i, lnus, lnvs, lnb, err, errors = state

        # update each f^j
        ln_Kv = logsumexp(
            lnvs[:, None, :] + lnK[None, :, :],
            axis=2,
        )
        lnus = log_measures - ln_Kv
        # compute log(K^T u^j) in a stable way
        ln_Ktu = logsumexp(
            lnus[:, :, None] + lnK[None, :, :],
            axis=1,
        )
        # update barycenter log(b)
        lnb = jnp.sum(lam * ln_Ktu, axis=0)
        lnb = lnb - logsumexp(lnb)   # now renormalize
        # update each g^j
        lnvs = lnb[None, :] - ln_Ktu

        def compute_err(_):
            new_err, _details = marginal_error_from_log(log_measures,
                                                        lnK, lnus, lnvs, lnb)
            return new_err
        err = lax.cond(
            (i % error_check_every) == 0,
            compute_err,
            lambda _: err,
            operand=None,
        )
        errors = errors.at[i].set(err)
        return (i+1, lnus, lnvs, lnb, err, errors)

    lnus = jnp.zeros_like(measures)
    lnvs = jnp.zeros_like(measures)
    errors = jnp.full((maxiter,), jnp.nan)
    err0 = jnp.asarray(jnp.inf)
    init_state = (jnp.asarray(0), lnus, lnvs, lnb, err0, errors)
    final_state = lax.while_loop(cond_fn, body_fn, init_state)
    iterations, lnus, lnvs, lnb, err, errors = final_state
    b = jnp.exp(lnb)
    b /= b.sum()
    diagnostics = {
            'iterations': iterations,
            'error': err,
            **({
                'ln_u': lnus,
                'ln_v': lnvs,
                'ln_b': lnb,
                'errors': errors,
                } if return_diagnostics else {}),
            }
    return b, diagnostics


def marginal_error_from_log(
    log_measures: jnp.ndarray,  # (J, N) = log(a_j)
    lnK: jnp.ndarray,           # (N, N) = -C/reg
    lnus: jnp.ndarray,          # (J, N) = log(u_j)
    lnvs: jnp.ndarray,          # (J, N) = log(v_j)
    lnb: jnp.ndarray,           # (N,)   = log(b)
    eps: float = 1e-30,
):
    # ln(K v_j) and ln(K^T u_j)
    lnKv = logsumexp(lnvs[:, None, :] + lnK[None, :, :], axis=2)    # (J, N)
    lnKtu = logsumexp(lnus[:, :, None] + lnK[None, :, :], axis=1)   # (J, N)

    # implied marginals in normal domain
    a_hat = jnp.exp(lnus + lnKv)                 # (J, N) should match exp(log_measures)
    b_hat = jnp.exp(lnvs + lnKtu)                # (J, N) should match exp(lnb)[None,:]

    a = jnp.exp(log_measures)                    # (J, N)
    b = jnp.exp(lnb)                             # (N,)

    # # L1 errors per measure
    # err_a_per_j = jnp.sum(jnp.abs(a_hat - a), axis=1)           # (J,)
    # err_b_per_j = jnp.sum(jnp.abs(b_hat - b[None, :]), axis=1)  # (J,)
    # L2 errors per measure
    err_a_per_j = jnp.sum(jnp.power(a_hat - a, 2), axis=1)           # (J,)
    err_b_per_j = jnp.sum(jnp.power(b_hat - b[None, :], 2), axis=1)  # (J,)

    err_a = jnp.max(err_a_per_j)
    err_b = jnp.max(err_b_per_j)
    err = jnp.maximum(err_a, err_b)

    return err, {"err_a": err_a, "err_b": err_b}

def _hist_on_grid(scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Nearest-bin histogram on fixed grid, returns probability vector (m,)."""
    scores = np.asarray(scores, dtype=float)
    grid = np.asarray(grid, dtype=float)

    edges = (grid[:-1] + grid[1:]) / 2.0
    idx = np.digitize(scores, edges)  # 0..m-1
    counts = np.bincount(idx, minlength=len(grid)).astype(float)
    s = counts.sum()
    if s <= 0:
        return np.ones(len(grid), dtype=float) / len(grid)
    return counts / s

def barycentric_projection_from_log_scalings_stable(
    ln_u: np.ndarray,      # (m,)
    ln_v: np.ndarray,      # (m,)
    lnK: np.ndarray,       # (m,m)
    mu: np.ndarray,        # (m,)
    grid: np.ndarray,      # (m,)
    *,
    eps_mass: float = 1e-16,
) -> np.ndarray:
    """
    Stable barycentric projection:
      T[i] = sum_j pi[i,j] * y[j] / mu[i], pi = diag(exp(ln_u)) exp(lnK) diag(exp(ln_v))
    Works even if grid has negative values (splits pos/neg).
    """
    ln_u = np.asarray(ln_u, dtype=float)
    ln_v = np.asarray(ln_v, dtype=float)
    lnK  = np.asarray(lnK,  dtype=float)
    mu   = np.asarray(mu,   dtype=float)
    y    = np.asarray(grid, dtype=float)

    m = y.size
    if ln_u.shape != (m,) or ln_v.shape != (m,) or mu.shape != (m,) or lnK.shape != (m, m):
        raise ValueError("Shapes must be ln_u:(m,), ln_v:(m,), mu:(m,), lnK:(m,m), grid:(m,)")

    # log pi_ij = ln_u_i + lnK_ij + ln_v_j
    log_pi = ln_u[:, None] + lnK + ln_v[None, :]  # (m,m)

    # Handle y with sign by splitting
    pos = y > 0
    neg = y < 0

    # numerator_i = sum_j exp(log_pi_ij) * y_j
    # = sum_{pos} exp(log_pi_ij + log(y_j)) - sum_{neg} exp(log_pi_ij + log(-y_j))
    num = np.zeros(m, dtype=float)

    if np.any(pos):
        log_y_pos = np.log(y[pos])
        # logsumexp over j in pos
        num_pos = np.exp(logsumexp(log_pi[:, pos] + log_y_pos[None, :], axis=1))
        num += num_pos

    if np.any(neg):
        log_y_neg = np.log(-y[neg])
        num_neg = np.exp(logsumexp(log_pi[:, neg] + log_y_neg[None, :], axis=1))
        num -= num_neg

    # divide by mu (avoid empty bins)
    T = np.empty_like(num)
    good = mu > eps_mass
    T[good] = num[good] / mu[good]
    T[~good] = y[~good]  # fallback identity on empty bins
    return T

def make_map_monotone(grid, T):
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    return ir.fit_transform(grid, T)

def apply_map_1d(scores: np.ndarray, grid: np.ndarray, T_on_grid: np.ndarray) -> np.ndarray:
    """Apply grid-defined map via linear interpolation."""
    scores = np.asarray(scores, dtype=float)
    grid = np.asarray(grid, dtype=float)
    T_on_grid = np.asarray(T_on_grid, dtype=float)

    s = np.clip(scores, grid[0], grid[-1])
    return np.interp(s, grid, T_on_grid)


def build_barycentric_projection_maps_from_calib(
    S_calib: pd.DataFrame,
    p_calib: np.ndarray,
    grid: np.ndarray,
    lnK: np.ndarray,
    lnus: np.ndarray,   # (G,m)
    lnvs: np.ndarray,   # (G,m)
    *,
    order: str = "insertion",  # "insertion" (dict order) or "sorted"
):
    """
    Returns:
      maps: dict[key] -> T_g_on_grid (m,)
      keys: list of keys in the exact order used to index lnus/lnvs
      mu_dict: dict[key] -> mu_g histogram (m,)
    """
    masks = create_marginal_masks(S_calib)

    # Choose a deterministic ordering of groups
    keys = list(masks.keys())
    if order == "sorted":
        keys = sorted(keys, key=lambda k: tuple(k))
    elif order == "insertion":
        # Python 3.7+ preserves insertion order; keep as-is
        pass
    else:
        raise ValueError("order must be 'insertion' or 'sorted'")

    G = len(keys)
    m = len(grid)

    lnus = np.asarray(lnus, dtype=float)
    lnvs = np.asarray(lnvs, dtype=float)

    if lnus.shape != (G, m) or lnvs.shape != (G, m):
        raise ValueError(f"Expected lnus/lnvs shape {(G,m)}, got {lnus.shape} and {lnvs.shape}")

    # Build mu_g from calibration logits
    mu_dict = {}
    for k in keys:
        mu_dict[k] = _hist_on_grid(p_calib[masks[k]], grid)

    # Build T_g using your log-domain projection
    maps = {}
    for gi, k in enumerate(keys):
        maps[k] = barycentric_projection_from_log_scalings_stable(
            ln_u=lnus[gi],
            ln_v=lnvs[gi],
            lnK=lnK,
            mu=mu_dict[k],
            grid=grid,
        )
        maps[k] = make_map_monotone(grid, maps[k])

    return maps, keys, mu_dict

def apply_fair_maps_to_test_logits(
    p_test: np.ndarray,
    S_test: pd.DataFrame,
    grid: np.ndarray,
    maps: dict,
    alpha: float = 0.0,
    *,
    unseen: str = "identity",  # what if a test group combo wasn't in calib
):
    """
    Returns fair logits p_test_fair.
    """
    p_test = np.asarray(p_test, dtype=float)
    masks_test = create_marginal_masks(S_test)

    p_fair = p_test.copy()

    for key, mask in masks_test.items():
        if key in maps:
            p_fair[mask] = (1 - alpha) * apply_map_1d(p_test[mask], grid, maps[key]) + alpha * p_test[mask]
        else:
            if unseen == "identity":
                p_fair[mask] = p_test[mask]
            else:
                raise ValueError("unseen must be 'identity' (or implement your own fallback)")

    return p_fair

def monge_maps_fit(
    S_calib: pd.DataFrame,
    p_calib: np.ndarray,
    sinkhorn_reg: float,
    sinkhorn_tol: float = 1e-6,
    sinkhorn_maxiter: int = 10000,
    n_bins=None,
    pad: float = 1e-6,
    cost: str = "l2",
):  
    if n_bins is None:
        n_bins, *_ = fast_bins_choice(S_calib)
    
    grid = make_grid(p_calib, n_bins, pad)
    marginals, weights = create_marginals(S_calib, p_calib, grid)

    if cost == "l2":
        C = ot.utils.dist(grid.reshape((-1, 1)), grid.reshape((-1, 1)))
    elif cost == "l1":
        C = ot.utils.dist(grid.reshape((-1, 1)), grid.reshape((-1, 1)), metric='cityblock')
    else:
        raise ValueError("cost must be 'l2' or 'l1'")
    
    
    C = C / C.max()
    
    _, diagnostics = barycenter_sinkhorn(
        marginals, 
        C, 
        weights, 
        reg=sinkhorn_reg, 
        tol=sinkhorn_tol,
        maxiter=sinkhorn_maxiter
    )
    lnvs = diagnostics['ln_v']
    lnus = diagnostics['ln_u']

    maps, _, _ = build_barycentric_projection_maps_from_calib(
        S_calib=S_calib,      # only sensitive cols, length = len(p_calib)
        p_calib=p_calib,      # logits on calib
        grid=grid,            # (m,)
        lnK=-C/sinkhorn_reg,              # (m|,m)
        lnus=lnus,            # (G,m) log u from barycenter
        lnvs=lnvs,            # (G,m) log v from barycenter
        order="insertion",    # MUST match how you built lnus/lnvs
    )

    return maps, grid