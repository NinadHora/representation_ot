"""
Phase 2: Permutation test and bootstrap CI for categorical coherence.

Computes the full pairwise distance matrix (17,955 pairs), then tests
H0: rho = 1 via label shuffling. No Sinkhorn recomputation per permutation.
"""

import numpy as np
import csv
import os
import time
from collections import defaultdict, Counter
from itertools import combinations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data import load_subjects
from core.sinkhorn import wasserstein_distance

OUTPUT_DIR = os.path.expanduser("~/ot_faces/output/stats")
EPSILON = 0.1
SEED = 42
N_PERM = 1000
N_BOOT = 1000


def full_distance_matrix(measures, eps):
    """Compute W_eps for all C(n,2) subject pairs. Runs once (~8s for n=190)."""
    sids = sorted(measures.keys())
    n = len(sids)
    D = np.zeros((n, n))
    total = n * (n - 1) // 2
    done = 0
    t0 = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            d = wasserstein_distance(
                measures[sids[i]][0], measures[sids[j]][0],
                measures[sids[i]][1], measures[sids[j]][1], eps=eps)
            D[i, j] = D[j, i] = d
            done += 1
            if done % 2000 == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{total} ({elapsed:.0f}s)")

    print(f"  {total} pairs in {time.time() - t0:.0f}s")
    return sids, D


def compute_ratio(D, sids, subjects, label_key, min_n=2):
    """Compute rho_G = mean(W_intra_G) / mean(W_inter) for each group G."""
    groups = defaultdict(list)
    for i, sid in enumerate(sids):
        groups[subjects[sid][label_key]].append(i)
    groups = {k: v for k, v in groups.items() if len(v) >= min_n}

    intra = {l: np.array([D[a, b] for a, b in combinations(idx, 2)])
             for l, idx in groups.items()}

    labels = sorted(groups)
    inter = []
    for k, la in enumerate(labels):
        for lb in labels[k + 1:]:
            inter.extend(D[a, b] for a in groups[la] for b in groups[lb])
    inter = np.array(inter)

    mu_inter = np.mean(inter)
    ratios = {l: np.mean(d) / mu_inter for l, d in intra.items()}
    return ratios, intra, inter, groups


def permutation_test(D, sids, subjects, label_key):
    """Shuffle labels N_PERM times, recompute rho. Returns p-values."""
    rng = np.random.RandomState(SEED)
    obs, obs_intra, obs_inter, groups = compute_ratio(D, sids, subjects, label_key)

    # Flatten to sub-matrix of valid subjects
    indices, labels_arr = [], []
    for label in sorted(groups):
        for idx in groups[label]:
            indices.append(idx)
            labels_arr.append(label)
    indices = np.array(indices)
    labels_arr = np.array(labels_arr)
    D_sub = D[np.ix_(indices, indices)]

    null = {l: [] for l in groups}
    t0 = time.time()

    for p in range(N_PERM):
        pl = rng.permutation(labels_arr)
        g = defaultdict(list)
        for i, l in enumerate(pl):
            g[l].append(i)

        intra_mu = {}
        for l, idx in g.items():
            if len(idx) >= 2:
                intra_mu[l] = np.mean([D_sub[a, b] for a, b in combinations(idx, 2)])

        sl = sorted(g)
        inter_vals = []
        for k, la in enumerate(sl):
            for lb in sl[k + 1:]:
                inter_vals.extend(D_sub[a, b] for a in g[la] for b in g[lb])
        mu_inter = np.mean(inter_vals)

        for l in g:
            if l in intra_mu and mu_inter > 0:
                null[l].append(intra_mu[l] / mu_inter)

        if (p + 1) % 200 == 0:
            print(f"    {p + 1}/{N_PERM} ({time.time() - t0:.0f}s)")

    results = {}
    for l in sorted(obs):
        nd = np.array(null[l])
        pv = np.mean(nd <= obs[l])
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "n.s."
        results[l] = {"rho": obs[l], "p": pv, "sig": sig, "null": nd,
                       "n": len(groups[l]), "n_pairs": len(obs_intra[l])}
        print(f"  {l:<20} rho={obs[l]:.4f}  p={pv:.4f} {sig}")

    return results, groups


def bootstrap_ci(D, sids, subjects, label_key, groups):
    """Bootstrap 95% CI for rho by resampling distance pairs."""
    rng = np.random.RandomState(SEED + 999)

    intra_d = {l: np.array([D[a, b] for a, b in combinations(idx, 2)])
               for l, idx in groups.items()}
    labels = sorted(groups)
    inter_d = []
    for k, la in enumerate(labels):
        for lb in labels[k + 1:]:
            inter_d.extend(D[a, b] for a in groups[la] for b in groups[lb])
    inter_d = np.array(inter_d)

    boots = {l: [] for l in groups}
    for _ in range(N_BOOT):
        mi = np.mean(rng.choice(inter_d, len(inter_d), replace=True))
        if mi <= 0:
            continue
        for l, d in intra_d.items():
            boots[l].append(np.mean(rng.choice(d, len(d), replace=True)) / mi)

    results = {}
    for l in sorted(groups):
        b = np.array(boots[l])
        lo, hi = np.percentile(b, [2.5, 97.5])
        rho = np.mean(intra_d[l]) / np.mean(inter_d)
        results[l] = {"rho": rho, "lo": lo, "hi": hi, "contains_one": lo <= 1.0 <= hi}
        flag = "yes" if results[l]["contains_one"] else "NO"
        print(f"  {l:<20} [{lo:.4f}, {hi:.4f}]  contains 1.0? {flag}")

    return results


def save(perm, boot, label_key, D, sids, subjects, groups):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path = os.path.join(OUTPUT_DIR, f"test_{label_key}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n", "n_pairs", "rho", "p", "sig", "ci_lo", "ci_hi", "ci_contains_1"])
        for l in sorted(perm):
            p, b = perm[l], boot[l]
            w.writerow([l, p["n"], p["n_pairs"], f"{p['rho']:.6f}", f"{p['p']:.6f}",
                        p["sig"], f"{b['lo']:.6f}", f"{b['hi']:.6f}", b["contains_one"]])
    print(f"  {path}")

    # Null distributions
    path2 = os.path.join(OUTPUT_DIR, f"null_{label_key}.csv")
    labels = sorted(perm)
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i"] + labels)
        for i in range(N_PERM):
            w.writerow([i] + [f"{perm[l]['null'][i]:.6f}" for l in labels])
    print(f"  {path2}")

    # Distance matrix
    np.savez_compressed(os.path.join(OUTPUT_DIR, f"matrix_{label_key}.npz"),
                        D=D, sids=np.array(sids))


if __name__ == "__main__":
    subjects, measures = load_subjects()
    sids, D = full_distance_matrix(measures, EPSILON)

    for key in ["country", "native_language"]:
        print(f"\n=== {key.upper()} ===")
        perm, groups = permutation_test(D, sids, subjects, key)
        boot = bootstrap_ci(D, sids, subjects, key, groups)
        save(perm, boot, key, D, sids, subjects, groups)

    print("\nDone.")
