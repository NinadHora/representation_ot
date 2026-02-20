"""
Visualizations for the OT x CCv2 experiment.

Reads results from output/stats/ CSVs and generates plots.
Run after all pipeline phases have completed.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import csv

OUTPUT_DIR = os.path.expanduser("~/ot_faces/output/plots")
STATS_DIR = os.path.expanduser("~/ot_faces/output/stats")

COUNTRY_COLORS = {"brazil": "#009739", "india": "#FF9933",
                   "indonesia": "#FF0000", "mexico": "#006847"}

MONK_COLORS = {i: c for i, c in enumerate([
    "#f6ede4", "#f3e7db", "#f7e0c5", "#eacfa0", "#c9a87c",
    "#a07850", "#825c3a", "#604530", "#3a2a1d", "#2d2116"], 1)}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                      "axes.spines.right": False, "figure.dpi": 150})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_coherence_ratios():
    """Bar chart of rho by country and language."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key in [(ax1, "country"), (ax2, "native_language")]:
        rows = read_csv(os.path.join(STATS_DIR, f"test_{key}.csv"))
        rows.sort(key=lambda r: float(r["rho"]))
        labels = [r["group"] for r in rows]
        rhos = [float(r["rho"]) for r in rows]
        colors = [COUNTRY_COLORS.get(l, "#666") for l in labels]

        bars = ax.barh(labels, rhos, color=colors, alpha=0.85, height=0.6)
        ax.axvline(1.0, color="#999", ls=":", alpha=0.5)
        ax.axvline(0.8, color="#cc3333", ls="--", alpha=0.5)
        for i, r in enumerate(rhos):
            p = rows[i]["p"]
            sig = rows[i]["sig"]
            ax.text(r + 0.01, i, f"{r:.3f} {sig}", va="center", fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_title(f"Coherence ratio ({key})")
        ax.set_xlabel("ρ (1.0 = no coherence)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "coherence_ratios.png"), bbox_inches="tight")
    plt.close()


def plot_dp():
    """DP accuracy by country and Monk Scale side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # By country
    rows = read_csv(os.path.join(STATS_DIR, "dp_country.csv"))
    rows.sort(key=lambda r: float(r["accuracy"]))
    labels = [r["group"] for r in rows]
    accs = [float(r["accuracy"]) * 100 for r in rows]
    colors = [COUNTRY_COLORS.get(l, "#666") for l in labels]
    ax1.barh(labels, accs, color=colors, alpha=0.85, height=0.6)
    ax1.axvline(25, color="#999", ls="--", alpha=0.5, label="chance")
    for i, a in enumerate(accs):
        ax1.text(a + 1, i, f"{a:.1f}%", va="center", fontsize=9)
    ax1.set_xlim(0, 105)
    ax1.set_title("Accuracy by country")
    ax1.legend(fontsize=8)

    # By Monk
    rows = read_csv(os.path.join(STATS_DIR, "dp_monk.csv"))
    scales = [r["monk_scale"] if "monk_scale" in r else r.get("monk", r.get("group", "")) for r in rows]
    accs = [float(r["accuracy"]) * 100 for r in rows]
    ns = [int(r["total"]) for r in rows]
    x = range(len(scales))
    bars = ax2.bar(x, accs, color=[MONK_COLORS.get(i + 1, "#999") for i in range(len(scales))],
                   edgecolor="#333", lw=0.5)
    for i, (a, n) in enumerate(zip(accs, ns)):
        ax2.text(i, a + 2, f"n={n}", ha="center", fontsize=7)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(scales, fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.set_title("Accuracy by Monk Scale")
    ax2.set_ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "demographic_parity.png"), bbox_inches="tight")
    plt.close()


def plot_quadrants():
    """Four-quadrant cross-cutting distances."""
    rows = read_csv(os.path.join(STATS_DIR, "cross_cutting_quadrants.csv"))
    labels = [r["quadrant"].replace("_", "\n") for r in rows]
    means = [float(r["mean"]) for r in rows]
    ns = [int(r["n"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6"]
    bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.85)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + 0.01, f"{m:.3f}\n(n={n})", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean $W_\\varepsilon$")
    ax.set_title("Cross-cutting: country × Monk Scale")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cross_cutting.png"), bbox_inches="tight")
    plt.close()


def plot_distance_heatmap():
    """Heatmap from the full distance matrix."""
    data = np.load(os.path.join(STATS_DIR, "matrix_country.npz"), allow_pickle=True)
    D, sids = data["D"], list(data["sids"])

    # Load metadata for grouping
    import json
    ann_path = "/home/nina.dahora/dataset/casual_conversations_v2/annotations/CasualConversationsV2.json"
    if not os.path.exists(ann_path):
        print("  Skipping heatmap (no annotations)")
        return

    with open(ann_path) as f:
        annotations = json.load(f)
    meta = {}
    for ann in annotations:
        sid = str(ann["subject_id"]).zfill(4)
        if sid not in meta:
            meta[sid] = ann["geo_location"]["country"]

    countries = sorted(set(meta.get(s, "?") for s in sids))
    n = len(countries)
    M = np.zeros((n, n))

    for ci, ca in enumerate(countries):
        idx_a = [i for i, s in enumerate(sids) if meta.get(s) == ca]
        for cj, cb in enumerate(countries):
            idx_b = [i for i, s in enumerate(sids) if meta.get(s) == cb]
            dists = [D[a, b] for a in idx_a for b in idx_b if a != b]
            M[ci, cj] = np.mean(dists) if dists else 0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="YlOrRd", vmin=1.0, vmax=2.0)
    for i in range(n):
        for j in range(n):
            c = "white" if M[i, j] > 1.6 else "black"
            w = "bold" if i == j else "normal"
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", color=c, fontweight=w)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(countries, fontsize=10)
    ax.set_yticklabels(countries, fontsize=10)
    ax.set_title("Mean $W_\\varepsilon$ between countries\n(diagonal = intra-group)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="$W_\\varepsilon$")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distance_heatmap.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating plots...")

    for fn in [plot_coherence_ratios, plot_dp, plot_quadrants, plot_distance_heatmap]:
        try:
            fn()
            print(f"  {fn.__name__}: ok")
        except Exception as e:
            print(f"  {fn.__name__}: SKIP ({e})")

    print(f"Saved to {OUTPUT_DIR}")
