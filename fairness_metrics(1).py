"""
Phase 4: Demographic parity and conditional demographic parity.

Uses the pre-computed distance matrix for leave-one-out nearest-centroid
classification, then measures DP by country, DP by Monk Scale, conditional
DP (country | Monk), and counterfactual decomposition.
"""

import numpy as np
import json
import os
import csv
from collections import defaultdict, Counter

MATRIX_PATH = os.path.expanduser("~/ot_faces/output/stats/matrix_country.npz")
ANNOTATIONS_PATH = "/home/nina.dahora/dataset/casual_conversations_v2/annotations/CasualConversationsV2.json"
OUTPUT_DIR = os.path.expanduser("~/ot_faces/output/stats")


def load_matrix_and_meta():
    data = np.load(MATRIX_PATH, allow_pickle=True)
    D, sids = data["D"], list(data["sids"])

    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)

    meta = {}
    for ann in annotations:
        sid = str(ann["subject_id"]).zfill(4)
        if sid not in meta:
            meta[sid] = {
                "country": ann["geo_location"]["country"],
                "monk": ann["monk_skin_tone"]["scale"],
            }

    subjects = [{"sid": sid, "idx": i, **meta[sid]}
                for i, sid in enumerate(sids) if sid in meta]
    print(f"{len(subjects)} subjects")
    return D, sids, subjects


def classify_loo(D, sids, subjects):
    """Leave-one-out nearest centroid: predict country with smallest mean W_eps."""
    sid_idx = {sid: i for i, sid in enumerate(sids)}
    groups = defaultdict(list)
    for s in subjects:
        groups[s["country"]].append(s)

    preds = []
    for s in subjects:
        i = sid_idx[s["sid"]]
        d2g = {}
        for label, members in groups.items():
            dists = [D[i, sid_idx[m["sid"]]] for m in members if m["sid"] != s["sid"]]
            if dists:
                d2g[label] = np.mean(dists)
        pred = min(d2g, key=d2g.get)
        preds.append({"sid": s["sid"], "country": s["country"], "monk": s["monk"],
                       "true": s["country"], "pred": pred, "correct": pred == s["country"]})
    return preds


def dp(preds, key):
    """Demographic parity: accuracy by group, DP gap = max - min."""
    by_g = defaultdict(list)
    for p in preds:
        by_g[p[key]].append(p["correct"])

    print(f"\n  {'group':<20} {'acc':>8} {'n':>6}")
    results = {}
    for g in sorted(by_g):
        acc = np.mean(by_g[g])
        results[g] = {"acc": acc, "correct": sum(by_g[g]), "n": len(by_g[g])}
        print(f"  {str(g):<20} {acc:>8.1%} {len(by_g[g]):>6}")

    accs = [r["acc"] for r in results.values()]
    gap = max(accs) - min(accs)
    overall = np.mean([p["correct"] for p in preds])
    print(f"  {'overall':<20} {overall:>8.1%} {len(preds):>6}")
    print(f"  DP gap: {gap:.1%}")
    return results, gap


def conditional_dp(preds):
    """Accuracy by country conditioned on Monk Scale."""
    by_mc = defaultdict(list)
    for p in preds:
        by_mc[(p["monk"], p["country"])].append(p["correct"])

    monks = sorted(set(p["monk"] for p in preds))
    countries = sorted(set(p["country"] for p in preds))

    print(f"\n  {'monk':<8}" + "".join(f"{c:>15}" for c in countries) + f"{'gap':>10}")
    gaps = []
    for m in monks:
        row = f"  {m:<8}"
        accs = {}
        for c in countries:
            vals = by_mc.get((m, c), [])
            if len(vals) >= 2:
                a = np.mean(vals)
                accs[c] = a
                row += f"{a:>12.0%} ({len(vals):>2})"
            else:
                row += f"{'--':>12} ({len(vals):>2})"
        if len(accs) >= 2:
            g = max(accs.values()) - min(accs.values())
            gaps.append(g)
            row += f"{g:>9.0%}"
        else:
            row += f"{'--':>10}"
        print(row)

    if gaps:
        print(f"\n  Avg conditional DP gap: {np.mean(gaps):.1%} (over {len(gaps)} levels)")
    return gaps


def counterfactual(preds):
    """Reweight each country to the global Monk distribution."""
    monk_w = Counter(p["monk"] for p in preds)
    total = len(preds)
    monk_w = {m: c / total for m, c in monk_w.items()}

    by_cm = defaultdict(list)
    for p in preds:
        by_cm[(p["country"], p["monk"])].append(p["correct"])

    countries = sorted(set(p["country"] for p in preds))
    print(f"\n  {'country':<15} {'observed':>10} {'counterfactual':>15} {'delta':>10}")
    for c in countries:
        obs = np.mean([p["correct"] for p in preds if p["country"] == c])
        cf_num = sum(monk_w.get(m, 0) * np.mean(by_cm[(c, m)])
                     for m in monk_w if (c, m) in by_cm)
        cf_den = sum(monk_w.get(m, 0) for m in monk_w if (c, m) in by_cm)
        cf = cf_num / cf_den if cf_den > 0 else obs
        print(f"  {c:<15} {obs:>10.1%} {cf:>15.1%} {cf - obs:>+10.1%}")


def save(preds, dp_country, dp_monk, cond_gaps):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "dp_country.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "accuracy", "correct", "total"])
        for g, r in sorted(dp_country.items()):
            w.writerow([g, f"{r['acc']:.6f}", r["correct"], r["n"]])

    with open(os.path.join(OUTPUT_DIR, "dp_monk.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["monk", "accuracy", "correct", "total"])
        for g, r in sorted(dp_monk.items()):
            w.writerow([g, f"{r['acc']:.6f}", r["correct"], r["n"]])

    with open(os.path.join(OUTPUT_DIR, "conditional_dp.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "gap"])
        for i, g in enumerate(cond_gaps):
            w.writerow([i, f"{g:.6f}"])

    with open(os.path.join(OUTPUT_DIR, "predictions_full.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "country", "monk", "pred", "correct"])
        for p in preds:
            w.writerow([p["sid"], p["country"], p["monk"], p["pred"], p["correct"]])


if __name__ == "__main__":
    D, sids, subjects = load_matrix_and_meta()

    preds = classify_loo(D, sids, subjects)
    acc = np.mean([p["correct"] for p in preds])
    print(f"Overall: {acc:.1%} ({sum(p['correct'] for p in preds)}/{len(preds)})")

    print("\n--- DP BY COUNTRY ---")
    dp_country, _ = dp(preds, "country")

    print("\n--- DP BY MONK SCALE ---")
    dp_monk, _ = dp(preds, "monk")

    print("\n--- CONDITIONAL DP ---")
    cond_gaps = conditional_dp(preds)

    print("\n--- COUNTERFACTUAL ---")
    counterfactual(preds)

    save(preds, dp_country, dp_monk, cond_gaps)
    print("\nDone.")
