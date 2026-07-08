"""
Phase 1: Exploratory OT classification on CCv2 embeddings.

Computes sampled pairwise distances, intra/inter ratios, and
prototype-based classification by country and language.
"""

import numpy as np
import csv
import os
import time
from collections import defaultdict

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data import load_subjects
from core.sinkhorn import wasserstein_distance

OUTPUT_DIR = os.path.expanduser("~/ot_faces/output")
EPSILON = 0.1
SEED = 42
TRAIN_RATIO = 0.7


def pairwise_distances(measures, subjects, label_key, eps, n_pairs=15):
    """Sample n_pairs intra- and inter-group Wasserstein distances."""
    rng = np.random.RandomState(SEED)

    groups = defaultdict(list)
    for sid in measures:
        groups[subjects[sid][label_key]].append(sid)

    def sample_distances(sids_a, sids_b, n):
        dists = []
        for _ in range(n):
            sa = sids_a[rng.randint(len(sids_a))]
            sb = sids_b[rng.randint(len(sids_b))]
            if sa == sb:
                continue
            d = wasserstein_distance(
                measures[sa][0], measures[sb][0],
                measures[sa][1], measures[sb][1], eps=eps)
            dists.append(d)
        return dists

    intra = {}
    for label, sids in groups.items():
        if len(sids) < 2:
            continue
        intra[label] = sample_distances(sids, sids, n_pairs)
        print(f"  {label}: intra mean={np.mean(intra[label]):.4f}")

    inter = {}
    labels = sorted(groups)
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            inter[(la, lb)] = sample_distances(groups[la], groups[lb], n_pairs)

    return intra, inter, groups


def classify(measures, subjects, groups, label_key, eps):
    """Prototype-based classification with 70/30 train/test split."""
    rng = np.random.RandomState(SEED)

    # Split
    train, test = {}, {}
    for label, sids in groups.items():
        rng.shuffle(sids)
        k = int(TRAIN_RATIO * len(sids))
        train[label] = sids[:k]
        test[label] = sids[k:]

    # Build prototypes (concatenated embeddings, subsampled to 200)
    prototypes = {}
    for label, sids in train.items():
        embs = np.concatenate([measures[s][0] for s in sids])
        if len(embs) > 200:
            idx = rng.choice(len(embs), 200, replace=False)
            embs = embs[idx]
        prototypes[label] = (embs, np.ones(len(embs)) / len(embs))

    # Classify
    predictions = []
    for label, sids in test.items():
        for sid in sids:
            dists = {l: wasserstein_distance(measures[sid][0], p[0], measures[sid][1], p[1], eps=eps)
                     for l, p in prototypes.items()}
            pred = min(dists, key=dists.get)
            predictions.append({
                "sid": sid, "true": label, "pred": pred,
                "correct": pred == label, "monk": subjects[sid]["monk"],
            })

    acc = np.mean([p["correct"] for p in predictions])
    print(f"  Overall: {acc:.1%} ({sum(p['correct'] for p in predictions)}/{len(predictions)})")
    for label in sorted(groups):
        sub = [p for p in predictions if p["true"] == label]
        if sub:
            a = np.mean([p["correct"] for p in sub])
            print(f"    {label}: {a:.1%} ({sum(p['correct'] for p in sub)}/{len(sub)})")

    return predictions


def save(predictions, intra, inter, label_key):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"predictions_{label_key}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid", "true", "pred", "monk"])
        w.writeheader()
        for p in predictions:
            w.writerow({k: p[k] for k in ["sid", "true", "pred", "monk"]})
    print(f"  Saved: {path}")


if __name__ == "__main__":
    t0 = time.time()
    subjects, measures = load_subjects()

    for key in ["country", "native_language"]:
        print(f"\n--- {key.upper()} ---")
        intra, inter, groups = pairwise_distances(measures, subjects, key, EPSILON)

        global_inter = np.mean([np.mean(v) for v in inter.values()])
        print(f"\n  Ratios (intra/inter, global inter={global_inter:.4f}):")
        for label in sorted(intra):
            r = np.mean(intra[label]) / global_inter
            print(f"    {label}: {r:.3f}")

        preds = classify(measures, subjects, groups, key, EPSILON)
        save(preds, intra, inter, key)

    print(f"\nDone in {time.time() - t0:.0f}s")
