"""
Phase 3: Cross-cutting analysis — phenotype vs nationality in embedding space.

Reuses the pre-computed distance matrix from permutation_test.py.
Classifies all 17,955 pairs into four quadrants by (country x Monk Scale)
and tests which axis better organizes the space.
"""

import numpy as np
import json
import os
import csv
from collections import defaultdict, Counter
from itertools import combinations

MATRIX_PATH = os.path.expanduser("~/ot_faces/output/stats/matrix_country.npz")
ANNOTATIONS_PATH = "/home/nina.dahora/dataset/casual_conversations_v2/annotations/CasualConversationsV2.json"
OUTPUT_DIR = os.path.expanduser("~/ot_faces/output/stats")
SEED = 42


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
    print(f"{len(subjects)} subjects, {D.shape[0]}x{D.shape[1]} matrix")
    return D, sids, subjects


def four_quadrants(D, sids, subjects):
    """Classify all pairs into same/diff country x same/diff tone."""
    sid_idx = {sid: i for i, sid in enumerate(sids)}
    Q = {"sc_st": [], "sc_dt": [], "dc_st": [], "dc_dt": []}

    for a, b in combinations(subjects, 2):
        d = D[sid_idx[a["sid"]], sid_idx[b["sid"]]]
        sc = a["country"] == b["country"]
        st = a["monk"] == b["monk"]
        key = ("sc" if sc else "dc") + "_" + ("st" if st else "dt")
        Q[key].append(d)

    names = {"sc_st": "same country, same tone",
             "sc_dt": "same country, diff tone",
             "dc_st": "diff country, same tone",
             "dc_dt": "diff country, diff tone"}

    for k in ["sc_st", "sc_dt", "dc_st", "dc_dt"]:
        v = np.array(Q[k])
        print(f"  {names[k]:<30} mean={np.mean(v):.4f}  std={np.std(v):.4f}  n={len(v)}")

    sc_dt, dc_st = np.mean(Q["sc_dt"]), np.mean(Q["dc_st"])
    winner = "nationality" if sc_dt < dc_st else "phenotype"
    print(f"\n  Aggregate winner: {winner} (delta={abs(sc_dt - dc_st):.4f})")
    return Q


def cell_distances(D, sids, subjects):
    """Mean distance for every (country_a, monk_a) x (country_b, monk_b) cell."""
    sid_idx = {sid: i for i, sid in enumerate(sids)}
    by_cell = defaultdict(list)
    for s in subjects:
        by_cell[(s["country"], s["monk"])].append(s["sid"])

    results = {}
    cells = sorted(by_cell.keys())
    for i, ca in enumerate(cells):
        for cb in cells[i:]:
            pairs = ([(a, b) for a in by_cell[ca] for b in by_cell[cb] if a != b]
                     if ca == cb else
                     [(a, b) for a in by_cell[ca] for b in by_cell[cb]])
            if not pairs:
                continue
            dists = [D[sid_idx[a], sid_idx[b]] for a, b in pairs]
            results[(ca, cb)] = {"mean": np.mean(dists), "n": len(dists)}
    return results


def permutation_test_phenotype_vs_nationality(D, sids, subjects, n_perm=1000):
    """Shuffle country labels (tone fixed), test if nationality > phenotype."""
    rng = np.random.RandomState(SEED)
    sid_idx = {sid: i for i, sid in enumerate(sids)}

    def compute_diff(subjs):
        sc_dt, dc_st = [], []
        for a, b in combinations(subjs, 2):
            d = D[sid_idx[a["sid"]], sid_idx[b["sid"]]]
            sc = a["country"] == b["country"]
            st = a["monk"] == b["monk"]
            if sc and not st:
                sc_dt.append(d)
            elif not sc and st:
                dc_st.append(d)
        if sc_dt and dc_st:
            return np.mean(sc_dt) - np.mean(dc_st)
        return 0.0

    obs = compute_diff(subjects)
    countries = [s["country"] for s in subjects]
    null = []
    for p in range(n_perm):
        perm = rng.permutation(countries)
        null.append(compute_diff([{**s, "country": c} for s, c in zip(subjects, perm)]))
        if (p + 1) % 200 == 0:
            print(f"    {p+1}/{n_perm}")

    null = np.array(null)
    pv = np.mean(null >= obs)
    print(f"  observed={obs:.4f}  null={np.mean(null):.4f}±{np.std(null):.4f}  p={pv:.4f}")
    return obs, pv, null


def brazil_deep_dive(D, sids, subjects):
    """Within-Brazil distances by Monk Scale; cross-country same-tone comparisons."""
    sid_idx = {sid: i for i, sid in enumerate(sids)}
    br = [s for s in subjects if s["country"] == "brazil"]
    br_by_monk = defaultdict(list)
    for s in br:
        br_by_monk[s["monk"]].append(s["sid"])

    print(f"\n  Brazil: {len(br)} subjects, Monk distribution: "
          + ", ".join(f"{m}:{len(v)}" for m, v in sorted(br_by_monk.items())))

    # Within-Brazil sub-group distances
    monks = sorted(m for m, v in br_by_monk.items() if len(v) >= 2)
    print("\n  Within-Brazil distances:")
    for i, ma in enumerate(monks):
        for mb in monks[i:]:
            if ma == mb:
                pairs = list(combinations(br_by_monk[ma], 2))
            else:
                pairs = [(a, b) for a in br_by_monk[ma] for b in br_by_monk[mb]]
            if pairs:
                dists = [D[sid_idx[a], sid_idx[b]] for a, b in pairs]
                label = "within" if ma == mb else f"{ma}-{mb}"
                print(f"    Brazil {label}: {np.mean(dists):.4f} (n={len(dists)})")

    # Cross-country, same tone
    non_br = defaultdict(list)
    for s in subjects:
        if s["country"] != "brazil":
            non_br[(s["country"], s["monk"])].append(s["sid"])

    print("\n  Cross-country (same tone):")
    for monk in monks:
        br_sids = br_by_monk[monk]
        within = [D[sid_idx[a], sid_idx[b]] for a, b in combinations(br_sids, 2)]
        if not within:
            continue
        w = np.mean(within)
        for (country, m), others in sorted(non_br.items()):
            if m != monk or len(others) < 2:
                continue
            cross = [D[sid_idx[a], sid_idx[b]] for a in br_sids for b in others]
            c = np.mean(cross)
            tag = "CLOSER" if c < w else "farther"
            print(f"    Br{monk} within={w:.4f} vs Br{monk}-{country}{monk}={c:.4f} [{tag}]")


def save(quadrants, cells, obs_diff, pv):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "cross_cutting_quadrants.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quadrant", "mean", "std", "n"])
        for k, v in quadrants.items():
            v = np.array(v)
            w.writerow([k, f"{np.mean(v):.6f}", f"{np.std(v):.6f}", len(v)])

    with open(os.path.join(OUTPUT_DIR, "cross_cutting_cells.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country_a", "monk_a", "country_b", "monk_b", "mean", "n"])
        for (ka, kb), r in sorted(cells.items()):
            w.writerow([ka[0], ka[1], kb[0], kb[1], f"{r['mean']:.6f}", r["n"]])

    with open(os.path.join(OUTPUT_DIR, "phenotype_vs_nationality.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["observed_diff", "p_value"])
        w.writerow([f"{obs_diff:.6f}", f"{pv:.6f}"])


if __name__ == "__main__":
    D, sids, subjects = load_matrix_and_meta()

    print("\n--- FOUR QUADRANTS ---")
    Q = four_quadrants(D, sids, subjects)

    print("\n--- CELL DISTANCES ---")
    cells = cell_distances(D, sids, subjects)

    print("\n--- PERMUTATION TEST ---")
    obs, pv, null = permutation_test_phenotype_vs_nationality(D, sids, subjects)

    print("\n--- BRAZIL ---")
    brazil_deep_dive(D, sids, subjects)

    save(Q, cells, obs, pv)
    print("\nDone.")
