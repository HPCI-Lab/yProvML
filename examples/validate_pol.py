#!/usr/bin/env python3
import os
import numpy as np
import argparse

# === TROVA I POL FILE NELLA STRUTTURA ===
def find_pol(path):
    path = os.path.join(path, "artifacts_GR0")
    pol_items = [p for p in os.listdir(path) if "_pol_" in p]
    if not pol_items:
        raise FileNotFoundError(f"No POL files found in {path}")
    pol_path = os.path.join(path, pol_items[0])

    if pol_path.endswith(".csv"):
        return pol_path
    else:
        # Directory con più file .npy
        return [os.path.join(pol_path, f)
                for f in os.listdir(pol_path)
                if os.path.isfile(os.path.join(pol_path, f)) and f.endswith(".npy")]

# === CALCOLA LA SOGLIA AL 95° PERCENTILE ===
def calculate_threshold(distances_list):
    all_distances = []
    for distances in distances_list:
        all_distances.extend(distances)
    threshold = np.percentile(all_distances, 95)
    print(f"[INFO] Computed 95th percentile threshold: {threshold:.6f}")
    return threshold

def dynamic_threshold(pols1, pols2, mode='combined'):
    if len(pols1) != len(pols2):
        raise ValueError(f"List lengths differ: {len(pols1)} vs {len(pols2)}")

    distances = []
    threshold = None
    for pol1, pol2 in zip(pols1, pols2): 
        p1 = np.load(pol1)
        p2 = np.load(pol2)
        threshold = np.mean(p1) + 3 * np.std(p1)
        distances.append(abs(np.linalg.norm(p1) - np.linalg.norm(p2)))
    distances = np.array(distances)
    
    # Dynamic thresholding
    if mode == 'relative':
        print(max(distances), min(distances), threshold)
        score = 0.0 if np.any(distances < threshold) else 1.0
    elif mode == 'quantile':
        threshold = np.quantile(distances, 0.95)
        score = np.mean(distances < threshold)
    else: 
        raise Exception(f"Unknown mode: {mode}")

    return 1.0 - float(score)

# === MAIN ===
def main(run_official, run_noseed, mode):
    # Trova file .npy
    pols_official = find_pol(run_official)
    pols_noseed = find_pol(run_noseed)

    score = dynamic_threshold(pols_official, pols_noseed, mode=mode)

    if score > 0.99:    
        print(f"✅ Safe / equivalent run ({score})")
    else:
        print(f"❌ Divergent or uncertified run ({score})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate multiple PVS4Provenance runs with automatic threshold")
    parser.add_argument("run_official", help="Path to official run folder")
    parser.add_argument("run_claimed", help="Path to unconfirmed run")
    parser.add_argument("mode", help="[relative|quantile]")
    args = parser.parse_args()

    main(args.run_official, args.run_claimed, args.mode)
