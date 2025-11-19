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

def dot_distance(weights1, weights2): 
    return (1 - np.dot(weights1, weights2) / (np.linalg.norm(weights1) * np.linalg.norm(weights1)))#.cpu().numpy()


def dynamic_threshold(pols1, pols2, mode='relative'):
    if len(pols1) != len(pols2):
        raise ValueError(f"List lengths differ: {len(pols1)} vs {len(pols2)}")

    distances = []
    threshold = None
    for pol1, pol2 in zip(pols1, pols2): 
        p1 = np.load(pol1)
        p2 = np.load(pol2)
        threshold = np.mean(p1) + 3 * np.std(p1)#max(threshold, np.mean(p1) + 3 * np.std(p1))
        p1, p2 = np.linalg.norm(p1), np.linalg.norm(p2)
        distances.append(dot_distance(p1, p2))#abs(p1 - p2))
    distances = np.array(distances)
    # import pandas as pd
    # threshold = pd.Series([np.mean(np.load(p)) for p in pols1]).diff()
    # threshold = threshold.max() + np.std(threshold) * 3
    
    # Dynamic thresholding
    if mode == 'relative':
        print(max(distances), min(distances), threshold)
        perc = np.mean(distances > threshold)
        print(f"PERC: {perc}")
        score = 0.0 if np.any(distances > threshold) else 1.0
    elif mode == 'quantile':
        threshold = np.quantile(distances, 0.95) #+ 3*np.std(distances)
        print(max(distances), min(distances), threshold)
        perc = np.mean(distances > threshold)
        print(f"PERC: {perc}")
        score = 0.0 if np.any(distances > threshold) else 1.0
    else: 
        raise Exception(f"Unknown mode: {mode}")

    return float(score)

# === MAIN ===
def main(run_official, run_noseed):
    # Trova file .npy
    pols_official = find_pol(run_official)
    pols_noseed = find_pol(run_noseed)

    score = dynamic_threshold(pols_official, pols_noseed, mode="quantile")

    if score > 0.99:    
        print(f"✅ Safe / equivalent run ({score})")
    else:
        print(f"❌ Divergent or uncertified run ({score})")

# from utils import parameter_distance
# def main(run_official, run_noseed):
#     # Trova file .npy
#     pols_official = find_pol(run_official)
#     pols_noseed = find_pol(run_noseed)

#     order = ['1', '2', 'inf', 'cos']
#     threshold = [1000, 30, 0.5, 1.0]

#     if not isinstance(order, list):
#         order = [order]
#         threshold = [threshold]
#     else:
#         assert len(order) == len(threshold)

#     dist_list = [[] for i in range(len(order))]
#     for i in range(len(pols_official)): 
#         pol_official = np.load(pols_official[i])
#         pol_noseed = np.load(pols_noseed[i])
#         res = parameter_distance(pol_official, pol_noseed, order=order)
#         for j in range(len(order)):
#             dist_list[j].append(res[j])

#     dist_list = np.array(dist_list)
#     for k in range(len(order)):
#         print(f"Distance metric: {order[k]} || threshold: {threshold[k]}")
#         print(f"Average distance: {np.average(dist_list[k])}")
#         above_threshold = np.sum(dist_list[k] > threshold[k])
#         if above_threshold == 0:
#             print("None of the steps is above the threshold, the proof-of-learning is valid.")
#         else:
#             print(f"{above_threshold} / {dist_list[k].shape[0]} "
#                 f"({100 * np.average(dist_list[k] > threshold[k])}%) "
#                 f"of the steps are above the threshold, the proof-of-learning is invalid.")
#     return dist_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate multiple PVS4Provenance runs with automatic threshold")
    parser.add_argument("run_official", help="Path to official run folder")
    parser.add_argument("run_claimed", help="Path to unconfirmed run")
    # parser.add_argument("mode", help="[relative|quantile]")
    args = parser.parse_args()

    main(args.run_official, args.run_claimed)
