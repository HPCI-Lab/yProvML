import pandas as pd
import numpy as np
import argparse
import os

def validate_exact(pols1, pols2): 
    f1 = pd.read_csv(pols1, sep=";")
    f2 = pd.read_csv(pols2, sep=";")

    f1["Context.TRAINING"] = f1["Context.TRAINING"].map(lambda x: eval(x))
    f2["Context.TRAINING"] = f2["Context.TRAINING"].map(lambda x: eval(x))
    proof1 = pd.DataFrame(f1["Context.TRAINING"].tolist(), columns=["step", "loss", "weights_hash"])
    proof2 = pd.DataFrame(f2["Context.TRAINING"].tolist(), columns=["step", "loss", "weights_hash"])

    for step in range(len(proof1)):
        p1 = proof1.iloc[step]
        p2 = proof2.iloc[step]
        if p1["weights_hash"] != p2["weights_hash"]:
            print(f"❌ Mismatch at step {step} in validate_exact")
            return False
    else:
        print("✅ Perfect match — proof validated!")
    return True

def find_pol(path): 
    pol1 = os.path.join(path, [p for p in os.listdir(path) if "_pol_" in p][0])
    if pol1.endswith(".csv"): 
        return pol1
    else: 
        pols1 = [os.path.join(pol1, f) for f in os.listdir(pol1)]
        return pols1

def validate_threshold(pols1, pols2): 
    distaces = []
    for pol1, pol2 in zip(pols1, pols2): 
        p1 = np.load(pol1)
        p2 = np.load(pol2)
        distaces.append(np.linalg.norm(p1 - p2))

    import matplotlib.pyplot as plt
    plt.plot(distaces)
    plt.show()


def main(f1, f2): 
    proof_path1 = os.path.join(f1, "metrics_GR0")
    proof_path2 = os.path.join(f2, "metrics_GR0")
    pols1 = find_pol(proof_path1)
    pols2 = find_pol(proof_path2)
    
    valid = validate_exact(pols1, pols2)

    proof_path1 = os.path.join(f1, "artifacts_GR0")
    proof_path2 = os.path.join(f2, "artifacts_GR0")
    pols1 = find_pol(proof_path1)
    pols2 = find_pol(proof_path2)

    valid = validate_threshold(pols1, pols2)



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('filename_1')
    parser.add_argument('filename_2')
    args = parser.parse_args()
    main(args.filename_1, args.filename_2)