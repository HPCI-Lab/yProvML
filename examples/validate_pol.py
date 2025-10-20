import pandas as pd
import argparse

def main(f1, f2): 
    f1 = pd.read_csv(f1, sep=";")
    f2 = pd.read_csv(f2, sep=";")

    f1["Context.TRAINING"] = f1["Context.TRAINING"].map(lambda x: eval(x))
    f2["Context.TRAINING"] = f2["Context.TRAINING"].map(lambda x: eval(x))
    proof1 = pd.DataFrame(f1["Context.TRAINING"].tolist(), columns=["step", "loss", "weights_hash"])
    proof2 = pd.DataFrame(f2["Context.TRAINING"].tolist(), columns=["step", "loss", "weights_hash"])
    for step in range(len(proof1)):
        p1 = proof1.iloc[step]
        p2 = proof2.iloc[step]
        if p1["weights_hash"] != p2["weights_hash"]:
            print(f"❌ Mismatch at step {step}")
            break
    else:
        print("✅ Perfect match — proof validated!")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('filename_1')
    parser.add_argument('filename_2')
    args = parser.parse_args()
    main(args.filename_1, args.filename_2)