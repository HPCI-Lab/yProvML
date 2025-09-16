import os
import argparse
from prov.model import ProvDocument
from prov4ml.utils.file_utils import custom_prov_to_dot

def unify_prov_json(experiment_name: str, prov_save_path: str = "prov", output_json: str = None,
                    make_dot: bool = False, make_svg: bool = False):
    """
    Collect all prov_*.json files under prov_save_path/<experiment_name>_* dirs
    and merge them into a single PROV-JSON.
    """
    from glob import glob

    run_dirs = glob(os.path.join(prov_save_path, f"{experiment_name}_*"))
    docs = []
    for d in run_dirs:
        for f in os.listdir(d):
            if f.startswith("prov_") and f.endswith(".json"):
                path = os.path.join(d, f)
                with open(path, "r") as fh:
                    docs.append(ProvDocument.deserialize(fh))

    if not docs:
        raise RuntimeError(f"No prov JSON files found for {experiment_name} in {prov_save_path}")

    # union all documents
    master = ProvDocument()
    for d in docs:
        master.update(d)

    if output_json is None:
        output_json = os.path.join(prov_save_path, f"{experiment_name}_UNIFIED.json")

    with open(output_json, "w") as f:
        master.serialize(f, format="json")

    if make_dot:
        dot_file = output_json.replace(".json", ".dot")
        with open(dot_file, "w") as f:
            f.write(custom_prov_to_dot(master).to_string())

    if make_svg:
        os.system(f"dot -Tsvg {output_json.replace('.json', '.dot')} -o {output_json.replace('.json', '.svg')}")

    return output_json

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser(description="Unify multiple PROV-JSON runs into one file")
    p.add_argument("--experiment", required=True, help="Experiment name prefix (e.g., my_experiment)")
    p.add_argument("--root", default="prov", help="Base provenance directory")
    p.add_argument("--output", default=None, help="Path for unified PROV-JSON")
    p.add_argument("--dot", action="store_true", help="Also write a .dot graph")
    p.add_argument("--svg", action="store_true", help="Also render an .svg via Graphviz")
    args = p.parse_args()

    out = unify_prov_json(args.experiment, args.root, args.output, args.dot, args.svg)
    print(f"Unified prov JSON written to {out}")

if __name__ == "__main__":
    main()
