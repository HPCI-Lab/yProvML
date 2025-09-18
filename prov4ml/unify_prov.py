import os
import argparse
from glob import glob
from prov.model import ProvDocument, ProvActivity, ProvAgent, QualifiedName, Namespace
from prov4ml.utils.file_utils import custom_prov_to_dot
import re

# ----------------------------- Helpers -----------------------------

HOUSEKEEPING_LOCALPARTS = {
    "local_rank", "node_rank"
}

def _root_activities(d: ProvDocument):
    roots = []
    for rec in d.get_records():
        if isinstance(rec, ProvActivity):
            lvl = None
            for (qname, val) in rec.attributes:
                if isinstance(qname, QualifiedName) and qname.localpart == "level":
                    try:
                        if int(val) == 0:
                            lvl = 0
                    except Exception:
                        pass
            if lvl == 0:
                roots.append(rec)
    if not roots:
        for rec in d.get_records():
            if isinstance(rec, ProvActivity) and str(rec.identifier).startswith("context:"):
                roots.append(rec)
    return roots

def _agents(d: ProvDocument):
    return [rec for rec in d.get_records() if isinstance(rec, ProvAgent)]

def _collect_params_tuples_from_root(root_act):
    items = []
    for (qname, val) in root_act.attributes:
        # copy ANY attribute that is not in housekeeping
        if hasattr(qname, "localpart"):
            lp = qname.localpart
            if lp in HOUSEKEEPING_LOCALPARTS:
                continue
            ns = qname.namespace
            if getattr(ns, "prefix", None) and getattr(ns, "uri", None):
                items.append((ns.prefix, ns.uri, lp, str(val)))
            else:
                items.append(("param", "urn:param", lp, str(val))) 
    return items

def _attach_params(unified: ProvDocument, activity, params_tuples):
    """Ensure prefixes exist; attach attributes to 'activity' as QualifiedNames."""
    # base namespaces
    unified.add_namespace("context", "context")
    unified.add_namespace("prov", "http://www.w3.org/ns/prov#")
    unified.add_namespace("xsd", "http://www.w3.org/2000/10/XMLSchema#")
    # register needed prefixes
    for (prefix, uri, _, _) in params_tuples:
        try:
            unified.add_namespace(prefix, uri)
        except Exception:
            pass
    # build attributes
    attr_dict = {}
    for (prefix, _, lp, val) in params_tuples:
        qn_str = f"{prefix}:{lp}"
        try:
            qn = unified.valid_qualified_name(qn_str) or unified.mandatory_valid_qname(qn_str)
        except Exception:
            unified.add_namespace("param", "urn:param")
            qn = unified.mandatory_valid_qname(f"param:{lp}")
        attr_dict[qn] = val
    if attr_dict:
        activity.add_attributes(attr_dict)

def _qname_or(unified: ProvDocument, raw: str, fallback_qname: str):
    """Try to reuse a record's original identifier; otherwise use a safe fallback like context:..."""
    qn = unified.valid_qualified_name(raw)
    if qn is not None:
        return qn
    return unified.mandatory_valid_qname(fallback_qname)

def _agent_qname(unified: ProvDocument, ag_id: str):
    """Make a safe agent QualifiedName."""
    qn = unified.valid_qualified_name(ag_id)
    if qn:
        return qn
    import re
    unified.add_namespace("user", "urn:user")
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', ag_id) or "unknown"
    return unified.mandatory_valid_qname(f"user:{safe}")


# ----------------------------- unify -----------------------------
def unify_prov_json(
    experiment_name: str,
    prov_save_path: str = "prov",
    output_json: str | None = None,
    make_dot: bool = False,
    make_svg: bool = False,
    level: str = "high",
):
    """
    Unify PROV-JSONs for an experiment.

    level = "high": full merged PROV-JSON (default).
    level = "low" : compact doc with only a single experiment activity,
                    associated user agent(s), and merged parameters.
    """
    # Look for prov/<exp>_* and prov/<exp>
    run_dirs = sorted(
        d for d in (
            glob(os.path.join(prov_save_path, f"{experiment_name}_*"))
            + [os.path.join(prov_save_path, experiment_name)]
        )
        if os.path.isdir(d)
    )

    docs = []
    for d in run_dirs:
        for f in os.listdir(d):
            if f.startswith("prov_") and f.endswith(".json"):
                path = os.path.join(d, f)
                with open(path, "r") as fh:
                    docs.append(ProvDocument.deserialize(fh))

    if not docs:
        raise RuntimeError(f"No prov JSON files found for '{experiment_name}' in '{prov_save_path}'")

    # ------------- low level (compact) -------------
    if level == "low":
        # Build a compact doc that still keeps EACH RUN separate.
        unified = ProvDocument()
        unified.add_namespace("context", "context")
        unified.add_namespace("prov", "http://www.w3.org/ns/prov#")
        unified.add_namespace("xsd", "http://www.w3.org/2000/10/XMLSchema#")
        unified.add_namespace("yProv4ML", "yProv4ML")
        unified.add_namespace("param", "urn:param")  # used as fallback for bare attrs

        # Parent summary node
        parent = unified.activity(f"context:{experiment_name}_UNIFIED")

        run_idx = 0
        for d in docs:
            roots = _root_activities(d)
            if not roots:
                continue
            # take first root per doc (typical: one root)
            r = roots[0]

            # Child node per run: try to keep the original id; else fallback to RUN{idx}
            child_id_raw = str(r.identifier)
            child_qn = _qname_or(unified, child_id_raw, f"context:{experiment_name}_RUN{run_idx}")
            child = unified.activity(child_qn)
            child.wasInformedBy(parent)

            # Copy ALL parameters (except housekeeping) from THIS run's root
            params_this_run = _collect_params_tuples_from_root(r)
            _attach_params(unified, child, params_this_run)

            # Associate THIS run's agents to THIS child
            for ag in _agents(d):
                ag_qn = _agent_qname(unified, str(ag.identifier))
                a = unified.agent(ag_qn)
                child.wasAssociatedWith(a)

            run_idx += 1

        # output
        out_dir = os.path.join(prov_save_path, experiment_name)
        os.makedirs(out_dir, exist_ok=True)
        if not output_json:
            output_json = os.path.join(out_dir, f"prov_{experiment_name}_UNIFIED_low.json")

        with open(output_json, "w") as f:
            f.write(unified.serialize(indent=2))

        if make_dot or make_svg:
            dot_file = output_json.replace(".json", ".dot")
            with open(dot_file, "w") as f:
                f.write(custom_prov_to_dot(unified).to_string())
            if make_svg:
                os.system(f"dot -Tsvg {dot_file} -o {output_json.replace('.json', '.svg')}")
        return output_json


    # ------------- high level (full merge) -------------
    master = ProvDocument()
    for d in docs:
        master.update(d)

    out_dir = os.path.join(prov_save_path, experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    if output_json is None:
        output_json = os.path.join(out_dir, f"prov_{experiment_name}_UNIFIED.json")

    with open(output_json, "w") as f:
        f.write(master.serialize(indent=2))

    if make_dot or make_svg:
        dot_file = output_json.replace(".json", ".dot")
        with open(dot_file, "w") as f:
            f.write(custom_prov_to_dot(master).to_string())
        if make_svg:
            os.system(f"dot -Tsvg {dot_file} -o {output_json.replace('.json', '.svg')}")
    return output_json

# ----------------------------- CLI -----------------------------
def main():
    p = argparse.ArgumentParser(description="Unify multiple PROV-JSON runs into one file")
    p.add_argument("--experiment", required=True, help="Experiment name prefix (e.g., experiment_name)")
    p.add_argument("--root", default="prov", help="Base provenance directory")
    p.add_argument("--output", default=None, help="Path for unified PROV-JSON")
    p.add_argument("--dot", action="store_true", help="Also write a .dot graph")
    p.add_argument("--svg", action="store_true", help="Also render an .svg via Graphviz")
    p.add_argument("--level", choices=["high", "low"], default="high",
                   help="high = full merged PROV (default), low = compact with only user, experiment, parameters")
    args = p.parse_args()

    out = unify_prov_json(
        experiment_name=args.experiment,
        prov_save_path=args.root,
        output_json=args.output,
        make_dot=args.dot,
        make_svg=args.svg,
        level=args.level,
    )
    print(f"Unified prov JSON written to {out}")

if __name__ == "__main__":
    main()
