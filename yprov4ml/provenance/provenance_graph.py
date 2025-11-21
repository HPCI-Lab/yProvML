
import os
import sys
import prov
import prov.model as prov
from rocrate.rocrate import ROCrate
from pathlib import Path

from yprov4ml.constants import PROV4ML_DATA
from yprov4ml.datamodel.metric_type import get_file_type

def create_prov_document() -> prov.ProvDocument:
    
    doc = PROV4ML_DATA.root_provenance_doc
    # run_activity = get_activity(doc, "context:" + PROV4ML_DATA.EXPERIMENT_NAME)
    file_type = get_file_type(PROV4ML_DATA.metrics_file_type)

    for (name, ctx) in PROV4ML_DATA.metrics.keys():
        metric_file_path = os.path.join(PROV4ML_DATA.METRIC_DIR, name + "_" + str(ctx) + f"_GR{PROV4ML_DATA.global_rank}" + file_type)
        s = PROV4ML_DATA.metrics[(name, ctx)].source
        e = PROV4ML_DATA.add_artifact(name,metric_file_path,0,ctx, is_input=False, log_copy_in_prov_directory=False)
        
        e.add_attributes({
            f'{PROV4ML_DATA.PROV_PREFIX}:context': str(ctx),
            f'{PROV4ML_DATA.PROV_PREFIX}:source': str(s)
        })

    return doc

def get_properties_from_file(file : str):
    if file.endswith(".dot"): 
        return {
            "name": "pygraphviz provenance graph file",
            "encodingFormat": "application/dot"
        }
    elif file.endswith(".csv"): 
        return {
            "name": "metric",
            "encodingFormat": "text/csv"
        }
    elif file.endswith(".svg"): 
        return {
            "name": "pygraphviz svg provenance graph file",
            "encodingFormat": "image/svg+xml"
        }
    elif file.endswith(".json") and "/" not in file: 
        return {
            "name": "provenance JSON file",
            "encodingFormat": "text/json"
        }
    elif file.endswith(".json") and "/" in file: 
        return {
            "name": "JSON property description",
            "encodingFormat": "text/json"
        }
    elif file.endswith(".pt") or file.endswith(".pth"): 
        return {
            "name": "pytorch model checkpoint",
            "encodingFormat": "application/octet-stream"
        }
    elif file.endswith(".py"): 
        return {
            "name": "python source file",
            "encodingFormat": "text/plain"
        }
    else: 
        return {
            "name": file,
            "encodingFormat": f"{file.split('.')[-1]}",
        }

def create_rocrate_in_dir(directory): 
    crate = ROCrate()

    for (d, _, fs) in os.walk(directory): 
        for f in fs: 
            file_path = os.path.join(d, f)
            if Path(file_path).exists():
                property = get_properties_from_file(file_path)
                property["@type"] = "File" 
                property["@id"] = file_path
                crate.add_file(file_path, dest_path=file_path, properties=property)

    # crate.write("exp_crate")
    crate.write_zip(f"{directory}.zip")