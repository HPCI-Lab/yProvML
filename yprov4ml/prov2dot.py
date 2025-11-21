
import os
import argparse
from prov.model import ProvDocument
from typing import Optional

from yprov4ml.utils.file_utils import custom_prov_to_dot

def main(prov_file : str, out_file : Optional[str]): 
    if not prov_file.endswith(".json"): 
        prov_file += ".json"
    if out_file is None:
        out_file = prov_file.replace(".json", ".dot") 
    if not out_file.endswith(".dot"): 
        out_file += ".dot"

    doc = ProvDocument()
    with open(prov_file, 'r') as f:
        doc = ProvDocument.deserialize(f)

    path_dot = os.path.join(".", out_file)
    with open(path_dot, 'w') as prov_dot:
        prov_dot.write(custom_prov_to_dot(doc).to_string())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a PROV-JSON file to a DOT file')
    parser.add_argument('--prov_json', type=str, help='The PROV-JSON file to convert')
    parser.add_argument('--output', type=str, help='The output DOT file', default=None)
    args = parser.parse_args()
    
    main(args.prov_json, args.output)