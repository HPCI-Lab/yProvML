
import os
import argparse
from typing import Optional

def main(dot_file : str, out_file : Optional[str]): 
    if not dot_file.endswith(".dot"): 
        dot_file += ".dot"

    if out_file is None: 
        out_file = dot_file.replace(".dot", ".svg")
    if not out_file.endswith(".svg"): 
        out_file += ".svg"

    path_svg = os.path.join(".", out_file)
    os.system(f"dot -Tsvg {dot_file} > {path_svg}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a DOT file to a SVG file')
    parser.add_argument('--dot', type=str, help='The DOT file to convert')
    parser.add_argument('--output', type=str, help='The output SVG file', default=None)
    args = parser.parse_args()

    main(args.dot, args.output)