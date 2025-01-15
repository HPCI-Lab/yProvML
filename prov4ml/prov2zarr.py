import json
import os
import zarr
import argparse
import numpy as np

if __package__ == None:
    print('This file must be run as a module using "python -m prov4ml.prov2netCDF -h" and not directly.')
    exit()

from prov4ml.utils.prov_getters import get_metrics, get_metric_numpy
from prov4ml.utils.compress_utils import compress_file, print_file_size

def json_to_zarr(json_file, zarr_file):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    metrics = get_metrics(data, "TRAINING")
    metrics = [get_metric_numpy(data, m) for m in metrics] # [[epochs, values, time, size], [.., .., .., ..], ...]

    # Determine groups dimension
    unique_sizes = list(set([len(m[0]) for m in metrics]))

    # Stack metrics with equal sizes
    epochs = {}
    values = {}
    times = {}
    num_metrics_per_size = {}    
    for metric in metrics: 
        size = metric[3]

        # Save how many metrics we have for each size
        if size in num_metrics_per_size: 
            num_metrics_per_size[size] += 1
        else: 
            num_metrics_per_size[size] = 1
            flag = True

        # Stack
        if flag:
            epochs[size] = metric[0].copy()
            values[size] = metric[1].copy()
            times[size] = metric[2].copy()
            flag = False
        else:
            epochs[size] = np.vstack((epochs[size], metric[0]))
            values[size] = np.vstack((values[size], metric[1]))
            times[size] = np.vstack((times[size], metric[2]))

    # Resize metrics with only one row
    for size in unique_sizes:
        if epochs[size].ndim == 1:
            epochs[size].resize(1, size)
            values[size].resize(1, size)
            times[size].resize(1, size)

    # Create zarr file
    dataset = zarr.open(zarr_file, mode='w')

    groups = {size: dataset.create_group(f"metric_granularity_{size}") for size in unique_sizes}

    # Populate dataset
    for size, group in groups.items(): 

        chunks = size if size < 10000 else 10000 # To refine

        group.create_dataset(name='epochs', shape=(num_metrics_per_size[size],size), chunks=(num_metrics_per_size[size],chunks), dtype='i4')
        group.create_dataset(name='values', shape=(num_metrics_per_size[size],size), chunks=(num_metrics_per_size[size],chunks), dtype='f4')
        group.create_dataset(name='timestamps', shape=(num_metrics_per_size[size],size), chunks=(num_metrics_per_size[size],chunks), dtype='i8')

        group['epochs'] = epochs[size]
        group['values'] = values[size]
        group['timestamps'] = times[size]

    # Add metadata

    dataset.attrs['description'] = 'lol'
    print(dataset.attrs['description'])

    print(dataset.info)
    print(dataset.tree())

    print(f'Zarr file "{zarr_file}" created successfully.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='input file path, must be .json', required=True)
    parser.add_argument('-o', '--output', help='output file path, if missing defaults to <input>.zarr', required=False)

    args = parser.parse_args()
    
    input_file: str = os.path.abspath(args.input)
    output_file: str

    if not os.path.isfile(input_file):
        print("Input is not a valid file")
        exit()

    if not input_file.endswith('.json'):
        print("File type must be .json")
        exit()

    if args.output:
        output_file = os.path.abspath(args.output)

        if not output_file.endswith('.zarr'):
            output_file += '.zarr'
    else:
        output_file = input_file.replace('.json', '.zarr')

    return input_file, output_file

if __name__ == "__main__":

    input_file, output_file = parse_args()

    json_to_zarr(input_file, output_file)
    compress_file(output_file, output_file)

    print_file_size(input_file)
    print_file_size(output_file)
    print_file_size(output_file + '.tar.gz')
