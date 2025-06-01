import pytest
import numpy as np
import pickle
import atexit
import shutil
import os
import zarr
import numcodecs
import zarr.codecs
import numcodecs.zarr3
import netCDF4
import itertools

write_dir = os.path.join('test', 'write')
read_dir = os.path.join('test', 'read')

dataset = None
chunk_sizes = set()
zarr_compressors = set()
netcdf4_compressors = set()
data_distributions = set()

def move_datasets(): # Move and rename files to avoid cache-related side effects in benchmarks.
    if not os.path.exists(write_dir):
        raise SystemExit(f"Missing {write_dir} directory. Execute the write benchmark first.")
    
    # if os.path.exists(read_dir):
    #     shutil.rmtree(read_dir)
    
    os.makedirs(read_dir, exist_ok=True)
    for file_name in os.listdir(write_dir):
        match file_name[-3:]:
            case 'arr':
                name = file_name[:-5]
                params = name.split('_')
                data_distributions.add(params[2])
                chunk_sizes.add(int(params[3]))
                if params[4] == 'blosc':
                    zarr_compressors.add('blosc_' + params[5])
                else:
                    zarr_compressors.add(params[4])
                shutil.move(os.path.join(write_dir, file_name), os.path.join(read_dir, name + '_read.zarr'))

            case '.nc':
                name = file_name[:-3]
                params = name.split('_')
                data_distributions.add(params[2])
                chunk_sizes.add(int(params[3]))
                netcdf4_compressors.add(params[4])
                shutil.move(os.path.join(write_dir, file_name), os.path.join(read_dir, name + '_read.nc'))

            case 'txt':
                name = file_name[:-4]
                params = name.split('_')
                data_distributions.add(params[2])
                chunk_sizes.add(int(params[3]))
                shutil.move(os.path.join(write_dir, file_name), os.path.join(read_dir, name + '_read.txt'))

move_datasets()

@pytest.mark.benchmark(group="zarr_read_benchmark")
@pytest.mark.parametrize("chunk_size", chunk_sizes)
@pytest.mark.parametrize("compressor_name", zarr_compressors)
@pytest.mark.parametrize("data_distribution", data_distributions)
def test_zarr_read_benchmark(benchmark, chunk_size, compressor_name, data_distribution):
    global dataset

    file_name = os.path.join('test', 'read', f'zarr_test_{data_distribution}_{chunk_size}_{compressor_name}_read.zarr')

    def setup():
        global dataset
        if dataset:
            dataset.store.close()

        dataset = zarr.open(file_name, mode='r')


    def read_data():
        global dataset
        for key in dataset.array_keys():
            array = dataset[key]
            shape = array.shape

            for chunk_index in range(0, shape[0], chunk_size):
                chunk = array[chunk_index : chunk_index + chunk_size]
                for i in range(chunk_size):
                    chunk[i]

    benchmark.pedantic(read_data, setup=setup, iterations=1, rounds=5, warmup_rounds=1)

    dataset.store.close()
    dataset = None


@pytest.mark.benchmark(group="netcdf_read_benchmark")
@pytest.mark.parametrize("chunk_size", chunk_sizes)
@pytest.mark.parametrize("compressor_name", netcdf4_compressors)
@pytest.mark.parametrize("data_distribution", ['uniform', 'normal', 'exponential'])
def test_netcdf_read_benchmark(benchmark, chunk_size, compressor_name, data_distribution):
    global dataset

    file_name = os.path.join('test', 'read', f'netcdf_test_{data_distribution}_{chunk_size}_{compressor_name}_read.nc')

    def setup():
        global dataset
        if dataset:
            dataset.close()

        dataset = netCDF4.Dataset(file_name, mode='r')

    def read_data():
        global dataset
        for key in dataset.variables:
            var = dataset.variables[key]
            shape = var.shape

            for chunk_index in range(0, shape[0], chunk_size):
                chunk = var[chunk_index : chunk_index + chunk_size]
                for i in range(chunk_size):
                    chunk[i]

    benchmark.pedantic(read_data, setup=setup, iterations=1, rounds=5, warmup_rounds=1)

    dataset.close()
    dataset = None

@pytest.mark.benchmark(group="txt_read_benchmark")
@pytest.mark.parametrize("chunk_size", chunk_sizes)
@pytest.mark.parametrize("data_distribution", data_distributions)
def test_txt_read_benchmark(benchmark, chunk_size, data_distribution):
    global dataset

    file_name = os.path.join('test', 'read', f'txt_test_{data_distribution}_{chunk_size}_read.txt')

    def setup():
        global dataset
        if dataset:
            dataset.close()

        dataset = open(file_name, 'r')

    def read_data():
        global dataset
        eof = False
        while not eof:
            for i in range(chunk_size):
                line = dataset.readline()
                if not line:
                    eof = True
                    break
                
                epochs, values, timestamps = line.strip().split(',')
                epochs = int(epochs)
                values = float(values)
                timestamps = int(timestamps)

    benchmark.pedantic(read_data, setup=setup, iterations=1, rounds=5, warmup_rounds=1)

    dataset.close()
    dataset = None
