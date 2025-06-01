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

total_size = 100000
chunk_sizes = [50000, 10000]
data_distributions = ['uniform', 'normal', 'exponential']

dataset = None
distribution = {}
current_data_distribution = ''

data_path = {
    'uniform': os.path.join('data', 'uniform_data.pkl'),
    'normal': os.path.join('data', 'normal_data.pkl'),
    'exponential': os.path.join('data', 'exponential_data.pkl')
}

def delete_distribution():
    os.remove(data_path['uniform'])
    os.remove(data_path['normal'])
    os.remove(data_path['exponential'])

def generate_distribution():
    # Uniform distribution
    data = {
        'epochs': np.random.randint(low=1, high=1000, size=total_size).astype(np.int32),
        'values': np.random.uniform(low=0.0, high=100.0, size=total_size).astype(np.float32),
        'timestamps': np.random.randint(low=1, high=1_000_000, size=total_size).astype(np.int32)
    }
    pickle.dump(data, open(data_path['uniform'], 'wb'))

    # Normal distribution
    data = {
        'epochs': np.abs(np.random.normal(loc=500, scale=200, size=total_size)).astype(np.int32),
        'values': np.abs(np.random.normal(loc=50.0, scale=20.0, size=total_size)).astype(np.float32),
        'timestamps': np.abs(np.random.normal(loc=500_000, scale=200_000, size=total_size)).astype(np.int32)
    }
    pickle.dump(data, open(data_path['normal'], 'wb'))

    # Exponential distribution
    data = {
        'epochs': np.random.exponential(scale=300.0, size=total_size).astype(np.int32),
        'values': np.random.exponential(scale=30.0, size=total_size).astype(np.float32),
        'timestamps': np.random.exponential(scale=300_000.0, size=total_size).astype(np.int32)
    }
    pickle.dump(data, open(data_path['exponential'], 'wb'))

def load_distribution(data_distribution) -> dict:
    global distribution
    global current_data_distribution

    if data_distribution != current_data_distribution:
        current_data_distribution = data_distribution

        match data_distribution:
            case 'uniform':
                distribution = pickle.load(open(data_path['uniform'], 'rb'))

            case 'normal':
                distribution = pickle.load(open(data_path['normal'], 'rb'))

            case 'exponential':
                distribution = pickle.load(open(data_path['exponential'], 'rb'))

    return distribution

atexit.register(delete_distribution)
generate_distribution()
if os.path.exists('test'):
    shutil.rmtree('test')

compression_level = 5
zarr_compressors = {
    'uncompressed': None,
    'blosc_lz4': zarr.codecs.BloscCodec(cname='lz4', clevel=compression_level, shuffle=zarr.codecs.BloscShuffle.shuffle),
    'blosc_lz4hc': zarr.codecs.BloscCodec(cname='lz4hc', clevel=compression_level, shuffle=zarr.codecs.BloscShuffle.shuffle),
    'blosc_blosclz': zarr.codecs.BloscCodec(cname='blosclz', clevel=compression_level, shuffle=zarr.codecs.BloscShuffle.shuffle),
    'blosc_zstd': zarr.codecs.BloscCodec(cname='zstd', clevel=compression_level, shuffle=zarr.codecs.BloscShuffle.shuffle),
    'blosc_zlib': zarr.codecs.BloscCodec(cname='zlib', clevel=compression_level, shuffle=zarr.codecs.BloscShuffle.shuffle),
    'bz2': numcodecs.zarr3.BZ2(level=compression_level),
    'gzip': zarr.codecs.GzipCodec(level=compression_level),
    'lzma': numcodecs.zarr3.LZMA(preset=compression_level),
    'lz4': numcodecs.zarr3.LZ4(acceleration=compression_level),
    'zlib': numcodecs.zarr3.Zlib(level=compression_level),
    'zstd': zarr.codecs.ZstdCodec(level=compression_level),
}

netcdf4_compressors = {
    'uncompressed': None,
    'zlib': 'zlib'
}

@pytest.mark.benchmark(group="zarr_write_benchmark")
@pytest.mark.parametrize("chunk_size", chunk_sizes)
@pytest.mark.parametrize("compressor_name, compressor", zarr_compressors.items(), ids=[name for name in zarr_compressors.keys()])
@pytest.mark.parametrize("data_distribution", data_distributions)
def test_zarr_write_benchmark(benchmark, chunk_size, compressor_name, compressor, data_distribution):
    global dataset

    data = load_distribution(data_distribution)
    file_name = os.path.join('test', 'write', f'zarr_test_{data_distribution}_{chunk_size}_{compressor_name}.zarr')

    def setup():
        global dataset
        if os.path.exists(file_name):
            dataset.store.close()
            shutil.rmtree(file_name)

        dataset = zarr.open(file_name, mode='w')

        for key, dtype in [('epochs', np.int32), ('values', np.float32), ('timestamps', np.int32)]:
            dataset.create_array(
                name=key,
                shape=(0,),
                chunks=(chunk_size,),
                dtype=dtype,
                compressors=compressor
            )

    def write_data():
        global dataset
        num_chunks = total_size // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            for key in data.keys():
                dataset[key].append(data[key][start:end])

    benchmark.pedantic(write_data, setup=setup, iterations=1, rounds=5, warmup_rounds=1)

    dataset.store.close()

    total_size_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(file_name)
        for f in filenames
    )
    size_mb = total_size_bytes / (1024 * 1024)

    benchmark.extra_info["disk_size_mb"] = round(size_mb, 3)

@pytest.mark.benchmark(group="netcdf_write_benchmark")
@pytest.mark.parametrize("chunk_size", chunk_sizes)
@pytest.mark.parametrize("compressor_name, compressor", netcdf4_compressors.items(), ids=[name for name in netcdf4_compressors.keys()])
@pytest.mark.parametrize("data_distribution", data_distributions)
def test_netcdf_write_benchmark(benchmark, chunk_size, compressor_name, compressor, data_distribution):
    global dataset

    data = load_distribution(data_distribution)
    file_name = os.path.join('test', 'write', f'netcdf_test_{data_distribution}_{chunk_size}_{compressor_name}.nc')

    def setup():
        global dataset
        if os.path.exists(file_name):
            dataset.close()
            os.remove(file_name)

        dataset = netCDF4.Dataset(file_name, 'w')

        dataset.createDimension('time', None)
        dataset.createVariable('epochs', 'i4', ('time',), compression=compressor, complevel=compression_level)
        dataset.createVariable('values', 'f4', ('time',), compression=compressor, complevel=compression_level)
        dataset.createVariable('timestamps', 'i4', ('time',), compression=compressor, complevel=compression_level)

    def write_data():
        global dataset
        num_chunks = total_size // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            current_size = dataset.dimensions['time'].size
            new_size = current_size + chunk_size
            for key in data.keys():
                dataset.variables[key][current_size:new_size] = data[key][start:end]

    benchmark.pedantic(write_data, setup=setup, iterations=1, rounds=5, warmup_rounds=1)

    dataset.close()

    total_size_bytes = os.path.getsize(file_name)
    size_mb = total_size_bytes / (1024 * 1024)

    benchmark.extra_info["disk_size_mb"] = round(size_mb, 3)

@pytest.mark.benchmark(group="txt_write_benchmark")
@pytest.mark.parametrize("chunk_size", chunk_sizes)
@pytest.mark.parametrize("data_distribution", data_distributions)
def test_txt_write_benchmark(benchmark, chunk_size, data_distribution):
    global dataset

    data = load_distribution(data_distribution)
    file_name = os.path.join('test', 'write', f'txt_test_{data_distribution}_{chunk_size}.txt')

    def setup():
        global dataset
        if os.path.exists(file_name):
            dataset.close()
            os.remove(file_name)

        dataset = open(file_name, 'w')

    def write_data():
        global dataset
        num_chunks = total_size // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            for epoch, value, timestamp in zip(data['epochs'][start:end], data['values'][start:end], data['timestamps'][start:end]):
                dataset.write(f"{epoch}, {value}, {timestamp}\n")

    benchmark.pedantic(write_data, setup=setup, iterations=1, rounds=5, warmup_rounds=1)

    dataset.close()

    total_size_bytes = os.path.getsize(file_name)
    size_mb = total_size_bytes / (1024 * 1024)

    benchmark.extra_info["disk_size_mb"] = round(size_mb, 3)