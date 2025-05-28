import pytest
import numpy as np
import os
import zarr
import numcodecs
import zarr.codecs
import numcodecs.zarr3

total_size = 100000
chunk_sizes = [5000, 10000]

datasets = {}

compressors = {
    'uncompressed': None,
    'blosc_lz4': zarr.codecs.BloscCodec(cname='lz4'),
    'blosc_lz4hc': zarr.codecs.BloscCodec(cname='lz4hc'),
    'blosc_blosclz': zarr.codecs.BloscCodec(cname='blosclz'),
    'blosc_zstd': zarr.codecs.BloscCodec(cname='zstd'),
    'blosc_zlib': zarr.codecs.BloscCodec(cname='zlib'),
    'bz2': numcodecs.zarr3.BZ2(),
    'gzip': zarr.codecs.GzipCodec(level=5),
    'lzma': numcodecs.zarr3.LZMA(),
    'lz4': numcodecs.zarr3.LZ4(),
    'zlib': numcodecs.zarr3.Zlib(),
    'zstd': zarr.codecs.ZstdCodec(level=5),
}

def generate_chunks():
    global datasets
    for chunk_size in chunk_sizes:
        num_chunks = total_size // chunk_size
        chunks_data = []
        for _ in range(num_chunks):
            chunk_i4 = np.random.randint(0, 1000, size=chunk_size, dtype=np.int32)
            chunk_f4 = np.random.rand(chunk_size).astype(np.float32)
            chunk_i8 = np.random.randint(0, 1_000_000, size=chunk_size, dtype=np.int32)
            chunks_data.append((chunk_i4, chunk_f4, chunk_i8))
        datasets[chunk_size] = chunks_data

generate_chunks()

@pytest.mark.parametrize("chunk_size", chunk_sizes)
@pytest.mark.parametrize("compressor_name, compressor", compressors.items(), ids=[name for name in compressors.keys()])
def test_zarr_write_benchmark(benchmark, chunk_size, compressor_name, compressor):
    global datasets

    chunks_data = datasets[chunk_size]

    folder = os.path.join("test", f"zarr_test_{compressor_name}_chunk_{chunk_size}")
    dataset = zarr.open(folder, mode='w')

    for key, dtype in [('arr_i4', np.int32), ('arr_f4', np.float32), ('arr_i8', np.int64)]:
        dataset.create_array(
            name=key,
            shape=(0,),
            chunks=(chunk_size,),
            dtype=dtype,
            compressors=compressor
        )

    def write_data():
        for chunk_i4, chunk_f4, chunk_i8 in chunks_data:
            for key, data in zip(['arr_i4', 'arr_f4', 'arr_i8'], [chunk_i4, chunk_f4, chunk_i8]):
                dataset[key].append(data)

    benchmark.pedantic(write_data, iterations=1, rounds=5, warmup_rounds=1)

    total_size_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(folder)
        for f in filenames
    )
    size_mb = total_size_bytes / (1024 * 1024)

    benchmark.extra_info["disk_size_mb"] = round(size_mb, 3)
