import numcodecs.zarr3
import numpy as np
import zarr
import zarr.codecs
import numcodecs
import time
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

total_size = 10_000_000
chunk_sizes = [100_000, 500_000, 1_000_000]

compressors = {
    'raw_uncompressed': None,
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

all_results = []

for chunk_size in tqdm(chunk_sizes):
    num_chunks = total_size // chunk_size

    chunks_data = []
    for _ in range(num_chunks):
        chunk_i4 = np.random.randint(0, 1000, size=chunk_size, dtype=np.int32)
        chunk_f4 = np.random.rand(chunk_size).astype(np.float32)
        chunk_i8 = np.random.randint(0, 1_000_000, size=chunk_size, dtype=np.int64)
        chunks_data.append((chunk_i4, chunk_f4, chunk_i8))

    for name, compressor in compressors.items():
        folder = f'zarr_test_{name}_chunk_{chunk_size}'
        shutil.rmtree(folder, ignore_errors=True)

        dataset = zarr.open(folder, mode='w')

        for key, dtype in [('arr_i4', np.int32), ('arr_f4', np.float32), ('arr_i8', np.int64)]:
            dataset.create_array(
                name=key,
                shape=(0,),
                chunks=(chunk_size,),
                dtype=dtype,
                compressors=compressor
            )

        t0 = time.time()
        for chunk_i4, chunk_f4, chunk_i8 in chunks_data:
            for key, data in zip(['arr_i4', 'arr_f4', 'arr_i8'], [chunk_i4, chunk_f4, chunk_i8]):
                dataset[key].append(data)
        t1 = time.time()

        total_size_bytes = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(folder)
            for f in filenames
        )

        all_results.append({
            'chunk_size': chunk_size,
            'compressor': name,
            'time_sec': round(t1 - t0, 4),
            'size_kb': round(total_size_bytes / 1024, 2),
        })



df = pd.DataFrame(all_results)

uncompressed = df[df['compressor'] == 'raw_uncompressed'][['chunk_size', 'size_kb']]
uncompressed = uncompressed.rename(columns={'size_kb': 'uncompressed_size_kb'})
df = df.merge(uncompressed, on='chunk_size')
df['compression_ratio'] = df['uncompressed_size_kb'] / df['size_kb']

df['chunk_size_str'] = df['chunk_size'].astype(str)

compressors_list = sorted(df['compressor'].unique())

colors = plt.get_cmap('tab20').colors
markers = ['o', 's', '^', 'D', '*', 'x', '+', 'v', '<', '>', 'p', 'h']
linestyles = ['-', '--', ':', '-.']

color_map = {compressor: colors[i % len(colors)] for i, compressor in enumerate(compressors_list)}
marker_map = {compressor: markers[i % len(markers)] for i, compressor in enumerate(compressors_list)}
linestyle_map = {compressor: linestyles[i % len(linestyles)] for i, compressor in enumerate(compressors_list)}

def plot_metric(metric, ylabel, title):
    plt.figure(figsize=(12, 6))
    for compressor in compressors_list:
        subset = df[df['compressor'] == compressor]
        plt.plot(subset['chunk_size_str'], subset[metric],
                 marker=marker_map[compressor],
                 linestyle=linestyle_map[compressor],
                 color=color_map[compressor],
                 label=compressor)

    plt.title(title)
    plt.xlabel('Chunk Size')
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', linestyle='-', color='gray', alpha=0.5)
    plt.xticks(df['chunk_size_str'].unique())
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

plot_metric('time_sec', 'Time (seconds)', 'Write time vs Chunk Size')

plot_metric('compression_ratio', 'Compression ratio', 'Compression ratio vs Chunk Size')

plt.show()
