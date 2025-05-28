# Benchmarks

This directory contains benchmark tests aimed at evaluating the performance and efficiency of different compressors when used with Zarr and NetCDF datasets.

The goal is to identify the most suitable compression algorithms for our specific use case, in terms of speed and compression ratio.

## Requirements

The benchmarks are implemented using [pytest](https://docs.pytest.org/) along with the [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) plugin.

To install the necessary dependencies, run:

```bash
cd benchmark
pip install -r requirements.txt
```

## Running the Benchmarks

Once the dependencies are installed, you can run the benchmarks using:

```bash
pytest zarr_write_benchmark.py --benchmark-save=zarr_write
```