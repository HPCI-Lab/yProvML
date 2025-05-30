# Benchmarks

This directory contains benchmark tests aimed at evaluating the performance and efficiency of different compressors when used with Zarr and NetCDF datasets.

The goal is to identify the most suitable compression algorithms for our specific use case, in terms of speed and compression ratio.

Benchmarks are executed automatically by pytest, which parameterizes the variables chunk size, data distribution, and compression algorithm, running a test for every possible combination. Each test includes a warm-up round, and the test time is taken as the average of 5 runs.

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
pytest write_benchmark.py --benchmark-save=write_benchmark
```