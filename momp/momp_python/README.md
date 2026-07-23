# MOMP Python

Standalone Python command-line app for running a MOMP-style motif-only search
without importing the SCAMPI package.

This app provides a Python port of the pruning/downsampling `momp_v9.m`
algorithm next to the MATLAB implementation in `../momp_matlab/`. It also keeps
a MATLAB-style `mpx_v2` matrix-profile search as a comparison mode.

## Install

From this directory:

```bash
python -m pip install -e .
```

Optional dependencies:

- `scipy` for reading MATLAB `.mat` files.
- `matplotlib` for `--plot`.
- `numba` for compiling the MPX diagonal recurrence loops.

## Run

```bash
python -m momp_python --input path/to/data.csv --motif-length 512
python -m momp_python --input path/to/data.npy --motif-length 512
python -m momp_python --input path/to/data.mat --mat-key recorddata --motif-length 512
python -m momp_python --input path/to/data.npy --motif-length 512 --algorithm mpx-v2
```

The app writes JSON by default:

```bash
python -m momp_python \
  --input ../example.mat \
  --mat-key recorddata \
  --motif-length 512 \
  --output result.json \
  --plot result.pdf
```

Output fields:

- `distance`: z-normalized Euclidean distance of the best motif pair.
- `locations`: zero-based motif start positions.
- `locations_matlab`: one-based motif start positions for MATLAB comparison.
- `motif_length`, `series_length`, `runtime_seconds`, and app metadata.
- `iterations` for the `momp-v9` pruning/downsampling path.

## Benchmark MOMP `.mat` Datasets

The benchmark command compares the direct `mpx-v2` matrix-profile baseline
with the `momp-v9` pruning/downsampling port on prefixes of large datasets from
`../../datasets/momp`.

```bash
python -m momp_python.benchmark \
  --data-dir ../../datasets/momp \
  --datasets EOG_one_hour_50_Hz.mat,Challenge2009TestSetA_101a.mat \
  --prefix-lengths 5000,10000 \
  --motif-lengths 256
```

Outputs are written incrementally to:

- `momp/momp_python/results/momp_python_dataset_benchmark.csv`
- `momp/momp_python/results/momp_python_dataset_benchmark.jsonl`
