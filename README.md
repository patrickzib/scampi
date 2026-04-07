# SCAMPI - Fast Pattern Discovery in Massive Time Series

This repository provides the reference implementation and experimental material 
for the paper **“SCAMPI - Fast Pattern Discovery in Massive Time Series”**.

## Repository Structure

This repository contains the full framework, benchmark datasets, and reproducible 
experiments used in the evaluation.


- `motiflets/`  
  Core implementation of the k-Motiflets algorithm.

- `notebooks/`  
  Jupyter notebooks demonstrating typical use cases and reproducing paper figures.

- `datasets/momp/`  
  large-Scale Benchmark time series datasets used throughout the paper.  
  Please see https://sites.google.com/view/momp2024, too

- `tests/`  
  Unit tests for the implementation, and running the benchmarks
 
- `tests/csvs/`  
  Raw experimental results for all competing methods.

## SCAMPI (SCalable Anytime Mining of Patterns In time series)

This paper introduces SCAMPI 
(scalable Anytime Mining of Pat terns under Euclidean Distance). 

## Installation

The easiest is to use pip to install motiflets.

### a) Install using pip
```
BLINDED
```

You can also install  the project from source.

### b) Build from Source

Clone the repository:

```bash
git clone https://github.com/patrickzib/scampi.git
cd scampi
````

Install the package:

```bash
pip install .
```

## Usage Example

```python
from scampi.scampi import *

ml = SCAMPI(
    ds_name,  # dataset name
    series,   # time series data
    n_jobs,    # number of CPU cores
    scampi_max_memory = "1 GB"  # max-memory to use for SCAMPI
)

k_max = 20  # maximum motif set size to consider
motif_length = 100  # length of the motifs to search for
top_N = 10  # number of top motif sets to return

dists, candidates, elbow_points = ml.fit_k_elbow(
    k_max,
    motif_length,
    top_N=top_N
)

ml.plot_motifset()
```
