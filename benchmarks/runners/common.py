"""Shared helpers for dataset-specific benchmark runners."""

from pathlib import Path

import matplotlib
import matplotlib as mpl
import pandas as pd

from benchmarks import utils as ut
from scampi.scampi import SCAMPI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
mpl.rcParams["figure.dpi"] = 150


def original_path(filename):
    return DATASETS_ROOT / "original" / filename


def experiments_path(filename):
    return DATASETS_ROOT / "experiments" / filename


def read_single_column_csv(path, ds_name, header=None):
    series = pd.read_csv(path, header=header).squeeze("columns")
    print(f"Loaded dataset {ds_name} with length {len(series)}")
    return ds_name, series


def plot_dataset(read_data, output_path, backend=None, max_points=10_000):
    ds_name, series = read_data()
    kwargs = {"backend": backend} if backend is not None else {}
    ml = SCAMPI(ds_name, series, **kwargs)
    ml.plot_dataset(max_points=max_points, path=output_path)


def run_scale_n(
        read_data,
        n_range,
        l_range,
        k_max,
        backends=("scampi",),
        delta=0.1,
        subsampling=None,
        scampi_max_memory="2GB",
        **kwargs):
    for backend in backends:
        ut.test_motiflets_scale_n(
            read_data,
            n_range,
            l_range,
            k_max,
            backend=backend,
            subsampling=subsampling,
            scampi_delta=delta,
            scampi_max_memory=scampi_max_memory,
            **kwargs
        )


def make_scale_n_runner(
        read_data,
        n_range,
        l_range,
        scampi_max_memory="2GB"):
    def run_motiflets_scale_n(
            backends=("scampi",),
            delta=0.1,
            k_max=10,
            subsampling=None):
        run_scale_n(
            read_data,
            n_range,
            l_range,
            k_max,
            backends=backends,
            delta=delta,
            subsampling=subsampling,
            scampi_max_memory=scampi_max_memory)

    return run_motiflets_scale_n
