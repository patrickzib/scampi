import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

from scampi.distances import (
    sliding_mean_std,
    znormed_euclidean_distance,
    znormed_euclidean_distance_single,
)
from scampi.scampi import _argknn, _sliding_dot_product


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
DEFAULT_LENGTHS = [512, 1024, 2048, 4096]
RESULT_COLUMNS = [
    "length",
    "motif length",
    "backend",
    "time in s",
    "memory in MB",
    "motiflet",
    "extent",
]


class ConverterUtils:
    @staticmethod
    def add_common_args(parser):
        parser.add_argument(
            "--datasets",
            help=(
                "Comma-separated dataset keys. Defaults depend on the converter; "
                "most use all known datasets by length."
            ),
        )
        parser.add_argument(
            "--lengths",
            type=lambda value: ConverterUtils.parse_csv(value, int),
            default=DEFAULT_LENGTHS,
            help="Comma-separated motif lengths. Defaults to 512,1024,2048,4096.",
        )
        parser.add_argument("--k-max", type=int, default=10)
        parser.add_argument(
            "--data-path",
            help="Directory containing MOMP .mat files.",
        )

    @staticmethod
    def parse_csv(value, cast=str):
        if value is None:
            return None
        return [cast(item.strip()) for item in value.split(",") if item.strip()]

    @staticmethod
    def parse_array(cell):
        if isinstance(cell, np.ndarray):
            return cell.astype(np.int32)
        if not isinstance(cell, str) or cell.strip() in {"", "[]"}:
            return np.array([], dtype=np.int32)
        cleaned = cell.strip().strip("[]")
        return np.fromstring(cleaned, sep=" ", dtype=np.int32)

    @staticmethod
    def selected_datasets(args, filenames, default=None):
        if default is None:
            lengths = np.array([properties[-1] for properties in filenames.values()])
            datasets = [list(filenames.keys())[idx] for idx in np.argsort(lengths)]
        else:
            datasets = list(default)

        if args.datasets:
            requested = ConverterUtils.parse_csv(args.datasets)
            unknown = sorted(set(requested) - set(filenames.keys()))
            if unknown:
                raise ValueError(
                    f"Unknown datasets: {unknown}. Available datasets: {list(filenames)}"
                )
            datasets = [dataset for dataset in datasets if dataset in requested]

        return datasets

    @staticmethod
    def result_frame():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    @staticmethod
    def output_filename(output_dir, pattern, ds_name, k_max):
        return Path(output_dir) / pattern.format(ds_name=ds_name, k_max=k_max)

    @staticmethod
    def indexed_csv(path):
        path = Path(path)
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        return frame.set_index("motif length", drop=False)

    @staticmethod
    def metric_for_length(frame, motif_length, metric):
        if frame is not None and motif_length in frame.index:
            return frame.loc[motif_length, metric]
        return None

    @staticmethod
    def extrapolate_metric(frame, motif_length, metric):
        last_length = int(frame["motif length"].values[-1])
        factor = motif_length / last_length
        multiplier = factor * 2 if metric == "time in s" else factor
        value = float(frame[metric].values[-1]) * multiplier
        print(
            f"  Extrapolating {metric} for m={motif_length} from "
            f"last known m={last_length}: factor={factor}, value={value}"
        )
        return value

    @staticmethod
    def write_csv(frame, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(f"  Wrote {output_path}")

    @staticmethod
    def compute_knn(
            ts,
            motiflets,
            motif_length,
            neighbors,
            include_seed_set=False):
        motiflets = np.asarray(motiflets, dtype=np.int32)
        if len(motiflets) == 0:
            return [], np.inf
        best_motiflet, min_extent = _compute_knn(
            ts.copy(),
            motiflets,
            motif_length,
            neighbors,
        )
        if include_seed_set and len(motiflets) == neighbors:
            seed_extent = _pairwise_extent_1d(
                ts,
                motiflets,
                motif_length,
                znormed_euclidean_distance_single,
                sliding_mean_std(ts, motif_length),
            )
            if min_extent > seed_extent:
                return motiflets, seed_extent
        return best_motiflet, min_extent

    @staticmethod
    def exact_refine_from_seeds(ts, seeds, motif_length, k_max, **kwargs):
        return ConverterUtils.compute_knn(
            ts,
            seeds,
            motif_length,
            k_max - 1,
            **kwargs,
        )


@njit(cache=True, parallel=True)
def _compute_knn(
        ts,
        motiflets,
        m,
        k,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
):
    halve_m = np.int32(m * slack)
    n = ts.shape[-1] - m + 1
    preprocessing = distance_preprocessing(ts, m)

    knns = np.zeros((len(motiflets), k), dtype=np.int32)
    extents = np.zeros(len(motiflets), dtype=np.float64)

    for i in prange(len(motiflets)):
        start = motiflets[i]
        if start < len(ts) - m + 1:
            dot_rolled = _sliding_dot_product(
                ts[start:start + m],
                ts,
            )
            dist = distance(dot_rolled, n, m, preprocessing, start, halve_m)
            knns[i] = _argknn(dist, k, m, slack=slack)
            extents[i] = _pairwise_extent_1d(
                ts, knns[i], m, distance_single, preprocessing)
        else:
            extents[i] = np.inf

    min_pos = np.argmin(extents)
    return knns[min_pos], extents[min_pos]


@njit(cache=True)
def _pairwise_extent_1d(series, motifset_pos, motif_length,
                        distance_single, preprocessing):
    if -1 in motifset_pos:
        return np.inf

    motifset_extent = np.float64(0.0)
    for ii in np.arange(len(motifset_pos) - 1):
        i = motifset_pos[ii]
        a = series[i:i + motif_length]
        for jj in np.arange(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]
            b = series[j:j + motif_length]
            dist = distance_single(a, b, i, j, preprocessing)
            motifset_extent = max(motifset_extent, dist)
    return motifset_extent


def parser(description):
    return argparse.ArgumentParser(description=description)
