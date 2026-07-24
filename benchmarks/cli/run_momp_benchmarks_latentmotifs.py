"""Run LatentMotif MOMP benchmarks.

Examples
--------
List available LatentMotif radius modes:
    python benchmarks/cli/run_momp_benchmarks_latentmotifs.py --list-radius-modes

Run LatentMotif with the default 2r radius mode on all datasets:
    python benchmarks/cli/run_momp_benchmarks_latentmotifs.py

Run selected datasets and motif lengths:
    python benchmarks/cli/run_momp_benchmarks_latentmotifs.py \
        --datasets EOG_one_hour_50_Hz,Challenge2009TestSetA_101a \
        --lengths 512,1024 \
        --radius-mode r2

Run LatentMotif and store raw, unrefined motifs:
    python benchmarks/cli/run_momp_benchmarks_latentmotifs.py \
        --no-exact-refine

Run against a specific server data directory:
    python benchmarks/cli/run_momp_benchmarks_latentmotifs.py \
        --data-path /vol/fob-wbib-vol2/wbi/schaefpa/motiflets/momp \
        --local-n full

This runner is intentionally restricted to LatentMotif. It reads reference
pyattimo/SCAMPI extents, derives the LatentMotif radius from the selected
radius mode, and writes results below ``tests/results`` by default.
"""

import argparse
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

if "NUMBA_CACHE_DIR" not in os.environ:
    cache_name = f"scampi-numba-{os.getuid()}-{os.getpid()}"
    os.environ["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / cache_name)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import psutil

from benchmarks.cli.run_momp_benchmarks import parse_csv, selected_filenames


RADIUS_MODES = ["2r", "r2"]
HPC_DATA_PATH = "/vol/fob-wbib-vol2/wbi/schaefpa/motiflets/momp"
RUN_LOCAL = not (os.path.exists(HPC_DATA_PATH) and os.path.isdir(HPC_DATA_PATH))
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
DEFAULT_REFERENCE_DIR = RESULTS_DIR / "pyattimo_0.7.0_20GB"


def default_lengths():
    if RUN_LOCAL:
        return [2 ** 9]
    return [2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LatentMotif MOMP benchmarks."
    )
    parser.add_argument(
        "--list-radius-modes",
        action="store_true",
        help="Print available LatentMotif radius modes and exit.",
    )
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset keys. Defaults to all datasets by length.",
    )
    parser.add_argument(
        "--lengths",
        type=lambda value: parse_csv(value, int),
        default=default_lengths(),
        help="Comma-separated motif lengths. Defaults to 512 locally, 512..4096 on HPC.",
    )
    parser.add_argument(
        "--data-path",
        help="Directory containing MOMP .mat files. Forces full-data mode.",
    )
    parser.add_argument(
        "--local-n",
        default=None,
        help="Local-mode time series length cap. Use 'full' to disable the cap.",
    )
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "Numba thread count for LatentMotif. Use -1 to keep Numba's "
            "default thread count."
        ),
    )
    parser.add_argument(
        "--exact-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refine LatentMotif seed positions with exact k-NN distances and "
            "store the optimized motiflet/extent. Enabled by default; use "
            "--no-exact-refine to store the raw LatentMotif motif set."
        ),
    )
    parser.add_argument(
        "--radius-mode",
        choices=RADIUS_MODES,
        default="2r",
        help=(
            "How to derive the LatentMotif radius from the reference extent r. "
            "'2r' uses 2 * radius_factor * (r + epsilon); "
            "'r2' uses radius_factor * (r + epsilon) ** 2."
        ),
    )
    parser.add_argument(
        "--radius-factor",
        type=float,
        default=1.0,
        help="Additional multiplier applied to the selected radius formula.",
    )
    parser.add_argument("--radius-epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory containing pyattimo/SCAMPI reference CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for LatentMotif result CSVs. Defaults to "
            "tests/results/latentmotifs_<radius-mode>."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute rows even if an existing output CSV already has finite extents.",
    )
    return parser.parse_args()


def default_output_dir(radius_mode):
    return RESULTS_DIR / f"latentmotifs_{radius_mode}"


def output_dir(args):
    if args.output_dir is not None:
        return args.output_dir
    return default_output_dir(args.radius_mode)


def radius_from_reference(reference_extent, args):
    radius_base = reference_extent + args.radius_epsilon
    if args.radius_mode == "2r":
        return 2.0 * args.radius_factor * radius_base
    if args.radius_mode == "r2":
        return args.radius_factor * radius_base ** 2
    raise ValueError(f"Unknown radius mode: {args.radius_mode}")


def local_length(series, local_n, run_local):
    if not run_local:
        return len(series)
    if local_n == "full":
        return len(series)
    if local_n is None:
        return 10_000
    return int(local_n)


def output_filename(output_dir, ds_name, k_max):
    return output_dir / f"scalability_n_{ds_name}_{k_max}_latentmotifs.csv"


def backend_label(args):
    label = (
        f"LatentMotif radius_mode={args.radius_mode} "
        f"radius_factor={args.radius_factor}"
    )
    if args.exact_refine:
        label = f"{label} exact_refine"
    return label


def reference_filename(reference_dir, ds_name, k_max):
    return reference_dir / f"scalability_n_{ds_name}_{k_max}_pyattimo_delta_0.1.csv"


def load_existing_extents(filename, lengths, overwrite):
    if overwrite or not filename.exists():
        return np.full(len(lengths), np.inf, dtype=np.float64)

    existing = pd.read_csv(filename)
    if "motif length" not in existing or "extent" not in existing:
        return np.full(len(lengths), np.inf, dtype=np.float64)

    motif_lengths = pd.to_numeric(existing["motif length"], errors="coerce")
    extents = np.full(len(lengths), np.inf, dtype=np.float64)
    for i, length in enumerate(lengths):
        rows = existing[motif_lengths == int(length)]
        if rows.empty:
            continue
        values = pd.to_numeric(rows["extent"], errors="coerce")
        finite_values = values[np.isfinite(values)]
        if not finite_values.empty:
            extents[i] = float(finite_values.iloc[-1])
    return extents


def load_existing_results(filename, overwrite):
    columns = [
        "length",
        "motif length",
        "backend",
        "time in s",
        "memory in MB",
        "extent",
        "motiflet",
        "elbows",
    ]
    if overwrite or not filename.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(filename)


def compute_extent(ts, motif_set, motif_length):
    from scampi.distances import map_distances
    from scampi.scampi import get_pairwise_extent_raw

    distance_preprocessing, _, distance_single = map_distances("znormed_ed")
    preprocessing = np.array(
        [distance_preprocessing(ts, motif_length)],
        dtype=np.float64,
    )

    if len(motif_set) == 0:
        return np.inf

    return get_pairwise_extent_raw(
        ts.reshape(1, -1),
        motif_set,
        motif_length,
        distance_single=distance_single,
        preprocessing=preprocessing,
    )


def run_latentmotif(ds_name, series, lengths, args, local_n, run_local):
    from competitors.latentmotifs import LatentMotif

    output_path = output_filename(args.output_dir, ds_name, args.k_max)
    reference_path = reference_filename(args.reference_dir, ds_name, args.k_max)

    print(f"  Reference: {reference_path}")
    print(f"  Output:    {output_path}")

    reference = pd.read_csv(reference_path)[["motif length", "extent"]]
    reference.set_index("motif length", inplace=True)

    existing_extents = load_existing_extents(output_path, lengths, args.overwrite)
    df_results = load_existing_results(output_path, args.overwrite)

    n = local_length(series, local_n, run_local)
    if run_local and local_n != "full":
        print(f"  Local cap: n={n}")
    ts = series[:n]

    for i, motif_length in enumerate(lengths):
        if not np.isinf(existing_extents[i]):
            print(f"  Skipping length {motif_length}; finite result already exists.")
            continue

        if motif_length not in reference.index:
            print(f"  Skipping length {motif_length}; no reference extent available.")
            continue

        reference_extent = reference.loc[motif_length, "extent"]
        radius = radius_from_reference(reference_extent, args)
        print(
            f"  Running LatentMotif: m={motif_length} "
            f"radius_mode={args.radius_mode} "
            f"reference_extent={reference_extent} radius={radius}"
        )

        pid = os.getpid()
        process = psutil.Process(pid)

        start = time.time()
        lm = LatentMotif(
            n_patterns=1,
            wlen=motif_length,
            radius=radius,
            n_starts=args.n_starts,
            n_jobs=args.n_jobs,
        )
        lm.fit(ts)
        motif_set = np.asarray(lm.prediction_indices_[0], dtype=np.int32)

        print(f"    Patterns: {lm.patterns_.shape[0]}")
        print(f"    Locations: {motif_set.shape[0]}")

        if args.exact_refine:
            from benchmarks.converters.utils import ConverterUtils

            motif_set, extent = ConverterUtils.exact_refine_from_seeds(
                ts,
                motif_set.astype(np.int32),
                motif_length,
                args.k_max,
            )
            duration = time.time() - start
            memory_usage = process.memory_info().rss / (1024 * 1024)
        else:
            duration = time.time() - start
            memory_usage = process.memory_info().rss / (1024 * 1024)
            extent = compute_extent(ts, motif_set, motif_length)

        print(
            f"    Completed in {duration:0.2f}s "
            f"memory={memory_usage:0.2f}MB extent={extent}"
        )
        del lm

        current = [
            ts.shape[-1],
            motif_length,
            backend_label(args),
            duration,
            memory_usage,
            float(extent),
            motif_set,
            -1,
        ]

        df_results = df_results[df_results["motif length"] != motif_length]
        df_results.loc[len(df_results.index)] = current
        df_results.sort_values("motif length", inplace=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False)


def main():
    args = parse_args()
    if args.list_radius_modes:
        print("\n".join(RADIUS_MODES))
        return

    args.output_dir = output_dir(args)

    from benchmarks import utils as ut

    local_n = args.local_n
    if local_n is None and not RUN_LOCAL:
        local_n = "full"
    ut.configure_paths(data_path=args.data_path, local_n=local_n)

    print("Method: latentmotifs")
    print(f"Motif lengths: {args.lengths}")
    print(f"n_starts: {args.n_starts}")
    print(f"radius_mode: {args.radius_mode}")
    print(f"radius_factor: {args.radius_factor}")

    for filename in selected_filenames(args, ut):
        ds_name, _, _, _ = ut.filenames[filename]
        print(f"\nDataset: {filename} ({ds_name})")
        data = ut.read_mat(filename)

        try:
            run_latentmotif(
                filename,
                data,
                args.lengths,
                args,
                local_n,
                ut.run_local,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
