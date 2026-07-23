"""Benchmark MPX v2 against the MOMP v9 port on MOMP .mat files."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from .core import mpx_v2_motif_pair
from .io import load_series, to_jsonable
from .momp_v9 import momp_v9


DEFAULT_DATASETS = [
    "EOG_one_hour_50_Hz.mat",
    "Challenge2009TestSetA_101a.mat",
    "swtAttack7.mat",
]

FIELDNAMES = [
    "dataset",
    "mat_key",
    "algorithm",
    "original_length",
    "prefix_length",
    "motif_length",
    "distance",
    "location_0",
    "location_1",
    "location_0_matlab",
    "location_1_matlab",
    "runtime_seconds",
    "iterations",
    "final_pruning",
    "status",
    "error",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the MPX v2 baseline and the MOMP v9 Python port "
            "on MATLAB datasets from datasets/momp."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/momp"),
        help="Directory containing MOMP .mat files.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated .mat filenames to benchmark.",
    )
    parser.add_argument(
        "--algorithms",
        default="mpx-v2,momp-v9",
        help="Comma-separated algorithms: mpx-v2,momp-v9.",
    )
    parser.add_argument(
        "--motif-lengths",
        default="256",
        help="Comma-separated motif lengths.",
    )
    parser.add_argument(
        "--prefix-lengths",
        default="5000,10000",
        help=(
            "Comma-separated prefix lengths. This benchmarks prefixes from "
            "large .mat datasets."
        ),
    )
    parser.add_argument(
        "--initial-downsample-rate",
        type=int,
        default=64,
        help="Initial downsample rate for momp-v9.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("momp/momp_python/results"),
        help="Directory for benchmark CSV and JSONL outputs.",
    )
    parser.add_argument(
        "--name",
        default="momp_python_dataset_benchmark",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing outputs before running.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not run a small JIT warm-up before measuring.",
    )
    return parser.parse_args(argv)


def split_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def load_series_with_key(path):
    if path.suffix.lower() != ".mat":
        return load_series(path), None

    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError("Reading .mat files requires scipy") from exc

    mat = sio.loadmat(path)
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            return load_series(path, mat_key=key), key
    raise ValueError("no numeric array found in .mat file")


def existing_keys(path):
    if not path.exists():
        return set()

    keys = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "ok":
                keys.add(row_key(row))
    return keys


def row_key(row):
    return (
        row["dataset"],
        row.get("mat_key") or "",
        row["algorithm"],
        int(row["prefix_length"]),
        int(row["motif_length"]),
    )


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, sort_keys=True)
        handle.write("\n")


def run_algorithm(algorithm, series, motif_length, initial_downsample_rate):
    if algorithm == "mpx-v2":
        return mpx_v2_motif_pair(series, motif_length)
    if algorithm == "momp-v9":
        return momp_v9(
            series,
            motif_length,
            verbose=False,
            initial_downsample_rate=initial_downsample_rate,
        )
    raise ValueError(f"unknown algorithm: {algorithm}")


def make_row(dataset, mat_key, algorithm, original_length, prefix_length, motif_length):
    return {
        "dataset": dataset,
        "mat_key": mat_key or "",
        "algorithm": algorithm,
        "original_length": int(original_length),
        "prefix_length": int(prefix_length),
        "motif_length": int(motif_length),
    }


def warmup_algorithms(algorithms, motif_lengths, initial_downsample_rate):
    if not motif_lengths:
        return

    motif_length = max(motif_lengths)
    n = max(5000, 4 * motif_length, 4 * initial_downsample_rate)
    rng = np.random.default_rng(0)
    series = rng.normal(size=n)
    print(f"Warming up algorithms on synthetic n={n}, m={motif_length}")
    for algorithm in algorithms:
        try:
            run_algorithm(algorithm, series, motif_length, initial_downsample_rate)
        except Exception as exc:
            print(f"  Warm-up skipped for {algorithm}: {type(exc).__name__}: {exc}")


def main(argv=None):
    args = parse_args(argv)
    datasets = split_csv(args.datasets)
    algorithms = split_csv(args.algorithms)
    motif_lengths = split_csv(args.motif_lengths, int)
    prefix_lengths = split_csv(args.prefix_lengths, int)

    unknown = sorted(set(algorithms) - {"mpx-v2", "momp-v9"})
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}")

    if not args.no_warmup:
        warmup_algorithms(algorithms, motif_lengths, args.initial_downsample_rate)

    csv_path = args.output_dir / f"{args.name}.csv"
    jsonl_path = args.output_dir / f"{args.name}.jsonl"
    if args.overwrite:
        csv_path.unlink(missing_ok=True)
        jsonl_path.unlink(missing_ok=True)

    seen = existing_keys(csv_path)
    print(f"Writing CSV:   {csv_path}")
    print(f"Writing JSONL: {jsonl_path}")

    for dataset in datasets:
        path = args.data_dir / dataset
        if not path.exists():
            print(f"Skipping missing dataset: {path}")
            continue

        series, mat_key = load_series_with_key(path)
        original_length = len(series)
        key_msg = f" key={mat_key}" if mat_key else ""
        print(f"Dataset {dataset}:{key_msg} length={original_length}")

        for motif_length in motif_lengths:
            for prefix_length in prefix_lengths:
                if prefix_length > original_length:
                    print(
                        f"  Skipping prefix {prefix_length}: "
                        f"dataset has only {original_length} samples"
                    )
                    continue
                if prefix_length < 2 * motif_length:
                    print(
                        f"  Skipping m={motif_length}, n={prefix_length}: "
                        "need at least 2 * motif_length samples"
                    )
                    continue

                prefix = np.ascontiguousarray(series[:prefix_length])
                for algorithm in algorithms:
                    row = make_row(
                        dataset,
                        mat_key,
                        algorithm,
                        original_length,
                        prefix_length,
                        motif_length,
                    )
                    key = row_key(row)
                    if key in seen:
                        print(
                            f"  Already done: {dataset} {algorithm} "
                            f"n={prefix_length} m={motif_length}"
                        )
                        continue
                    print(
                        f"  Running {algorithm}: n={prefix_length} "
                        f"m={motif_length}"
                    )
                    try:
                        start = time.time()
                        result = run_algorithm(
                            algorithm,
                            prefix,
                            motif_length,
                            args.initial_downsample_rate,
                        )
                        iterations = getattr(result, "iterations", [])
                        row.update({
                            "distance": float(result.distance),
                            "location_0": int(result.locations[0]),
                            "location_1": int(result.locations[1]),
                            "location_0_matlab": int(result.locations_matlab[0]),
                            "location_1_matlab": int(result.locations_matlab[1]),
                            "runtime_seconds": float(result.runtime_seconds),
                            "iterations": len(iterations),
                            "final_pruning": (
                                float(iterations[-1]["pruning"])
                                if iterations else ""
                            ),
                            "status": "ok",
                            "error": "",
                        })
                        payload = {
                            **row,
                            "wall_seconds": time.time() - start,
                            "result": result.to_dict(),
                        }
                        print(
                            f"    done in {row['runtime_seconds']:0.3f}s "
                            f"distance={row['distance']:0.6g} "
                            f"locations=({row['location_0']}, {row['location_1']})"
                        )
                    except Exception as exc:
                        row.update({
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        payload = {**row}
                        print(f"    error: {row['error']}")

                    append_csv(csv_path, row)
                    append_jsonl(jsonl_path, payload)
                    if row["status"] == "ok":
                        seen.add(key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
