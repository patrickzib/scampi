"""Run synthetic planted-motif benchmarks with STUMPY motif sets.

This script mirrors the synthetic SCAMPI series-length benchmark but uses
``stumpy.stump`` for the matrix profile and ``stumpy.motifs`` for motif-set
extraction. Full array-valued results are appended to JSONL, while scalar
plotting values and recovery metrics are written to a CSV summary.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import psutil

from benchmarks.cli.run_momp_benchmarks import parse_csv
from benchmarks.synthetic import (
    compute_ground_truth_extent,
    generate_random_walk_with_planted_motif,
    json_dumps,
    select_best_motiflet,
    to_jsonable,
)


RESULTS_DIR = PROJECT_ROOT / "tests" / "results" / "synthetic_stumpy"
EXPERIMENT = "series-length"
SUMMARY_COLUMNS = [
    "n",
    "motif length",
    "n instances",
    "noise",
    "seed",
    "backend",
    "normalize",
    "top_n",
    "selected_k",
    "time in s",
    "memory in MB",
    "ground truth extent",
    "selected extent",
    "precision",
    "recall",
    "f_score",
    "true positives",
    "false positives",
    "false negatives",
    "position tolerance",
    "ground truth positions",
    "selected motiflet",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run synthetic planted-motif benchmarks with STUMPY."
    )
    parser.add_argument(
        "--series-lengths",
        type=lambda value: parse_csv(value, int),
        default=parse_csv(
            "10000,50000,100000,250000,500000,750000,1000000", int),
        help="Comma-separated time-series lengths to benchmark.",
    )
    parser.add_argument(
        "--noise-levels",
        type=lambda value: parse_csv(value, float),
        default=parse_csv("0", float),
        help="Comma-separated Gaussian noise levels added to motif instances.",
    )
    parser.add_argument(
        "--seeds",
        type=lambda value: parse_csv(value, int),
        default=parse_csv("1,2,3", int),
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--fixed-motif-length",
        type=int,
        default=2048,
        help="Motif length used for the series-length sweep.",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=10,
        help="Number of planted motif instances and selected motif-set size.",
    )
    parser.add_argument("--motif-amplitude", type=float, default=5.0)
    parser.add_argument("--motif-cycles", type=float, default=2.0)
    parser.add_argument("--random-walk-sigma", type=float, default=1.0)
    parser.add_argument(
        "--position-tolerance",
        type=int,
        default=None,
        help="Position matching tolerance. Defaults to motif_length // 2.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use STUMPY z-normalized distances.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Maximum number of motif sets requested from STUMPY.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for JSONL full results and CSV summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute rows already present in the summary CSV.",
    )
    return parser.parse_args()


def output_paths(output_dir):
    return (
        output_dir / f"{EXPERIMENT}_runs.jsonl",
        output_dir / f"{EXPERIMENT}_summary.csv",
    )


def run_key(n, motif_length, noise, seed, args):
    return (
        int(n),
        int(motif_length),
        int(args.n_instances),
        float(noise),
        int(seed),
        bool(args.normalize),
        int(args.top_n),
    )


def existing_keys(summary_path):
    if not summary_path.exists():
        return set()
    df = pd.read_csv(summary_path)
    if df.empty:
        return set()
    return {
        (
            int(row["n"]),
            int(row["motif length"]),
            int(row["n instances"]),
            float(row["noise"]),
            int(row["seed"]),
            parse_bool(row["normalize"]),
            int(row["top_n"]),
        )
        for _, row in df.iterrows()
    }


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def can_implant_non_overlapping(n, motif_length, n_instances):
    return int(n) >= int(motif_length) * int(n_instances)


def load_stumpy():
    try:
        import stumpy
    except ImportError as exc:
        raise RuntimeError(
            "STUMPY is required for this benchmark. Install project "
            "dependencies or run `python -m pip install stumpy`."
        ) from exc
    return stumpy


def matrix_profile_values(mp):
    if hasattr(mp, "P_"):
        return np.asarray(mp.P_, dtype=np.float64)
    arr = np.asarray(mp)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError("Unexpected STUMPY matrix profile shape")
    return arr[:, 0].astype(np.float64)


def motifs_to_motiflets(motif_indices, motif_distances, selected_k):
    motiflet_candidates = []
    extent_candidates = []

    indices = np.asarray(motif_indices)
    if indices.ndim == 0:
        indices = indices.reshape(1, 1)
    elif indices.ndim == 1:
        indices = indices.reshape(1, -1)

    distances = np.asarray(motif_distances, dtype=np.float64)
    if distances.ndim == 0:
        distances = distances.reshape(1, 1)
    elif distances.ndim == 1:
        distances = distances.reshape(1, -1)

    for rank, candidate in enumerate(indices):
        positions = [int(pos) for pos in np.ravel(candidate) if int(pos) >= 0]
        if len(positions) > selected_k:
            positions = positions[:selected_k]
        motiflet_candidates.append(np.asarray(positions, dtype=np.int32))

        if rank < distances.shape[0]:
            finite = distances[rank][np.isfinite(distances[rank])]
            extent = float(np.max(finite)) if finite.size else math.inf
        else:
            extent = math.inf
        extent_candidates.append(extent)

    motiflets = [None] * (selected_k + 1)
    extents = [None] * (selected_k + 1)
    motiflets[selected_k] = motiflet_candidates
    extents[selected_k] = np.asarray(extent_candidates, dtype=np.float64)
    return motiflets, extents


def current_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_one(n, motif_length, noise, seed, args):
    stumpy = load_stumpy()

    series, ground_truth, motif = generate_random_walk_with_planted_motif(
        n=n,
        motif_length=motif_length,
        n_instances=args.n_instances,
        motif_noise=noise,
        random_walk_sigma=args.random_walk_sigma,
        motif_amplitude=args.motif_amplitude,
        motif_cycles=args.motif_cycles,
        seed=seed,
    )
    tolerance = (
        args.position_tolerance
        if args.position_tolerance is not None
        else max(1, motif_length // 2)
    )

    start = time.time()
    mp = stumpy.stump(
        series,
        m=motif_length,
        ignore_trivial=True,
        normalize=args.normalize,
    )
    profile = matrix_profile_values(mp)
    motif_distances, motif_indices = stumpy.motifs(
        series,
        profile,
        min_neighbors=max(1, args.n_instances - 1),
        max_matches=args.n_instances,
        max_motifs=args.top_n,
        normalize=args.normalize,
    )
    duration = time.time() - start
    memory_usage = current_rss_mb()

    motiflets, extents = motifs_to_motiflets(
        motif_indices,
        motif_distances,
        args.n_instances,
    )
    selected = select_best_motiflet(
        motiflets,
        extents,
        args.n_instances,
        ground_truth,
        tolerance,
    )
    metrics = selected["metrics"]
    ground_truth_extent = compute_ground_truth_extent(
        series,
        ground_truth,
        motif_length,
    )

    full_record = {
        "experiment": EXPERIMENT,
        "n": n,
        "motif_length": motif_length,
        "n_instances": args.n_instances,
        "noise": noise,
        "seed": seed,
        "backend": "stumpy",
        "normalize": args.normalize,
        "top_n": args.top_n,
        "time_in_s": duration,
        "memory_in_mb": memory_usage,
        "ground_truth_positions": ground_truth,
        "ground_truth_extent": ground_truth_extent,
        "clean_motif": motif,
        "matrix_profile": profile,
        "motif_distances": motif_distances,
        "motif_indices": motif_indices,
        "extents": extents,
        "motiflets": motiflets,
        "selected_k": args.n_instances,
        "selected_rank": selected["rank"],
        "selected_extent": selected["extent"],
        "selected_motiflet": selected["motiflet"],
        "position_tolerance": tolerance,
        **metrics,
    }

    summary_row = {
        "n": n,
        "motif length": motif_length,
        "n instances": args.n_instances,
        "noise": noise,
        "seed": seed,
        "backend": "stumpy",
        "normalize": args.normalize,
        "top_n": args.top_n,
        "selected_k": args.n_instances,
        "time in s": duration,
        "memory in MB": memory_usage,
        "ground truth extent": ground_truth_extent,
        "selected extent": selected["extent"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f_score": metrics["f_score"],
        "true positives": metrics["true_positives"],
        "false positives": metrics["false_positive_count"],
        "false negatives": metrics["false_negative_count"],
        "position tolerance": tolerance,
        "ground truth positions": json_dumps(ground_truth),
        "selected motiflet": json_dumps(selected["motiflet"]),
    }

    return full_record, summary_row


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(record), allow_nan=False) + "\n")


def append_summary(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=SUMMARY_COLUMNS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    args = parse_args()
    try:
        load_stumpy()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, summary_path = output_paths(args.output_dir)
    if args.overwrite:
        jsonl_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    completed = set() if args.overwrite else existing_keys(summary_path)

    print("Method: stumpy")
    print(f"Experiment: {EXPERIMENT}")
    print(f"Series lengths: {args.series_lengths}")
    print(f"Motif length: {args.fixed_motif_length}")
    print(f"Noise levels: {args.noise_levels}")
    print(f"Seeds: {args.seeds}")
    print(f"Normalize: {args.normalize}")
    print(f"Top N: {args.top_n}")
    print(f"Output: {args.output_dir}")

    for n in args.series_lengths:
        motif_length = args.fixed_motif_length
        if not can_implant_non_overlapping(n, motif_length, args.n_instances):
            print(
                f"Skipping {EXPERIMENT}: n={n} m={motif_length} "
                f"instances={args.n_instances}; non-overlapping planted "
                f"motifs require n >= {motif_length * args.n_instances}."
            )
            continue

        for noise in args.noise_levels:
            for seed in args.seeds:
                key = run_key(n, motif_length, noise, seed, args)
                if key in completed:
                    print(
                        f"Skipping {EXPERIMENT}: n={n} m={motif_length} "
                        f"noise={noise} seed={seed}; summary row exists."
                    )
                    continue

                print(
                    f"Running {EXPERIMENT}: n={n} m={motif_length} "
                    f"instances={args.n_instances} noise={noise} seed={seed}"
                )
                try:
                    full_record, summary_row = run_one(
                        n,
                        motif_length,
                        noise,
                        seed,
                        args,
                    )
                    append_jsonl(jsonl_path, full_record)
                    append_summary(summary_path, summary_row)
                    completed.add(key)
                    print(
                        f"  Done in {summary_row['time in s']:0.2f}s "
                        f"F1={summary_row['f_score']:0.3f} "
                        f"extent={summary_row['selected extent']}"
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(f"Error: {exc}")
                    print(traceback.format_exc())


if __name__ == "__main__":
    main()
