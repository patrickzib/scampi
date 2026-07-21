"""Run synthetic planted-motif SCAMPI benchmarks.

Examples
--------
Run the time-series length scalability experiment:
    python benchmarks/cli/run_synthetic_scampi_benchmarks.py \
        --experiment series-length

Run the motif-length scalability experiment:
    python benchmarks/cli/run_synthetic_scampi_benchmarks.py \
        --experiment motif-length

Run the SCAMPI memory-budget experiment:
    python benchmarks/cli/run_synthetic_scampi_benchmarks.py \
        --experiment memory-budget

Run several noise levels and seeds:
    python benchmarks/cli/run_synthetic_scampi_benchmarks.py \
        --experiment series-length \
        --noise-levels 0,0.05,0.1,0.2 \
        --seeds 1,2,3 \
        --scampi-deltas 0.05,0.1

Defaults
--------
The default experiment is ``series-length``. It uses one fixed motif length
``512`` and sweeps these time-series lengths:
    10000, 50000, 100000, 250000, 500000, 750000, 1000000

The ``motif-length`` experiment uses one fixed time-series length ``200000``
and sweeps these motif lengths:
    256, 512, 1024, 2048, 4096

The ``memory-budget`` experiment uses fixed time-series length ``200000`` and
fixed motif length ``512``, and sweeps these SCAMPI max-memory budgets:
    1GB, 2GB, 4GB, 8GB

Only ``memory-budget`` sweeps ``--scampi-max-memories``. The ``series-length``
and ``motif-length`` experiments each use the single fixed
``--scampi-max-memory`` value.

All experiments use these shared defaults:
    noise levels: 0, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0
    seeds: 1, 2, 3
    planted motif instances: 5
    motif amplitude: 5.0
    motif cycles: 2.0
    random-walk sigma: 1.0
    k max: 10
    top N: 1
    SCAMPI deltas: 0.1
    SCAMPI max memory: 2 GB
    exact refine: false

This script exercises the SCAMPI/pyattimo backend.
Full array-valued results are appended to JSONL files, while scalar plotting values
and recovery metrics are written to CSV summaries.
"""

import argparse
import json
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

from benchmarks.cli.run_momp_benchmarks import parse_csv
from benchmarks.synthetic import (
    generate_random_walk_with_planted_motif,
    json_dumps,
    select_best_motiflet,
    to_jsonable,
)


RESULTS_DIR = PROJECT_ROOT / "tests" / "results" / "synthetic_scampi"
EXPERIMENTS = ["series-length", "motif-length", "memory-budget"]
SUMMARY_COLUMNS = [
    "experiment",
    "n",
    "motif length",
    "n instances",
    "noise",
    "seed",
    "backend",
    "scampi_delta",
    "scampi_max_memory",
    "scampi_exact_refine",
    "selected_k",
    "time in s",
    "memory in MB",
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
    "plot path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run synthetic planted-motif benchmarks with SCAMPI only."
    )
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS + ["all"],
        default="series-length",
        help="Experiment to run. Use 'all' for all sweeps.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="Print available experiment names and exit.",
    )
    parser.add_argument(
        "--series-lengths",
        type=lambda value: parse_csv(value, int),
        default=parse_csv(
            "10000,50000,100000,250000,500000,750000,1000000", int),
        help="Comma-separated n values for --experiment series-length.",
    )
    parser.add_argument(
        "--motif-lengths",
        type=lambda value: parse_csv(value, int),
        default=parse_csv("256,512,1024,2048,4096", int),
        help="Comma-separated motif lengths for --experiment motif-length.",
    )
    parser.add_argument(
        "--fixed-series-length",
        type=int,
        default=200_000,
        help="n used for the motif-length sweep.",
    )
    parser.add_argument(
        "--fixed-motif-length",
        type=int,
        default=512,
        help="Motif length used for the series-length sweep.",
    )
    parser.add_argument(
        "--noise-levels",
        type=lambda value: parse_csv(value, float),
        default=parse_csv("0,0.05,0.1,0.2,0.25,0.5,0.75,1.0", float),
        help="Comma-separated Gaussian noise levels added to motif instances.",
    )
    parser.add_argument(
        "--seeds",
        type=lambda value: parse_csv(value, int),
        default=parse_csv("1,2,3", int),
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--n-instances", type=int, default=5)
    parser.add_argument("--motif-amplitude", type=float, default=5.0)
    parser.add_argument("--motif-cycles", type=float, default=2.0)
    parser.add_argument("--random-walk-sigma", type=float, default=1.0)
    parser.add_argument(
        "--position-tolerance",
        type=int,
        default=None,
        help="Position matching tolerance. Defaults to motif_length // 2.",
    )
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--slack", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--scampi-deltas", type=lambda v: parse_csv(v, float),
                        default=[0.1])
    parser.add_argument("--scampi-max-memory", default="2 GB")
    parser.add_argument(
        "--scampi-max-memories",
        type=lambda value: parse_csv(value, str),
        default=parse_csv("1GB,2GB,4GB,8GB"),
        help="Comma-separated memory budgets for --experiment memory-budget.",
    )
    parser.add_argument("--scampi-exact-refine", action="store_true")
    parser.add_argument(
        "--plot-generated",
        action="store_true",
        help="Save a SCAMPI plot of each generated series and selected motiflet.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for generated plots. Defaults to <output-dir>/plots.",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=2_000,
        help="Maximum points passed to SCAMPI plotting downsampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for JSONL full results and CSV summaries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute rows already present in the summary CSV.",
    )
    return parser.parse_args()


def experiment_grid(args, experiment):
    if experiment == "series-length":
        for n in args.series_lengths:
            yield n, args.fixed_motif_length
    elif experiment == "motif-length":
        for motif_length in args.motif_lengths:
            yield args.fixed_series_length, motif_length
    elif experiment == "memory-budget":
        yield args.fixed_series_length, args.fixed_motif_length
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def memory_budgets(args, experiment):
    if experiment == "memory-budget":
        return args.scampi_max_memories
    return [args.scampi_max_memory]


def output_paths(output_dir, experiment):
    return (
        output_dir / f"{experiment}_runs.jsonl",
        output_dir / f"{experiment}_summary.csv",
    )


def run_key(
        experiment,
        n,
        motif_length,
        noise,
        seed,
        scampi_delta,
        scampi_max_memory,
        args):
    return (
        experiment,
        int(n),
        int(motif_length),
        int(args.n_instances),
        float(noise),
        int(seed),
        float(scampi_delta),
        str(scampi_max_memory),
        bool(args.scampi_exact_refine),
    )


def existing_keys(summary_path):
    if not summary_path.exists():
        return set()
    df = pd.read_csv(summary_path)
    if df.empty:
        return set()
    return {
        (
            row["experiment"],
            int(row["n"]),
            int(row["motif length"]),
            int(row["n instances"]),
            float(row["noise"]),
            int(row["seed"]),
            float(row["scampi_delta"]),
            str(row["scampi_max_memory"]),
            parse_bool(row["scampi_exact_refine"]),
        )
        for _, row in df.iterrows()
    }


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def plot_path(
        args,
        experiment,
        n,
        motif_length,
        noise,
        seed,
        scampi_delta,
        scampi_max_memory):
    plot_dir = args.plot_dir if args.plot_dir is not None else args.output_dir / "plots"
    filename = (
        f"{experiment}"
        f"_n_{n}"
        f"_m_{motif_length}"
        f"_noise_{format_token(noise)}"
        f"_seed_{seed}"
        f"_delta_{format_token(scampi_delta)}"
        f"_memory_{format_token(scampi_max_memory)}"
        ".pdf"
    )
    return plot_dir / filename


def format_token(value):
    return str(value).replace("-", "m").replace(".", "p").replace(" ", "")


def ground_truth_frame(positions, motif_length):
    intervals = [[int(pos), int(pos + motif_length)] for pos in positions]
    return pd.DataFrame({"Ground Truth": [intervals]})


def save_generated_plot(
        path,
        ds_name,
        series,
        motif_length,
        ground_truth,
        selected_motiflet,
        max_points):
    """Save a SCAMPI plot for generated data and recovered positions."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from scampi.plotting import plot_motifset

    motifsets = None
    if selected_motiflet:
        motifsets = [np.array(selected_motiflet, dtype=np.int32)]

    fig, _ = plot_motifset(
        ds_name,
        series,
        motifsets=motifsets,
        motif_length=motif_length,
        ground_truth=ground_truth_frame(ground_truth, motif_length),
        max_points=max_points,
        show=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_one(
        experiment,
        n,
        motif_length,
        noise,
        seed,
        scampi_delta,
        scampi_max_memory,
        args):
    from scampi.scampi import SCAMPI

    if args.k_max < args.n_instances:
        raise ValueError("k_max must be >= n_instances so selected_k is available")

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
    ds_name = (
        f"synthetic_{experiment}_n_{n}_m_{motif_length}"
        f"_noise_{noise}_seed_{seed}"
    )

    start = time.time()
    model = SCAMPI(
        ds_name,
        series,
        backend="scampi",
        n_jobs=args.n_jobs,
        slack=args.slack,
        scampi_delta=scampi_delta,
        scampi_max_memory=scampi_max_memory,
        scampi_exact_refine=args.scampi_exact_refine,
    )
    extents, motiflets, elbows = model.fit_k_elbow(
        args.k_max,
        motif_length,
        top_N=args.top_n,
        plot_elbows=False,
        plot_motifs_as_grid=False,
    )
    duration = time.time() - start

    selected = select_best_motiflet(
        motiflets,
        extents,
        args.n_instances,
        ground_truth,
        tolerance,
    )
    metrics = selected["metrics"]
    saved_plot_path = None
    if args.plot_generated:
        saved_plot_path = plot_path(
            args,
            experiment,
            n,
            motif_length,
            noise,
            seed,
            scampi_delta,
            scampi_max_memory,
        )
        save_generated_plot(
            saved_plot_path,
            ds_name,
            series,
            motif_length,
            ground_truth,
            selected["motiflet"],
            args.plot_max_points,
        )
        print(f"  Plot: {saved_plot_path}")

    full_record = {
        "experiment": experiment,
        "n": n,
        "motif_length": motif_length,
        "n_instances": args.n_instances,
        "noise": noise,
        "seed": seed,
        "backend": "scampi",
        "scampi_delta": scampi_delta,
        "scampi_max_memory": scampi_max_memory,
        "scampi_exact_refine": args.scampi_exact_refine,
        "slack": args.slack,
        "k_max": args.k_max,
        "top_n": args.top_n,
        "time_in_s": duration,
        "memory_in_mb": model.memory_usage,
        "ground_truth_positions": ground_truth,
        "clean_motif": motif,
        "extents": extents,
        "motiflets": motiflets,
        "elbows": elbows,
        "selected_k": args.n_instances,
        "selected_rank": selected["rank"],
        "selected_extent": selected["extent"],
        "selected_motiflet": selected["motiflet"],
        "plot_path": str(saved_plot_path) if saved_plot_path is not None else None,
        "position_tolerance": tolerance,
        **metrics,
    }

    summary_row = {
        "experiment": experiment,
        "n": n,
        "motif length": motif_length,
        "n instances": args.n_instances,
        "noise": noise,
        "seed": seed,
        "backend": "scampi",
        "scampi_delta": scampi_delta,
        "scampi_max_memory": scampi_max_memory,
        "scampi_exact_refine": args.scampi_exact_refine,
        "selected_k": args.n_instances,
        "time in s": duration,
        "memory in MB": model.memory_usage,
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
        "plot path": str(saved_plot_path) if saved_plot_path is not None else "",
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
    if args.list_experiments:
        print("\n".join(EXPERIMENTS))
        return

    experiments = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Method: scampi")
    print(f"Experiments: {experiments}")
    print(f"Noise levels: {args.noise_levels}")
    print(f"Seeds: {args.seeds}")
    print(f"SCAMPI deltas: {args.scampi_deltas}")
    print(f"Output: {args.output_dir}")

    for experiment in experiments:
        print(f"{experiment} SCAMPI max memory values: "
              f"{memory_budgets(args, experiment)}")
        jsonl_path, summary_path = output_paths(args.output_dir, experiment)
        if args.overwrite:
            jsonl_path.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
        completed = set() if args.overwrite else existing_keys(summary_path)

        for n, motif_length in experiment_grid(args, experiment):
            for noise in args.noise_levels:
                for seed in args.seeds:
                    for scampi_delta in args.scampi_deltas:
                        for scampi_max_memory in memory_budgets(args, experiment):
                            key = run_key(
                                experiment,
                                n,
                                motif_length,
                                noise,
                                seed,
                                scampi_delta,
                                scampi_max_memory,
                                args,
                            )
                            if key in completed:
                                print(
                                    f"Skipping {experiment}: n={n} "
                                    f"m={motif_length} noise={noise} seed={seed} "
                                    f"delta={scampi_delta} "
                                    f"memory={scampi_max_memory}; "
                                    f"summary row exists."
                                )
                                continue

                            print(
                                f"Running {experiment}: n={n} m={motif_length} "
                                f"noise={noise} seed={seed} "
                                f"delta={scampi_delta} memory={scampi_max_memory}"
                            )
                            try:
                                full_record, summary_row = run_one(
                                    experiment,
                                    n,
                                    motif_length,
                                    noise,
                                    seed,
                                    scampi_delta,
                                    scampi_max_memory,
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
