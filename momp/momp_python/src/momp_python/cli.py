"""Command-line interface for the standalone MOMP Python app."""

import argparse
import sys

from .core import mpx_v2_motif_pair
from .io import load_series, write_json
from .momp_v9 import momp_v9
from .plotting import save_motif_pair_plot


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run standalone Python MOMP-style motif-pair search."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input time series file: .npy, .csv, .txt, .tsv, or .mat.",
    )
    parser.add_argument(
        "--mat-key",
        default=None,
        help="Variable key to read from a MATLAB .mat file.",
    )
    parser.add_argument(
        "--motif-length",
        "-m",
        type=int,
        required=True,
        help="Motif/subsequence length.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["momp-v9", "mpx-v2"],
        default="momp-v9",
        help=(
            "Algorithm to run. 'momp-v9' ports the MATLAB pruning/downsampling "
            "loop; 'mpx-v2' runs the MATLAB-style matrix profile baseline."
        ),
    )
    parser.add_argument(
        "--initial-downsample-rate",
        type=int,
        default=64,
        help="Initial MOMP downsampling rate. MATLAB momp_v9 hard-codes 64.",
    )
    parser.add_argument(
        "--exclusion-zone",
        type=int,
        default=None,
        help="Minimum start-position separation for mpx-v2. Defaults to motif_length // 2.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional plot path, e.g. result.pdf or result.png.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable stdout summary.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    series = load_series(args.input, mat_key=args.mat_key)
    if args.algorithm == "momp-v9":
        result = momp_v9(
            series,
            args.motif_length,
            verbose=not args.quiet,
            initial_downsample_rate=args.initial_downsample_rate,
        )
    else:
        result = mpx_v2_motif_pair(
            series,
            args.motif_length,
            exclusion_zone=args.exclusion_zone,
        )

    payload = {
        "input": args.input,
        "mat_key": args.mat_key,
        "requested_algorithm": args.algorithm,
        **result.to_dict(),
    }

    if args.output:
        write_json(args.output, payload)

    if args.plot:
        save_motif_pair_plot(args.plot, series, result)

    if not args.quiet:
        print(
            f"distance={result.distance:0.6g} "
            f"locations={result.locations} "
            f"locations_matlab={result.locations_matlab} "
            f"time={result.runtime_seconds:0.3f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
