import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from benchmarks.converters.utils import ConverterUtils, RESULTS_DIR, parser


DEFAULT_DATASETS = ["stator_winding"]


def parse_args():
    arg_parser = parser(
        "Convert per-length MOMP CSVs by recomputing exact k-NN motiflets."
    )
    ConverterUtils.add_common_args(arg_parser)
    arg_parser.set_defaults(k_max=3)
    arg_parser.add_argument(
        "--input-dir",
        type=Path,
        default=RESULTS_DIR / "momp",
        help="Directory containing momp_<length>.csv input files.",
    )
    arg_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "momp_pair",
        help="Directory for converted MOMP pair CSVs.",
    )
    return arg_parser.parse_args()


def output_filename(output_dir, ds_name):
    return output_dir / f"MOMP_{ds_name}.csv"


def read_length_inputs(input_dir, lengths):
    frames = {}
    for motif_length in lengths:
        path = input_dir / f"momp_{motif_length}.csv"
        if path.exists():
            frames[motif_length] = pd.read_csv(path).set_index("dataset")
        else:
            print(f"Missing input file for m={motif_length}: {path}")
    return frames


def row_for_length(frames, ds_name, motif_length):
    frame = frames.get(motif_length)
    if frame is None or ds_name not in frame.index:
        return None
    return frame.loc[ds_name]


def convert_dataset(ds_name, lengths, k_max, frames, output_dir, ut):
    print(f"Converting {ds_name}")
    ts = ut.read_mat(ds_name)
    df = ConverterUtils.result_frame()

    for motif_length in lengths:
        row = row_for_length(frames, ds_name, motif_length)
        if row is None:
            print(f"  Skipping m={motif_length}; no row for {ds_name}")
            continue

        print(f"  Processing m={motif_length}")
        motiflets = ConverterUtils.parse_array(row["motiflet"])
        best_motiflet, min_extent = ConverterUtils.exact_refine_from_seeds(
            ts,
            motiflets,
            motif_length,
            k_max,
            include_seed_set=True,
        )
        df.loc[len(df.index)] = [
            len(ts),
            motif_length,
            "MOMP",
            row["time in s"],
            -1,
            best_motiflet,
            min_extent,
        ]

    ConverterUtils.write_csv(df, output_filename(output_dir, ds_name))


def main():
    args = parse_args()
    from benchmarks import utils as ut

    ut.configure_paths(data_path=args.data_path)
    frames = read_length_inputs(args.input_dir, args.lengths)
    for ds_name in ConverterUtils.selected_datasets(
            args, ut.filenames, default=DEFAULT_DATASETS):
        convert_dataset(
            ds_name,
            args.lengths,
            args.k_max,
            frames,
            args.output_dir,
            ut,
        )


if __name__ == "__main__":
    main()
