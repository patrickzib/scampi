import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from numba import set_num_threads

from benchmarks.converters.utils import ConverterUtils, RESULTS_DIR, parser


def parse_args():
    arg_parser = parser(
        "Convert pyattimo CSVs by recomputing exact k-NN motiflets from "
        "stored pyattimo seed positions."
    )
    ConverterUtils.add_common_args(arg_parser)
    arg_parser.add_argument(
        "--input-dir",
        type=Path,
        default=RESULTS_DIR / "pyattimo_0.7.0_8GB",
        help="Directory containing pyattimo CSVs.",
    )
    arg_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "pyattimo_converted",
        help="Directory for converted pyattimo CSVs.",
    )
    arg_parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Numba thread count for the refinement kernel.",
    )
    return arg_parser.parse_args()


def input_filename(input_dir, ds_name, k_max):
    return input_dir / f"scalability_n_{ds_name}_{k_max}_pyattimo_delta_0.1.csv"


def output_filename(output_dir, ds_name, k_max):
    return output_dir / f"scalability_n_{ds_name}_{k_max}_pyattimo_delta_0.1.csv"


def convert_dataset(ds_name, lengths, k_max, input_dir, output_dir, ut):
    input_path = input_filename(input_dir, ds_name, k_max)
    df_attimo = ConverterUtils.indexed_csv(input_path)
    if df_attimo is None:
        print(f"Skipping {ds_name}; missing {input_path}")
        return

    print(f"Converting {ds_name}")
    ts = ut.read_mat(ds_name)
    df = ConverterUtils.result_frame()

    for motif_length in lengths:
        print(f"  Processing m={motif_length}", flush=True)
        if motif_length in df_attimo.index:
            current_motiflets = ConverterUtils.parse_array(
                df_attimo.loc[motif_length, "motiflet"])
            current_extent = df_attimo.loc[motif_length, "extent"]
            best_motiflet, min_extent = ConverterUtils.exact_refine_from_seeds(
                ts, current_motiflets, motif_length, k_max)
            if min_extent > current_extent:
                min_extent = current_extent
                best_motiflet = current_motiflets
            time_value = df_attimo.loc[motif_length, "time in s"]
            memory_value = df_attimo.loc[motif_length, "memory in MB"]
        else:
            best_motiflet = []
            min_extent = np.inf
            time_value = ConverterUtils.extrapolate_metric(
                df_attimo, motif_length, "time in s")
            memory_value = ConverterUtils.extrapolate_metric(
                df_attimo, motif_length, "memory in MB")

        df.loc[len(df.index)] = [
            len(ts),
            motif_length,
            "attimo",
            time_value,
            memory_value,
            best_motiflet,
            min_extent,
        ]

    ConverterUtils.write_csv(df, output_filename(output_dir, ds_name, k_max))


def main():
    args = parse_args()
    set_num_threads(args.num_threads)
    from benchmarks import utils as ut

    ut.configure_paths(data_path=args.data_path)
    for ds_name in ConverterUtils.selected_datasets(args, ut.filenames):
        convert_dataset(
            ds_name,
            args.lengths,
            args.k_max,
            args.input_dir,
            args.output_dir,
            ut,
        )


if __name__ == "__main__":
    main()
