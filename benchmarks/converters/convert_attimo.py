import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from benchmarks.converters.utils import ConverterUtils, RESULTS_DIR, parser


DEFAULT_DATASETS = [
    "HAR_Ambient_Sensor_Data",
    "Challenge2009TestSetA_101a",
    "swtAttack38",
]


def parse_args():
    arg_parser = parser(
        "Convert ATTIMO CSVs by recomputing exact k-NN motiflets from "
        "stored ATTIMO seed positions."
    )
    ConverterUtils.add_common_args(arg_parser)
    arg_parser.set_defaults(k_max=3)
    arg_parser.add_argument(
        "--input-dir",
        type=Path,
        default=RESULTS_DIR / "attimo",
        help="Directory containing ATTIMO CSVs.",
    )
    arg_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "attimo_pair",
        help="Directory for converted ATTIMO CSVs.",
    )
    return arg_parser.parse_args()


def input_filename(input_dir, ds_name, k_max):
    return input_dir / f"scalability_n_{ds_name}_{k_max}_pyattimo_delta_0.1.csv"


def output_filename(output_dir, ds_name, k_max):
    return output_dir / f"scalability_n_{ds_name}_{k_max}_attimo.csv"


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
        print(f"  Processing m={motif_length}")
        if motif_length in df_attimo.index:
            motiflets = ConverterUtils.parse_array(
                df_attimo.loc[motif_length, "motiflet"])
            best_motiflet, min_extent = ConverterUtils.exact_refine_from_seeds(
                ts, motiflets, motif_length, k_max)
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
    from benchmarks import utils as ut

    ut.configure_paths(data_path=args.data_path)
    datasets = ConverterUtils.selected_datasets(
        args, ut.filenames, default=DEFAULT_DATASETS)
    for ds_name in datasets:
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
