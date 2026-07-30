import numpy as np
import pandas as pd

from benchmarks.runners import common

N_RANGE = [173_875]
L_RANGE = [8192, 4096, 2048, 1024]
SCAMPI_MAX_MEMORY = "2GB"


def read_data(selection=None):
    desc_filename = common.experiments_path("pamap_desc.txt")
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []
    for idx, row in enumerate(desc_file):
        if selection is not None and idx not in selection: continue

        (ts_name, window_size), change_points = row[:2], row[2:]
        if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
        ts = np.load(file=common.experiments_path("pamap_data.npz"))[ts_name]

        df.append(
            (ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return "PAMAP", pd.DataFrame.from_records(
        df, columns=["name", "window_size", "change_points", "time_series"]).time_series[0]


def test_plot_data():
    selection = [126]  # Outdoor

    ds_name, series = read_data(selection)
    ts = series
    print(f"Loaded dataset PAMAP with length {len(ts)}")

    common.plot_dataset(
        lambda: (ds_name, ts), "results/images/pamap_data.pdf")


run_motiflets_scale_n = common.make_scale_n_runner(
    read_data, N_RANGE, L_RANGE, SCAMPI_MAX_MEMORY)


def main():
    print("running")
    run_motiflets_scale_n()

if __name__ == "__main__":
    main()
