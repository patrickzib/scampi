import pandas as pd

from benchmarks.runners import common

N_RANGE = [650_000]
L_RANGE = [8192, 4096, 2048, 1024]
SCAMPI_MAX_MEMORY = "2GB"


def read_data():
    series = pd.read_csv(
        common.experiments_path("arrhythmia_subject231_channel0.csv"))
    ds_name = "Arrhythmia"
    print(f"Loaded dataset {ds_name} with length {series.shape}")
    return ds_name, series.iloc[:, 0].T


def test_plot_data():
    common.plot_dataset(read_data, "results/images/arrhythmia_data.pdf")


run_motiflets_scale_n = common.make_scale_n_runner(
    read_data, N_RANGE, L_RANGE, SCAMPI_MAX_MEMORY)


def main():
    print("running")
    run_motiflets_scale_n()


if __name__ == "__main__":
    main()
