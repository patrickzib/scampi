import pandas as pd
import scipy.io as sio

from benchmarks.runners import common
from scampi.scampi import SCAMPI

N_RANGE = [3 * 10 ** 6]
L_RANGE = [32, 64, 23]
SCAMPI_MAX_MEMORY = "2GB"


def test_plot_data():
    ds_name, series = read_penguin_1m()
    series = series.iloc[497699 - 5000: 497699 + 5000, 0].T

    common.plot_dataset(
        lambda: (ds_name, series), "results/images/penguin_data.pdf")


def read_penguin_1m_channel0():
    ds_name, series = read_penguin_1m(channel="X-Acc")
    return ds_name, series.T


def read_penguin_1m(channel=None):
    series = pd.read_csv(common.experiments_path("penguin.txt"),
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguin1M"

    if channel is not None:
        return ds_name, series[[channel]]

    return ds_name, series


def read_penguin_3m():
    ds_name = "Penguin3M"
    test = sio.loadmat(common.DATASETS_ROOT / "PeVAMmotif" / "penguinLabel.mat")
    series = test["data"].T
    return ds_name, pd.DataFrame(series[2, :]).T


def test_plotting():
    ds_name, ts = read_penguin_1m()
    ts = ts.iloc[497699 - 50_000: 497699 + 50_000, -2]

    mm = SCAMPI(ds_name, ts)
    mm.plot_dataset(path="results/images/penguin_data_raw.pdf")


def run_motiflets_scale_n(
        backends=("scampi",),
        delta=0.1,
        use_1m=False,
        k_max=10,
        subsampling=None):
    common.run_scale_n(
        read_penguin_1m_channel0 if use_1m else read_penguin_3m,
        N_RANGE,
        L_RANGE,
        k_max,
        backends=backends,
        delta=delta,
        subsampling=subsampling,
        scampi_max_memory=SCAMPI_MAX_MEMORY)


def main():
    print("running")
    run_motiflets_scale_n()


if __name__ == "__main__":
    main()
