from benchmarks.runners import common

N_RANGE = [2049280]
L_RANGE = [8192, 4096, 2048, 1024]
SCAMPI_MAX_MEMORY = "2GB"

def read_data():
    return common.read_single_column_csv(
        common.original_path("GAP.csv"), "GAP")

def test_plot_data():
    common.plot_dataset(read_data, "results/images/gap_data.pdf")


run_motiflets_scale_n = common.make_scale_n_runner(
    read_data, N_RANGE, L_RANGE, SCAMPI_MAX_MEMORY)


def main():
    print("running")
    run_motiflets_scale_n()

if __name__ == "__main__":
    main()
