import traceback

from benchmarks.runners import run_gap as gap
from benchmarks.runners import run_pamap as pamap
from benchmarks.runners import run_penguin as penguin
from benchmarks.runners import run_astro as astro
from benchmarks.runners import run_dishwasher as dishwasher
from benchmarks.runners import run_eeg_physiodata as eeg
from benchmarks.runners import run_arrhythmia as arrhythmia


BENCHMARKS = [
    # (module, extra keyword arguments)
    # (penguin, {"use_1m": False}),
    # (astro, {}),
    # (arrhythmia, {}),
    # (dishwasher, {}),
    # (eeg, {}),
    # (gap, {}),
    # (pamap, {}),
    (penguin, {"use_1m": True}),
]


def run_safe(module, backends, delta, k_max, **kwargs):
    try:
        module.run_motiflets_scale_n(
            backends=backends, delta=delta, k_max=k_max, **kwargs)
    except Exception as e:
        print(traceback.format_exc())
    except BaseException as e:
        print(f"Caught a panic: {e}")


def main():
    backends = ["scampi"]
    deltas = [0.1]
    k_maxs = [10, 20, 30, 40]

    for delta in deltas:
        for k_max in k_maxs:
            print(f"Using delta {delta}")
            for module, kwargs in BENCHMARKS:
                run_safe(module, backends, delta, k_max, **kwargs)

if __name__ == "__main__":
    main()
