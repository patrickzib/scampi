"""Run MOMP benchmark sweeps.

Examples
--------
List available method presets:
    python tests/run_momp_benchmarks.py --list-methods

Run the default method (currently FAISS-HNSW) on all datasets:
    python tests/run_momp_benchmarks.py

Run selected methods and datasets:
    python tests/run_momp_benchmarks.py \
        --datasets EOG_one_hour_50_Hz,WindTurbine \
        --lengths 512,1024,2048 \
        --methods scampi,faiss-hnsw,faiss-ivf \
        --k-max 3

Sweep parameter grids by passing comma-separated values:
    python tests/run_momp_benchmarks.py \
        --methods faiss-hnsw \
        --faiss-M 32,64 \
        --search-radius 10 \
        --faiss-efConstruction 300,500 \
        --faiss-efSearch 400,800

Run against a specific server data directory:
    python tests/run_momp_benchmarks.py \
        --data-path /vol/fob-wbib-vol2/wbi/schaefpa/motiflets/momp \
        --methods faiss-hnsw

Available method presets are listed in AVAILABLE_METHODS below. Each selected
method expands to one or more BenchmarkRun entries from the relevant parameter
grid. Results are written to tests/results, and local runs use a 10,000 point
slice per dataset to keep smoke checks manageable unless --local-n full is set.
"""

import argparse
import itertools
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

if "NUMBA_CACHE_DIR" not in os.environ:
    cache_name = f"scampi-numba-{os.getuid()}-{os.getpid()}"
    os.environ["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / cache_name)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


HPC_DATA_PATH = "/vol/fob-wbib-vol2/wbi/schaefpa/motiflets/momp"
RUN_LOCAL = not (os.path.exists(HPC_DATA_PATH) and os.path.isdir(HPC_DATA_PATH))

DEFAULT_METHODS = ["faiss-hnsw"]
AVAILABLE_METHODS = [
    "scampi",
    "faiss-hnsw",
    "faiss-lsh",
    "faiss-ivf",
    "faiss-ivfpq",
    "faiss-ivfpq-hnsw",
    "pynndescent",
    "annoy",
    "scalable",
    "scalable-subsampling",
]


@dataclass(frozen=True)
class BenchmarkRun:
    label: str
    backend: str
    kwargs: dict


def parse_csv(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_optional_int_csv(value):
    def cast(item):
        if item.lower() in {"none", "auto"}:
            return None
        return int(item)

    return parse_csv(value, cast)


def parse_methods(value):
    methods = parse_csv(value)
    unknown = sorted(set(methods) - set(AVAILABLE_METHODS))
    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}. Available methods: {AVAILABLE_METHODS}"
        )
    return methods


def build_runs(args):
    runs = []
    methods = parse_methods(args.methods)

    if "scampi" in methods:
        for delta in args.scampi_deltas:
            runs.append(BenchmarkRun(
                label=(
                    f"scampi delta={delta} "
                    f"exact_refine={args.scampi_exact_refine}"
                ),
                backend="scampi",
                kwargs={
                    "scampi_delta": delta,
                    "scampi_max_memory": args.scampi_max_memory,
                    "scampi_exact_refine": args.scampi_exact_refine,
                },
            ))

    if "faiss-hnsw" in methods:
        for M, ef_construction, ef_search in itertools.product(
                args.faiss_M,
                args.faiss_efConstruction,
                args.faiss_efSearch):
            runs.append(BenchmarkRun(
                label=(
                    f"faiss HNSW M={M} "
                    f"efConstruction={ef_construction} "
                    f"efSearch={ef_search} "
                    f"search_radius={args.search_radius}"
                ),
                backend="faiss",
                kwargs={
                    "faiss_index": "HNSW",
                    "faiss_M": M,
                    "faiss_efConstruction": ef_construction,
                    "faiss_efSearch": ef_search,
                    "search_radius": args.search_radius,
                },
            ))

    if "faiss-lsh" in methods:
        for nbits in args.faiss_nbits:
            runs.append(BenchmarkRun(
                label=(
                    f"faiss LSH nbits={nbits} "
                    f"search_radius={args.search_radius}"
                ),
                backend="faiss",
                kwargs={
                    "faiss_index": "LSH",
                    "faiss_nbits": nbits,
                    "search_radius": args.search_radius,
                },
            ))

    if "faiss-ivf" in methods:
        for nlist, nprobe in itertools.product(args.faiss_nlist, args.faiss_nprobe):
            kwargs = {
                "faiss_index": "IVF",
                "faiss_nprobe": nprobe,
                "search_radius": args.search_radius,
            }
            if nlist is not None:
                kwargs["faiss_nlist"] = nlist
            runs.append(BenchmarkRun(
                label=(
                    f"faiss IVF nlist={nlist or 'sqrt(n)'} "
                    f"nprobe={nprobe} "
                    f"search_radius={args.search_radius}"
                ),
                backend="faiss",
                kwargs=kwargs,
            ))

    if "faiss-ivfpq" in methods:
        runs.extend(build_pq_runs("IVFPQ", args))

    if "faiss-ivfpq-hnsw" in methods:
        runs.extend(build_pq_runs("IVFPQ+HNSW", args))

    if "pynndescent" in methods:
        for values in itertools.product(
                args.pynndescent_n_neighbors,
                args.pynndescent_leaf_size,
                args.pynndescent_pruning_degree_multiplier,
                args.pynndescent_diversify_prob,
                args.pynndescent_n_search_trees,
                args.pynndescent_search_epsilon):
            runs.append(BenchmarkRun(
                label="pynndescent",
                backend="pynndescent",
                kwargs={
                    "pynndescent_n_neighbors": values[0],
                    "pynndescent_leaf_size": values[1],
                    "pynndescent_pruning_degree_multiplier": values[2],
                    "pynndescent_diversify_prob": values[3],
                    "pynndescent_n_search_trees": values[4],
                    "pynndescent_search_epsilon": values[5],
                },
            ))

    if "annoy" in methods:
        for n_trees, search_k in itertools.product(
                args.annoy_n_trees,
                args.annoy_search_k):
            runs.append(BenchmarkRun(
                label=f"annoy n_trees={n_trees} search_k={search_k}",
                backend="annoy",
                kwargs={
                    "annoy_n_trees": n_trees,
                    "annoy_search_k": search_k,
                },
            ))

    if "scalable" in methods:
        runs.append(BenchmarkRun(
            label="scalable",
            backend="scalable",
            kwargs={},
        ))

    if "scalable-subsampling" in methods:
        for subsampling in args.subsampling:
            runs.append(BenchmarkRun(
                label=f"scalable subsampling={subsampling}",
                backend="scalable",
                kwargs={"subsampling": subsampling},
            ))

    return runs


def build_pq_runs(faiss_index, args):
    runs = []
    for nlist, nprobe, pq_m, pq_nbits in itertools.product(
            args.faiss_nlist,
            args.faiss_nprobe,
            args.faiss_pq_m,
            args.faiss_pq_nbits):
        kwargs = {
            "faiss_index": faiss_index,
            "faiss_nprobe": nprobe,
            "faiss_pq_nbits": pq_nbits,
            "search_radius": args.search_radius,
        }
        if nlist is not None:
            kwargs["faiss_nlist"] = nlist
        if pq_m is not None:
            kwargs["faiss_pq_m"] = pq_m

        runs.append(BenchmarkRun(
            label=(
                f"faiss {faiss_index} nlist={nlist or 'sqrt(n)'} "
                f"nprobe={nprobe} pq_m={pq_m or 'auto'} "
                f"pq_bits={pq_nbits} search_radius={args.search_radius}"
            ),
            backend="faiss",
            kwargs=kwargs,
        ))
    return runs


def selected_filenames(args, ut):
    lengths = np.array([properties[-1] for properties in ut.filenames.values()])
    filenames = [list(ut.filenames.keys())[idx] for idx in np.argsort(lengths)]

    if args.datasets:
        requested = parse_csv(args.datasets)
        unknown = sorted(set(requested) - set(ut.filenames.keys()))
        if unknown:
            raise ValueError(
                f"Unknown datasets: {unknown}. Available datasets: {list(ut.filenames)}"
            )
        filenames = [filename for filename in filenames if filename in requested]

    return filenames


def default_lengths():
    if RUN_LOCAL:
        return [2 ** 9]
    return [2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MOMP benchmark variants with selectable methods."
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated methods to run. Use --list-methods to inspect "
            "available names."
        ),
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print available method names and exit.",
    )
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset keys. Defaults to all datasets by length.",
    )
    parser.add_argument(
        "--lengths",
        type=lambda value: parse_csv(value, int),
        default=default_lengths(),
        help="Comma-separated motif lengths. Defaults to 512 locally, 512..4096 on HPC.",
    )
    parser.add_argument(
        "--data-path",
        help="Directory containing MOMP .mat files. Forces full-data mode.",
    )
    parser.add_argument(
        "--local-n",
        default=None,
        help="Local-mode time series length cap. Use 'full' to disable the cap.",
    )
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--scampi-deltas", type=lambda v: parse_csv(v, float),
                        default=[0.1])
    parser.add_argument("--scampi-max-memory", default="2 GB")
    parser.add_argument(
        "--scampi-exact-refine",
        action="store_true",
        help="Refine pyattimo seed positions with exact k-NN distances.",
    )

    parser.add_argument("--faiss-M", type=lambda v: parse_csv(v, int),
                        default=[64])
    parser.add_argument("--faiss-efConstruction", type=lambda v: parse_csv(v, int),
                        default=[500])
    parser.add_argument("--faiss-efSearch", type=lambda v: parse_csv(v, int),
                        default=[400])
    parser.add_argument("--faiss-nlist", type=parse_optional_int_csv,
                        default=[None])
    parser.add_argument("--faiss-nprobe", type=lambda v: parse_csv(v, int),
                        default=[10])
    parser.add_argument("--faiss-nbits", type=lambda v: parse_csv(v, int),
                        default=[4])
    parser.add_argument(
        "--search-radius",
        type=int,
        default=10,
        help=(
            "Candidate shortlist multiplier for approximate vector methods. "
            "The backend requests search_radius * k raw neighbors before "
            "exclusion-zone post-processing."
        ),
    )
    parser.add_argument("--faiss-pq-m", type=parse_optional_int_csv,
                        default=[None])
    parser.add_argument("--faiss-pq-nbits", type=lambda v: parse_csv(v, int),
                        default=[8])

    parser.add_argument("--pynndescent-n-neighbors", type=lambda v: parse_csv(v, int),
                        default=[60])
    parser.add_argument("--pynndescent-leaf-size", type=lambda v: parse_csv(v, int),
                        default=[48])
    parser.add_argument(
        "--pynndescent-pruning-degree-multiplier",
        type=lambda v: parse_csv(v, float),
        default=[2.0],
    )
    parser.add_argument("--pynndescent-diversify-prob",
                        type=lambda v: parse_csv(v, float),
                        default=[0.0])
    parser.add_argument("--pynndescent-n-search-trees",
                        type=lambda v: parse_csv(v, int),
                        default=[1])
    parser.add_argument("--pynndescent-search-epsilon",
                        type=lambda v: parse_csv(v, float),
                        default=[0.2])

    parser.add_argument("--annoy-n-trees", type=lambda v: parse_csv(v, int),
                        default=[100])
    parser.add_argument("--annoy-search-k", type=lambda v: parse_csv(v, int),
                        default=[-1])
    parser.add_argument("--subsampling", type=lambda v: parse_csv(v, int),
                        default=[8, 16])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_methods:
        print("\n".join(AVAILABLE_METHODS))
        return

    import utils as ut
    local_n = args.local_n
    if local_n is None and not RUN_LOCAL:
        local_n = "full"
    ut.configure_paths(data_path=args.data_path, local_n=local_n)

    runs = build_runs(args)
    print(f"Selected methods: {[run.label for run in runs]}")
    print(f"Motif lengths: {args.lengths}")

    for filename in selected_filenames(args, ut):
        ds_name, _, _, _ = ut.filenames[filename]
        print(f"\nDataset: {filename} ({ds_name})")
        data = ut.read_mat(filename)

        for run in runs:
            kwargs = dict(run.kwargs)
            subsampling = kwargs.pop("subsampling", None)
            print(f"\n  Method: {run.label}", flush=True)
            ut.run_safe(
                filename,
                data,
                args.lengths,
                args.k_max,
                run.backend,
                subsampling=subsampling,
                n_jobs=args.n_jobs,
                local_n=local_n,
                **kwargs,
            )


if __name__ == "__main__":
    main()
