import os
import psutil
import numpy as np


SCAMPI_TOP_N_PYATTIMO = "pyattimo"
SCAMPI_TOP_N_MASK = "mask"
SCAMPI_TOP_N_STRATEGIES = {SCAMPI_TOP_N_PYATTIMO, SCAMPI_TOP_N_MASK}


class PyAttimoError(ImportError):
    """Raised when the pyattimo SCAMPI backend cannot be loaded."""


class SCAMPINearestNeighbors:
    """SCAMPI/pyattimo-based motiflet backend.

    Parameters
    ----------
    m : int
        Motif length.
    k_max : int
        Maximum motiflet support index to request from pyattimo. The backend
        asks pyattimo for ``support=k_max - 1`` and stores returned motiflets by
        ``mot.support``.
    top_k : int, default=1
        Number of motiflet candidates to keep per support.
    slack : float, default=0.5
        Exclusion-zone factor. The exclusion zone passed to pyattimo is
        ``int(m * slack)``.
    verbose : bool, default=False
        Whether to print pyattimo progress and exact-refinement diagnostics.
    **kwargs
        scampi_delta : float or None, default=0.1
            Approximation delta passed to pyattimo. If truthy, pyattimo is run
            with ``stop_on_threshold=True`` and ``fraction_threshold=log(n)/n``.
        scampi_max_memory : str, default="2 GB"
            Maximum memory string passed to pyattimo.
        scampi_exact_refine : bool, default=False
            If true, each pyattimo motiflet candidate is treated as a set of
            seed positions. For each seed, the backend recomputes exact
            z-normalized k-nearest neighbors and replaces the pyattimo candidate
            only when the exact pairwise extent is smaller. This is more
            expensive because it performs full sliding-dot-product searches for
            the seed positions.
    """
    def __init__(
            self,
            m,
            k_max,
            top_k=1,
            slack=0.5,
            verbose=False,
            elbow_deviation=1.0,
            filter=True,
            **kwargs):

        self.m = m
        self.k_max = k_max
        self.slack = slack
        self.top_k = top_k
        self.verbose = verbose
        self.elbow_deviation = elbow_deviation
        self.filter = filter

        self.scampi_delta = kwargs.get("scampi_delta", 0.1)
        self.scampi_max_memory = kwargs.get("scampi_max_memory", "2 GB")
        self.scampi_exact_refine = kwargs.get("scampi_exact_refine", False)
        self.scampi_top_n_strategy = kwargs.get(
            "scampi_top_n_strategy", SCAMPI_TOP_N_PYATTIMO)

        if self.top_k < 1:
            raise ValueError("top_N must be >= 1")
        if self.scampi_top_n_strategy not in SCAMPI_TOP_N_STRATEGIES:
            raise ValueError('scampi_top_n_strategy must be "pyattimo" or "mask"')

        consumed_kwargs = {
            "scampi_delta",
            "scampi_max_memory",
            "scampi_exact_refine",
            "scampi_top_n_strategy",
        }
        unused_kwargs = {
            key: value for key, value in kwargs.items()
            if key not in consumed_kwargs
        }

        print(
            "SCAMPI kwargs: "
            f"delta={self.scampi_delta}, "
            f"max_memory={self.scampi_max_memory}, "
            f"exact_refine={self.scampi_exact_refine}, "
            f"scampi_top_n_strategy={self.scampi_top_n_strategy}, "
            f"unused={unused_kwargs}"
        )

    def compute_knns(self, X):
        """Compute k-nearest neighbors using SCAMPI motiflet discovery."""

        if X.shape[0] != 1:
            raise ValueError("SCAMPI can handle univariate data, only.")

        try:
            import pyattimo
        except ImportError as e:
            raise PyAttimoError(f"Failed to import SCAMPI: {str(e)}") from e

        pid = os.getpid()
        process = psutil.Process(pid)

        ts = X.flatten()
        if self.scampi_top_n_strategy == SCAMPI_TOP_N_MASK:
            return self._compute_masked_knns(ts, pyattimo, process)

        return self._compute_pyattimo_knns(ts, self.top_k, pyattimo, process)

    def _compute_pyattimo_knns(self, ts, top_k, pyattimo, process):
        """Run one pyattimo motiflet discovery pass."""
        n = ts.shape[-1] - self.m + 1
        k_motiflet_distances = np.full((self.k_max, top_k), np.inf, dtype=np.float64)
        k_motiflet_candidates = np.empty(self.k_max, dtype=object)

        for i in range(len(k_motiflet_candidates)):
            k_motiflet_candidates[i] = []

        memory_usage = 0.0

        # Prepare common arguments
        attimo_args = {
            'ts': ts,
            'w': self.m,
            'top_k': top_k,
            'support': self.k_max - 1,
            'exclusion_zone': int(self.m * self.slack),
            'max_memory': self.scampi_max_memory,
            'observability_file': None  # "observe.csv"
        }

        if self.scampi_delta:
            attimo_args.update({
                'delta': self.scampi_delta,
                'stop_on_threshold': True,
                'fraction_threshold': np.log(n) / n
            })

            if self.verbose:
                print(f"\tSCAMPI: Setting "
                      f"\n\t\tw={self.m}, "
                      f"\n\t\tdelta={self.scampi_delta}, "
                      f"\n\t\tsupport={attimo_args['support']}, "
                      f"\n\t\tmax_memory={attimo_args['max_memory']}, "
                      f"\n\t\texclusion_zone={attimo_args['exclusion_zone']}, "
                      f"\n\t\ttop_k={attimo_args['top_k']}, "
                      f"\n\t\tscampi_top_n_strategy={self.scampi_top_n_strategy}, "
                      f"\n\t\tstop_on_threshold={attimo_args['stop_on_threshold']}, "
                      # f"\n\t\tfraction_threshold=log(n)/n", flush=True
                      , flush=True)

        m_iter = None

        try:
            m_iter = pyattimo.MotifletsIterator(**attimo_args)

            if self.verbose:
                print("\tComputing scampi with SCAMPI...", flush=True)

            for mot in m_iter:
                if self.verbose:
                    print(f"\t\t{mot}", flush=True)

                test_k = mot.support
                if test_k < self.k_max:
                    motiflet = np.array(mot.indices, dtype=np.int32)
                    extent = mot.extent ** 2

                    if self.scampi_exact_refine and test_k > 0:
                        refined_motiflet, refined_extent = compute_knn(
                            ts,
                            motiflet,
                            self.m,
                            test_k,
                            slack=self.slack
                        )
                        if self.verbose:
                            print(
                                "\t\tExact refine: "
                                f"k={test_k} "
                                f"extent={extent:.6g} -> {refined_extent:.6g} "
                                f"changed={not np.array_equal(motiflet, refined_motiflet)}",
                                flush=True
                            )
                        if refined_extent < extent:
                            motiflet = refined_motiflet
                            extent = refined_extent

                    rank = len(k_motiflet_candidates[test_k])
                    if rank >= top_k:
                        continue

                    k_motiflet_distances[test_k][rank] = extent

                    # TODO: expose mot.lower_bound for confidence scores
                    k_motiflet_candidates[test_k].append(motiflet)

            if self.verbose:
                print(f"\t{len(k_motiflet_candidates[-1])}-Motiflet"
                      f"\n\t\tPos: {k_motiflet_candidates[-1]} "
                      f"\n\t\tExtent: {k_motiflet_distances[-1]}", flush=True)

            memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

            times = m_iter.timings()
            print(f"\tTime for Build {times.get('repetition_setup_s', 'n/a')}")
            print(f"\tTime for Test {times.get('pair_discovery_s', 'n/a')}")

        except Exception as e:
            print(f"SCAMPI computation failed: {str(e)}", flush=True)

        if m_iter:
            del m_iter

        return k_motiflet_distances, k_motiflet_candidates, memory_usage

    def _compute_masked_knns(self, ts, pyattimo, process):
        """Run repeated top-1 pyattimo passes, masking each rank's elbows."""
        from scampi.scampi import find_and_filter_elbow_points

        k_motiflet_distances = np.full((self.k_max, self.top_k), np.inf, dtype=np.float64)
        k_motiflet_candidates = np.empty(self.k_max, dtype=object)

        for i in range(len(k_motiflet_candidates)):
            k_motiflet_candidates[i] = []

        timestamp_mask = np.zeros(ts.shape[-1], dtype=bool)
        rng = np.random.default_rng(1234)
        memory_usage = 0.0

        rank_range = range(self.top_k)
        if self.verbose:
            try:
                from tqdm.auto import tqdm
                rank_range = tqdm(
                    rank_range,
                    total=self.top_k,
                    desc="SCAMPI masked top-N",
                )
            except ImportError:
                pass

        for rank in rank_range:
            if self.verbose:
                print(f"\tSCAMPI masked top-N rank {rank + 1}/{self.top_k}", flush=True)

            run_ts = _apply_timestamp_mask(
                ts,
                timestamp_mask,
                self.m,
                rng,
            )
            run_distances, run_candidates, run_memory = self._compute_pyattimo_knns(
                run_ts, 1, pyattimo, process)
            memory_usage = max(memory_usage, run_memory)

            empty = np.array([], dtype=np.int32)
            for k, candidates in enumerate(run_candidates):
                k_motiflet_distances[k, rank] = run_distances[k, 0]
                k_motiflet_candidates[k].append(candidates[0] if candidates else empty)

            elbows = find_and_filter_elbow_points(
                run_distances[:, 0],
                run_candidates,
                self.m,
                rank=0,
                filter=self.filter,
                elbow_deviation=self.elbow_deviation,
            )
            for k in elbows[:1]:
                for pos in run_candidates[k][0]:
                    timestamp_mask[
                        int(pos - 2 * self.m):int(pos + 3 * self.m)
                    ] = True
            print("masked timestamps:", np.sum(timestamp_mask))

        return k_motiflet_distances, k_motiflet_candidates, memory_usage


def _moving_average_fill(series, window):
    ret = np.cumsum(series, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret / window


def _apply_timestamp_mask(series, timestamp_mask, window, rng):
    mask = timestamp_mask.astype(series.dtype, copy=False)
    mavg = _moving_average_fill(series, window) * mask
    noise = rng.normal(scale=np.abs(mavg), size=mask.shape) * mask
    noise += mavg
    reset_series = series * (1 - mask)
    return reset_series + noise


def compute_knn(ts, motiflet_seeds, m, k, slack=0.5):
    """Refine pyattimo seed positions into an exact k-neighbor motiflet.

    Each seed position is used as a query subsequence. The function computes its
    exact non-overlapping k nearest neighbors under z-normalized Euclidean
    distance, evaluates the exact pairwise extent of that k-motiflet, and
    returns the candidate with the smallest extent.
    """
    from scampi.distances import (
        sliding_mean_std,
        znormed_euclidean_distance,
        znormed_euclidean_distance_single,
    )
    from scampi.scampi import _argknn, _sliding_dot_product

    if len(motiflet_seeds) == 0 or k <= 0:
        return np.full(max(k, 0), -1, dtype=np.int32), np.inf

    halve_m = np.int32(m * slack)
    n = ts.shape[-1] - m + 1
    preprocessing = sliding_mean_std(ts, m)

    knns = np.full((len(motiflet_seeds), k), -1, dtype=np.int32)
    extents = np.full(len(motiflet_seeds), np.inf, dtype=np.float64)

    for i, start in enumerate(motiflet_seeds):
        if 0 <= start < n:
            dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
            dist = znormed_euclidean_distance(
                dot_rolled, n, m, preprocessing, start, halve_m)
            knn = _argknn(dist, k, m, slack=slack)
            knns[i, :len(knn)] = knn
            extents[i] = get_pairwise_extent_raw_1d(
                ts,
                knns[i],
                m,
                znormed_euclidean_distance_single,
                preprocessing
            )

    min_pos = np.argmin(extents)
    return knns[min_pos], extents[min_pos]


def get_pairwise_extent_raw_1d(
        series, motifset_pos, motif_length,
        distance_single, preprocessing):
    """Compute the exact pairwise extent for one univariate motif set.

    Returns ``np.inf`` when the motif set contains invalid ``-1`` positions.
    """
    if -1 in motifset_pos:
        return np.inf

    motifset_extent = np.float64(0.0)
    for ii in np.arange(len(motifset_pos) - 1):
        i = motifset_pos[ii]
        a = series[i:i + motif_length]

        for jj in np.arange(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]
            b = series[j:j + motif_length]
            dist = distance_single(a, b, i, j, preprocessing)
            motifset_extent = max(motifset_extent, dist)

    return motifset_extent
