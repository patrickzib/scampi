# -*- coding: utf-8 -*-

import os
import time
from warnings import simplefilter

import numpy as np
import psutil
from numba import set_num_threads, njit, prange, get_num_threads

from scampi.distances import znormed_euclidean_distance_single, sliding_mean_std

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

STD_THRESHOLD = 1e-8

index_strategies = [
    "faiss",
    "annoy",
    "pynndescent"
]


class VectorSearchNearestNeighbors:
    """Approximate vector-search backend for motiflet k-NN candidates.

    The backend z-normalizes all subsequences of length ``m``, queries an
    approximate nearest-neighbor index, restores the original time-series
    offsets after an internal shuffle, and applies the motiflet exclusion zone
    before returning exact z-normalized distances for the selected neighbors.

    Parameters
    ----------
    m : int
        Motif length.
    k : int
        Number of non-overlapping neighbors to return per subsequence. The
        returned arrays have shape ``(n - m + 1, k)``.
    index_strategy : {"faiss", "annoy", "pynndescent"}, default="faiss"
        Approximate nearest-neighbor implementation to use.
    search_radius : int, default=5
        Multiplier for the raw candidate shortlist size. Approximate vector
        backends query at least ``search_radius * k`` neighbors before
        exclusion-zone post-processing.
    slack : float, default=0.5
        Exclusion-zone factor around accepted neighbors.
    n_jobs : int, default=4
        Worker/thread count. Values below 1 use ``os.cpu_count()``.
    verbose : bool, default=True
        Whether to print index parameters and post-processing diagnostics.
    **kwargs
        random_state : int or None, default=42
            Seed for the deterministic window shuffle and approximate backend
            randomness where supported.
        faiss_index : {"HNSW", "LSH", "IVF", "IVFPQ"}
            FAISS index type. Required when ``index_strategy="faiss"``.
        faiss_M : int, default=64
            HNSW graph degree.
        faiss_efConstruction : int, default=500
            HNSW construction parameter.
        faiss_efSearch : int, default=800
            HNSW search parameter. The effective value is at least
            ``search_radius * k``.
        faiss_nlist : int or None, default=None
            Number of IVF cells. ``None`` uses ``sqrt(n_windows)``.
        faiss_nprobe : int, default=32
            Number of IVF cells to probe.
        faiss_nbits : int, default=4
            LSH bits multiplier. The actual LSH bit count is
            ``faiss_nbits * m``.
        faiss_pq_m : int or None, default=None
            Number of product-quantizer subquantizers. If omitted, a divisor of
            the vector dimension up to 64 is chosen.
        faiss_pq_nbits : int, default=8
            Bits per product-quantizer code.
        annoy_n_trees : int, default=10
            Number of Annoy trees.
        annoy_search_k : int, default=-1
            Annoy search effort.
        pynndescent_n_neighbors : int, default=10
            Number of neighbors for NNDescent graph construction. The effective
            value is at least ``search_radius * k``.
        pynndescent_leaf_size : int, default=24
            NNDescent tree leaf size.
        pynndescent_pruning_degree_multiplier : float, default=1.0
            NNDescent graph pruning multiplier.
        pynndescent_diversify_prob : float, default=1.0
            NNDescent diversification probability.
        pynndescent_n_search_trees : int, default=1
            NNDescent search tree count.
    """

    def __init__(
            self,
            m,
            k,
            index_strategy="faiss",
            search_radius=5,
            slack=0.5,
            n_jobs=4,
            verbose=True,
            **kwargs):

        self.m = m
        self.k = k
        self.index_strategy = index_strategy
        self.search_radius = search_radius
        self.slack = slack
        self.n_jobs = n_jobs
        self.n_jobs = os.cpu_count() if self.n_jobs < 1 else self.n_jobs

        self.verbose = verbose
        self.random_state = kwargs.get("random_state", 42)

        #### faiss
        self.faiss_index = kwargs.get("faiss_index")
        self.M = kwargs.get("faiss_M", 64)
        self.efConstruction = kwargs.get("faiss_efConstruction", 500)
        self.efSearch = kwargs.get("faiss_efSearch", 800)
        self.efSearch = max(self.search_radius * self.k, self.efSearch)

        # number of clusters/cells
        self.nlist = kwargs.get("faiss_nlist")
        if self.nlist:
            self.nlist = int(self.nlist)

        # number of cells to search
        self.nprobe = kwargs.get("faiss_nprobe", 32)

        # number of bits used for hashing (resolution)
        self.nBits = kwargs.get("faiss_nbits", 4)
        self.pq_m = kwargs.get("faiss_pq_m")
        self.pq_nbits = kwargs.get("faiss_pq_nbits", 8)

        #### annoy

        self.annoy_n_trees = kwargs.get("annoy_n_trees", 10)
        self.annoy_search_k = kwargs.get("annoy_search_k", -1)

        #### pynndescent

        self.pynndescent_n_neighbors = kwargs.get("pynndescent_n_neighbors", 10)
        self.pynndescent_n_neighbors = max(
            self.pynndescent_n_neighbors,
            self.search_radius * self.k,
        )
        self.pynndescent_leaf_size = kwargs.get("pynndescent_leaf_size", 24)
        self.pynndescent_pruning_degree_multiplier = kwargs.get(
            "pynndescent_pruning_degree_multiplier", 1.0)
        self.pynndescent_diversify_prob = kwargs.get("pynndescent_diversify_prob", 1.0)
        self.pynndescent_n_search_trees = kwargs.get("pynndescent_n_search_trees", 1)

    def compute_knns(self, X):
        """Compute approximate k-nearest neighbors with exact post-processing.

        Parameters
        ----------
        X : np.ndarray
            Univariate time series with shape ``(1, n)`` or a flattenable
            one-dimensional array.

        Returns
        -------
        tuple
            ``(D_exact, knns_exact, index_create_time, index_search_time,
            post_process_time, memory_usage)`` where ``D_exact`` contains exact
            z-normalized distances for the post-processed neighbors and
            ``knns_exact`` contains original subsequence offsets. Missing
            neighbors are encoded as ``-1`` with distance ``np.inf``.

        Notes
        -----
        FAISS, Annoy, and PyNNDescent are retried with candidate shortlist sizes
        ``r``, ``2r``, and ``4r`` when exclusion-zone post-processing finds
        zero complete rows. The retry stops early as soon as at least one row
        has all ``k`` valid neighbors.
        """
        if X.shape[0] != 1:
            raise ValueError("Vector backends can handle univariate data, only.")

        # Set the number of threads for Numba
        self.previous_jobs = get_num_threads()
        set_num_threads(self.n_jobs)

        try:
            pid = os.getpid()
            self.process = psutil.Process(pid)

            if X.ndim > 1:
                X = X.flatten()

            X_windows = znorm_windows(X, self.m)

            # We must shuffle
            permutation = np.arange(len(X_windows), dtype=np.int32)
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(permutation)
            X_windows = X_windows[permutation]

            retry_multipliers = (1, 2, 4, 8)
            retry_count = len(retry_multipliers)
            index_create_time = 0.0
            index_search_time = 0.0
            memory_usage = 0.0
            post_process_time = time.time()

            if self.index_strategy == "faiss":
                original_search_radius = self.search_radius
                original_ef_search = self.efSearch

                try:
                    for attempt, multiplier in enumerate(retry_multipliers):
                        self.search_radius = original_search_radius * multiplier
                        self.efSearch = max(self.efSearch, self.search_radius * self.k)

                        if self.verbose:
                            print(
                                f"    Attempt {attempt + 1}/{retry_count} FAISS: "
                                f"search_radius={self.search_radius} "
                                f"query_k={self.search_radius * self.k}"
                            )

                        D, create_time, search_time, knns, attempt_memory \
                            = self.process_faiss(X_windows)
                        D, knns = restore_original_indices(D, knns, permutation)
                        D_exact, knns_exact = apply_exclusion_zone(
                            X,
                            self.m,
                            D,
                            knns,
                            self.k,
                            slack=self.slack
                        )
                        complete_rows = np.sum(np.all(knns_exact >= 0, axis=1))

                        index_create_time += create_time
                        index_search_time += search_time
                        memory_usage = max(memory_usage, attempt_memory)

                        if self.verbose:
                            print(
                                f"    Attempt {attempt + 1}/{retry_count} "
                                f"after exclusion zone: "
                                f"{complete_rows}/{len(knns_exact)}"
                            )

                        if (complete_rows > 0
                                or attempt == retry_count - 1):
                            break
                finally:
                    self.search_radius = original_search_radius
                    self.efSearch = original_ef_search

            elif self.index_strategy == "annoy":
                original_search_radius = self.search_radius

                try:
                    for attempt, multiplier in enumerate(retry_multipliers):
                        self.search_radius = original_search_radius * multiplier

                        if self.verbose:
                            print(
                                f"    Attempt {attempt + 1}/{retry_count} Annoy: "
                                f"search_radius={self.search_radius} "
                                f"query_k={self.search_radius * self.k}"
                            )

                        D, create_time, search_time, knns, attempt_memory \
                            = self.process_annoy(X_windows)
                        D, knns = restore_original_indices(D, knns, permutation)
                        D_exact, knns_exact = apply_exclusion_zone(
                            X,
                            self.m,
                            D,
                            knns,
                            self.k,
                            slack=self.slack
                        )
                        complete_rows = np.sum(np.all(knns_exact >= 0, axis=1))

                        index_create_time += create_time
                        index_search_time += search_time
                        memory_usage = max(memory_usage, attempt_memory)

                        if self.verbose:
                            print(
                                f"    Attempt {attempt + 1}/{retry_count} "
                                f"after exclusion zone: "
                                f"{complete_rows}/{len(knns_exact)}"
                            )

                        if (complete_rows > 0
                                or attempt == retry_count - 1):
                            break
                finally:
                    self.search_radius = original_search_radius

            elif self.index_strategy == "pynndescent":
                original_n_neighbors = self.pynndescent_n_neighbors

                try:
                    for attempt, multiplier in enumerate(retry_multipliers):
                        self.pynndescent_n_neighbors = (
                            original_n_neighbors * multiplier
                        )

                        if self.verbose:
                            print(
                                f"    Attempt {attempt + 1}/{retry_count} "
                                "PyNNDescent: "
                                f"n_neighbors={self.pynndescent_n_neighbors}"
                            )

                        D, create_time, search_time, knns, attempt_memory \
                            = self.process_pynndescent(X_windows)
                        D, knns = restore_original_indices(D, knns, permutation)
                        D_exact, knns_exact = apply_exclusion_zone(
                            X,
                            self.m,
                            D,
                            knns,
                            self.k,
                            slack=self.slack
                        )
                        complete_rows = np.sum(np.all(knns_exact >= 0, axis=1))

                        index_create_time += create_time
                        index_search_time += search_time
                        memory_usage = max(memory_usage, attempt_memory)

                        if self.verbose:
                            print(
                                f"    Attempt {attempt + 1}/{retry_count} "
                                f"after exclusion zone: "
                                f"{complete_rows}/{len(knns_exact)}"
                            )

                        if (complete_rows > 0
                                or attempt == retry_count - 1):
                            break
                finally:
                    self.pynndescent_n_neighbors = original_n_neighbors

            else:
                raise ValueError(
                    f"Unknown indexing strategy: {self.index_strategy}. "
                    f"Available strategies: {index_strategies}"
                )

            if self.index_strategy not in ["faiss", "annoy", "pynndescent"]:
                # Post-process the results to filter out distances and neighbors
                if self.verbose:
                    print("    Applying exclusion zone")

                D, knns = restore_original_indices(D, knns, permutation)
                D_exact, knns_exact = apply_exclusion_zone(
                    X,
                    self.m,  # :window_size
                    D,
                    knns,
                    self.k,
                    slack=self.slack
                )
                complete_rows = np.sum(np.all(knns_exact >= 0, axis=1))

            if self.verbose:
                #print(f"    First neighbors: {knns_exact[0]}")
                #print(f"    Last neighbors:  {knns_exact[-1]}")
                print(
                    f"    Complete neighbor rows: "
                    f"{complete_rows}/{len(knns_exact)}"
                )

            post_process_time = time.time() - post_process_time
            # print(f"\tPost-processing took {post_process_time:.3f} seconds.")

            print(
                f"    Vector search time: create={index_create_time:.3f}s "
                f"search={index_search_time:.3f}s "
                f"post_process={post_process_time:.3f}s"
            )

            return (D_exact, knns_exact, index_create_time,
                    index_search_time, post_process_time, memory_usage)

        finally:
            set_num_threads(self.previous_jobs)

    def process_annoy(self, X_windows):
        """Build and query an Annoy index on shuffled z-normalized windows.

        The query count is ``self.search_radius * self.k``. Returned neighbor
        indices refer to the shuffled window order and must be restored by
        ``restore_original_indices`` before motiflet post-processing.
        """
        import annoy

        d = X_windows.shape[-1]
        query_k = self.search_radius * self.k
        # https://github.com/spotify/annoy
        index_create_time = time.time()
        index = annoy.AnnoyIndex(d, metric="euclidean")

        if self.verbose:
            print(f"\tannoy")
            print(f"\tn_trees:  {self.annoy_n_trees}")
            print(f"\tsearch_k:  {self.annoy_search_k}")
            print(f"\tsearch_radius:  {self.search_radius}")
            print(f"\tquery_k:  {query_k}")

        for i, X in enumerate(X_windows):
            index.add_item(i, X)

        index.build(self.annoy_n_trees, n_jobs=self.n_jobs)
        index_create_time = time.time() - index_create_time

        index_search_time = time.time()
        # no method to query multiple samples at the same time
        knns = np.zeros((len(X_windows), query_k), dtype=np.int32)
        D = np.zeros((len(X_windows), query_k), dtype=np.float32)
        for i, X in enumerate(X_windows):
            knns[i], D[i] = index.get_nns_by_vector(
                X, query_k, self.annoy_search_k, include_distances=True)

        index_search_time = time.time() - index_search_time

        memory_usage = self.process.memory_info().rss / (1024 * 1024)  # MB

        del index
        return D, index_create_time, index_search_time, knns, memory_usage

    def process_pynndescent(self, X_windows):
        """Build NNDescent and return its neighbor graph for the windows."""
        import pynndescent

        # https://pynndescent.readthedocs.io/en/latest/api.html
        index_create_time = time.time()
        index = pynndescent.NNDescent(
            X_windows,
            metric="euclidean",
            low_memory=False,
            n_neighbors=self.pynndescent_n_neighbors,
            leaf_size=self.pynndescent_leaf_size,
            pruning_degree_multiplier=self.pynndescent_pruning_degree_multiplier,
            diversify_prob=self.pynndescent_diversify_prob,
            n_search_trees=self.pynndescent_n_search_trees,
            random_state=self.random_state,
            n_jobs=self.n_jobs
            # compressed=True,
            # verbose=True,
        )

        if self.verbose:
            print(f"\tpynndescent")
            print(f"\tn_neighbors:  {self.pynndescent_n_neighbors}")
            print(f"\tleaf_size: {self.pynndescent_leaf_size}")
            print(
                f"\tpruning_degree_multiplier: {self.pynndescent_pruning_degree_multiplier}")
            print(f"\tdiversify_prob: {self.pynndescent_diversify_prob}")
            print(f"\tn_search_trees: {self.pynndescent_n_search_trees}")

        index_create_time = time.time() - index_create_time

        # We can then extract the nearest neighbors of each training sample by
        # using the neighbor_graph attribute.
        index_search_time = time.time()
        knns, D = index.neighbor_graph
        index_search_time = time.time() - index_search_time

        memory_usage = self.process.memory_info().rss / (1024 * 1024)  # MB

        del index
        return D, index_create_time, index_search_time, knns, memory_usage

    def process_faiss(self, X_windows):
        """Build and query the configured FAISS index.

        The query count is ``self.search_radius * self.k``. Returned neighbor
        indices refer to the shuffled window order and must be restored by
        ``restore_original_indices`` before motiflet post-processing.
        """

        import faiss
        faiss.omp_set_num_threads(self.n_jobs)
        X_windows = np.ascontiguousarray(X_windows, dtype=np.float32)

        # Compute distances using the lower bounding representation
        index_create_time = time.time()
        d = X_windows.shape[-1]

        if self.faiss_index:

            if self.faiss_index == "LSH":
                # setup our HNSW parameters
                n_bits = self.nBits * d  # total number of bits

                # number of neighbours we add to each vertex
                if self.verbose:
                    print(
                        f"    FAISS LSH: nBits={n_bits} search_radius={self.search_radius}")

                index = faiss.IndexLSH(d, n_bits)

            elif self.faiss_index == "HNSW":
                # setup our HNSW parameters

                # number of neighbours we add to each vertex
                if self.verbose:
                    print(
                        f"    FAISS HNSW: M={self.M} "
                        f"efConstruction={self.efConstruction} "
                        f"efSearch={self.efSearch} "
                        f"search_radius={self.search_radius}"
                    )

                index = faiss.IndexHNSWFlat(d, self.M)
                index.hnsw.efConstruction = self.efConstruction
                index.hnsw.efSearch = self.efSearch  # TODO: reset every time needed?

            elif self.faiss_index == "IVF":
                # setup our IVF parameters

                if not self.nlist:
                    # number of clusters/cells set to sqrt(n)
                    self.nlist = int(np.sqrt(X_windows.shape[0]))

                if self.verbose:
                    print(
                        f"    FAISS IVF: nlist={self.nlist} "
                        f"nprobe={self.nprobe} "
                        f"search_radius={self.search_radius}"
                    )

                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_L2)
                index.train(X_windows)

                index.nprobe = self.nprobe  # TODO: reset every time needed?

            elif self.faiss_index == "IVFPQ":
                # setup our IVF-PQ parameters

                if not self.nlist:
                    # number of clusters/cells set to sqrt(n)
                    self.nlist = int(np.sqrt(X_windows.shape[0]))

                if self.verbose:
                    print(
                        f"    FAISS IVFPQ: nlist={self.nlist} "
                        f"nprobe={self.nprobe} "
                        f"search_radius={self.search_radius}"
                    )

                mm, nbits = self._faiss_pq_params(d)
                if self.verbose:
                    print(f"    PQ: m={mm} bits={nbits}")

                factory_string = f"IVF{int(self.nlist)},PQ{mm}x{nbits}"
                index = faiss.index_factory(d, factory_string, faiss.METRIC_L2)
                index.train(X_windows)

                index.nprobe = self.nprobe  # TODO: reset every time needed?

            else:
                raise ValueError(
                    'Unknown FAISS index' + self.faiss_index + '.' +
                    'Use "HNSW", "IVF", "IVFPQ", "LSH".')
        else:
            raise ValueError(
                'faiss_index not set. Use "HNSW", "IVF", "IVFPQ", "LSH".')

        index.add(X_windows)
        index_create_time = time.time() - index_create_time

        # Find k-nearest neighbors based on the lower bound distances
        index_search_time = time.time()
        D, knns = index.search(
            X_windows,
            self.search_radius * self.k
        )

        index_search_time = time.time() - index_search_time
        # print(f"\tIndexing search took {index_search_time:.3f} seconds.")

        faiss.omp_set_num_threads(self.previous_jobs)
        memory_usage = self.process.memory_info().rss / (1024 * 1024)  # MB

        # cleanup
        if 'quantizer' in locals():
            del quantizer
        del index

        return D, index_create_time, index_search_time, knns, memory_usage

    def _faiss_pq_params(self, d):
        """Return valid FAISS product-quantizer parameters for dimension ``d``."""
        nbits = int(self.pq_nbits)
        if self.pq_m is not None:
            mm = int(self.pq_m)
            if d % mm != 0:
                raise ValueError(
                    f"faiss_pq_m={mm} must divide vector dimension {d}."
                )
            return mm, nbits

        mm = min(64, d)
        while d % mm != 0:
            mm -= 1
        return mm, nbits


@njit(fastmath=True, cache=True)
def make_windows(X, window_size, n_chunks, chunk_size):
    """Create fixed-size windows by stepping through ``X`` in chunks."""
    X_windows = np.full((n_chunks, window_size), np.inf, dtype=X.dtype)
    for i in range(n_chunks):
        start = i * chunk_size
        X_windows[i] = X[start:start + window_size]
    return X_windows


def znorm_windows(X, window_size):
    """Return all sliding windows of ``X`` z-normalized by window statistics."""
    num_inst = X.shape[0] - window_size + 1
    X_windows = X[np.arange(window_size)[None, :] + np.arange(num_inst)[:, None]]

    mean, std = sliding_mean_std(X, window_size)
    X_lb = (X_windows - mean[:, np.newaxis]) / std[:, np.newaxis]

    return X_lb


@njit(cache=True)
def restore_original_indices(D_shuffled, knns_shuffled, permutation):
    """Map distances and neighbor indices from shuffled to original order."""
    D = np.empty_like(D_shuffled)
    knns = np.full(knns_shuffled.shape, -1, dtype=knns_shuffled.dtype)

    for shuffled_row in range(len(permutation)):
        original_row = permutation[shuffled_row]
        D[original_row] = D_shuffled[shuffled_row]

        for i in range(knns_shuffled.shape[1]):
            shuffled_neighbor = knns_shuffled[shuffled_row, i]
            if shuffled_neighbor >= 0:
                knns[original_row, i] = permutation[shuffled_neighbor]

    return D, knns


# FIXME: adding fastmath=True breaks the code???
@njit(cache=True, parallel=True)
def apply_exclusion_zone(X, m, D_lb, knns_lb, k, slack=0.5):
    """Apply motiflet exclusion-zone filtering to approximate neighbor lists.

    Parameters
    ----------
    X : np.ndarray
        Original one-dimensional time series.
    m : int
        Motif length.
    D_lb : np.ndarray
        Approximate/lower-bound distances for each query row.
    knns_lb : np.ndarray
        Candidate neighbor offsets for each query row.
    k : int
        Number of valid non-overlapping neighbors to keep.
    slack : float, default=0.5
        Exclusion-zone factor around accepted neighbors.

    Returns
    -------
    tuple
        ``(D, knns)`` with exact z-normalized distances and filtered neighbor
        offsets. Rows that cannot be filled keep ``-1`` positions and
        ``np.inf`` distances.
    """
    # compute size of the exclusion zone
    means, stds = sliding_mean_std(X, m)
    halve_m = np.int32(slack * m)

    n = D_lb.shape[0]

    D_knn = np.full((n, k), np.inf, dtype=np.float64)
    knns = np.full((n, k), -1, dtype=np.int32)

    # The knns_lb - list can be overlapping, Thus apply slack (exclusion zone)
    # to take top-k neighbors
    for order in prange(n):
        dist_pos = knns_lb[order]
        dist_sort = np.copy(D_lb[order])
        excluded = np.zeros(n, dtype=np.bool_)

        # top-k counter
        k_idx = 0
        # go through the list, applying exclusion zone
        for i in range(len(dist_sort)):
            arg_pos = np.argmin(dist_sort)
            pos = dist_pos[arg_pos]
            d = dist_sort[arg_pos]
            dist_sort[arg_pos] = np.inf

            # check if the position is not within some exclusion zone to a previously
            # chosen index
            if (0 <= pos < n and
                    (not excluded[pos]) and
                    (not np.isnan(d)) and
                    (not np.isinf(d))):
                D_knn[order, k_idx] = d
                knns[order, k_idx] = np.int32(pos)

                # exclude all trivial matches and itself
                excluded[max(0, pos - halve_m): min(pos + halve_m, n)] = True
                k_idx += 1

            # We found the top-k elements
            if k_idx >= k:
                break

    D = np.zeros((n, D_knn.shape[-1]), dtype=np.float64)
    for i in prange(len(knns)):
        query = X[i:i + m]
        knn = knns[i]
        for a in range(len(knn)):
            j = knn[a]
            if j > -1:
                # Re-rank based on z-normalized Euclidean distance
                D[i, a] = znormed_euclidean_distance_single(
                    query, X[j:j + m], i, j, (means, stds))
            else:
                D[i, a] = np.inf

    return D, knns
