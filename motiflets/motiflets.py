# -*- coding: utf-8 -*-
"""Compute k-Motiflets.

"""

__author__ = ["patrickzib"]

import os
import logging
from ast import literal_eval
from os.path import exists

os.environ["RUST_LOG"] = "error"
logging.basicConfig(level=logging.CRITICAL)
pyattimo_logger = logging.getLogger('pyattimo')
pyattimo_logger.setLevel(logging.CRITICAL)

import math

import pandas as pd
from numba import objmode
from numba.typed import Dict, List
from scipy.signal import argrelextrema
from scipy.stats import zscore

import motiflets.plotting as pl
from motiflets.knn_vector_backend import *
from motiflets.knn_scampi_backend import *
from motiflets.distances import *
from motiflets.maxheap import MaxHeap


class Motiflets:

    def __init__(
            self,
            ds_name,
            series,
            ground_truth=None,
            elbow_deviation=1.00,
            distance="znormed_ed",
            slack=0.5,
            n_jobs=-1,
            backend="default",
            **kwargs
    ):
        """Computes the AU_EF plot to extract the best motif lengths

            This is the method to find and plot the characteristic motif-lengths, for k in
            [2...k_max], using the area AU-EF plot.

            Details are given within the paper 5.2 Learning Motif Length l.

            Parameters
            ----------
            ds_name: String
                Name of the time series for displaying
            series: array-like
                the TS
            ground_truth: pd.Series
                Ground-truth information as pd.Series.
            elbow_deviation : float, default=1.00
                The minimal absolute deviation needed to detect an elbow.
                It measures the absolute change in deviation from k to k+1.
                1.05 corresponds to 5% increase in deviation.
            distance: str (default="znormed_ed")
                The name of the distance function to be computed.
                Available options are:
                    - 'znormed_ed' or 'znormed_euclidean' for z-normalized ED
                    - 'ed' or 'euclidean' for the "normal" ED.
            slack: float
                Defines an exclusion zone around each subsequence to avoid trivial matches.
                Defined as percentage of m. E.g. 0.5 is equal to half the window length.
            n_jobs : int
                Number of jobs to be used.
            backend : String, default="default"
                The backend to use. As of now 'scalable', 'scampi' and 'default' are supported.
                Use 'default' for the original exact implementation with excessive memory,
                Use 'scalable' for a scalable, exact implementation with less memory,
                Use 'scampi' for a fast, scalable but approximate implementation.

            Returns
            -------
            best_motif_length: int
                The motif length that maximizes the AU-EF.
            """
        self.ds_name = ds_name
        self.series = series
        self.elbow_deviation = elbow_deviation
        self.slack = slack
        self.ground_truth = ground_truth

        n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
        self.n_jobs = n_jobs

        # distance function used
        self.distance_preprocessing, self.distance, self.distance_single \
            = map_distances(distance)
        self.backend = backend

        self.motif_length_range = None
        self.motif_length = 0
        self.all_extrema = []
        self.all_elbows = []
        self.all_top_motiflets = []
        self.all_dists = []
        self.kwargs = kwargs

        self.motif_length = 0
        self.memory_usage = 0
        self.k_max = 0
        self.dists = []
        self.motiflets = []
        self.elbow_points = []

    def fit_motif_length(
            self,
            k_max,
            motif_length_range,
            subsample=2,
            plot=True
    ):
        """Computes the AU_EF plot to extract the best motif lengths

            This is the method to find and plot the characteristic motif-lengths, for k in
            [2...k_max], using the area AU-EF plot.

            Details are given within the paper 5.2 Learning Motif Length l.

            Parameters
            ----------
            k_max: int
                use [2...k_max] to compute the elbow plot.
            motif_length_range: array-like
                the interval of lengths
            subsample: int (default=2)
                the subsample factor

            Returns
            -------
            best_motif_length: int
                The motif length that maximizes the AU-EF.

            """

        self.motif_length_range = motif_length_range
        self.k_max = k_max

        # turn into 2d array
        index, data_raw = pd_series_to_numpy(self.series)

        header = " in " + self.series.index.name if isinstance(
            self.series, pd.Series) and self.series.index.name != None else ""

        # discretizes ranges
        motif_length_range = np.int32(motif_length_range)

        self.motif_length, _, au_ef, elbow, top_motiflets, _ = \
            find_au_ef_motif_length(
                data_raw, k_max,
                motif_length_range=motif_length_range,
                n_jobs=self.n_jobs,
                elbow_deviation=self.elbow_deviation,
                slack=self.slack,
                subsample=subsample,
                distance=self.distance,
                distance_single=self.distance_single,
                distance_preprocessing=self.distance_preprocessing,
                backend=self.backend,
                **self.kwargs)

        if plot:
            pl.plot_window_lengths(self.ds_name, au_ef, header, motif_length_range)

        return self.motif_length

    def fit_k_elbow(
            self,
            k_max,
            motif_length=None,  # if None, use best_motif_length
            filter=True,
            top_N=1,
            plot_elbows=True,
            plot_motifs_as_grid=True,
            plot_method_name=None,
            plot_ground_truth=None,
    ):
        """Plots the elbow-plot for k-Motiflets.

            This is the method to find and plot the characteristic k-Motiflets within range
            [2...k_max] for given a `motif_length` using elbow-plots.

            Details are given within the paper Section 5.1 Learning meaningful k.

            Parameters
            ----------
            k_max: int
                use [2...k_max] to compute the elbow plot (user parameter).
            motif_length: int
                the length of the motif (user parameter)
            filter: bool, default=True
                filters overlapping motiflets from the result,
            top_N : int, default=1
                Number of best motiflets to return per k.
            plot_elbows: bool, default=False
                plots the elbow ploints into the plot
            plot_motifs_as_grid: bool, default=True
                plot_plots the motiflets as grid into the plot
            plot_ground_truth: pd.Series (default=None)
                Ground-truth information as pd.Series.
            plot_method_name: str, default=None
                The name of the method to be plotted in the title when plotting
                motiflets as grid.

            Returns
            -------
            Tuple
                dists:          distances for each k in [2...k_max]
                candidates:     motifset-candidates for each k
                elbow_points:   elbow-points

            """
        self.k_max = k_max
        self.top_N = top_N

        if motif_length is None:
            motif_length = self.motif_length
        else:
            self.motif_length = motif_length

        # turn into 2d array
        data = convert_to_2d(self.series)
        _, raw_data = pd_series_to_numpy(data)

        self.dists, self.motiflets, self.elbow_points, _, self.memory_usage \
            = search_k_motiflets_elbow(
            k_max,
            raw_data,
            motif_length,
            n_jobs=self.n_jobs,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
            filter=filter,
            distance=self.distance,
            distance_single=self.distance_single,
            distance_preprocessing=self.distance_preprocessing,
            backend=self.backend,
            top_N=self.top_N,
            **self.kwargs)

        if plot_elbows:
            pl._plot_elbow_points(
                self.ds_name, raw_data, motif_length,
                self.elbow_points, self.motiflets, self.dists)

        if plot_motifs_as_grid:
            if data.shape[0] == 1:
                pl.plot_grid_motiflets(
                    self.ds_name,
                    raw_data,
                    self.motiflets,
                    self.elbow_points,
                    self.dists,
                    motif_length,
                    method_name=plot_method_name,
                    show_elbows=False,
                    font_size=24,
                    ground_truth=plot_ground_truth)
            else:
                pl.plot_motifset(
                    self.ds_name,
                    data,
                    motifsets=candidates[elbows[0]][0],
                    motif_length=motif_length,
                    ground_truth=plot_ground_truth,
                    show=True)

        return self.dists, self.motiflets, self.elbow_points

    def plot_dataset(self, max_points=10_000, path=None):
        fig, ax = pl.plot_dataset(
            self.ds_name,
            self.series,
            max_points=max_points,
            show=path is None,
            ground_truth=self.ground_truth)

        if path is not None:
            plt.savefig(path)
            plt.show()

        return fig, ax

    def plot_motifset(self, max_points=10_000, path=None, elbow_point=None):
        """Plots the motif set for a given elbow point."""

        if self.dists is None or self.motiflets is None or self.elbow_points is None:
            raise Exception("Please call fit_k_elbow first.")

        if elbow_point is None:
            elbow_point = self.elbow_points[0][-1]

        fig, ax = pl.plot_motifset(
            self.ds_name,
            self.series,
            max_points=max_points,
            motifsets=self.motiflets[elbow_point].reshape((1, -1)),
            motif_length=self.motif_length,
            show=path is None)

        if path is not None:
            plt.savefig(path)
            plt.show()

        return fig, ax

    def flatten_data(self, max_items=None):
        return flatten_elbows(
            self.elbow_points,
            self.motiflets,
            self.dists,
            max_items=max_items)


def as_series(data, index_range, index_name):
    """Converts a time series to a series with an index.

    Parameters
    ----------
    data : array-like
        The time series raw data as numpy array
    index_range :
        The index to use
    index_name :
        The name of the index to use (e.g. time)

    Returns
    -------
    series : PD.Series

    """
    series = pd.Series(data=data, index=index_range)
    series.index.name = index_name
    return series


def resample(data, sampling_factor=10000):
    """Resamples a time series to roughly `sampling_factor` points.

    The method searches a factor to skip every i-th point.

    Parameters
    ----------
    data : array-like
        The time series data
    sampling_factor :
        The rough size of the time series after sampling

    Returns
    -------
    Tuple
        data :
            The raw data after sampling
        factor : int
            The factor used to sample the time series

    """
    factor = 1
    if len(data) > sampling_factor:
        factor = np.int32(len(data) / sampling_factor)
        data = data[::factor]
    return data, factor


def convert_to_2d(
        series
):
    if series.ndim == 1:
        # print('Warning: The input dimension must be 2d.')
        if isinstance(series, pd.Series):
            series = series.to_frame().T
        elif isinstance(series, (np.ndarray, np.generic)):
            series = series.reshape(1, -1)
    if series.shape[0] > series.shape[1]:
        raise ValueError(f'Warning: The input shape is wrong: {series.shape}. '
                         'Dimensions should be on rows. '
                         'Try transposing the input.')

    return series


def flatten_elbows(elbow_points, candidates, dists, max_items=None):
    if not isinstance(elbow_points, list):
        return elbow_points, candidates, dists

    items = []
    for rank in range(len(elbow_points)):
        elbows = elbow_points[rank]
        for k in elbows:
            if candidates[k] is None:
                continue
            items.append((k, rank, dists[k, rank]))

    if max_items is not None:
        items.sort(key=lambda item: (-item[0], item[2]))
        items = items[:max_items]

    flat_candidates = []
    flat_dists = []
    for k, rank, dist in items:
        flat_candidates.append(candidates[k][rank])
        flat_dists.append(dist)

    flat_elbows = np.arange(len(flat_candidates), dtype=np.int32)
    return (
        flat_elbows,
        np.array(flat_candidates, dtype=object),
        np.array(flat_dists, dtype=np.float64),
    )


@njit(fastmath=True, cache=True)
def compute_paa(ts, segment_size):
    segments = np.int32(np.ceil(len(ts) / segment_size))

    paa = np.zeros(segments, dtype=np.float64)
    paa_segment_sizes = np.zeros(segments, dtype=np.int32)  # needed for lower bounding

    for i in np.arange(0, segments, dtype=np.int32):
        start = i * segment_size
        end = min((i + 1) * segment_size, len(ts))
        paa[i] = ts[start:end].mean()
        paa_segment_sizes[i] = end - start

    return paa, paa_segment_sizes


def read_ground_truth(dataset):
    """Reads the ground-truth data for the time series.

    Parameters
    ----------
    dataset : String
        Name of the dataset

    Returns
    -------
    Series : pd.Series
        A series of ground-truth data

    """
    file = '../datasets/ground_truth/' + dataset.split(".")[0] + "_gt.csv"
    if exists(file):
        print(file)
        series = pd.read_csv(file, index_col=0)

        for i in range(0, series.shape[0]):
            series.iloc[i] = series.iloc[i].apply(literal_eval)

        return series
    return None


def read_dataset_with_index(dataset, sampling_factor=10000):
    """Reads a time series with an index (e.g. time) and resamples.

    Parameters
    ----------
    dataset : String
        File location.
    sampling_factor :
        The time series is sampled down to roughly this number of points by skipping
        every other point.

    Returns
    -------
    Tuple
        data : pd.Series
            The time series (z-score applied) with the index.
        gt : pd:series
            Ground-truth, if available as `dataset`_gt file

    """
    full_path = '../datasets/ground_truth/' + dataset
    data = pd.read_csv(full_path, index_col=0).squeeze('columns')
    print("Dataset Original Length n: ", len(data))

    data, factor = resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))

    data[:] = zscore(data)

    gt = read_ground_truth(dataset)
    if gt is not None:
        if factor > 1:
            for column in gt:
                gt[column] = gt[column].transform(
                    lambda l: (np.array(l)) // factor)
        return data, gt
    else:
        return data


def pd_series_to_numpy(data):
    """Converts a PD.Series to two numpy arrays by extracting the raw data and index.

    Parameters
    ----------
    data : array or PD.Series
        the TS

    Returns
    -------
    Tuple
        data_index : array_like
            The index of the time series
        data_raw :
            The raw data of the time series

    """
    data = convert_to_2d(data)

    if isinstance(data, pd.Series):
        data_raw = data.values
        data_index = data.index
    elif isinstance(data, pd.DataFrame):
        data_raw = data.values
        data_index = data.columns
    else:
        data_raw = data
        data_index = np.arange(data.shape[-1])
    try:
        return (data_index.astype(np.float64), data_raw.astype(np.float64, copy=False))
    except TypeError:  # datetime index cannot be cast to float64
        return (data_index, data_raw.astype(np.float64, copy=False))


def read_dataset(dataset, sampling_factor=10000):
    """ Reads a dataset and resamples.

    Parameters
    ----------
    dataset : String
        File location.
    sampling_factor :
        The time series is sampled down to roughly this number of points by skipping
        every other point.

    Returns
    -------
    data : array-like
        The time series with z-score applied.

    """
    full_path = '../datasets/' + dataset
    data = pd.read_csv(full_path).T
    data = np.array(data)[0]
    print("Dataset Original Length n: ", len(data))

    data, factor = resample(data, sampling_factor)
    print("Dataset Sampled Length n: ", len(data))

    return zscore(data)


# fastmath=True,
@njit(cache=True)
def _sliding_dot_product(query, time_series):
    """Compute a sliding dot-product using the Fourier-Transform

    Parameters
    ----------
    query : array-like
        first time series, typically shorter than ts
    time_series : array-like
        second time series, typically longer than query.

    Returns
    -------
    dot_product : array-like
        The result of the sliding dot-product
    """
    m = len(query)
    n = len(time_series)
    if m > n:
        raise ValueError("query longer than time_series")

    # Reverse query for cross-correlation and cast to float64
    q_rev = query[::-1]
    t = time_series

    # Next power-of-two ≥ n + m  (good for FFT speed)
    total = n + m
    exponent = math.ceil(math.log2(total))

    L = 1 << exponent
    q_pad = np.concatenate((q_rev, np.zeros(L - m, dtype=q_rev.dtype)))
    t_pad = np.concatenate((t, np.zeros(L - n, dtype=t.dtype)))

    with objmode(conv='float64[:]'):
        conv = np.fft.irfft(np.fft.rfft(q_pad) * np.fft.rfft(t_pad))

    # Trim to the valid sliding-dot range
    return conv[m - 1: n]


@njit(nogil=True, cache=True, parallel=True)
def compute_distances_with_knns_full(
        time_series,
        m,
        k,
        exclude_trivial_match=True,
        n_jobs=4,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std
):
    """Compute the full Distance Matrix between all pairs of subsequences.

        Computes pairwise distances between n-m+1 subsequences, of length, extracted from
        the time series, of length n.

        Z-normed ED is used for distances.

        This implementation is in O(n^2) by using the sliding dot-product.

        Parameters
        ----------
        time_series : array-like
            The time series
        m : int
            The window length
        k : int
            Number of nearest neighbors
        exclude_trivial_match : bool (default: True)
            Trivial matches will be excluded if this parameter is set
        n_jobs : int (default: 4)
            Number of jobs to be used.
        slack: float (default: 0.5)
            Defines an exclusion zone around each subsequence to avoid trivial matches.
            Defined as percentage of m. E.g. 0.5 is equal to half the window length.
        distance: callable (default: znormed_euclidean_distance)
                The distance function to be computed.
        distance_preprocessing: callable (default: sliding_mean_std)
                The distance preprocessing function to be computed.

        Returns
        -------
        D : 2d array-like
            The O(n^2) z-normed ED distances between all pairs of subsequences
        knns : 2d array-like
            The k-nns for each subsequence

    """
    # Input dim must be 2d
    assert time_series.ndim == 2, "Dimensionality is not correct"

    dims = time_series.shape[0]
    n = np.int32(time_series.shape[-1] - m + 1)
    n_jobs = max(1, min(n // 8, n_jobs))  # Cannot use more jobs than length of the ts

    halve_m = 0
    if exclude_trivial_match:
        halve_m = np.int32(m * slack)

    D = np.zeros((n, n), dtype=np.float64)
    knns = np.full((n, k), -1, dtype=np.int32)

    bin_size = np.int32(np.ceil(time_series.shape[-1] / n_jobs))

    for idx in prange(n_jobs):
        start = idx * bin_size
        end = min(start + bin_size, n)

        for d in np.arange(dims):
            ts = time_series[d]
            preprocessing = distance_preprocessing(ts, m)
            dot_first = _sliding_dot_product(ts[:m], ts)

            dot_prev = None
            for order in np.arange(start, end):
                if order == start:
                    # O(n log n) operation
                    dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
                else:
                    # constant time O(1) operations
                    dot_rolled = np.roll(dot_prev, 1) \
                                 + ts[order + m - 1] * ts[m - 1:n + m] \
                                 - ts[order - 1] * np.roll(ts[:n], 1)
                    dot_rolled[0] = dot_first[order]

                dist = distance(dot_rolled, n, m, preprocessing, order, halve_m)
                D[order] += dist
                dot_prev = dot_rolled

        for order in np.arange(start, end):
            knn = _argknn(D[order], k, m, slack=slack)
            knns[order, :len(knn)] = knn

    return D, knns


@njit(fastmath=True, cache=True)
def compute_upper_bound(
        ts, D_knn, knns, k, m,
        distance_single, preprocessing,
):
    kth_extent = np.zeros(k, dtype=np.float64)
    kth_extent[0] = np.inf

    for kk in range(1, len(kth_extent)):
        # kk is the kk-th NN
        # The motiflet candidate has thus kk+1 elements (including the query itself)
        best_knn_pos = np.argmin(D_knn[:, kk])
        candidate = knns[best_knn_pos, :kk + 1]
        kth_extent[kk] = get_pairwise_extent_raw(
            ts, candidate, m,
            distance_single,
            preprocessing)

        # extent must be within the diameter of the sphere
        kth_nn_min = np.min(D_knn[:, kk])
        if kth_extent[kk] > 4 * kth_nn_min or kth_extent[kk] < kth_nn_min:
            kth_extent[kk] = kth_nn_min

        # assert kth_extent[kk] <= 4 * kth_nn_min
        # assert kth_extent[kk] >= kth_nn_min

    return kth_extent


@njit(nogil=True, cache=True, parallel=True)
def compute_distances_with_knns(
        time_series,
        m,
        k,
        exclude_trivial_match=True,
        n_jobs=4,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std
):
    """ Compute the full Distance Matrix between all pairs of subsequences of a
        multivariate time series.

        Computes pairwise distances between n-m+1 subsequences, of length, extracted
        from the time series, of length n.

        Z-normed ED is used for distances.

        This implementation is in O(n^2) by using the sliding dot-product.

        Parameters
        ----------
        time_series : array-like
            The time series
        m : int
            The window length
        k : int
            Number of nearest neighbors
        exclude_trivial_match : bool (default: True)
            Trivial matches will be excluded if this parameter is set
        n_jobs : int (default: 4)
            Number of jobs to be used.
        slack: float (default: 0.5)
            Defines an exclusion zone around each subsequence to avoid trivial matches.
            Defined as percentage of m. E.g. 0.5 is equal to half the window length.
        distance: callable (default: znormed_euclidean_distance)
                The distance function to be computed.
        distance_preprocessing: callable (default: sliding_mean_std)
                The distance preprocessing function to be computed.

        Returns
        -------
        D : 2d array-like
            The O(n^2) z-normed ED distances between all pairs of subsequences
        knns : 2d array-like
            The k-nns for each subsequence

    """
    # Input dim must be 2d
    assert time_series.ndim == 2, "Dimensionality is not correct"

    dims = time_series.shape[0]
    n = np.int32(time_series.shape[-1] - m + 1)
    n_jobs = max(1, min(n // 8, n_jobs))  # Cannot use more jobs than length of the ts

    halve_m = 0
    if exclude_trivial_match:
        halve_m = np.int32(m * slack)

    D_knn = np.zeros((n, k), dtype=np.float64)
    knns = np.full((n, k), -1, dtype=np.int32)

    bin_size = np.int32(np.ceil(time_series.shape[-1] / n_jobs))

    preprocessing = []
    dot_first = []

    for d in np.arange(dims):
        ts = time_series[d]
        preprocessing.append(distance_preprocessing(ts, m))
        dot_first.append(_sliding_dot_product(ts[:m], ts))

    # first pass, computing the k-nns
    for idx in prange(n_jobs):
        dot_rolled = np.zeros((dims, n), dtype=np.float64)
        dot_prev = np.zeros((dims, n), dtype=np.float64)

        start = np.int32(idx * bin_size)
        end = np.int32(min(start + bin_size, n))

        for order in np.arange(start, end, ):
            dist = np.zeros(n, dtype=np.float64)
            for d in np.arange(dims):
                ts = time_series[d, :]
                if order == start:
                    # O(n log n) operation
                    dot_rolled[d] = _sliding_dot_product(ts[start:start + m], ts)
                else:
                    # constant time O(1) operations
                    dot_rolled[d] = np.roll(dot_prev[d], 1) \
                                    + ts[order + m - 1] * ts[m - 1:n + m] \
                                    - ts[order - 1] * np.roll(ts[:n], 1)
                    dot_rolled[d][0] = dot_first[d][order]

                dists = distance(dot_rolled[d], n, m, preprocessing[d], order, halve_m)
                for i in range(len(dists)):
                    dist[i] += dists[i]
                dot_prev[d] = dot_rolled[d]

            knn = _argknn(dist, k, m, slack=slack)
            D_knn[order, :len(knn)] = dist[knn]
            knns[order, :len(knn)] = knn

    return D_knn, knns


@njit(fastmath=True, cache=True)
def get_radius(D_full, motifset_pos):
    """Computes the radius of the passed motif set (motiflet).

    Parameters
    ----------
    D_full : 2d array-like
        The distance matrix
    motifset_pos : array-like
        The motif set start-offsets

    Returns
    -------
    motiflet_radius : float
        The radius of the motif set
    """
    motiflet_radius = np.inf

    for ii in range(len(motifset_pos) - 1):
        i = motifset_pos[ii]
        current = np.float64(0.0)
        for jj in range(1, len(motifset_pos)):
            if i != jj:
                j = motifset_pos[jj]
                current = max(current, D_full[i, j])
        motiflet_radius = min(current, motiflet_radius)

    return motiflet_radius


@njit(fastmath=True, cache=True, nogil=True)
def get_pairwise_extent(D_full, motifset_pos, upperbound=np.inf):
    """Computes the extent of the motifset using pre-computed distances.

    Parameters
    ----------
    D_full : 2d array-like
        The distance matrix
    motifset_pos : array-like
        The motif set start-offsets
    upperbound : float, default: np.inf
        Upper bound on the distances. If passed, will apply admissible pruning
        on distance computations, and only return the actual extent, if it is lower
        than `upperbound`

    Returns
    -------
    motifset_extent : float
        The extent of the motif set, if smaller than `upperbound`, else np.inf
    """

    if -1 in motifset_pos:
        return np.inf

    motifset_extent = np.float64(0.0)

    for ii in range(len(motifset_pos) - 1):
        i = motifset_pos[ii]

        for jj in range(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]

            motifset_extent = max(motifset_extent, D_full[i][j])
            if motifset_extent > upperbound:
                return np.inf

    return motifset_extent


@njit(cache=True, nogil=True)
def get_pairwise_extent_raw(
        series, motifset_pos, motif_length,
        distance_single, preprocessing, upperbound=np.inf):
    """Computes the extent of the motifset via pairwise comparisons.

    Parameters
    ----------
    series : array-like
        The time series
    motifset_pos : array-like
        The motif set start-offsets
    motif_length : int
        The motif length
    upperbound : float, default: np.inf
        Upper bound on the distances. If passed, will apply admissible pruning
        on distance computations, and only return the actual extent, if it is lower
        than `upperbound`

    Returns
    -------
    motifset_extent : float
        The extent of the motif set, if smaller than `upperbound`, else np.inf
    """

    if -1 in motifset_pos:
        return np.inf

    motifset_extent = np.float64(0.0)

    for ii in np.arange(len(motifset_pos) - 1):
        i = motifset_pos[ii]
        a = series[:, i:i + motif_length]

        for jj in np.arange(ii + 1, len(motifset_pos)):
            j = motifset_pos[jj]
            b = series[:, j:j + motif_length]

            dist = np.float64(0.0)
            for dim in np.arange(series.shape[0]):
                dist += distance_single(a[dim], b[dim], i, j, preprocessing[dim])

            motifset_extent = max(motifset_extent, dist)
            if motifset_extent > upperbound:
                return np.inf

    return motifset_extent


# FIXME: adding fastmath=True breaks the code???
@njit(cache=True)  # fastmath breaks the CODE: np.isinf does not work???
def _argknn(
        dist, k, m, lowest_dist=np.inf, slack=0.5):
    """Finds the closest k-NN non-overlapping subsequences in candidates.

    Parameters
    ----------
    dist : array-like
        the distances
    k : int
        The k in k-NN
    m : int
        The window-length
    lowest_dist : float
        Used for admissible pruning
    slack: float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.

    Returns
    -------
    idx : the <= k subsequences within `lowest_dist`

    """
    halve_m = np.int32(m * slack)
    dists = np.copy(dist).astype(np.float32)

    new_k = np.int32(min(len(dist) - 1, 2 * k))
    dist_pos = np.argpartition(dist, new_k)[:new_k]
    dist_sort = dist[dist_pos]

    idx = []  # there may be less than k, thus use a list

    # go through the partitioned list
    for i in range(len(dist_sort)):
        p = np.argmin(dist_sort)
        pos = dist_pos[p]
        dist_sort[p] = np.inf

        if (not np.isnan(dists[pos])
                and (not np.isinf(dists[pos]))
                and (dists[pos] < lowest_dist)):
            idx.append(pos)

            # exclude all trivial matches and itself
            dists[max(0, pos - halve_m): min(pos + halve_m, len(dists))] = np.inf

        if len(idx) == k:
            break

    # if not enough elements found, go through the rest
    for i in range(len(idx), k):
        pos = np.argmin(dists)
        if (not np.isnan(dists[pos])
                and (not np.isinf(dists[pos]))
                and (dists[pos] < lowest_dist)):
            idx.append(pos)

            # exclude all trivial matches
            dists[max(0, pos - halve_m): min(pos + halve_m, len(dists))] = np.inf
        else:
            break

    return np.array(idx, dtype=np.int32)


@njit(cache=True, nogil=True)
def get_approximate_k_motiflet(
        ts, m, k, D, knns,
        distance_single=None,
        preprocessing=None,
        use_D_full=True,
        upper_bound=np.inf,
        top_N=1
):
    """Compute the approximate k-Motiflets.

    Details are given within the paper Section 4.2 Approximate k-Motiflet Algorithm.

    Parameters
    ----------
    ts : array-like
        The raw time seres
    m : int
        The motif length
    k : int
        The k in k-Motiflets
    D : 2d array-like
        The distance matrix
    knns: 2d array-like
        The k-NNs for each subsequence
    use_D_full : bool
        If True, uses the full distance matrix D for computing the extent of the motiflet.
        If False, uses pairwise distances computed from the time series.
    upper_bound : float
        Used for admissible pruning
    top_N : int
        Number of best motiflets to return

    Returns
    -------
    Tuple
        motiflet_candidates : np.array
            The (approximate) best motiflets found
        motiflet_dists:
            The extents of the motiflets found
        motiflet_all_candidates : np.array
            All candidates found during the search, with k-NNs for each subsequence
            in the time series. The first k elements are the k-NNs, the rest is -1.
    """
    n = ts.shape[-1] - m + 1
    motiflet_all_candidates = np.full((n, k), -1, dtype=np.int32)
    heap = MaxHeap(top_N, k)

    # allow subsequence itself
    # Fill diagonal with 0
    knn_distances = np.zeros(n, dtype=np.float64)
    if use_D_full:
        for i in range(len(D)):
            D[i][i] = 0.0
        for i in np.arange(n):
            knn_distances[i] = D[i][knns[i, k - 1]]
    else:
        for i in np.arange(n):
            knn_distances[i] = D[i][k - 1]

    # order by increasing k-nn distance
    best_order = np.argsort(knn_distances)
    current_bound = upper_bound

    for i, order in enumerate(best_order):
        idx = knns[order, :k]
        motiflet_all_candidates[i, :min(k, len(idx))] = idx

        if len(idx) >= k and idx[-1] >= 0:
            bound_check = heap.heap_dist[0] if heap.size == top_N else np.inf
            if knn_distances[order] <= bound_check:
                if use_D_full:
                    # get_pairwise_extent requires the full distance matrix
                    motiflet_extent = get_pairwise_extent(D, idx, bound_check)
                else:
                    # get_pairwise_extent_raw does pairwise comparisons
                    motiflet_extent = get_pairwise_extent_raw(
                        ts, idx, m, distance_single, preprocessing, bound_check)

                if motiflet_extent <= bound_check:
                    # Search for overlap - if there is a just a single overlap, replace the
                    # motiflet with the largest distance in the heap
                    overlap_pos = []
                    for j in np.arange(heap.size):
                        if not _check_unique(idx, heap.heap_candidates[j], m):
                            # if heap.heap_dist[j] > heap.heap_dist[overlap_pos]:
                            overlap_pos.append(j)

                    if (len(overlap_pos) == 0) and (heap.size < top_N):
                        heap.push(motiflet_extent, idx)
                    elif len(overlap_pos) <= 1:
                        replace_pos = overlap_pos[0] if len(overlap_pos) == 1 else 0
                        if motiflet_extent < heap.heap_dist[replace_pos]:
                            # print(idx, "replacing", heap.heap_candidates[replace_pos], "overlap?", overlap_pos!=-1)
                            heap.replace_at(replace_pos, motiflet_extent, idx)
            else:
                # There is no point in continuing, as the distances are sorted
                # and the next k-NN will have a larger distance.
                break

    motiflet_candidates_sorted, motiflet_dists = heap.sorted_entries()
    return motiflet_candidates_sorted, motiflet_dists, motiflet_all_candidates


@njit(fastmath=True, cache=True)
def _check_unique(motifset_1, motifset_2, motif_length):
    """Check for overlaps between two motif sets.

    Two motif sets overlapp, if more than m/2 subsequences overlap from motifset 1.

    Parameters
    ----------
    motifset_1 : array-like
        Positions of the smaller motif set.
    motifset_2 : array-like
        Positions of the larger motif set.
    motif_length : int
        The length of the motif. Overlap exists, if 25% of two subsequences overlap.

    Returns
    -------
    True, if there are at least m/2 subsequences with an overlap of 25%, else False.
    """
    count = 0
    for a in motifset_1:  # smaller motiflet
        for b in motifset_2:  # larger motiflet
            if abs(a - b) < (motif_length / 4):
                count = count + 1
                break

        if count >= len(motifset_1) / 2:
            return False
    return True


def filter_unique(elbow_points, candidates, motif_length):
    """Filters the list of candidate elbows for only the non-overlapping motifsets.

    This method applied a duplicate detection by filtering overlapping motif sets.
    Two candidate motif sets overlap, if at least m/2 subsequences of the smaller
    motifset overlapp with the larger motifset. Only the largest non-overlapping
    motif sets are retained.

    Parameters
    ----------
    elbow_points : array-like
        List of possible k's for elbow-points.
    candidates : 2d array-like
        List of motif sets for each k
    motif_length : int
        Length of the motifs, needed for checking overlaps.

    Returns
    -------
    filtered_ebp : array-like
        The set of non-overlapping elbow points.

    """
    filtered_ebp = []
    for i in range(len(elbow_points)):
        unique = True
        for j in range(i + 1, len(elbow_points)):
            unique = _check_unique(
                candidates[elbow_points[i]], candidates[elbow_points[j]], motif_length)
            if not unique:
                break
        if unique:
            filtered_ebp.append(elbow_points[i])

    return np.array(filtered_ebp)


def filter_unique_across_ranks(elbow_points_per_rank, candidates, dists, motif_length):
    """Filters overlapping motifsets across multiple ranks.

    The candidates are ordered by motifset size (k) descending, then distance ascending.
    Overlapping motifsets are removed, keeping the first encountered candidate.

    Parameters
    ----------
    elbow_points_per_rank : list of array-like
        List of elbow points for each rank.
    candidates : array-like
        Motifset candidates for each k. Each entry is (top_N, k) if top_N > 1.
    dists : array-like
        Distances for each k and rank, shape (k_max, top_N).
    motif_length : int
        Length of the motifs, needed for checking overlaps.

    Returns
    -------
    filtered_elbows : list of np.array
        Filtered elbow points for each rank.
    """
    filtered = [[] for _ in range(len(elbow_points_per_rank))]
    items = []

    for rank, elbows in enumerate(elbow_points_per_rank):
        for k in elbows:
            if k < len(candidates):
                items.append((int(k), float(dists[k, rank]), rank))

    items.sort(key=lambda item: (-item[0], item[1]))

    accepted = []
    for k, _, rank in items:
        motifset = candidates[k][rank]
        unique = True
        for acc_rank, acc_k in accepted:
            if not _check_unique(motifset, candidates[acc_k][acc_rank], motif_length):
                unique = False
                break
        if unique:
            filtered[rank].append(k)
            accepted.append((rank, k))

    return [np.array(sorted(rank_elbows), dtype=np.int32) for rank_elbows in filtered]


@njit(fastmath=True, cache=True)
def find_elbow_points(dists, alpha=2, elbow_deviation=1.00):
    """Finds elbow-points in the elbow-plot (extent over each k).

    Parameters
    ----------
    dists : array-like
        The extends for each k.
    alpha : float
        A threshold used to detect an elbow-point in the distances.
        It measures the relative change in deviation from k-1 to k to k+1.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.

    Returns
    -------
    elbow_points : the elbow-points in the extent-function
    """
    elbow_points = set()
    elbow_points.add(2)  # required for numba to have a type
    elbow_points.clear()

    peaks = np.zeros(len(dists))
    for i in range(3, len(peaks) - 1):
        if (dists[i] != np.inf and
                dists[i + 1] != np.inf and
                dists[i - 1] != np.inf):

            m1 = (dists[i + 1] - dists[i]) + 0.00001
            m2 = (dists[i] - dists[i - 1]) + 0.00001

            # avoid detecting elbows in near constant data
            # TODO adding this removes reproducability
            # if dists[i - 1] == dists[i]:
            #    m2 = 1.0  # peaks[i] = 0

            if (dists[i] > 0) and (dists[i + 1] / dists[i] > elbow_deviation):
                peaks[i] = (m1 / m2)

    elbow_points = []

    while True:
        p = np.argmax(peaks)
        if peaks[p] > alpha:
            elbow_points.append(p)
            peaks[p - 1:p + 2] = 0
        else:
            break

    if len(elbow_points) == 0:
        elbow_points.append(2)

    return np.sort(np.array(list(set(elbow_points))))


def find_au_ef_motif_length(
        data,
        k_max,
        motif_length_range,
        n_jobs=4,
        elbow_deviation=1.00,
        slack=0.5,
        subsample=2,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
        backend="default",
        **kwargs):
    """Computes the Area under the Elbow-Function within an of motif lengths.

    Parameters
    ----------
    data : array-like
        The time series.
    k_max : int
        The interval of k's to compute the area of a single AU_EF.
    motif_length_range : array-like
        The range of lengths to compute the AU-EF.
    n_jobs : int
        Number of jobs to be used.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    slack: float
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.
    distance: callable
        The distance function to be computed.
    distance_preprocessing: callable
        The distance preprocessing function to be computed.
    backend : String, default="default"
        The backend to use. As of now 'scampi', 'scalable', and 'default'
        are supported.
        Use 'default' for the original exact implementation with excessive memory,
        Use 'scalable' for a scalable, exact implementation with less memory,

    Returns
    -------
    Tuple
        minimum : array-like
            The minumum found
        all_minima : array-like
            All local minima found
        au_efs : array-like
            For each length in the interval, the AU_EF.
        elbows :
            Largest k (largest elbow) found
        top_motiflets :
            The motiflet for the largest k for each length.

    """
    # apply sampling for speedup only
    if subsample > 1:
        if data.ndim >= 2:
            data = data[:, ::subsample]
        else:
            data = data[::subsample]

    # in reverse order
    au_efs = np.full(len(motif_length_range), np.inf, dtype=object)
    elbows = np.zeros(len(motif_length_range), dtype=object)
    top_motiflets = np.zeros(len(motif_length_range), dtype=object)
    dists = np.zeros(len(motif_length_range), dtype=object)

    # TODO parallelize?
    for i, m in enumerate(motif_length_range[::-1]):
        m_sub = m // subsample
        if m_sub < data.shape[-1] and m_sub >= 2:
            dist, candidates, elbow_points, _, memory_usage = search_k_motiflets_elbow(
                k_max,
                data,
                m_sub,
                n_jobs=n_jobs,
                elbow_deviation=elbow_deviation,
                slack=slack,
                distance=distance,
                distance_single=distance_single,
                distance_preprocessing=distance_preprocessing,
                backend=backend,
                top_N=1,
                kwargs=kwargs)

            # flatten the data types
            dist = dist.squeeze(1)
            elbow_points = np.array(elbow_points).flatten()
            candidates_rank = np.empty(len(candidates), dtype=object)
            for c in range(len(candidates)):
                if candidates[c] is not None:
                    candidates_rank[c] = candidates[c][0]

            dists_ = dist[(~np.isinf(dist)) & (~np.isnan(dist))]
            if dists_.max() - dists_.min() == 0:
                au_efs[i] = 1.0
            else:
                au_efs[i] = (((dists_ - dists_.min()) / (
                        dists_.max() - dists_.min())).sum()
                             / len(dists_))

            elbow_points = filter_unique(elbow_points, candidates_rank, m_sub)

            if len(elbow_points > 0):
                elbows[i] = elbow_points
                top_motiflets[i] = candidates_rank[elbow_points]
            else:
                # we found only the pair motif
                elbows[i] = [2]
                top_motiflets[i] = [candidates_rank[2]]

                # no elbow can be found, ignore this part
                au_efs[i] = 1.0

            dists[i] = dist

    # reverse order
    au_efs = np.array(au_efs, dtype=np.float64)[::-1]
    elbows = elbows[::-1]
    dists = dists[::-1]
    top_motiflets = top_motiflets[::-1] * subsample

    # Minima in AU_EF
    minimum = motif_length_range[np.nanargmin(au_efs)]
    au_ef_minima = argrelextrema(au_efs, np.less_equal, order=subsample)

    # Maxima in the EF
    return (minimum, au_ef_minima, au_efs,
            elbows, top_motiflets, dists)


def search_k_motiflets_elbow(
        k_max,
        data,
        motif_length='auto',
        motif_length_range=None,
        elbow_deviation=1.00,
        filter=True,
        slack=0.5,
        n_jobs=4,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
        backend="default",
        top_N=1,
        **kwargs
):
    """Computes the elbow-function.

    This is the method to find the characteristic k-Motiflets within range
    [2...k_max] for given a `motif_length` using elbow-plots.

    Details are given within the paper Section 5.1 Learning meaningful k.

    Parameters
    ----------
    k_max : int
        use [2...k_max] to compute the elbow plot (user parameter).
    data : array-like
        the TS
    motif_length : int (default='auto')
        the length of the motif (user parameter) or
        `motif_length == 'AU_EF'` or `motif_length == 'auto'`.
    motif_length_range : array-like (default=None)
        Can be used to determine to length of the motif set automatically.
        If a range is passed and `motif_length == 'auto'`, the best window length
        is first determined, prior to computing the elbow-plot.
    approximate_motiflet_pos : array-like (default=None)
        An initial estimate of the positions of the k-Motiflets for each k in the
        given range [2...k_max]. Will be used for bounding distance computations.
    elbow_deviation : float, default=1.00 (user parameter)
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    filter: bool, default=True (user parameter)
        filters overlapping motiflets from the result,
    slack: float (default=0.5)
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.
    n_jobs : int (default=4)
        Number of jobs to be used.
    distance: callable (default=znormed_euclidean_distance)
            The distance function to be computed.
    distance_preprocessing: callable (default=sliding_mean_std)
            The distance preprocessing function to be computed.
    backend : String, default="default"
        The backend to use. As of now 'scampi', 'scalable' and
        'default' are supported.
        Use 'default' for the original exact implementation with excessive memory,
        Use 'scalable' for a scalable, exact implementation with less memory,
    top_N : int
        Number of best motiflets to return per k.

    Returns
    -------
    Tuple
        dists :
            distances for each k in [2...k_max]
        candidates :
            motifset-candidates for each k
        elbow_points :
            elbow-points per rank when top_N > 1
        m : int
            best motif length
    """
    n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
    previous_jobs = get_num_threads()
    set_num_threads(n_jobs)

    # convert to 2d array
    _, data_raw = pd_series_to_numpy(data)

    # auto motif size selection
    if motif_length == 'AU_EF' or motif_length == 'auto':
        if motif_length_range is None:
            print("Warning: no valid motiflet range set")
            assert False
        m, _, _, _, _, _ = find_au_ef_motif_length(
            data, k_max, motif_length_range,
            n_jobs=n_jobs,
            elbow_deviation=elbow_deviation,
            slack=slack,
            backend=backend,
            **kwargs)
        m = np.int32(m)
    elif isinstance(motif_length, int) or \
            isinstance(motif_length, np.int32) or \
            isinstance(motif_length, np.int64):
        m = motif_length
    else:
        print("Warning: no valid motif_length set - use 'auto' for automatic selection")
        assert False

    pid = os.getpid()
    process = psutil.Process(pid)

    # non-overlapping motifs only
    n = data_raw.shape[-1] - m + 1

    if m <= 0:
        raise ValueError("motif_length must be > 0")
    if slack <= 0:
        raise ValueError("slack must be > 0")

    k_max_ = max(3, min(int(n // (m * slack)), k_max))

    # non-overlapping motifs only
    k_motiflet_distances = np.full((k_max_, top_N), np.inf, dtype=np.float64)
    k_motiflet_candidates = np.empty(k_max_, dtype=object)

    if backend in ["faiss", "annoy", "pynndescent",
                   "scampi", "default", "scalable"]:

        backend = check_valid_backend(backend, data_raw, n)

        # print(f"Using backend: {backend} for k-Motiflet search with motif length: {m}")
        # print(f"Jobs used: {n_jobs}")

        if backend == "scampi":
            backend_imlp = SCAMPINearestNeighbors(
                m, k_max_,
                top_k=top_N,
                slack=slack,
                **kwargs)

            k_motiflet_distances, k_motiflet_candidates, memory_usage \
                = backend_imlp.compute_knns(data_raw)

        else:
            if backend in ["faiss", "pynndescent", "annoy"]:
                backend_imlp = VectorSearchNearestNeighbors(
                    m, k_max_,
                    index_strategy=backend,
                    search_radius=5,
                    slack=slack,
                    n_jobs=n_jobs,
                    **kwargs
                )

                (D_full,
                 knns,
                 index_create_time,
                 index_search_time,
                 post_process_time,
                 memory_usage) = backend_imlp.compute_knns(data_raw)

            else:
                if backend == "scalable":
                    # uses pairwise comparisons to compute the distances
                    call_to_distances = compute_distances_with_knns
                else:
                    # computes the full matrix
                    call_to_distances = compute_distances_with_knns_full

                D_full, knns = call_to_distances(
                    data_raw, m, k_max_,
                    n_jobs=n_jobs,
                    slack=slack,
                    distance=distance,
                    distance_single=distance_single,
                    distance_preprocessing=distance_preprocessing
                )

                memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

            preprocessing = compute_preprocessing(data_raw, distance_preprocessing, m)

            upper_bound = np.inf
            for test_k in np.arange(k_max_ - 1, 1, -1):
                candidates, candidate_dists, _ = get_approximate_k_motiflet(
                    data_raw, m, test_k, D_full, knns,
                    distance_single=distance_single,
                    preprocessing=preprocessing,
                    use_D_full=(backend in ["default"]),
                    upper_bound=upper_bound,
                    top_N=top_N,
                )
                candidate_dist = candidate_dists[0]
                k_motiflet_distances[test_k, :len(candidate_dists)] = candidate_dists
                k_motiflet_candidates[test_k] = candidates
                upper_bound = min(candidate_dist, upper_bound)

            del D_full
            del knns
    else:
        raise ValueError(
            'Unknown backend: ' + backend + '. ' +
            'Use "scalable", "faiss", "pynndescent", "annoy", '
            '"scampi", or "default".')

    # smoothen the line to make it monotonically increasing
    k_motiflet_distances[0:2] = k_motiflet_distances[2]
    for i in range(len(k_motiflet_distances) - 1, 2, -1):
        k_motiflet_distances[i - 1] = (
            np.minimum(k_motiflet_distances[i], k_motiflet_distances[i - 1]))

    elbow_points = []
    for rank in range(top_N):
        eb = find_elbow_points(
            k_motiflet_distances[:, rank], elbow_deviation=elbow_deviation)

        if filter:
            candidates_rank = np.empty(len(k_motiflet_candidates), dtype=object)
            for e in eb:
                candidates_rank[e] = k_motiflet_candidates[e][rank]
            eb = filter_unique(eb, candidates_rank, m)

        elbow_points.append(eb)

    set_num_threads(previous_jobs)

    return k_motiflet_distances, k_motiflet_candidates, elbow_points, m, memory_usage


def compute_preprocessing(data_raw, distance_preprocessing, m):
    preprocessing = []
    for dim in np.arange(data_raw.shape[0]):
        preprocessing.append(distance_preprocessing(data_raw[dim], m))
    preprocessing = np.array(preprocessing, dtype=np.float64)
    return preprocessing


def check_valid_backend(backend, data_raw, n):
    """ Switch to LSH-backend, when length is >150_000 and univariate. """
    if ((n >= 125_000) and (data_raw.shape[0] == 1)
            and (backend in ["default", "scalable"])):
        print(f"Setting 'scampi' backend for distance computations. "
              f"Old Backend: '{backend}'")
        backend = "scampi"

    elif backend == "default":
        """ Switch to scalable matrix representation when length is >25_000 or 4 GB. """
        if data_raw.shape[0] == 1:
            recommend_scalable = n >= 25_000
        else:
            d = data_raw.shape[0]
            scalable_gb = ((n ** 2) * d) * 32 / (1024 ** 3) / 8
            recommend_scalable = scalable_gb > 4.0

        if recommend_scalable:
            print(f"Setting 'scalable' backend for distance computations. "
                  f"Old Backend: '{backend}'")
            backend = "scalable"

    return backend
