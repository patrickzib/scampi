# -*- coding: utf-8 -*-
"""
LatentMotif motif discovery algorithm wrapper.
Origin: https://github.com/grrvlr/TSMD/
"""

import numpy as np

import warnings

warnings.filterwarnings('ignore')

from numba import njit, prange, set_num_threads


class LatentMotif(object):
    """LatentMotif algorithm for motif discovery.

    Parameters
    ----------
    n_patterns : int 
        Number of patterns to detect.
    radius : float
        Threshold factor for pattern inclusion.
    wlen : int
        Window length.
    alpha : float, optional (default=1.0)
        Regularization parameter.
    learning_rate : float, optional (default=0.1)
        Learning rate. 
    n_iterations :int, optional (default=100): 
        Number of gradient iteration.
    n_starts : int, optional (default=10): 
        Number of trials. 
    verbose : bool, optional (default=False) : 
        Verbose. 
    Attributes
    ----------
    prediction_mask_ : np.ndarray of shape (n_patterns, n_samples)
        Binary mask indicating the presence of motifs across the signal.  
        Each row corresponds to one discovered motif, and each column to a time step.  
        A value of 1 means the motif is present at that time step, and 0 means it is not.
    """

    def __init__(self, n_patterns: int, wlen: int, radius: float, alpha=1.0,
                 learning_rate=0.1, n_iterations=100, n_starts=1,
                 verbose=False, chunk_size=1024, use_numba=True,
                 n_jobs=-1) -> None:

        self.n_patterns = n_patterns
        self.wlen = wlen
        self.radius = radius
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_starts = n_starts
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.use_numba = use_numba
        self.n_jobs = n_jobs

    def _configure_numba_threads(self):
        if not self.use_numba or self.n_jobs is None or self.n_jobs == -1:
            return
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be -1, None, or a positive integer.")
        set_num_threads(self.n_jobs)

    @staticmethod
    def _sliding_mean_std(signal: np.ndarray, wlen: int):
        cumsum = np.concatenate(([0.0], np.cumsum(signal, dtype=np.float64)))
        cumsum_sq = np.concatenate(([0.0], np.cumsum(signal ** 2, dtype=np.float64)))

        window_sum = cumsum[wlen:] - cumsum[:-wlen]
        window_sum_sq = cumsum_sq[wlen:] - cumsum_sq[:-wlen]
        mean = window_sum / wlen
        var = window_sum_sq / wlen - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))
        return mean, std

    def _iter_normalized_windows(self):
        for start in range(0, self.set_size_, self.chunk_size):
            end = min(start + self.chunk_size, self.set_size_)
            segment = self.signal_[start:end + self.wlen - 1]
            windows = np.lib.stride_tricks.sliding_window_view(segment, self.wlen)
            windows = (
                (windows - self.window_mean_[start:end, np.newaxis])
                / self.window_std_[start:end, np.newaxis]
            )
            yield start, end, windows

    def _freq(self, patterns: np.ndarray) -> float:  # verified
        """Compute the frequency score of the given patterns.

        Parameters
        ----------
        patterns : np.ndarray 
            Array of shape (n_patterns, wlen) representing the patterns.

        Returns
        -------
        freq : float
            Frequency score. Measures the similarity of the given patterns to the internal set.
        """
        if self.use_numba:
            return _freq_numba(
                self.signal_,
                self.window_mean_,
                self.window_std_,
                patterns,
                self.alpha,
                self.radius,
                self.n_patterns,
                self.set_size_,
                self.wlen,
            )

        exp_dist_sum = 0.0
        for _, _, windows in self._iter_normalized_windows():
            dist = np.sum(
                (windows[:, np.newaxis, :] - patterns[np.newaxis, ...]) ** 2,
                axis=2)
            exp_dist_sum += np.sum(np.exp(-self.alpha / self.radius * dist))

        freq = 1 / (self.n_patterns * self.set_size_) * exp_dist_sum
        return freq

    def _pen(self, patterns):  # verified
        """Compute a penalty score between patterns.

        Parameters
        ----------
        patterns : np.ndarray 
            Array of shape (n_patterns, wlen) representing the patterns.

        Returns
        -------
        pen : float
            Penalty score. 
        """
        if self.n_patterns > 1:
            dist = np.sum((patterns[:, np.newaxis, :] - patterns[np.newaxis, ...]) ** 2, axis=2)
            pen_m = np.where(dist < 2 * self.radius, (1 - dist / (2 * self.radius)) ** 2, 0)
            pen = 2 / (self.n_patterns * (self.n_patterns - 1)) * np.sum(np.triu(pen_m, k=1))
        else:
            pen = 0
        return pen

    def _score(self, patterns):  # verified
        """Compute the overall score for the given patterns.
        The score is defined as the frequency minus the penalty.

        Parameters
        ----------
        patterns : np.ndarray 
            Array of shape (n_patterns, wlen) representing the patterns.

        Returns
        -------
        score: float 
            Overall score
        """
        return self._freq(patterns) - self._pen(patterns)

    def _freq_derivative(self, patterns):
        """Compute the derivative of the frequency score with respect to the patterns.

        Parameters
        ----------
        patterns : np.ndarray 
            Array of shape (n_patterns, wlen) representing the patterns.

        Returns
        -------
        div_freq : float
            Frequency score derivative
        """
        if self.use_numba:
            return _freq_derivative_numba(
                self.signal_,
                self.window_mean_,
                self.window_std_,
                patterns,
                self.alpha,
                self.radius,
                self.n_patterns,
                self.set_size_,
                self.wlen,
            )

        div_sum = np.zeros_like(patterns)
        for _, _, windows in self._iter_normalized_windows():
            diff = windows[:, np.newaxis, :] - patterns[np.newaxis, ...]
            exp_dist = np.exp(-self.alpha / self.radius * np.sum(diff ** 2, axis=2))
            div_sum += np.sum(exp_dist[..., np.newaxis] * diff, axis=0)

        div_freq = (
            -2 * self.alpha
            / (self.n_patterns * self.set_size_ * self.radius)
            * div_sum
        )
        return div_freq

    def _pattern_distances(self, patterns: np.ndarray) -> np.ndarray:
        if self.use_numba:
            return _pattern_distances_numba(
                self.signal_,
                self.window_mean_,
                self.window_std_,
                patterns,
                patterns.shape[0],
                self.set_size_,
                self.wlen,
            )

        dist = np.empty((self.set_size_, patterns.shape[0]), dtype=np.float64)
        for start, end, windows in self._iter_normalized_windows():
            dist[start:end] = np.sum(
                (windows[:, np.newaxis, :] - patterns[np.newaxis, ...]) ** 2,
                axis=2)
        return dist

    def _pen_derivative(self, patterns):
        """Compute the derivative of the penalty score with respect to the patterns.

        Parameters
        ----------
        patterns : np.ndarray 
            Array of shape (n_patterns, wlen) representing the patterns.

        Returns
        -------
        div_pen : float
            Penalty score derivative
        """
        diff = patterns[:, np.newaxis, :] - patterns[np.newaxis, ...]
        dist = np.sum(diff ** 2, axis=2)
        pen_m = np.where(dist < 2 * self.radius, 2 * self.radius - dist, 0)
        div_pen = -2 / (self.radius ** 2 * self.n_patterns * (self.n_patterns - 1)) * np.sum(pen_m[..., np.newaxis] * diff, axis=0)
        return div_pen

    def fit(self, signal: np.ndarray) -> None:
        """Fit LatentMotif
        
        Parameters
        ----------
        signal : numpy array of shape (n_samples, )
            The input samples (time series length).
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # initialization
        self._configure_numba_threads()
        self.signal_ = signal.astype(np.float64, copy=False)
        self.window_mean_, self.window_std_ = self._sliding_mean_std(
            self.signal_, self.wlen)
        self.set_size_ = self.signal_.shape[0] - self.wlen + 1
        self.score_ = -np.inf
        self.patterns_ = np.zeros((self.n_patterns, self.wlen))

        if self.verbose:
            print("Start Trials")
        for i in range(self.n_starts):
            patterns, score = self.one_fit_()
            # if self.verbose:
            print(f"Trial: {i + 1}/{self.n_starts}, score : {score}")
            if np.isinf(score) or np.isnan(score):
                print(f"Adjusted radius to {self.radius} due to invalid score.")
                self.radius *= 2
            if score > self.score_:
                print(f"New best score found: {score}")
                self.score_ = score
                self.patterns_ = patterns

        if self.verbose:
            print(f"Successfully finished, best score: {self.score_}")

        return self

    def one_fit_(self):

        patterns = np.random.randn(self.n_patterns, self.wlen)
        rate_adapt = np.zeros((self.n_patterns, self.wlen))

        for i in range(self.n_iterations):
            if self.n_patterns > 1:
                div = self._freq_derivative(patterns) - self._pen_derivative(patterns)
            else:
                div = self._freq_derivative(patterns)
            rate_adapt += div ** 2
            patterns -= self.learning_rate / np.sqrt(rate_adapt) * div

            if self.verbose:
                print(
                    f"Iteration: {i + 1}/{self.n_iterations}, score: {self._score(patterns)} ")

        score = self._score(patterns)
        return patterns, score

    @property
    def prediction_indices_(self) -> list:
        dist = self._pattern_distances(self.patterns_)
        idx_lsts = []
        for line in dist.T:
            idxs = np.arange(line.shape[0])
            idx_lst = []
            t_distance = np.min(line)
            while t_distance < self.radius:
                # try:
                # local next neighbor
                t_idx = np.argmin(line)
                t_distance = line[t_idx]
                if line[t_idx] < self.radius:
                    idx_lst.append(idxs[t_idx])
                    # remove window
                    remove_idx = np.arange(max(0, t_idx - self.wlen + 1),
                                           min(len(line), t_idx + self.wlen))
                    line[remove_idx] = np.inf

                # except:
                # break
            idx_lsts.append(idx_lst)

        return idx_lsts

    @property
    def prediction_mask_(self) -> np.ndarray:
        idx_lsts = self.prediction_indices_
        mask = np.zeros((self.n_patterns, self.signal_.shape[0]))
        for i, p_idx in enumerate(idx_lsts):
            for idx in p_idx:
                mask[i, idx:idx + self.wlen] = 1

                # remove null lines
        mask = mask[~np.all(mask == 0, axis=1)]

        return mask, idx_lsts


@njit(cache=True, parallel=True)
def _freq_numba(signal, mean, std, patterns, alpha, radius,
                n_patterns, set_size, wlen):
    exp_dist_sum = 0.0
    for i in prange(set_size):
        for p in range(n_patterns):
            dist = 0.0
            for d in range(wlen):
                value = (signal[i + d] - mean[i]) / std[i]
                diff = value - patterns[p, d]
                dist += diff * diff
            exp_dist_sum += np.exp(-alpha / radius * dist)

    return exp_dist_sum / (n_patterns * set_size)


@njit(cache=True, parallel=True)
def _freq_derivative_numba(signal, mean, std, patterns, alpha, radius,
                           n_patterns, set_size, wlen):
    weights = np.empty((set_size, n_patterns), dtype=np.float64)

    for i in prange(set_size):
        for p in range(n_patterns):
            dist = 0.0
            for d in range(wlen):
                value = (signal[i + d] - mean[i]) / std[i]
                diff = value - patterns[p, d]
                dist += diff * diff
            weights[i, p] = np.exp(-alpha / radius * dist)

    div_freq = np.empty((n_patterns, wlen), dtype=np.float64)
    scale = -2 * alpha / (n_patterns * set_size * radius)

    for flat_idx in prange(n_patterns * wlen):
        p = flat_idx // wlen
        d = flat_idx - p * wlen
        total = 0.0
        for i in range(set_size):
            value = (signal[i + d] - mean[i]) / std[i]
            total += weights[i, p] * (value - patterns[p, d])
        div_freq[p, d] = scale * total

    return div_freq


@njit(cache=True, parallel=True)
def _pattern_distances_numba(signal, mean, std, patterns,
                             n_patterns, set_size, wlen):
    dist = np.empty((set_size, n_patterns), dtype=np.float64)

    for i in prange(set_size):
        for p in range(n_patterns):
            total = 0.0
            for d in range(wlen):
                value = (signal[i + d] - mean[i]) / std[i]
                diff = value - patterns[p, d]
                total += diff * diff
            dist[i, p] = total

    return dist
