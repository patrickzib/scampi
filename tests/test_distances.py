import numpy as np

from scampi.distances import (
    complexity_invariant_distance,
    complexity_invariant_distance_single,
    cosine_distance,
    cosine_distance_single,
    sliding_csum,
    sliding_csum_dcsum,
)


def _sliding_dot_products(ts, m, order):
    query = ts[order:order + m]
    return np.array([
        np.dot(query, ts[i:i + m])
        for i in range(len(ts) - m + 1)
    ], dtype=np.float64)


def _brute_force_cosine_distances(ts, m, order):
    query = ts[order:order + m]
    dist = np.empty(len(ts) - m + 1, dtype=np.float64)

    for i in range(len(dist)):
        candidate = ts[i:i + m]
        denom = np.linalg.norm(query) * np.linalg.norm(candidate)
        dist[i] = 1 - np.dot(query, candidate) / denom if denom > 0 else 1.0

    dist[order] = 0.0
    return dist


def _window_complexity(window):
    diff = window[:-1] - window[1:]
    return np.dot(diff, diff)


def _brute_force_cid_distances(ts, m, order):
    query = ts[order:order + m]
    query_complexity = _window_complexity(query)
    dist = np.empty(len(ts) - m + 1, dtype=np.float64)

    for i in range(len(dist)):
        candidate = ts[i:i + m]
        ed = np.dot(query - candidate, query - candidate)
        candidate_complexity = _window_complexity(candidate)
        min_complexity = min(query_complexity, candidate_complexity)
        if min_complexity > 0:
            cf = max(query_complexity, candidate_complexity) / min_complexity
        else:
            cf = 1.0
        dist[i] = ed * cf

    dist[order] = 0.0
    return dist


def test_cosine_distance_matches_brute_force_numpy():
    ts = np.array([1.0, 2.0, -1.0, 3.0, 0.5, -2.0, 4.0], dtype=np.float64)
    m = 3
    order = 2
    n = len(ts) - m + 1
    csumsq = sliding_csum(ts, m)
    dot_rolled = _sliding_dot_products(ts, m, order)

    actual = cosine_distance(dot_rolled, n, m, csumsq, order, 0)
    expected = _brute_force_cosine_distances(ts, m, order)

    np.testing.assert_allclose(actual, expected)


def test_cosine_distance_single_matches_brute_force_numpy():
    ts = np.array([1.0, 2.0, -1.0, 3.0, 0.5, -2.0, 4.0], dtype=np.float64)
    m = 3
    a_i = 1
    b_j = 4
    a = ts[a_i:a_i + m]
    b = ts[b_j:b_j + m]
    csumsq = sliding_csum(ts, m)

    actual = cosine_distance_single(a, b, a_i, b_j, csumsq)
    expected = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    np.testing.assert_allclose(actual, expected)


def test_cosine_distance_zero_norm_windows_are_finite():
    ts = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 2.0], dtype=np.float64)
    m = 3
    order = 0
    n = len(ts) - m + 1
    csumsq = sliding_csum(ts, m)
    dot_rolled = _sliding_dot_products(ts, m, order)

    actual = cosine_distance(dot_rolled, n, m, csumsq, order, 0)
    assert np.all(np.isfinite(actual))
    assert actual[order] == 0.0
    np.testing.assert_allclose(actual[1:], np.ones(n - 1))

    zero = ts[0:m]
    nonzero = ts[3:3 + m]
    single = cosine_distance_single(zero, nonzero, 0, 3, csumsq)
    assert np.isfinite(single)
    assert single == 1.0


def test_sliding_csum_dcsum_matches_window_complexity():
    ts = np.array([1.0, 3.0, 2.0, 7.0, 5.0, 4.0], dtype=np.float64)
    m = 3

    _, actual = sliding_csum_dcsum(ts, m)
    expected = np.array([
        _window_complexity(ts[i:i + m])
        for i in range(len(ts) - m + 1)
    ], dtype=np.float64)

    np.testing.assert_allclose(actual, expected)


def test_complexity_invariant_distance_matches_brute_force_numpy():
    ts = np.array([1.0, 3.0, 2.0, 7.0, 5.0, 4.0, 8.0], dtype=np.float64)
    m = 3
    order = 2
    n = len(ts) - m + 1
    preprocessing = sliding_csum_dcsum(ts, m)
    dot_rolled = _sliding_dot_products(ts, m, order)

    actual = complexity_invariant_distance(
        dot_rolled, n, m, preprocessing, order, 0)
    expected = _brute_force_cid_distances(ts, m, order)

    np.testing.assert_allclose(actual, expected)


def test_complexity_invariant_distance_single_matches_brute_force_numpy():
    ts = np.array([1.0, 3.0, 2.0, 7.0, 5.0, 4.0, 8.0], dtype=np.float64)
    m = 3
    a_i = 1
    b_j = 4
    a = ts[a_i:a_i + m]
    b = ts[b_j:b_j + m]
    preprocessing = sliding_csum_dcsum(ts, m)
    min_complexity = min(_window_complexity(a), _window_complexity(b))
    cf = max(_window_complexity(a), _window_complexity(b)) / min_complexity
    expected = np.dot(a - b, a - b) * cf

    actual = complexity_invariant_distance_single(a, b, a_i, b_j, preprocessing)

    np.testing.assert_allclose(actual, expected)


def test_complexity_invariant_distance_zero_complexity_windows_are_finite():
    ts = np.array([2.0, 2.0, 2.0, 1.0, 4.0, 1.0], dtype=np.float64)
    m = 3
    order = 0
    n = len(ts) - m + 1
    preprocessing = sliding_csum_dcsum(ts, m)
    dot_rolled = _sliding_dot_products(ts, m, order)

    actual = complexity_invariant_distance(
        dot_rolled, n, m, preprocessing, order, 0)
    expected = _brute_force_cid_distances(ts, m, order)

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected)

    constant = ts[0:m]
    nonconstant = ts[3:3 + m]
    single = complexity_invariant_distance_single(
        constant, nonconstant, 0, 3, preprocessing)
    assert np.isfinite(single)
    np.testing.assert_allclose(single, np.dot(constant - nonconstant,
                                             constant - nonconstant))
