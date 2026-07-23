import numpy as np
import pytest

from scampi.knn_scampi_backend import SCAMPINearestNeighbors


pytest.importorskip("pyattimo")


def make_refinable_series():
    rng = np.random.default_rng(13)
    series = rng.normal(size=1150).astype(np.float64)
    pattern = rng.normal(size=64)

    for pos, noise in [(20, 0.0), (400, 0.03), (800, 0.06), (1000, 0.09)]:
        series[pos:pos + 64] = pattern + noise * rng.normal(size=64)

    return series.reshape(1, -1)


def run_scampi_backend(data, exact_refine):
    backend = SCAMPINearestNeighbors(
        m=64,
        k_max=4,
        top_k=1,
        slack=0.5,
        verbose=False,
        scampi_delta=0.5,
        scampi_max_memory="2 GB",
        scampi_exact_refine=exact_refine,
    )
    return backend.compute_knns(data)


def test_scampi_exact_refine_updates_real_pyattimo_motiflet_and_extent():
    data = make_refinable_series()

    pyattimo_dists, pyattimo_candidates, _ = run_scampi_backend(
        data, exact_refine=False)
    refined_dists, refined_candidates, _ = run_scampi_backend(
        data, exact_refine=True)

    pyattimo_motiflet = pyattimo_candidates[3][0]
    refined_motiflet = refined_candidates[3][0]

    assert np.array_equal(
        pyattimo_motiflet,
        np.array([20, 400, 1000], dtype=np.int32),
    )
    assert np.array_equal(
        refined_motiflet,
        np.array([20, 400, 800], dtype=np.int32),
    )
    assert not np.array_equal(refined_motiflet, pyattimo_motiflet)
    assert refined_dists[3, 0] <= pyattimo_dists[3, 0]
