import numpy as np

from momp_python.core import mpx_v2_motif_pair
from momp_python.momp_v9 import crosscorr_profile, momp_v9, mpx_v2, moving_mean_compensated, _window_invnorm


def test_mpx_v2_wrapper_matches_raw_profile_on_planted_signal():
    rng = np.random.default_rng(1)
    series = rng.normal(size=1000)
    motif_length = 80
    motif = np.sin(np.linspace(0, 4 * np.pi, motif_length))
    series[120:200] = motif
    series[650:730] = motif + 0.01 * rng.normal(size=motif_length)

    profile, _, motifs = mpx_v2(series, motif_length // 2, motif_length)
    result = mpx_v2_motif_pair(series, motif_length)

    np.testing.assert_allclose(
        np.nanmin(profile),
        result.distance,
        rtol=1e-10,
        atol=1e-10,
    )
    assert tuple(motifs[:2, 0].astype(int)) == result.locations
    assert result.locations == (120, 650)


def test_momp_v9_uses_mpx_v2_profile_path():
    rng = np.random.default_rng(2)
    series = rng.normal(size=5000)
    motif_length = 256
    motif = np.sin(np.linspace(0, 8 * np.pi, motif_length))
    series[600:856] = motif
    series[3000:3256] = motif + 0.02 * rng.normal(size=motif_length)

    expected = mpx_v2_motif_pair(series, motif_length)
    result = momp_v9(series, motif_length, verbose=False)

    np.testing.assert_allclose(result.distance, expected.distance, rtol=1e-10, atol=1e-10)
    assert result.locations == expected.locations


def test_crosscorr_profile_fft_matches_direct_path():
    rng = np.random.default_rng(3)
    motif_length = 512
    series = rng.normal(size=5000)
    mu = moving_mean_compensated(series, motif_length)
    finite_windows = np.ones(len(mu), dtype=np.bool_)
    invnorm = _window_invnorm(series, mu, finite_windows, motif_length)
    idx = 1234

    direct = crosscorr_profile(
        series,
        mu,
        invnorm,
        idx,
        motif_length,
        fft_threshold=motif_length + 1,
    )
    fft = crosscorr_profile(
        series,
        mu,
        invnorm,
        idx,
        motif_length,
        fft_threshold=1,
    )

    np.testing.assert_allclose(fft, direct, rtol=1e-10, atol=1e-10)


def test_mpx_v2_fft_neighbor_path_keeps_planted_pair():
    rng = np.random.default_rng(4)
    motif_length = 512
    series = rng.normal(size=6000)
    motif = np.sin(np.linspace(0, 12 * np.pi, motif_length))
    series[900:900 + motif_length] = motif
    series[4100:4100 + motif_length] = motif + 0.01 * rng.normal(size=motif_length)

    profile, _, motifs = mpx_v2(series, motif_length // 2, motif_length)

    assert tuple(motifs[:2, 0].astype(int)) == (900, 4100)
    assert np.nanmin(profile) < 0.5
