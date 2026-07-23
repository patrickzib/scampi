"""Core motif-pair result helpers."""

from dataclasses import asdict, dataclass
import time

import numpy as np


@dataclass(frozen=True)
class MOMPResult:
    """Result of a motif-pair search."""

    distance: float
    locations: tuple[int, int]
    locations_matlab: tuple[int, int]
    motif_length: int
    series_length: int
    exclusion_zone: int
    runtime_seconds: float
    algorithm: str

    def to_dict(self):
        return asdict(self)


def mpx_v2_motif_pair(series, motif_length, exclusion_zone=None):
    """Find the best motif pair using the MATLAB-style MPX v2 profile."""
    from .momp_v9 import mpx_v2

    start_time = time.time()
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    m = int(motif_length)
    if exclusion_zone is None:
        exclusion_zone = m // 2
    exclusion_zone = int(exclusion_zone)

    profile, profile_idx, motifs = mpx_v2(x, exclusion_zone, m)
    if motifs.shape[1] == 0 or np.isnan(motifs[0, 0]) or np.isnan(motifs[1, 0]):
        raise ValueError("MPX v2 did not find a finite motif pair")

    i, j = sorted(motifs[:2, 0].astype(np.int64))
    distance = float(np.nanmin(profile))
    if not np.isfinite(distance):
        raise ValueError("MPX v2 did not find a finite motif pair")

    return MOMPResult(
        distance=distance,
        locations=(int(i), int(j)),
        locations_matlab=(int(i + 1), int(j + 1)),
        motif_length=m,
        series_length=len(x),
        exclusion_zone=exclusion_zone,
        runtime_seconds=time.time() - start_time,
        algorithm="mpx_v2",
    )
