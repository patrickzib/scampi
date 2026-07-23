"""Python port of the pruning/downsampling MOMP v9 MATLAB implementation."""

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

try:
    from numba import get_num_threads, get_thread_id, njit, prange
except ImportError:  # pragma: no cover - exercised only without numba
    def get_num_threads():
        return 1

    def get_thread_id():
        return 0

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range


@dataclass(frozen=True)
class MOMPPortResult:
    """Result of the MOMP v9 pruning/downsampling port."""

    distance: float
    locations: tuple[int, int]
    locations_matlab: tuple[int, int]
    motif_length: int
    series_length: int
    initial_downsample_rate: int
    iterations: list[dict]
    runtime_seconds: float
    algorithm: str

    def to_dict(self):
        return asdict(self)


def momp_v9(series, motif_length, verbose=True, initial_downsample_rate=64):
    """Run a faithful Python port of ``momp_v9.m``.

    Indices are zero-based in ``locations`` and one-based in
    ``locations_matlab``. The MATLAB code hard-codes ``dd = 64`` after briefly
    deriving a value from the motif length; this port keeps that default.
    """
    start_time = time.time()
    t_orig = np.asarray(series, dtype=np.float64).reshape(-1)
    m = int(motif_length)
    if m < 2:
        raise ValueError("motif_length must be at least 2")
    if len(t_orig) < 2 * m:
        raise ValueError("series must contain at least two motif-length windows")

    dd = int(initial_downsample_rate)
    if dd < 1 or dd & (dd - 1):
        raise ValueError("initial_downsample_rate must be a power of two")

    t_current = t_orig.copy()
    current_indices = np.arange(len(t_orig), dtype=np.int64)
    bsf = math.inf
    bsf_loc = (-1, -1)
    iterations = []

    ktip_start = time.time()
    ktip = ktip_v1(t_orig, m, dd)
    ktip_time = time.time() - ktip_start
    last_ktip = np.zeros(ktip.shape[0], dtype=np.float64)

    if verbose:
        print(f"T is length {len(t_orig)}, and m is set to {m}")
        print(f"Starting downsampling rate: {dd}")
        print("====== MOMP ======")

    while True:
        iter_start = time.time()
        if dd > 1:
            curr_ktip = ktip[:, int(math.log2(dd)) - 1]
        else:
            curr_ktip = last_ktip

        (lbmp,
         local_bsf_loc,
         uamp_time,
         lbmp_time) = upsample_approximate_mp(
            t_current,
            m,
            dd,
            current_indices,
            curr_ktip,
        )

        refine_start = time.time()
        bsf, bsf_loc = refine(t_orig, m, bsf, bsf_loc, local_bsf_loc, dd)
        refine_time = time.time() - refine_start

        prune_start = time.time()
        if dd > 1:
            t_pruned, pruned_indices = prune(t_orig, m, current_indices, lbmp, bsf)
        else:
            t_pruned = t_current
            pruned_indices = current_indices
        prune_time = time.time() - prune_start
        pruning = 1.0 - (len(t_pruned) / len(t_orig))

        elapsed = time.time() - start_time
        iterations.append({
            "downsample_rate": dd,
            "current_length": int(len(t_current)),
            "best_distance": float(bsf),
            "best_locations": [int(bsf_loc[0]), int(bsf_loc[1])],
            "pruning": float(pruning),
            "iteration_seconds": time.time() - iter_start,
            "uamp_seconds": uamp_time,
            "lbmp_seconds": lbmp_time,
            "refine_seconds": refine_time,
            "prune_seconds": prune_time,
        })

        if verbose:
            print(
                f"MOMP : Tpaa1in{dd} | BSF: {bsf:0.2f} "
                f"{{{bsf_loc[0] + 1}, {bsf_loc[1] + 1}}}| "
                f"pruning: {pruning:0.4f} | Time: {elapsed:0.2f}"
            )

        dd = dd // 2
        t_current = t_pruned
        current_indices = pruned_indices

        if dd < 1:
            break

    if bsf_loc == (-1, -1):
        raise ValueError("MOMP did not find a finite motif pair")

    if verbose:
        total = time.time() - start_time
        print("====== Profiling Summary ======")
        print(f"Tot KTIP time : {ktip_time:0.2f}s  ({ktip_time / total:0.2f} of momp time)")

    return MOMPPortResult(
        distance=float(bsf),
        locations=(int(bsf_loc[0]), int(bsf_loc[1])),
        locations_matlab=(int(bsf_loc[0] + 1), int(bsf_loc[1] + 1)),
        motif_length=m,
        series_length=len(t_orig),
        initial_downsample_rate=int(initial_downsample_rate),
        iterations=iterations,
        runtime_seconds=time.time() - start_time,
        algorithm="momp_v9_port",
    )


def ktip_v1(time_series, subseq_len, dsrate, minlag=0):
    """Port of ``ktip_v1.m``."""
    x = np.asarray(time_series, dtype=np.float64).reshape(-1).copy()
    m = int(subseq_len)
    n = len(x)
    if m < 2 or m > n:
        raise ValueError("subseq_len must be between 2 and len(time_series)")

    profile_rows = n - m + 1
    profile_cols = int(math.floor(math.log2(dsrate)))
    ktip = np.full((profile_rows, profile_cols), np.nan, dtype=np.float64)
    current = np.full(profile_rows, np.nan, dtype=np.float64)

    finite_windows = finite_window_mask(x, m)
    x[~np.isfinite(x)] = 0.0
    mu = moving_mean(x, m)
    sigma = moving_std_population(x, m)
    invsig = np.full(profile_rows, np.nan, dtype=np.float64)
    valid = finite_windows & (sigma > 1e-12)
    invsig[valid] = 1.0 / sigma[valid]

    df = np.empty(profile_rows, dtype=np.float64)
    dg = np.empty(profile_rows, dtype=np.float64)
    df[0] = 0.0
    dg[0] = 0.0
    df[1:] = 0.5 * (x[m:n] - x[0:n - m])
    dg[1:] = (x[m:n] - mu[1:profile_rows]) + (x[0:n - m] - mu[0:profile_rows - 1])

    ktip = _ktip_v1_kernel(x, mu, invsig, df, dg, current, ktip, m, int(dsrate), int(minlag))
    return np.sqrt(np.maximum(0.0, 2.0 * (m - ktip)))


@njit(cache=True)
def _ktip_v1_kernel(x, mu, invsig, df, dg, current, ktip, m, dsrate, minlag):
    n = len(x)
    next_capture = 2
    capture_col = 0

    for diag in range(minlag + 1, dsrate + 1):
        if diag + m - 1 > n:
            break

        cov = 0.0
        diag_start = diag - 1
        for k in range(m):
            cov += (x[diag_start + k] - mu[diag_start]) * (x[k] - mu[0])

        row_stop = n - m - diag + 2
        for row in range(row_stop):
            col = row + diag - 1
            cov = cov + df[row] * dg[col] + df[col] * dg[row]
            corr = cov * invsig[row] * invsig[col]
            if not math.isfinite(corr):
                continue
            if math.isnan(current[row]) or corr < current[row]:
                current[row] = corr
            if math.isnan(current[col]) or corr < current[col]:
                current[col] = corr

        if diag == next_capture and capture_col < ktip.shape[1]:
            for row in range(ktip.shape[0]):
                ktip[row, capture_col] = current[row]
            next_capture *= 2
            capture_col += 1

    return ktip


def upsample_approximate_mp(time_series, motif_length, dd, indices, ip):
    """Port of ``upsample_approximate_mp`` without plotting arrays."""
    uamp_start = time.time()
    x = np.asarray(time_series, dtype=np.float64).reshape(-1)
    n = len(x)

    if len(ip) == 0:
        ktip = np.zeros(0, dtype=np.float64)
    else:
        mask = indices < len(ip)
        ktip = ip[indices[mask]]

    pad = int(math.ceil(n / dd) * dd - n)
    if pad > 0:
        # MATLAB pads with randn. Fixed zeros keep this port deterministic and
        # only affect padded samples beyond the original series.
        x_pad = np.pad(x, (0, pad), mode="constant")
    else:
        x_pad = x

    if dd > 1:
        x_ds = paa(x_pad, n // dd)
    else:
        x_ds = x_pad
    m_ds = int(math.floor(motif_length / dd))
    if m_ds < 2:
        m_ds = 2

    amp, _, motifs = mpx_v2(x_ds, int(math.floor(m_ds / 2)), m_ds)
    if motifs.shape[1] == 0 or np.isnan(motifs[0, 0]) or np.isnan(motifs[1, 0]):
        raise ValueError("downsampled MPX did not find a motif pair")

    absf_loc_ds = motifs[:2, 0].astype(np.int64) * dd
    absf_loc_ds = np.clip(absf_loc_ds, 0, len(indices) - 1)
    absf_loc = indices[absf_loc_ds]
    uamp = math.sqrt(dd) * np.repeat(amp, dd)[:max(0, n - motif_length + 1)]
    uamp_time = time.time() - uamp_start

    lbmp_start = time.time()
    lbmp = comp_lb(n, motif_length, amp, ktip, dd)
    lbmp_time = time.time() - lbmp_start

    return lbmp, (int(absf_loc[0]), int(absf_loc[1])), uamp_time, lbmp_time


def comp_lb(n, motif_length, amp, ktip, dd):
    subseq_count = n - motif_length + 1
    ip = ktip[:subseq_count:dd]
    n_ip = len(ip)
    amp = amp[:n_ip]
    camp_ds = (math.sqrt(dd) * amp) - ip
    lbmp_ds = _comp_lb_downsampled(camp_ds, ip)
    return np.repeat(lbmp_ds, dd)[:subseq_count]


def _comp_lb_downsampled(camp_ds, ip):
    finite_ip = ip[np.isfinite(ip)]
    if len(ip) == 0:
        return np.empty(0, dtype=np.float64)
    if len(finite_ip) == 0:
        return np.full(len(ip), -np.inf, dtype=np.float64)

    min_value = np.min(finite_ip)
    lbmp_ds = camp_ds - min_value
    lbmp_ds[~np.isfinite(camp_ds)] = -np.inf

    unique_min = np.count_nonzero(finite_ip == min_value) == 1
    if unique_min:
        min_pos = np.flatnonzero(ip == min_value)[0]
        second_candidates = finite_ip[finite_ip != min_value]
        if len(second_candidates) == 0 or not np.isfinite(camp_ds[min_pos]):
            lbmp_ds[min_pos] = -np.inf
        else:
            lbmp_ds[min_pos] = camp_ds[min_pos] - np.min(second_candidates)
    return lbmp_ds


def refine(t_orig, motif_length, bsf, bsf_loc, approximate_locations, dd):
    i, j = sorted((int(approximate_locations[0]), int(approximate_locations[1])))
    m = int(motif_length)
    st1 = max(0, i - dd)
    end1 = min(len(t_orig), i + m + dd - 1)
    st2 = max(j - dd, end1)
    end2 = min(len(t_orig), j + m + dd - 1)
    if st1 >= end1 or st2 >= end2:
        return bsf, bsf_loc

    idx = np.concatenate((np.arange(st1, end1), np.arange(st2, end2)))
    temp = t_orig[idx]
    try:
        mp, _, motifs = mpx_v2(temp, m // 2, m)
    except ValueError:
        return bsf, bsf_loc

    if motifs.shape[1] == 0 or np.isnan(motifs[0, 0]) or np.isnan(motifs[1, 0]):
        return bsf, bsf_loc
    current = float(np.nanmin(mp))
    if current <= bsf:
        loc = idx[motifs[:2, 0].astype(np.int64)]
        return current, (int(loc[0]), int(loc[1]))
    return bsf, bsf_loc


def prune(t_orig, motif_length, indices, uamp, bsf):
    targets = np.flatnonzero(uamp <= bsf)
    if len(targets) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64)
    return idx_filter(t_orig, targets, indices, motif_length)


def idx_filter(t_orig, targets, indices, motif_length):
    margin = int(math.floor(motif_length / 4))
    gaps = np.flatnonzero(np.concatenate(([2], np.diff(targets))) > 1)
    pruned_parts = []
    pruned_index_parts = []
    last_pruned = -1

    for g_pos, gap in enumerate(gaps):
        next_gap = gaps[g_pos + 1] if g_pos + 1 < len(gaps) else len(targets)
        segment = targets[gap:next_gap]
        if len(segment) == 0:
            continue
        start = max(0, int(segment[0]) - margin, last_pruned + 1)
        end = min(int(segment[-1]) + motif_length + margin, len(indices) - 1)
        if end < start:
            continue
        selected = np.arange(start, end + 1, dtype=np.int64)
        last_pruned = int(selected[-1])
        pruned_index_parts.append(indices[selected])
        pruned_parts.append(t_orig[indices[selected]])

    if not pruned_parts:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64)

    return cat_segments(pruned_parts), np.concatenate(pruned_index_parts)


def cat_segments(parts):
    aligned_parts = []
    for part in parts:
        part = np.asarray(part, dtype=np.float64).reshape(-1)
        if len(part) == 0:
            continue
        if not aligned_parts:
            aligned_parts.append(part.copy())
        else:
            aligned_parts.append(part + (aligned_parts[-1][-1] - part[0]))
    if not aligned_parts:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(aligned_parts)


def mpx_v2(time_series, minlag, subseq_len):
    """Port of MATLAB ``mpx_v2.m`` for self-joins.

    The matrix profile is computed as maximal Pearson correlation along
    diagonals using the MATLAB difference-equation update, then converted to
    z-normalized Euclidean distance before returning.
    """
    x = np.asarray(time_series, dtype=np.float64).reshape(-1)
    m = int(subseq_len)
    minlag = int(minlag)
    subcount = len(x) - m + 1
    if m < 2 or subcount < 2:
        raise ValueError("subsequence length must be between 2 and len(time_series)")

    finite_windows = finite_window_mask(x, m)
    x_work = x.copy()
    x_work[~np.isfinite(x_work)] = 0.0

    mu = moving_mean_compensated(x_work, m)
    mus = moving_mean_compensated(x_work, m - 1)
    invnorm = _window_invnorm(x_work, mu, finite_windows, m)

    corr_profile, profile_idx = _mpx_correlation_profile(
        x_work,
        mu,
        mus,
        invnorm,
        finite_windows,
        m,
        minlag,
    )
    motifs = find_motifs(x_work, mu, invnorm, corr_profile.copy(), profile_idx, m, minlag)
    distance_profile = np.sqrt(np.maximum(0.0, 2.0 * m * (1.0 - corr_profile)))
    distance_profile[~np.isfinite(corr_profile)] = np.nan
    return distance_profile, profile_idx.astype(np.float64), motifs


@njit(cache=True, parallel=True)
def _window_invnorm(x, mu, finite_windows, m):
    subcount = len(mu)
    invnorm = np.empty(subcount, dtype=np.float64)
    for i in prange(subcount):
        if not finite_windows[i]:
            invnorm[i] = np.nan
            continue
        total = 0.0
        for j in range(m):
            diff = x[i + j] - mu[i]
            total += diff * diff
        norm = math.sqrt(total)
        if norm > 0.0 and math.isfinite(norm):
            invnorm[i] = 1.0 / norm
        else:
            invnorm[i] = np.nan
    return invnorm


@njit(parallel=True)
def _mpx_correlation_profile(x, mu, mus, invnorm, finite_windows, m, minlag):
    subcount = len(mu)
    dr_bwd = np.zeros(subcount, dtype=np.float64)
    dc_bwd = np.zeros(subcount, dtype=np.float64)
    dr_fwd = np.empty(subcount, dtype=np.float64)
    dc_fwd = np.empty(subcount, dtype=np.float64)

    for i in prange(1, subcount):
        dr_bwd[i] = x[i - 1] - mu[i - 1]
        dc_bwd[i] = x[i - 1] - mus[i]
    for i in prange(subcount):
        dr_fwd[i] = x[i + m - 1] - mu[i]
        dc_fwd[i] = x[i + m - 1] - mus[i]

    thread_count = get_num_threads()
    local_profile = np.full((thread_count, subcount), -1.0, dtype=np.float64)
    local_idx = np.full((thread_count, subcount), -1, dtype=np.int64)

    first_offset = max(1, minlag)
    for offset in prange(first_offset, subcount):
        tid = get_thread_id()
        cov = 0.0
        for k in range(m):
            cov += (x[offset + k] - mu[offset]) * (x[k] - mu[0])

        row_count = subcount - offset
        for row in range(row_count):
            col = row + offset
            if row > 0:
                cov = (
                    cov
                    - dr_bwd[row] * dc_bwd[col]
                    + dr_fwd[row] * dc_fwd[col]
                )

            corr = cov * invnorm[row] * invnorm[col]
            if not math.isfinite(corr):
                continue
            if corr > 1.0:
                corr = 1.0
            elif corr < -1.0:
                corr = -1.0

            if corr > local_profile[tid, row]:
                local_profile[tid, row] = corr
                local_idx[tid, row] = col
            if corr > local_profile[tid, col]:
                local_profile[tid, col] = corr
                local_idx[tid, col] = row

    profile = np.full(subcount, -1.0, dtype=np.float64)
    profile_idx = np.full(subcount, -1, dtype=np.int64)
    for i in prange(subcount):
        if not finite_windows[i]:
            profile[i] = np.nan
            continue
        best = -1.0
        best_idx = -1
        for tid in range(thread_count):
            value = local_profile[tid, i]
            if value > best:
                best = value
                best_idx = local_idx[tid, i]
        profile[i] = best
        profile_idx[i] = best_idx

    return profile, profile_idx


def find_motifs(time_series, mu, invnorm, corr_profile, profile_idx, subseq_len, exclusion_len):
    motif_count = 3
    radius = 2.0
    neighbor_count = 10
    motifs = np.full((neighbor_count + 2, motif_count), np.nan, dtype=np.float64)
    work = np.asarray(corr_profile, dtype=np.float64).copy()

    for motif_col in range(motif_count):
        finite = np.flatnonzero(np.isfinite(work))
        if len(finite) == 0:
            break
        mot_idx = int(finite[np.argmax(work[finite])])
        corr = float(work[mot_idx])
        neighbor_idx = int(profile_idx[mot_idx])
        if not math.isfinite(corr) or corr == -1.0 or neighbor_idx < 0:
            break

        first = min(mot_idx, neighbor_idx)
        second = max(mot_idx, neighbor_idx)
        motifs[0:2, motif_col] = [first, second]

        corr_neighbors = crosscorr_profile(time_series, mu, invnorm, mot_idx, subseq_len)
        corr_neighbors[~np.isfinite(work)] = np.nan

        if exclusion_len > 0:
            for idx in (first, second):
                begin = max(0, int(idx) - exclusion_len + 1)
                end = min(len(corr_neighbors), int(idx) + exclusion_len)
                corr_neighbors[begin:end] = np.nan

        for neighbor_row in range(2, neighbor_count + 2):
            finite_neighbors = np.flatnonzero(np.isfinite(corr_neighbors))
            if len(finite_neighbors) == 0:
                break
            neighbor = int(finite_neighbors[np.argmax(corr_neighbors[finite_neighbors])])
            neighbor_corr = float(corr_neighbors[neighbor])
            if (
                not math.isfinite(neighbor_corr)
                or (1.0 - neighbor_corr) >= radius * (1.0 - corr)
            ):
                break
            motifs[neighbor_row, motif_col] = neighbor
            if exclusion_len > 0:
                begin = max(0, neighbor - exclusion_len + 1)
                end = min(len(corr_neighbors), neighbor + exclusion_len)
                corr_neighbors[begin:end] = np.nan

        work[~np.isfinite(corr_neighbors)] = np.nan

    return motifs


def crosscorr_profile(time_series, mu, invnorm, idx, subseq_len, fft_threshold=None):
    """Compute the correlation profile for one motif candidate.

    This mirrors the MATLAB ``findMotifs`` choice: use a direct sliding dot
    product for short subsequences and FFT convolution for longer ones.
    """
    if fft_threshold is None:
        fft_threshold = max(1, get_num_threads()) * 128
    if subseq_len < fft_threshold:
        return _crosscorr_profile_direct(time_series, mu, invnorm, idx, subseq_len)
    return _crosscorr_profile_fft(time_series, mu, invnorm, idx, subseq_len)


@njit(cache=True, parallel=True)
def _crosscorr_profile_direct(time_series, mu, invnorm, idx, subseq_len):
    subcount = len(mu)
    out = np.empty(subcount, dtype=np.float64)
    ref_mu = mu[idx]
    ref_invnorm = invnorm[idx]
    for i in prange(subcount):
        cov = 0.0
        for j in range(subseq_len):
            cov += time_series[i + j] * (time_series[idx + j] - ref_mu)
        corr = cov * ref_invnorm * invnorm[i]
        if math.isfinite(corr):
            if corr > 1.0:
                corr = 1.0
            elif corr < -1.0:
                corr = -1.0
            out[i] = corr
        else:
            out[i] = np.nan
    return out


def _crosscorr_profile_fft(time_series, mu, invnorm, idx, subseq_len):
    subcount = len(mu)
    ref = (time_series[idx:idx + subseq_len] - mu[idx]) * invnorm[idx]
    pad_len = next_power_of_two(len(time_series) + subseq_len - 1)
    spectrum = np.fft.rfft(time_series, pad_len)
    ref_spectrum = np.fft.rfft(ref[::-1], pad_len)
    cov = np.fft.irfft(spectrum * ref_spectrum, pad_len)[subseq_len - 1:subseq_len - 1 + subcount]
    corr = cov * invnorm
    corr[~np.isfinite(corr)] = np.nan
    return np.clip(corr, -1.0, 1.0)


def next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def paa(series, num_coeff):
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    num_coeff = int(num_coeff)
    if num_coeff <= 0:
        raise ValueError("num_coeff must be positive")
    seg_len = len(series) // num_coeff
    if seg_len <= 0:
        raise ValueError("num_coeff must not exceed series length")
    return series[:num_coeff * seg_len].reshape(num_coeff, seg_len).mean(axis=1)


def moving_mean(series, window):
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(series, kernel, mode="valid")


@njit(cache=True)
def moving_mean_compensated(series, window):
    n = len(series)
    count = n - window + 1
    out = np.empty(count, dtype=np.float64)
    p = series[0]
    s = 0.0

    for i in range(1, window):
        x = p + series[i]
        z = x - p
        s = s + ((p - (x - z)) + (series[i] - z))
        p = x

    out[0] = p + s

    for i in range(window, n):
        x = p - series[i - window]
        z = x - p
        s = s + ((p - (x - z)) - (series[i - window] + z))
        p = x

        x = p + series[i]
        z = x - p
        s = s + ((p - (x - z)) + (series[i] - z))
        p = x

        out[i - window + 1] = p + s

    for i in range(count):
        out[i] = out[i] / window
    return out


def moving_std_population(series, window):
    mean = moving_mean(series, window)
    mean_sq = moving_mean(series * series, window)
    return np.sqrt(np.maximum(0.0, mean_sq - mean * mean))


def finite_window_mask(series, window):
    finite = np.isfinite(series).astype(np.int32)
    counts = np.convolve(finite, np.ones(int(window), dtype=np.int32), mode="valid")
    return counts == int(window)


def znormalize_windows(windows):
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True)
    out = np.full(windows.shape, np.nan, dtype=np.float64)
    valid = stds[:, 0] > 1e-12
    out[valid] = (windows[valid] - means[valid]) / stds[valid]
    return out
