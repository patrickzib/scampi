"""Synthetic planted-motif benchmark helpers."""

import json
import math

import numpy as np


def generate_random_walk_with_planted_motif(
        n,
        motif_length,
        n_instances,
        motif_noise,
        random_walk_sigma,
        motif_amplitude,
        seed,
        motif_cycles=2.0):
    """Create a random walk with non-overlapping noisy sine motif instances."""
    rng = np.random.default_rng(seed)
    series = np.cumsum(
        rng.normal(0.0, random_walk_sigma, size=n)).astype(np.float64)
    positions = choose_non_overlapping_positions(n, motif_length, n_instances, rng)

    x = np.linspace(0.0, 2.0 * np.pi * motif_cycles, motif_length)
    motif = motif_amplitude * np.sin(x)

    for pos in positions:
        noise = rng.normal(0.0, motif_noise, size=motif_length)
        baseline = series[pos]
        series[pos:pos + motif_length] = baseline + motif + noise

    return series, positions, motif


def choose_non_overlapping_positions(n, motif_length, n_instances, rng):
    """Choose sorted motif starts with at least one motif length separation."""
    if motif_length <= 0:
        raise ValueError("motif_length must be > 0")
    if n_instances <= 0:
        raise ValueError("n_instances must be > 0")
    if n < motif_length * n_instances:
        raise ValueError(
            "n must be at least motif_length * n_instances for non-overlap")

    positions = []
    max_start = n - motif_length
    for _ in range(10_000):
        if len(positions) == n_instances:
            break
        pos = int(rng.integers(0, max_start + 1))
        if all(abs(pos - other) >= motif_length for other in positions):
            positions.append(pos)

    if len(positions) < n_instances:
        positions = np.linspace(0, max_start, n_instances, dtype=np.int64).tolist()

    return sorted(int(pos) for pos in positions)


def select_best_motiflet(motiflets, extents, selected_k, ground_truth, tolerance):
    """Pick the best top-N motiflet at selected_k by F-score then extent."""
    candidates = candidates_at_k(motiflets, selected_k)
    distances = distances_at_k(extents, selected_k, len(candidates))

    best = None
    for rank, candidate in enumerate(candidates):
        recovered = clean_positions(candidate)
        metrics = match_positions(recovered, ground_truth, tolerance)
        extent = distances[rank] if rank < len(distances) else math.inf
        current = {
            "rank": rank,
            "motiflet": recovered,
            "extent": extent,
            "metrics": metrics,
        }
        if best is None:
            best = current
            continue
        if metrics["f_score"] > best["metrics"]["f_score"]:
            best = current
        elif metrics["f_score"] == best["metrics"]["f_score"]:
            if extent < best["extent"]:
                best = current

    if best is None:
        metrics = match_positions([], ground_truth, tolerance)
        return {
            "rank": None,
            "motiflet": [],
            "extent": math.inf,
            "metrics": metrics,
        }

    return best


def candidates_at_k(motiflets, selected_k):
    if selected_k >= len(motiflets) or motiflets[selected_k] is None:
        return []

    candidates = motiflets[selected_k]
    arr = np.asarray(candidates, dtype=object)
    if arr.ndim == 0:
        return []
    if arr.ndim == 1 and arr.size > 0 and np.isscalar(arr[0]):
        return [arr]
    return [candidate for candidate in candidates if candidate is not None]


def distances_at_k(extents, selected_k, n_candidates):
    if selected_k >= len(extents):
        return [math.inf] * n_candidates
    values = np.asarray(extents[selected_k], dtype=np.float64).reshape(-1)
    return [float(value) for value in values[:n_candidates]]


def clean_positions(positions):
    arr = np.asarray(positions).reshape(-1)
    return [int(pos) for pos in arr if int(pos) >= 0]


def match_positions(recovered, ground_truth, tolerance):
    """Greedily match recovered positions to ground truth within tolerance."""
    unmatched_truth = set(int(pos) for pos in ground_truth)
    matches = []
    false_positives = []

    for recovered_pos in sorted(int(pos) for pos in recovered):
        if not unmatched_truth:
            false_positives.append(recovered_pos)
            continue
        nearest = min(unmatched_truth, key=lambda truth: abs(truth - recovered_pos))
        if abs(nearest - recovered_pos) <= tolerance:
            unmatched_truth.remove(nearest)
            matches.append({
                "ground_truth": int(nearest),
                "recovered": int(recovered_pos),
                "absolute_error": int(abs(nearest - recovered_pos)),
            })
        else:
            false_positives.append(recovered_pos)

    tp = len(matches)
    fp = len(false_positives)
    fn = len(unmatched_truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "matched_ground_truth": [match["ground_truth"] for match in matches],
        "matched_recovered": [match["recovered"] for match in matches],
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": sorted(int(pos) for pos in unmatched_truth),
        "true_positives": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
    }


def to_jsonable(value):
    """Convert NumPy-heavy benchmark values into strict JSON values."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return finite_float(value)
    if isinstance(value, float):
        return finite_float(value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def finite_float(value):
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def json_dumps(value):
    return json.dumps(to_jsonable(value), separators=(",", ":"), sort_keys=True)
