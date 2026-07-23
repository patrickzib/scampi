"""Input/output helpers for the standalone MOMP Python app."""

import json
from pathlib import Path

import numpy as np


def load_series(path, mat_key=None):
    """Load a one-dimensional series from .npy, .csv/.txt, or .mat."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix in {".csv", ".txt", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        data = np.loadtxt(path, delimiter=delimiter)
    elif suffix == ".mat":
        data = load_mat(path, mat_key)
    else:
        raise ValueError(
            f"Unsupported input suffix '{path.suffix}'. "
            "Use .npy, .csv, .txt, .tsv, or .mat."
        )

    data = np.asarray(data, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("input series is empty after removing NaN/Inf values")
    return data


def load_mat(path, mat_key=None):
    """Load a numeric array from a MATLAB .mat file."""
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError("Reading .mat files requires scipy") from exc

    mat = sio.loadmat(path)
    if mat_key is not None:
        if mat_key not in mat:
            keys = sorted(k for k in mat if not k.startswith("__"))
            raise KeyError(f"MAT key '{mat_key}' not found. Available keys: {keys}")
        return mat[mat_key]

    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            return arr
    raise ValueError("no numeric array found in .mat file")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
