"""Plotting helpers for MOMP Python results."""

import os
from pathlib import Path
import tempfile


def save_motif_pair_plot(path, series, result):
    """Save a plot of the input series and the discovered motif pair."""
    try:
        if "MPLCONFIGDIR" not in os.environ:
            cache_name = f"momp-python-matplotlib-{os.getuid()}-{os.getpid()}"
            os.environ["MPLCONFIGDIR"] = str(
                Path(tempfile.gettempdir()) / cache_name)

        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib") from exc

    path = Path(path)
    m = result.motif_length
    i, j = result.locations

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(series, color="0.35", linewidth=0.8)
    for pos, color, label in [
        (i, "tab:red", "motif 1"),
        (j, "tab:blue", "motif 2"),
    ]:
        axes[0].plot(
            range(pos, pos + m),
            series[pos:pos + m],
            color=color,
            linewidth=1.8,
            label=label,
        )
    axes[0].set_title(
        f"MOMP Python motif pair: distance={result.distance:0.6g}, "
        f"locations={result.locations}"
    )
    axes[0].legend(loc="upper right")

    a = series[i:i + m]
    b = series[j:j + m]
    axes[1].plot(znorm(a), color="tab:red", linewidth=1.8, label="motif 1")
    axes[1].plot(znorm(b), color="tab:blue", linewidth=1.2, label="motif 2")
    axes[1].set_title("Z-normalized motif pair")
    axes[1].legend(loc="upper right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def znorm(values):
    std = values.std()
    if std <= 1e-12:
        return values * 0.0
    return (values - values.mean()) / std
