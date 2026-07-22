# -*- coding: utf-8 -*-
"""Plotting utilities.
"""

__author__ = ["patrickzib"]

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.stats import zscore
from tsdownsample import MinMaxLTTBDownsampler

import scampi.scampi as ml
from scampi.distances import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_dataset(
        ds_name,
        data,
        ground_truth=None,
        max_points=1000,
        show=True
):
    """Plots a time series.

    Parameters
    ----------
    ds_name: String
        The name of the time series
    data: array-like
        The time series
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    show: boolean
        Outputs the plot

    """
    return plot_motifset(
        ds_name,
        data,
        ground_truth=ground_truth,
        max_points=max_points,
        show=show
    )


def plot_motifset(
        ds_name,
        data,
        motifsets=None,
        max_points=2_000,
        motif_length=None,
        ground_truth=None,
        show=True):
    """Plots the data and the found motif sets.

    Parameters
    ----------
    ds_name: String,
        The name of the time series
    data: array-like
        The time series data
    motifsets: array like
        One found motif set
    motif_length: int
        The length of the motif
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    show: boolean
        Outputs the plot

    """
    # set_sns_style(font_size)
    # sns.set(font_scale=3)
    sns.set(font="Calibri")
    sns.set_style("white")

    # turn into 2d array
    data = ml.convert_to_2d(data)

    if motifsets is not None:
        git_ratio = [4]
        for _ in range(len(motifsets)):
            git_ratio.append(1)

        fig, axes = plt.subplots(2, 1 + len(motifsets),
                                 sharey="row",
                                 sharex=False,
                                 figsize=(
                                     10 + 2 * len(motifsets),
                                     5 + (data.shape[0] + len(motifsets)) // 2),
                                 squeeze=False,
                                 gridspec_kw={
                                     'width_ratios': git_ratio,
                                     'height_ratios': [10, 3]})  # 5 for rolling stone?
    elif ground_truth is not None:
        fig, axes = plt.subplots(2, 1,
                                 sharey="row",
                                 sharex=False,
                                 figsize=(20, 5 + data.shape[0] // 2),
                                 squeeze=False,
                                 gridspec_kw={
                                     'width_ratios': [4],
                                     'height_ratios': [10, 1]})
    else:
        fig, axes = plt.subplots(1, 1, squeeze=False,
                                 figsize=(20, 5 + data.shape[0] // 2))

    if ground_truth is None:
        ground_truth = []

    data_index, data_raw = ml.pd_series_to_numpy(data)
    data_raw_sampled, data_index_sampled = data_raw, data_index

    factor = 1
    motifsets_sampled = None
    if data_raw.shape[-1] > max_points:
        data_raw_sampled = np.zeros((data_raw.shape[0], max_points))
        for i in range(data_raw.shape[0]):
            index = MinMaxLTTBDownsampler().downsample(
                np.ascontiguousarray(data_raw[i]), n_out=max_points)
            data_raw_sampled[i] = data_raw[i, index]

        data_index_sampled = data_index[index]
        factor = max(1, data_raw.shape[-1] / data_raw_sampled.shape[-1])
        if motifsets is not None:
            motifsets_sampled = list(map(lambda x: np.int32(x // factor), motifsets))
    else:
        motifsets_sampled = motifsets

    color_offset = 1
    offset = 0
    tick_offsets = []
    axes[0, 0].set_title(ds_name, fontsize=22)

    for dim in range(data_raw.shape[0]):
        dim_raw = zscore(data_raw[dim])
        dim_raw_sampled = zscore(data_raw_sampled[dim])
        offset -= 1.2 * (np.max(dim_raw_sampled) - np.min(dim_raw_sampled))
        tick_offsets.append(offset)

        # dim_raw_sampled[dim_raw_sampled > 3] = 3
        _ = sns.lineplot(
            x=data_index_sampled,
            y=dim_raw_sampled + offset,
            ax=axes[0, 0],
            linewidth=0.5,
            color="gray",
            errorbar=("ci", None),
            estimator=None
        )

    sns.despine()

    y_labels = []
    gt_count = 0

    if motifsets is not None:
        for i, motifset in enumerate(motifsets_sampled):
            if motifset is not None:
                motif_length_sampled = np.int32(max(2, motif_length // factor))
                for a, pos in enumerate(motifset):
                    _ = sns.lineplot(
                        ax=axes[0, 0],
                        x=data_index_sampled[
                            pos: pos + motif_length_sampled],
                        y=dim_raw_sampled[
                              pos: pos + motif_length_sampled] + offset,
                        linewidth=3,
                        color=sns.color_palette("tab10")[
                            (color_offset + i) % len(sns.color_palette("tab10"))],
                        errorbar=("ci", None),
                        estimator=None)

                    motif_length_disp = motif_length

                    axes[0, 1 + i].set_title(
                        ("Motif Set " + str(i + 1)) + "\n" +
                        "k=" + str(len(motifset)) +
                        ", l=" + str(motif_length_disp),
                        fontsize=18)

                    motif_factor = 1
                    if motif_length_disp > max_points:
                        motif_factor = int(
                            max(1, np.floor(motif_length_disp / max_points)))
                        # print(f"factor {motif_factor}")

                    df = pd.DataFrame()
                    df["time"] = range(0, motif_length_disp, motif_factor)

                    for aa, pos in enumerate(motifsets[i]):
                        values = np.zeros(len(df["time"]), dtype=np.float32)
                        value = dim_raw[pos:pos + motif_length_disp:motif_factor]
                        values[:len(value)] = value

                        df[str(aa)] = (values - values.mean()) / (
                                values.std() + 1e-4) + offset

                    df_melt = pd.melt(df, id_vars="time")
                    _ = sns.lineplot(
                        ax=axes[0, 1 + i],
                        data=df_melt,
                        errorbar=("ci", 95),
                        # errorbar="se",
                        n_boot=3,
                        lw=1,
                        color=sns.color_palette("tab10")[
                            (color_offset + i) % len(sns.color_palette("tab10"))],
                        x="time",
                        y="value")

    if len(ground_truth) > 0:
        motif_set_count = 0 if motifsets is None else len(motifsets)

        for aaa, column in enumerate(ground_truth):
            for offsets in ground_truth[column]:
                for off in offsets:
                    ratio = 0.8
                    start = np.int32(off[0] // factor)
                    end = np.int32(off[1] // factor)
                    if end - 1 < dim_raw_sampled.shape[0]:
                        rect = Rectangle(
                            (data_index_sampled[start], 0),
                            data_index_sampled[end - 1] - data_index_sampled[start],
                            ratio,
                            facecolor=sns.color_palette("tab10")[
                                (color_offset + motif_set_count + aaa) %
                                len(sns.color_palette("tab10"))],
                            alpha=0.7
                        )

                        rx, ry = rect.get_xy()
                        cx = rx + rect.get_width() / 2.0
                        cy = ry + rect.get_height() / 2.0
                        axes[1, 0].annotate(column, (cx, cy),
                                            color='black',
                                            weight='bold',
                                            fontsize=12,
                                            ha='center',
                                            va='center')

                        axes[1, 0].add_patch(rect)

        gt_count = 1
        y_labels.append("Ground Truth")

    if motifsets is not None:
        for i, motif_set in enumerate(motifsets_sampled):
            if motif_set is not None:
                motif_length_sampled = np.int32(max(2, motif_length // factor))

                for pos in motif_set:
                    if pos + motif_length_sampled - 1 < dim_raw_sampled.shape[0]:
                        ratio = 0.8
                        rect = Rectangle(
                            (data_index_sampled[pos], -i - gt_count),
                            data_index_sampled[pos + motif_length_sampled - 1] -
                            data_index_sampled[pos],
                            ratio,
                            facecolor=sns.color_palette("tab10")[
                                (color_offset + i) % len(
                                    sns.color_palette("tab10"))],
                            alpha=0.7
                        )
                        axes[1, 0].add_patch(rect)

                label = (("Motif Set " + str(i + 1)))
                # label = "Motif Set, k=" + str(len(motifsets))
                y_labels.append(label)

    if len(y_labels) > 0:
        axes[1, 0].set_yticks(-np.arange(len(y_labels)) + 0.5)
        axes[1, 0].set_yticklabels(y_labels, fontsize=18)
        axes[1, 0].set_ylim([-abs(len(y_labels)) + 1, 1])
        axes[1, 0].set_xlim(axes[0, 0].get_xlim())
        axes[1, 0].set_xticklabels([])
        axes[1, 0].set_xticks([])

        if motifsets is not None:
            axes[1, 0].set_title("Positions", fontsize=22)

        for i in range(1, axes.shape[-1]):
            axes[1, i].remove()

    if isinstance(data, pd.DataFrame):
        axes[0, 0].set_yticks(tick_offsets)
        axes[0, 0].set_yticklabels(data.index, fontsize=18)
        axes[0, 0].set_xlabel("Time", fontsize=18)

        if motifsets is not None:
            axes[0, 1].set_yticks(tick_offsets)
            axes[0, 1].set_yticklabels(data.index, fontsize=18)
            axes[0, 1].set_xlabel("Length", fontsize=18)

    sns.despine()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_window_lengths(ds_name, au_ef, header, motif_length_range):
    """Plots the AU_EF values for different motif lengths.

    Parameters
    ----------
    au_ef: array-like
        The AU_EF values for each motif length.
    header: String
        The header to use for the x-axis (e.g. " in time").
    motif_length_range: array-like
        The motif lengths corresponding to the AU_EF values.
    """

    indices = ~np.isinf(au_ef)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax = sns.lineplot(
        x=motif_length_range[indices],
        y=au_ef[indices],
        label="AU_EF",
        ci=None, estimator=None)
    sns.despine()
    plt.tight_layout()
    ax.set_title("Best length on " + ds_name, size=20)
    ax.set(xlabel='Motif Length' + header,
           ylabel='Area under EF\n(lower is better)')

    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    # plt.legend(loc="best")
    fig.set_figheight(5)
    fig.set_figwidth(5)
    plt.show()


def _plot_elbow_points(
        ds_name,
        data,
        motif_length,
        elbow_points,
        motifset_candidates,
        dists):
    """Plots the elbow points found.

    Parameters
    ----------
    ds_name: String
        The name of the time series.
    data: array-like
        The time series data.
    motif_length: int
        The length of the motif.
    elbow_points: array-like
        The elbow points to plot.
    motifset_candidates: 2d array-like
        The motifset candidates. Will only extract those motif sets
        within elbow_points.
    dists: array-like
        The distances (extents) for each motif set
    """
    data_index, data_raw = ml.pd_series_to_numpy(data)

    # shows only first rank elbow point for now
    ebs = elbow_points[0]
    if data_raw.ndim == 1:
        data_raw = data_raw.reshape((1, -1))

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.set_title(ds_name + "\nElbow Points")
    ax.plot(range(2, len(np.sqrt(dists))), dists[2:, 0], "b", label="Extent")

    lim1 = plt.ylim()[0]
    lim2 = plt.ylim()[1]
    for elbow in ebs:
        ax.vlines(
            elbow, lim1, lim2,
            linestyles="--", label=str(elbow) + "-Motiflet"
        )
    ax.set(xlabel='Size (k)', ylabel='Extent')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # scampi = motifset_candidates[ebs][0]
    motiflets = [motifset_candidates[eb][0] for eb in ebs]
    for i, motiflet in enumerate(motiflets):
        if motiflet is not None:
            axins = ax.inset_axes(
                [(ebs[i] - 3) / (len(motifset_candidates) - 2), 0.7, 0.3, 0.3])

            df = pd.DataFrame()
            df["time"] = data_index[range(0, motif_length)]
            for aa, pos in enumerate(motiflet):
                # Shows only first dimension
                df[str(aa)] = zscore(data_raw[0, pos:pos + motif_length])

            df_melt = pd.melt(df, id_vars="time")

            _ = sns.lineplot(ax=axins, data=df_melt, x="time", y="value", ci=99,
                             n_boot=10, color=sns.color_palette("tab10")[i % 10])
            axins.set_xlabel("")
            axins.set_ylabel("")
            axins.xaxis.set_major_formatter(plt.NullFormatter())
            axins.yaxis.set_major_formatter(plt.NullFormatter())

    plt.show()


def plot_grid_motiflets(
        ds_name, data, motifsets_, elbow_points_, dist_,
        motif_length, font_size=20,
        max_items=None,
        ground_truth=None,
        method_name=None,
        method_names=None,
        show_elbows=False,
        max_points=2_000,
        color_palette=sns.color_palette("tab10"),
        grid_dim=None,
        plot_index=None):
    """Plots the characteristic motifs for each method along the time series.

    Parameters
    ----------
    ds_name: String
        The name of the time series
    data: array-like
        The time series data
    motifsets_: 2d array-like
        The motifset candidates
    elbow_points_: array-like
        The elbow points found. Only motif sets from the elbow points will be plotted.
    dist_: array-like
        The distances (extents) of the motif set candidates
    motif_length: int
        The motif length found.
    font_size: int
        Font-size to use for plotting.
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    method_name: String
        Name of a single method
    method_names: array-like
        Name of all methods
    show_elbows: bool
        Show an elbow plot
    color_palette:
        Color-palette to use
    grid_dim: int
        The dimensionality of the grid (number of columns)
    plot_index: int
        Plots only the passed methods in the given order

    """

    sns.set(font_scale=2)
    sns.set_style("white")
    sns.set_context("paper",
                    rc={"font.size": font_size,
                        "axes.titlesize": font_size - 8,
                        "axes.labelsize": font_size - 8,
                        "xtick.labelsize": font_size - 10,
                        "ytick.labelsize": font_size - 10, })

    label_cols = 2

    (dist, motifsets, elbow_points) \
        = ml.flatten_elbows(elbow_points_, motifsets_, dist_, max_items=max_items)

    count_plots = 3 if len(motifsets[elbow_points]) > 6 else 2
    if show_elbows:
        count_plots = count_plots + 1

    if ground_truth is None:
        ground_truth = []

    if grid_dim is None:
        if plot_index is not None:
            ll = len(plot_index)
        else:
            ll = len(elbow_points)
        grid_dim = int(max(2, np.ceil(ll / 2)))

    dims = int(np.ceil(len(elbow_points) / grid_dim)) + count_plots

    fig = plt.figure(constrained_layout=True, figsize=(10, dims * 2))
    gs = fig.add_gridspec(dims, grid_dim, hspace=0.8, wspace=0.4)

    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.set_title("(a) Dataset: " + ds_name + "")

    # turn into 2d array
    if data.ndim == 1:
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        elif isinstance(data, (np.ndarray, np.generic)):
            data = data.reshape(1, -1)

    data_index, data_raw = ml.pd_series_to_numpy(data)
    data_raw_sampled, data_index_sampled = data_raw, data_index

    factor = 1
    if data_raw.shape[-1] > max_points:
        data_raw_sampled = np.zeros((data_raw.shape[0], max_points))
        for i in range(data_raw.shape[0]):
            index = MinMaxLTTBDownsampler().downsample(
                np.ascontiguousarray(data_raw[i]), n_out=max_points)
            data_raw_sampled[i] = data_raw[i, index]

        data_index_sampled = data_index[index]
        factor = max(1, data_raw.shape[-1] / data_raw_sampled.shape[-1])
        if motifsets is not None:
            motifsets_sampled = np.array(
                list(map(lambda x: (x // factor) if x is not None else x, motifsets)),
                dtype=np.object_)
    else:
        motifsets_sampled = motifsets

    _ = sns.lineplot(x=data_index_sampled, y=data_raw_sampled[0], ax=ax_ts, linewidth=1)
    sns.despine()

    for aaa, column in enumerate(ground_truth):
        for offsets in ground_truth[column]:
            for pos, offset in enumerate(offsets):
                start = np.int32(offset[0] // factor)
                end = np.int32(offset[1] // factor)
                if pos == 0:
                    sns.lineplot(x=data_index_sampled[start:end],
                                 y=data_raw_sampled[0, start:end],
                                 label=column,
                                 color=color_palette[aaa + 1],
                                 ci=None, estimator=None
                                 )
                else:
                    sns.lineplot(x=data_index_sampled[start:end],
                                 y=data_raw_sampled[0, start:end],
                                 color=color_palette[aaa + 1],
                                 ci=None, estimator=None
                                 )

    if len(motifsets[elbow_points]) > 6:
        ax_bars = fig.add_subplot(gs[1:3, :], sharex=ax_ts)
        next_id = 3
    else:
        ax_bars = fig.add_subplot(gs[1, :], sharex=ax_ts)
        next_id = 2

    ax_bars.set_title("(b) Position of Top Motif Sets")

    if show_elbows:
        ax_elbow = fig.add_subplot(gs[next_id, :])
        ax_elbow.set_title("(c) Significant Elbow Points on " + ds_name)
        ax_elbow.plot(range(len(np.sqrt(dist))), dist, "b", label="Extent")
        lim1 = plt.ylim()[0]
        lim2 = plt.ylim()[1]
        for elbow in elbow_points:
            ax_elbow.vlines(
                elbow, lim1, lim2,
                label=str(elbow) + "-Motiflet"
            )
        ax_elbow.set(xlabel='Size (k)', ylabel='Extent')
        ax_elbow.xaxis.set_major_locator(MaxNLocator(integer=True))

    gs = fig.add_gridspec(dims, grid_dim)

    #### Hack to add a subplot title
    ax_title = fig.add_subplot(gs[count_plots, :])

    if (show_elbows):
        ax_title.set_title('(d) Shape of Top Motif Sets', pad=30)
    else:
        ax_title.set_title('(c) Shape of Top Motif Sets', pad=30)

    # Turn off axis lines and ticks of the big subplot 
    ax_title.tick_params(labelcolor=(1., 1., 1., 0.0),
                         top='off', bottom='off', left='off', right='off')
    ax_title.axis('off')
    ax_title._frameon = False
    sns.despine()
    ######

    # hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    y_labels = []
    ii = -1
    motiflets_sampled = motifsets_sampled[elbow_points]
    motiflets = motifsets[elbow_points]
    for i, motiflet in enumerate(motiflets_sampled):
        if motiflet is not None:
            motif_length_sampled = np.int32(max(2, motif_length // factor))

            plot_minature = (plot_index == None) or (i in plot_index)
            if plot_minature:
                ii = ii + 1
                off = int(ii / grid_dim)
                ax_motiflet = fig.add_subplot(gs[count_plots + off, ii % grid_dim])

            df = pd.DataFrame()
            df["time"] = data_index[range(0, motif_length)]

            for aa, pos in enumerate(motiflet):
                pos = np.int32(pos)
                ratio = 0.8
                rect = Rectangle(
                    (data_index_sampled[pos], -i),
                    data_index_sampled[pos + motif_length_sampled - 1] -
                    data_index_sampled[pos],
                    ratio,
                    facecolor=color_palette[
                        (len(ground_truth) + ii % grid_dim) % len(color_palette)],
                    # hatch=hatches[i],
                    alpha=0.7
                )
                ax_bars.add_patch(rect)

            for aa, pos in enumerate(motiflets[i]):
                df[str(aa)] = zscore(data_raw[0, pos:pos + motif_length])

            if method_name is not None:
                y_labels.append(method_name + "\nTop-" + str(i + 1))

            elif method_names is not None:
                y_labels.append(method_names[i])

            dists = ""
            if (dist is not None):
                dist = np.array(dist)
                dist[dist == float("inf")] = 0
                dists = str(dist[elbow_points[i]].astype(int))
                # dists = str(int(dist[elbow_points[i]]))

            label = ""
            # if method_names is not None:
            #    label =  method_names[elbow_points[i]]

            if plot_minature:
                df_melt = pd.melt(df, id_vars="time")
                _ = sns.lineplot(ax=ax_motiflet,
                                 data=df_melt,
                                 x="time", y="value",
                                 ci=99, n_boot=10,
                                 color=color_palette[
                                     (len(ground_truth) + ii % grid_dim) % len(
                                         color_palette)],
                                 label=label + "k=" + str(len(motiflet)) + ",d=" + dists
                                 )
                ax_motiflet.set_ylabel("")

                if isinstance(data, pd.Series):
                    ax_motiflet.set_xlabel(data.index.name)

                sns.despine()
                ax_motiflet.legend(loc="upper right")

            if method_names is not None:
                ax_bars.plot([], [], label=method_names[elbow_points[i]].split()[0],
                             linewidth=10,
                             color=color_palette[
                                 (len(ground_truth) + ii % grid_dim) % len(
                                     color_palette)])
                if plot_minature:
                    ax_motiflet.set_title(method_names[elbow_points[i]])

            elif method_name is not None:
                ax_bars.plot([], [], label=method_name, linewidth=10,
                             color=color_palette[
                                 (len(ground_truth) + ii % grid_dim) % len(
                                     color_palette)])
                if plot_minature:
                    ax_motiflet.set_title(method_name + " Top-" + str(i + 1))

            if show_elbows:
                axins = ax_elbow.inset_axes(
                    [elbow_points[i] / len(motifsets), 0.7, 0.1, 0.2])

                _ = sns.lineplot(ax=axins, data=df_melt, x="time", y="value",
                                 ci=0, n_boot=10,
                                 color=color_palette[
                                     (len(ground_truth) + ii % grid_dim) % len(
                                         color_palette)])
                axins.set_xlabel("")
                axins.set_ylabel("")
                axins.xaxis.set_major_formatter(plt.NullFormatter())
                axins.yaxis.set_major_formatter(plt.NullFormatter())

            if plot_minature:
                ax_motiflet.set_yticks([])

    ax_bars.set_yticks(-np.arange(len(y_labels)) + 0.5, )
    ax_bars.set_yticklabels(y_labels, fontsize=12)
    ax_bars.set_ylim([-len(motiflets) + 1, 1])
    # ax_bars.legend(loc="best")

    if (ground_truth is not None and len(ground_truth) > 0):
        ax_ts.legend(loc="upper left", ncol=label_cols)

    plt.tight_layout()
    gs.tight_layout(fig)
    plt.show()
