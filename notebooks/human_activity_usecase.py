import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    annotations = ["Walk very\nslow", "Normal\nWalk", "Nordic Walk", "Run", "Cycle", "Run", "Normal\nWalk", "Soccer", "Rope\nJump"]
    return (annotations,)


@app.cell
def _(load_pamap):
    series_df, changepoints = load_pamap()
    series = series_df.values[0,:]
    return changepoints, series


@app.cell
def _(plt, series):
    plt.figure(figsize=(12, 4))
    plt.plot(series)
    return


@app.cell
def _(series):
    n = series.shape[0]
    return


@app.cell
def _():
    k_max = 100
    return


@app.cell
def _():
    window_size = 160
    return


@app.cell
def _(SCAMPI, moving_average, np):
    class ScampiTopN(object):
        def __init__(self, series, k_max, window_size, top_N):
            self.series = series
            self.k_max = k_max
            self.window_size = window_size
            self.top_N = top_N
            self.found_motiflets = []
            self.mask = np.zeros_like(series)
            self.rng = np.random.default_rng(1234)

        def apply_mask(self):
            mavg = moving_average(self.series, self.window_size) * self.mask
            noise = self.rng.normal(scale=np.abs(mavg), size=self.mask.shape) * self.mask
            noise += mavg
            reset_series = self.series * (1 - self.mask)
            return reset_series + noise

        def __next__(self):
            if len(self.found_motiflets) >= self.top_N:
                return None
            series = self.apply_mask()
            ml = SCAMPI(
                "Human Activity",
                series,
                n_jobs=-1,
                backend="scampi",
                verbose=False,
                scampi_delta=0.1,
            )
            dists, motif_sets, elbow_points = ml.fit_k_elbow(
                k_max=self.k_max,
                motif_length=self.window_size,
                plot_elbows=False,
                plot_motifs_as_grid=False,
            )
            for elbow in elbow_points:
                motiflet = motif_sets[elbow][0][0]
                for i in motiflet:
                    self.mask[i-2*self.window_size:i+int(3*self.window_size)] = 1
            print("masked timestamps:", np.sum(self.mask))
            self.found_motiflets.append((dists, motif_sets, elbow_points))
            return dists[elbow_points], motif_sets[elbow_points]




    return (ScampiTopN,)


@app.cell
def _():
    # rwalk = series[2000:3000]
    # rtopn = ScampiTopN(rwalk, k_max=10, window_size=100, top_N=10)
    # rtopn.mask[200: 800] = 1

    # plt.plot(rwalk, label="original", lw=3)
    # masked = rtopn.apply_mask()
    # print(masked.shape, rwalk.shape)
    # plt.plot(masked, label="masked")
    # plt.legend()
    return


@app.cell
def _(ScampiTopN, series):
    topn = ScampiTopN(series, 100, 200, 12)
    iteration = 0
    while True:
        res = next(topn)
        if res is None:
            break
    return (topn,)


@app.cell
def _(ScampiTopN, annotations, changepoints, extent, np, plt, sns, topn):
    def plot_topn(topn: ScampiTopN, changepoints: np.array, annotations, fname=None):
        import matplotlib.gridspec as gridspec

        if topn.top_N <= 10:
            colors = sns.color_palette("tab10")
        else:
            colors = sns.color_palette("tab20")

        nrows = 1 + int(np.ceil(topn.top_N / 3))
        fig = plt.figure(figsize=(12, 2*nrows))

        # Create a GridSpec with 4 rows and 3 columns
        gs = gridspec.GridSpec(nrows, 3, figure=fig, hspace=0.4, wspace=0.3)

        # Top row: single plot spanning all 3 columns
        ax_top = fig.add_subplot(gs[0, :])
        # ax_top.set_title("Top Plot")
        ax_top.plot(topn.series, c="lightgray")
        for t in changepoints:
            ax_top.axvline(t, linestyle="dotted", c="lightgray")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_ylim(-180, 200)
        for annotation, (b, e) in zip(annotations, 
                                      zip(changepoints[:-1], changepoints[1:])):
            x = (b + e) / 2
            ax_top.annotate(annotation, xy=(x, 100), fontsize=10, ha="center")

        # Bottom 3x3 grid occupying rows 1-3

        toplot = []
        for res in topn.found_motiflets:
            dists, motif_sets, elbow_points = res
            hsm = motif_sets[elbow_points[-1]][0][0]
            d = extent(topn.series, topn.window_size, hsm)
            toplot.append((d, hsm))
        toplot = sorted(toplot)

        axes = []
        # for i, res in enumerate(topn.found_motiflets):
        for i, (d, hsm) in enumerate(toplot):
            if i >= len(colors):
                break
            row = i // 3 + 1
            col = i % 3
            dists, motif_sets, elbow_points = res
            # hsm = motif_sets[elbow_points[-1]][0][0]
            # d = extent(topn.series, topn.window_size, hsm)
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")
            mlabels = dict()
            for j, t in enumerate(hsm):
                idx = np.arange(t, t+topn.window_size)
                subseq = topn.series[idx]
                ax.plot(znorm(subseq), c=colors[i], alpha=0.3)
                ax_top.plot(idx, subseq, c=colors[i])
                l = annotations[np.searchsorted(changepoints, t)-1]
                mlabels[l] = mlabels.get(l, 0) + 1
                # plt.plot(idx, subseq, c=colors[iteration])
            activity_label = "\n".join([
                f"{k.replace("\n", " ")} (x{v})"
                for k, v in mlabels.items()
            ])
            ax.set_title(f"{i+1:02d}. {activity_label} (e={d:.2f})",
                         fontsize=10, ha="left", loc="left", va="top")
            axes.append(ax)

        plt.tight_layout(pad=0)

        if fname is not None:
            plt.savefig(fname, dpi=300, bbox_inches="tight")

        plt.show()

    plot_topn(topn, changepoints, annotations) 
    return (plot_topn,)


@app.cell
def _(annotations, changepoints, quantitative_evaluation, topn):
    quantitative_evaluation(topn, changepoints, annotations)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The 7-th motiflet appears between activities
    """)
    return


@app.cell
def _(ScampiTopN, series):
    long_topn = ScampiTopN(series, 100, 400, 12)
    while next(long_topn) is not None:
        pass
    return (long_topn,)


@app.cell
def _(annotations, changepoints, long_topn, quantitative_evaluation):
    quantitative_evaluation(long_topn, changepoints, annotations)
    return


@app.cell
def _(annotations, changepoints, long_topn, plot_topn):
    plot_topn(long_topn, changepoints, annotations, fname=f"human-activity-w{long_topn.window_size}.png") 
    return


@app.cell
def _(ScampiTopN, find_dominant_window_sizes2, series):
    long_topn_alt = ScampiTopN(series, 100, 4*find_dominant_window_sizes2(series), 12)
    while next(long_topn_alt) is not None:
        pass
    return (long_topn_alt,)


@app.cell
def _(annotations, changepoints, long_topn_alt, plot_topn):
    plot_topn(long_topn_alt, changepoints, annotations, fname=f"human-activity-w{long_topn_alt.window_size}.png") 
    return


@app.cell
def _(ScampiTopN, changepoints, series):
    last_two = series[changepoints[-3]:]
    last_two_topn = ScampiTopN(last_two, k_max=100, window_size=110, top_N=3)
    while next(last_two_topn) is not None:
        pass
    return last_two, last_two_topn


@app.cell
def _(annotations, changepoints, last_two_topn, quantitative_evaluation):
    quantitative_evaluation(last_two_topn, changepoints, annotations)
    return


@app.cell
def _(annotations, changepoints, last_two_topn, plot_topn):
    plot_topn(last_two_topn, changepoints=changepoints[-3:]-changepoints[-3], annotations=annotations[-2:], fname=f"human-activity-w{last_two_topn.window_size}-last-two.png")
    return


@app.cell
def _(ScampiTopN, find_dominant_window_sizes2, last_two):
    # last_two_alt = series[changepoints[-3]:]
    last_two_alt_topn = ScampiTopN(
        last_two, 
        k_max=100, 
        window_size=3*find_dominant_window_sizes2(last_two), 
        top_N=3
    )
    while next(last_two_alt_topn) is not None:
        pass
    return (last_two_alt_topn,)


@app.cell
def _(annotations, changepoints, last_two_alt_topn, plot_topn):
    plot_topn(last_two_alt_topn, changepoints=changepoints[-3:]-changepoints[-3], annotations=annotations[-2:], fname=f"human-activity-w{last_two_alt_topn.window_size}-last-two.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <hr />
    """)
    return


@app.cell
def _(find_dominant_window_sizes2, series):
    5*find_dominant_window_sizes2(series)
    return


@app.cell
def _(changepoints, find_dominant_window_sizes2, series):
    5*find_dominant_window_sizes2(series[changepoints[-3]:])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix: utilities
    """)
    return


@app.cell
def _():
    import marimo as mo
    import os
    import numpy as np
    import pandas as pd
    import scampi
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    import warnings
    warnings.simplefilter("ignore")
    return mo, np, os, pd, plt, sns


@app.cell
def _():
    from scampi.scampi import SCAMPI

    return (SCAMPI,)


@app.cell
def _(sns):
    colors = sns.color_palette("tab20")
    return (colors,)


@app.cell
def _(load_dataset, np, pd):
    def load_pamap():
        dataset="PAMAP"
        selection = [126, 127, 128] # Outdoor
        df_data = load_dataset(dataset)

        ts_name = df_data["name"].iloc[selection]
        ts = df_data.time_series.iloc[selection]
        cps = df_data.change_points.iloc[selection[0]]

        X = np.zeros((len(ts.values), len(ts.values[0])))
        for i, data in enumerate(ts.values):
            X[i] = data

        cps = np.concatenate([[0], cps, [X.shape[1]]])

        series = pd.DataFrame(data=X, index=ts_name)
        series.rename(index={'PAMAP_Outdoor_Subject8_IMU_Shoe_X-Acc': 'Shoe X-Acc', 
                             'PAMAP_Outdoor_Subject8_IMU_Shoe_Y-Acc': 'Shoe Z-Acc', 
                             'PAMAP_Outdoor_Subject8_IMU_Shoe_Z-Acc': 'Shoe Y-Acc'}, inplace=True)

        ## series = series.iloc[0, :]
        series = series.loc[["Shoe X-Acc"]]
        return series, cps

    return (load_pamap,)


@app.cell
def _(ScampiTopN, extent, np, pd):
    def quantitative_evaluation(topn: ScampiTopN,
                                changepoints: np.array,
                                annotations):
        w = topn.window_size
        motiflets = []
        section_motiflets = dict()
        for res in topn.found_motiflets:
            dists, motif_sets, elbow_points = res
            hsm = motif_sets[elbow_points[-1]][0][0]   # highest support motiflet
            d = extent(topn.series, topn.window_size, hsm)
            mlabels = dict()
            for t in hsm:
                l = annotations[np.searchsorted(changepoints, t)-1]
                mlabels[l] = mlabels.get(l, 0) + 1
            best_count, best_label = max([(v, k) for k, v in mlabels.items()])
            section = section_motiflets.get(best_label, [])
            section.extend(hsm)
            section_motiflets[best_label] = section

        results = []
        for section, motiflets in section_motiflets.items():
            total = 0
            covered = 0
            flags = np.zeros_like(topn.series)
            for t in motiflets:
                flags[t:t+w] = 1
            for l, (b, e) in zip(annotations, zip(changepoints[:-1], changepoints[1:])):
                if l == section:
                    covered += flags[b:e].sum()
                    total += e - b
            results.append(dict(section=section, total=total, covered=covered))
        results = pd.DataFrame(results)
        results["fraction"] = results["covered"] / results["total"]
        return results
            
            # motiflets.append(dict(extent=d, motiflet=hsm, best_label=best_label))
        # motiflets = pd.DataFrame(motiflets).sort_values("extent")
        # return motiflets
    return (quantitative_evaluation,)


@app.cell
def _(np, os, pd):
    def load_dataset(dataset, selection=None):
        desc_filename = f"datasets/{dataset}/desc.txt"
        desc_file = []

        with open(desc_filename, 'r') as file:
            for line in file.readlines(): desc_file.append(line.split(","))

        df = []

        for idx, row in enumerate(desc_file):
            if selection is not None and idx not in selection: continue
            (ts_name, window_size), change_points = row[:2], row[2:]
            if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
            path = f'datasets/{dataset}/'

            if os.path.exists(path + ts_name + ".txt"):
                ts = np.loadtxt(fname=path + ts_name + ".txt", dtype=np.float64)
            else:
                ts = np.load(file=path + "data.npz")[ts_name]

            df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

        return pd.DataFrame.from_records(df, columns=["name", "window_size", "change_points", "time_series"])

    return (load_dataset,)


@app.cell
def _(colors, cps, mapping, np, plt, sns, y_labels):
    def plot_data(series, annotations, ax=None):
        if not ax:
            fig, ax = plt.subplots(figsize=(20, 5))

        tick_offsets = []
        first_label = {}
        offset = 0
        factor = 16
        for dim in range(series.shape[0]):
            for i, (a, b) in enumerate(zip(cps[:-1], cps[1:])):
                dim_data = series.iloc[dim, a:b]
                label = annotations[i]    
                c = colors[mapping[label] % len(colors)]
                if not label in first_label:
                    first_label[label] = label
                    sns.lineplot(x=np.arange(a,b,factor), linewidth=1, y=dim_data[::factor]+offset, ax = ax, label=label, color=c)
                else: 
                    sns.lineplot(x=np.arange(a,b,factor), linewidth=1, y=dim_data[::factor]+offset, ax = ax, color=c)
                ax.axvline(x=b, color="black", linestyle="--", linewidth=1)    
            tick_offsets.append(offset)
            offset -= 1.2 * (series.iloc[dim].values.max() - series.iloc[dim].values.min())    

        ax.set_yticks(tick_offsets)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel("")
        sns.despine()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=10, fontsize=12)

        # ax.legend().set_visible(False)
        # ax.axis('off')
        plt.tight_layout()

    return


@app.cell
def _(np):
    def moving_average(a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret / n

    return (moving_average,)


@app.function
def znorm(x):
    return (x - x.mean()) / x.std()


@app.cell
def _(np):
    def zeucl(x, y):
        return np.linalg.norm(znorm(x) - znorm(y))

    return (zeucl,)


@app.cell
def _(zeucl):
    def extent(ts, w, motiflet):
        e = 0
        for i in motiflet:
            for j in motiflet:
                if j > i:
                    its = ts[i:i+w]
                    jts = ts[j:j+w]
                    e = max(e, zeucl(its, jts))
        return e

    return (extent,)


@app.cell
def _(np, plt):
    def find_dominant_window_sizes(X, offset=0.05):
        """Determine the Window-Size using dominant FFT-frequencies."""
        fourier = np.absolute(np.fft.fft(X))
        freqs = np.fft.fftfreq(X.shape[0], 1)
        print(freqs)
        print(fourier)
        plt.plot(fourier)
        # plt.semilogx()
        plt.show()

        coefs = []
        window_sizes = []

        for coef, freq in zip(fourier, freqs):
            if coef and freq > 0:
                coefs.append(coef)
                window_sizes.append(1 / freq)

        coefs = np.array(coefs)
        window_sizes = np.asarray(window_sizes, dtype=np.int64)

        idx = np.argsort(coefs)[::-1]
        return next(
            (
                int(window_size / 2)
                for window_size in window_sizes[idx]
                if window_size in range(20, int(X.shape[0] * offset))
            ),
            window_sizes[idx[0]],
        )

    return (find_dominant_window_sizes,)


@app.cell
def _(find_dominant_window_sizes, series):
    find_dominant_window_sizes(series)
    return


@app.cell
def _(changepoints, np, plt, series):
    def find_dominant_window_sizes2(X, offset=0.05, fs=100, f_min=1e-2):
        """Determine the Window-Size using dominant FFT-frequencies."""
        X = X - X.mean()
        fourier = np.fft.rfft(X)
        freqs = np.fft.rfftfreq(X.shape[0], d=1/fs)

        fourier = np.abs(fourier) / X.shape[0]             # Normalize amplitude
        fourier[1:-1] *= 2                          # Double non-DC, non-Nyquist bins
    
        plt.plot(freqs,fourier)
        # plt.xlim(0, fs / 2)
        plt.semilogx()

        mask = freqs >= f_min
        dominant_freq = freqs[mask][np.argmax(fourier[mask])]
        print("dominant frequency", dominant_freq, "corresponding window", fs/dominant_freq)
        plt.axvline(dominant_freq, color="red")
        ws = int(round(fs / dominant_freq))
        return ws

        # coefs = []
        # window_sizes = []

        # for coef, freq in zip(fourier, freqs):
        #     if coef and freq > 0:
        #         coefs.append(coef)
        #         window_sizes.append(fs / freq)

        # coefs = np.array(coefs)
        # window_sizes = np.asarray(window_sizes, dtype=np.int64)
    
        # idx = np.argsort(coefs)[::-1]
        # print(window_sizes[idx])
        # ws = next(
        #     (
        #         int(window_size)
        #         for window_size in window_sizes[idx]
        #         if window_size in range(20, int(X.shape[0] * offset))
        #     ),
        #     window_sizes[idx[0]],
        # )
        # print(ws)

        # plt.axvline(fs/(ws), color="green", zorder=-1)
    
        # plt.show()
        # return ws

    find_dominant_window_sizes2(series[changepoints[-3]:])
    return (find_dominant_window_sizes2,)


if __name__ == "__main__":
    app.run()
