"""figures: the report's pictures, drawn with matplotlib and returned as SVG text for report.html.
Every figure has one quiet look: a light chart surface, hairline grids, thin lines and bars, text in grey
inks, and one colour per symbol that follows the symbol from figure to figure.
Figures never print and never write files: the report puts them in the page"""

import io

import numpy as np
from matplotlib import rc_context
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, NullFormatter, StrMethodFormatter

from mi_race.config import Config
from mi_race.dataset import mean_signals

SURFACE = "#fcfcfb"                         # the chart surface
INK = "#0b0b0b"                             # primary text
SECONDARY = "#52514e"                       # labels and values
MUTED = "#898781"                           # tick labels
GRID = "#e1e0d9"                            # hairline gridlines
AXIS = "#c3c2b7"                            # the baseline
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",            # one colour per symbol, always in this order
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",    # one blue, light to dark:
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]               # for shares, none to all
DECODERS = {"random_forest": "random forest", "mlp": "MLP", "cnn": "CNN"}     # as in cli; a colour each, in order
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"   # the page's own sans
WIDTH = 7.2                                 # inches: 518 pt, the report's column
OVERLAY = 4                                 # at most this many signals share one plot; more get a panel each

STYLE = {
    "svg.fonttype": "none",                 # text stays text, in the page's font
    "svg.hashsalt": "mi-race",              # the same figure gives the same SVG
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],     # measures the text; the page draws it in FONT
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": SECONDARY,
    "axes.labelsize": 9,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.75,
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.75,
    "figure.facecolor": SURFACE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 0,                  # gridlines instead of ticks
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "lines.linewidth": 1.5,                 # 2 px on the page
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 8,
}


def symbol_color(u: int, n_symbols: int) -> str:
    """the colour of symbol u: its own while there are colours enough, else one colour for all (each
    symbol then has its own panel, named, so colour is not needed to tell them apart)"""
    return SERIES[u] if n_symbols <= len(SERIES) else SERIES[0]


def decoder_color(name: str) -> str:
    """the colour of a decoder, the same in every figure: random forest, MLP, CNN take the first three"""
    return SERIES[list(DECODERS).index(name)]


def plain_ticks(ax) -> None:
    """ticks as plain numbers (0, 0.5, 1; 0.001 on a log scale), at most 6 along the x axis"""
    ax.xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def svg(figure: Figure) -> str:
    """the figure as SVG text, without the XML header, ready to sit in an HTML page"""
    out = io.StringIO()
    figure.savefig(out, format="svg", metadata={"Date": None, "Creator": None})
    text = out.getvalue()
    return text[text.index("<svg"):].replace("'DejaVu Sans'", FONT)


def schedule_panel(ax, cfg: Config, u: int, width: float, columns: int = 1) -> None:
    """symbol u's releases on one axes: a bar at each release time, as high as its molecules, with the amounts
    written at the tips when there are only a few bursts and room between them for the numbers (the panel is
    one of 'columns' side by side)"""
    encoder = cfg.encoder
    vector = encoder.symbols[u]
    slots = [k for k, amount in enumerate(vector) if amount]
    ax.bar([k * encoder.slot_dt for k in slots], [vector[k] for k in slots], width=width,
           color=symbol_color(u, encoder.n_symbols), linewidth=0)
    points_per_second = (WIDTH * 72 - 60) / columns / cfg.channel.T          # about, along the time axis
    gaps = [(b - a) * encoder.slot_dt * points_per_second for a, b in zip(slots, slots[1:])]
    if len(slots) <= 3 and min(gaps, default=99) >= 16:                    # 16 pt: a 3-digit amount and air
        for k in slots:
            ax.annotate(str(vector[k]), (k * encoder.slot_dt, vector[k]), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=SECONDARY)
    ax.set_ylim(0, encoder.budget * 1.3)
    ax.set_yticks([0, encoder.budget])
    ax.set_xlim(-width, cfg.channel.T)
    ax.grid(axis="x", visible=False)
    plain_ticks(ax)


def bar_width(cfg: Config) -> float:
    """thin release bars: 60% of a slot, at most about 24 px wide"""
    return min(0.6 * cfg.encoder.slot_dt, cfg.channel.T / 24)


def release_schedules(cfg: Config) -> str:
    """the codebook as a picture: one row per symbol, a bar for each release at its time (slot k at
    k × slot_dt), as high as the molecules it releases. The time axis runs to T, the end of a run,
    so the picture shows how much of the run each symbol leaves for the molecules to arrive"""
    n = cfg.encoder.n_symbols
    with rc_context(STYLE):
        figure = Figure(figsize=(WIDTH, 0.5 + 0.55 * n), layout="constrained")
        axes = figure.subplots(n, 1, sharex=True, sharey=True, squeeze=False)[:, 0]
        for u, ax in enumerate(axes):
            schedule_panel(ax, cfg, u, bar_width(cfg))
            ax.set_ylabel(f"symbol {u}", rotation=0, ha="right", va="center", color=INK)
        axes[-1].set_xlabel("release time (s)")
        return svg(figure)


def compare_schedules(before: Config, after: Config, titles: tuple[str, str] = ("start", "found")) -> str:
    """two codebooks of one experiment side by side: one row per symbol, the codebook before on the left and
    after on the right, on the same time and molecule axes, so a moved burst shows as a bar in another place"""
    n = before.encoder.n_symbols
    with rc_context(STYLE):
        figure = Figure(figsize=(WIDTH, 0.7 + 0.55 * n), layout="constrained")
        figure.get_layout_engine().set(wspace=0.06)
        axes = figure.subplots(n, 2, sharex=True, sharey=True, squeeze=False)
        for column, (cfg, title) in enumerate(zip((before, after), titles)):
            for u in range(n):
                schedule_panel(axes[u, column], cfg, u, bar_width(cfg), columns=2)
            axes[0, column].set_title(title, loc="left", fontsize=8.5, color=INK)
            axes[-1, column].set_xlabel("release time (s)")
        for u in range(n):
            axes[u, 0].set_ylabel(f"symbol {u}", rotation=0, ha="right", va="center", color=INK)
        return svg(figure)


def mean_signal(cfg: Config, rows: np.ndarray) -> str:
    """what the receiver sees, on average: for each symbol, the molecules absorbed so far at each time,
    averaged over its runs in the dataset (the running sum of rx_0 ... rx_M). The curves are smooth even when
    a single interval holds a fraction of a molecule; a later release shifts a curve right, and its end is how
    many molecules arrive before T. Up to 4 symbols share one plot, each named at its end; more get a panel each"""
    signals = mean_signals(rows).cumsum(axis=1)
    n = len(signals)
    times = np.arange(signals.shape[1]) * cfg.channel.dt
    with rc_context(STYLE):
        if n <= OVERLAY:
            figure = Figure(figsize=(WIDTH, 2.8), layout="constrained")
            ax = figure.subplots()
            for u, signal in enumerate(signals):
                ax.plot(times, signal, color=symbol_color(u, n), label=f"symbol {u}")
            name_ends(ax, {f"symbol {u}": (times[-1], signal[-1]) for u, signal in enumerate(signals)})
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n, handlelength=1.5)
            axes = [ax]
        else:
            columns = 2
            figure = Figure(figsize=(WIDTH, 0.4 + 1.3 * -(-n // columns)), layout="constrained")
            figure.get_layout_engine().set(wspace=0.08)             # room between the columns' tick labels
            grid = figure.subplots(-(-n // columns), columns, sharex=True, sharey=True, squeeze=False)
            axes = list(grid.flat)
            for u, (ax, signal) in enumerate(zip(axes, signals)):
                ax.plot(times, signal, color=symbol_color(u, n))
                ax.set_title(f"symbol {u}", loc="left", fontsize=8.5, color=INK)
            for ax in axes[n:]:
                ax.set_visible(False)
        for ax in axes:
            ax.set_xlim(0, times[-1])
            ax.set_ylim(bottom=0)
            ax.grid(axis="x", visible=False)
            plain_ticks(ax)
        figure.supxlabel("time (s)", fontsize=9, color=SECONDARY)
        figure.supylabel(f"absorbed so far (mean of {len(rows) // n} runs)", fontsize=9, color=SECONDARY)
        return svg(figure)


def name_ends(ax, ends: dict, log: bool = False, strong: str | None = None) -> list[str]:
    """each curve's name just right of its last point (ends: name -> (x, y)), so it is named on the plot as well
    as in a legend; a name that would overlap one already placed is left out. log: the y axis is logarithmic,
    so closeness is measured in decades. strong: the one name in primary ink, placed first.
    Returns the names left out"""
    scale = (lambda y: np.log10(max(y, 1e-12))) if log else (lambda y: y)
    heights = [scale(y) for _, y in ends.values()]
    span_y = (max(heights) - min(heights)) if log else max(heights)
    span_x = max(x for x, _ in ends.values())
    placed, left_out = [], []
    for name in sorted(ends, key=lambda name: (name != strong, -ends[name][1])):
        x, y = ends[name]
        if any(abs(x - px) < 0.15 * (span_x or 1) and abs(scale(y) - scale(py)) < 0.07 * (span_y or 1)
               for px, py in placed):
            left_out.append(name)
            continue
        ax.annotate(name, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.5,
                    color=INK if name == strong else SECONDARY, annotation_clip=False)
        placed.append((x, y))
    return left_out


def training_loss(losses: dict) -> str | None:
    """how the network decoders learned: each one's training loss after every epoch (the mean cross-entropy,
    in nats, over the training runs), on a log scale so the late epochs show as well as the first.
    losses: decoder -> its losses, as in metrics.json. The forest has no epochs and is left out; None if
    no decoder has any"""
    curves = {name: np.maximum(values, 1e-12) for name, values in losses.items() if values}
    if not curves:
        return None
    with rc_context(STYLE):
        figure = Figure(figsize=(WIDTH, 2.6), layout="constrained")
        ax = figure.subplots()
        for name, values in curves.items():
            ax.plot(np.arange(1, len(values) + 1), values, color=decoder_color(name), label=DECODERS[name])
        last = max(len(values) for values in curves.values())
        name_ends(ax, {DECODERS[name]: (len(values), values[-1]) for name, values in curves.items()}, log=True)
        if len(curves) > 1:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(curves), handlelength=1.5)
        ax.set_yscale("log")
        ax.set_xlim(1, last)
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss (nats)")
        ax.grid(axis="x", visible=False)
        plain_ticks(ax)
        return svg(figure)


def confusion_matrices(matrices: dict) -> str:
    """what each decoder decided, against what was sent: one panel per decoder (matrices: decoder -> its
    confusion matrix, rows the symbol sent, columns the symbol decided). A cell is shaded by its share of the
    row, from light blue (none) to dark (all), so a dark diagonal means right decisions; up to 8 symbols the
    cell also shows its count of test runs. The panel title gives the share decided right"""
    cmap = LinearSegmentedColormap.from_list("share", RAMP)
    with rc_context(STYLE | {"axes.grid": False}):
        figure = Figure(figsize=(WIDTH, 2.9), layout="constrained")
        axes = figure.subplots(1, len(matrices), squeeze=False)[0]
        for ax, (name, matrix) in zip(axes, matrices.items()):
            counts = np.asarray(matrix)
            n = len(counts)
            shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
            image = ax.pcolormesh(shares, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1.5)
            if n <= 8:
                for i in range(n):
                    for j in range(n):
                        ax.text(j + 0.5, i + 0.5, str(counts[i, j]), ha="center", va="center", fontsize=7.5,
                                color="#ffffff" if shares[i, j] > 0.5 else INK)
            ax.set_aspect("equal")
            ax.set_xlim(0, n)
            ax.set_ylim(n, 0)                                       # symbol 0 at the top
            ticks = np.arange(n) + 0.5
            ax.set_xticks(ticks, [str(j) for j in range(n)])
            ax.set_yticks(ticks, [str(i) for i in range(n)])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel("decided")
            ax.set_ylabel("sent")
            ax.set_title(f"{DECODERS[name]} · {np.trace(counts) / counts.sum():.1%} right", loc="left",
                         fontsize=8.5, color=INK)
        if max(len(matrix) for matrix in matrices.values()) > 8:   # no counts in the cells: a scale instead
            figure.colorbar(image, ax=list(axes), shrink=0.8, label="share of the row").outline.set_visible(False)
        return svg(figure)


def learning_curves(curves: dict, best: str | None = None, ylabel: str = "check score (bits)") -> str:
    """how the search climbed: each run's score on the check datasets after every step (curves: run -> its
    points, as saved in search.json, each with 'steps' and 'mean'). The lines are steps, since a codebook only
    changes when a move is kept. The best run (best) is blue and named first; the other runs are grey, named
    where their ends are apart, else together as 'other runs'. All runs start from one codebook, so they share
    the first point"""
    with rc_context(STYLE):
        figure = Figure(figsize=(WIDTH, 2.8), layout="constrained")
        ax = figure.subplots()
        ends = {}
        for label, points in curves.items():
            steps, means = [p["steps"] for p in points], [p["mean"] for p in points]
            strong = label == best
            ax.plot(steps, means, drawstyle="steps-post", color=SERIES[0] if strong else MUTED,
                    linewidth=1.5 if strong else 1.0, zorder=3 if strong else 2)
            ax.plot(steps[-1], means[-1], "o", markersize=5, color=SERIES[0] if strong else MUTED,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4 if strong else 2)
            ends[f"{label} · best found" if strong else label] = (steps[-1], means[-1])
        first = next(iter(curves.values()))[0]
        ax.annotate(f"start · {first['mean']:.2f}", (0, first["mean"]), xytext=(5, -9), textcoords="offset points",
                    fontsize=7.5, color=SECONDARY)
        ax.set_xlim(0, max(x for x, _ in ends.values()))
        ax.set_ylim(bottom=0)
        left_out = name_ends(ax, ends, strong=f"{best} · best found" if best in curves else None)
        if left_out:                                    # grey runs ending too close together: one name for them
            x, y = max(ends[name][0] for name in left_out), min(ends[name][1] for name in left_out)
            ax.annotate("other runs" if len(left_out) > 1 else left_out[0], (x, y), xytext=(5, -10),
                        textcoords="offset points", va="center", fontsize=7.5, color=SECONDARY,
                        annotation_clip=False)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        plain_ticks(ax)
        return svg(figure)


def compare_signals(cfg: Config, before: np.ndarray, after: np.ndarray,
                    titles: tuple[str, str] = ("start", "found")) -> str:
    """the mean received signal of two codebooks of one experiment side by side (before and after: their
    datasets), on the same axes: the molecules absorbed so far, averaged over each symbol's runs. Curves that
    lie further apart are easier for a decoder to tell apart"""
    signals = [mean_signals(rows).cumsum(axis=1) for rows in (before, after)]
    n = len(signals[0])
    times = np.arange(signals[0].shape[1]) * cfg.channel.dt
    with rc_context(STYLE):
        figure = Figure(figsize=(WIDTH, 2.9), layout="constrained")
        figure.get_layout_engine().set(wspace=0.06)
        axes = figure.subplots(1, 2, sharey=True)
        for ax, curves, title in zip(axes, signals, titles):
            for u, curve in enumerate(curves):
                ax.plot(times, curve, color=symbol_color(u, n), label=f"symbol {u}")
            ax.set_title(title, loc="left", fontsize=8.5, color=INK)
            ax.set_xlim(0, times[-1])
            ax.set_ylim(bottom=0)
            ax.set_xlabel("time (s)")
            ax.grid(axis="x", visible=False)
            plain_ticks(ax)
        axes[0].set_ylabel("absorbed so far (mean)")
        figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=min(n, 8),
                      handlelength=1.5)
        return svg(figure)
