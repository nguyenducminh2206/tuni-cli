"""figures: the report's pictures, drawn with matplotlib and returned as SVG text for report.html.
Every figure has one quiet look: a light chart surface, hairline grids, thin lines and bars, text in grey
inks, and one colour per symbol that follows the symbol from figure to figure.
Figures never print and never write files: the report puts them in the page"""

import io

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

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


def svg(figure: Figure) -> str:
    """the figure as SVG text, without the XML header, ready to sit in an HTML page. Ticks are plain numbers
    (0, 0.5, 1), at most 6 on an axis"""
    for ax in figure.axes:
        ax.xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    out = io.StringIO()
    figure.savefig(out, format="svg", metadata={"Date": None, "Creator": None})
    text = out.getvalue()
    return text[text.index("<svg"):].replace("'DejaVu Sans'", FONT)


def release_schedules(cfg: Config) -> str:
    """the codebook as a picture: one row per symbol, a bar for each release at its time (slot k at
    k × slot_dt), as high as the molecules it releases. The time axis runs to T, the end of a run,
    so the picture shows how much of the run each symbol leaves for the molecules to arrive"""
    encoder, T = cfg.encoder, cfg.channel.T
    n = encoder.n_symbols
    width = min(0.6 * encoder.slot_dt, T / 24)      # thin bars, at most about 24 px wide
    with rc_context(STYLE):
        figure = Figure(figsize=(WIDTH, 0.5 + 0.55 * n), layout="constrained")
        axes = figure.subplots(n, 1, sharex=True, sharey=True, squeeze=False)[:, 0]
        for u, (ax, vector) in enumerate(zip(axes, encoder.symbols)):
            slots = [k for k, amount in enumerate(vector) if amount]
            ax.bar([k * encoder.slot_dt for k in slots], [vector[k] for k in slots], width=width,
                   color=symbol_color(u, n), linewidth=0)
            if len(slots) <= 3:                     # a few bursts: their sizes, at the bar tips
                for k in slots:
                    ax.annotate(str(vector[k]), (k * encoder.slot_dt, vector[k]), xytext=(0, 2),
                                textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
                                color=SECONDARY)
            ax.set_ylabel(f"symbol {u}", rotation=0, ha="right", va="center", color=INK)
            ax.set_ylim(0, encoder.budget * 1.3)
            ax.set_yticks([0, encoder.budget])
            ax.grid(axis="x", visible=False)
        axes[-1].set_xlim(-width, T)
        axes[-1].set_xlabel("release time (s)")
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
            name_ends(ax, times, signals)
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
        figure.supxlabel("time (s)", fontsize=9, color=SECONDARY)
        figure.supylabel(f"absorbed so far (mean of {len(rows) // n} runs)", fontsize=9, color=SECONDARY)
        return svg(figure)


def name_ends(ax, times: np.ndarray, signals: np.ndarray) -> None:
    """'symbol u' just right of each curve's end, so a symbol is named on the plot as well as in the legend;
    a name that would overlap another one is left to the legend"""
    placed = []
    for u in np.argsort(-signals[:, -1]):
        y = signals[u, -1]
        if any(abs(y - other) < 0.07 * signals.max() for other in placed):
            continue
        ax.annotate(f"symbol {u}", (times[-1], y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=SECONDARY, annotation_clip=False)
        placed.append(y)
