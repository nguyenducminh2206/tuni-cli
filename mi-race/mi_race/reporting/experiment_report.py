"""Experiment report generator (`mi-race report`).

Produces an HTML report in the style of the reference
``Optimal_Encoder_for_Noisy_Channel`` project: per-symbol *clean messages*
(the release schedules), per-symbol *averaged received* signals (mean SSA
trajectory per compartment), and a confusion matrix with MI / accuracy.

The report is structured around one or more :class:`ExperimentResult` objects.
Right now only the **baseline** (hand-picked codebook) is produced; once the
learned encoder (Phase C) exists it returns a second result and the same
renderer lays them out side by side as before/after.

Unlike ``outputs/{model}/`` (overwritten every ``mi-race run``), the report
writes to its own ``experiments/<name>/`` bundle so results are not clobbered.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from ..analysis import info_from_confusion_matrix
from ..encoder.codebook import codebook_from_config
from ..train.data_prep import load_df_from_cfg, build_features_from_config
from ..train.registry import MODEL_REGISTRY, SUPPORTED_MODELS


_GROUP_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")


def _normalize_labels(y) -> np.ndarray:
    """Discrete 1D labels for sklearn metrics (mirrors orchestrator helper)."""
    arr = np.asarray(y)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.dtype.kind == "f":
        finite = np.isfinite(arr)
        if finite.all() and np.allclose(arr, np.round(arr)):
            return np.round(arr).astype(int)
        return arr.astype(str)
    return arr


@dataclass
class ExperimentResult:
    """One encoder configuration evaluated end-to-end through the channel."""

    label: str                      # e.g. "Baseline (hand-picked)" / "Optimized"
    accuracy: float
    mi_bits: float
    nmi_sqrt: float
    confusion: np.ndarray
    class_labels: list
    times: np.ndarray               # time axis for received signals
    avg_received: dict              # {comp_prefix: ndarray (n_symbols, n_steps)}
    codebook: dict = field(default_factory=dict)  # {sid: [(t, a), ...]}


# ---------------------------------------------------------------------------
# Running the experiment
# ---------------------------------------------------------------------------
def _detect_groups(columns) -> dict[str, list[str]]:
    """Group ``<prefix>_<idx>`` columns by prefix, each sorted by index."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for c in columns:
        m = _GROUP_RE.match(str(c))
        if not m:
            continue
        groups.setdefault(m.group("prefix"), []).append((int(m.group("idx")), c))
    return {p: [name for _i, name in sorted(items)] for p, items in groups.items()}


def _avg_received_by_symbol(df: pd.DataFrame, y_col: str, cfg: dict):
    """Mean trajectory per symbol per compartment group.

    Returns (avg_received, times) where avg_received maps each compartment
    prefix to an array of shape (n_symbols, n_steps).
    """
    groups = _detect_groups([c for c in df.columns if c != y_col])
    symbols = sorted(pd.Series(df[y_col]).dropna().unique().tolist())

    avg: dict[str, np.ndarray] = {}
    n_steps = 0
    for prefix, cols in groups.items():
        mat = np.zeros((len(symbols), len(cols)), dtype=float)
        for i, sid in enumerate(symbols):
            mat[i] = df.loc[df[y_col] == sid, cols].to_numpy(dtype=float).mean(axis=0)
        avg[prefix] = mat
        n_steps = len(cols)

    channel = cfg.get("channel", {})
    T = float(channel.get("T", 1.0))
    times = np.linspace(0.0, T, n_steps) if n_steps else np.arange(0)
    return avg, times


def run_baseline_experiment(cfg: dict, model_name: str) -> ExperimentResult:
    """Train the decoder once on the configured dataset and collect metrics."""
    data_cfg = cfg["data"]
    df = load_df_from_cfg(data_cfg)

    y_col = data_cfg.get("y_col")
    if not y_col or y_col not in df.columns:
        raise SystemExit(f"[mi-race] report: y_col '{y_col}' not in dataset columns.")

    y = df[y_col].to_numpy()
    y_disc = _normalize_labels(y)

    spec = MODEL_REGISTRY[model_name]
    feature_df = None
    if spec.needs_features:
        feature_df, _ = build_features_from_config(df, cfg)

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {}).get(model_name, {})
    standardize = bool(train_cfg.get("standardize", True))
    random_state = int(train_cfg.get("random_state", 42))
    stratify = y_disc if train_cfg.get("stratify", True) else None
    counts = pd.Series(y).value_counts().sort_index()

    y_test, y_pred, _proba = spec.adapter(
        df, feature_df, y, y_disc, train_cfg, model_cfg,
        standardize, random_state, stratify, counts, quiet=True,
    )

    y_test_m = _normalize_labels(y_test)
    y_pred_m = _normalize_labels(y_pred)
    labels = sorted(pd.Series(_normalize_labels(y)).dropna().unique().tolist())
    cm = confusion_matrix(y_test_m, y_pred_m, labels=labels)
    info = info_from_confusion_matrix(cm, labels=labels)
    acc = float(accuracy_score(y_test_m, y_pred_m))

    avg, times = _avg_received_by_symbol(df, y_col, cfg)

    codebook: dict = {}
    if "channel" in cfg and isinstance(cfg["channel"], dict) and cfg["channel"].get("symbols"):
        codebook = codebook_from_config(cfg["channel"])

    return ExperimentResult(
        label="Baseline (hand-picked)",
        accuracy=acc,
        mi_bits=float(info["I"]),
        nmi_sqrt=float(info["NMI_sqrt"]),
        confusion=cm,
        class_labels=labels,
        times=times,
        avg_received=avg,
        codebook=codebook,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Shared matplotlib styling for a clean, consistent look across figures.
_PLOT_STYLE = {
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": "#bbbbbb",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
_ACCENT = "#2c6fb5"
_PALETTE = ["#2c6fb5", "#e8833a", "#3ba776", "#c0504d",
            "#8064a2", "#4bacc6", "#a0a0a0", "#d4b106"]


def _fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _grid(n: int):
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    return rows, cols


def plot_clean_messages(result: ExperimentResult, T: float) -> Optional[str]:
    """Per-symbol release schedule (the 'clean message'). None if no codebook."""
    if not result.codebook:
        return None
    import matplotlib.pyplot as plt

    sids = sorted(result.codebook.keys())
    rows, cols = _grid(len(sids))
    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.1 * rows), squeeze=False)
        flat = axes.flatten()
        for ax, sid in zip(flat, sids):
            pulses = result.codebook[sid]
            if pulses:
                ts = [t for t, _a in pulses]
                amps = [a for _t, a in pulses]
                marker, stem, base = ax.stem(ts, amps, basefmt=" ")
                stem.set_color(_ACCENT)
                stem.set_linewidth(2.0)
                marker.set_color(_ACCENT)
                marker.set_markersize(6)
            ax.set_xlim(-0.02 * T, T * 1.02)
            ax.set_ylim(bottom=0)
            ax.set_title(f"Symbol {sid}")
            ax.set_xlabel("Release time [s]")
            ax.set_ylabel("Amount")
        for ax in flat[len(sids):]:
            ax.axis("off")
        fig.tight_layout()
        return _fig_to_b64(fig)


def plot_avg_received(result: ExperimentResult) -> Optional[str]:
    """Per-symbol mean trajectory; one line per compartment group."""
    if not result.avg_received:
        return None
    import matplotlib.pyplot as plt

    n_symbols = len(result.class_labels)
    rows, cols = _grid(n_symbols)
    prefixes = sorted(result.avg_received.keys())
    t = result.times
    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.1 * rows), squeeze=False)
        flat = axes.flatten()
        for i, ax in enumerate(flat):
            if i >= n_symbols:
                ax.axis("off")
                continue
            for j, prefix in enumerate(prefixes):
                mat = result.avg_received[prefix]
                ax.plot(t, mat[i], label=prefix, linewidth=1.6,
                        color=_PALETTE[j % len(_PALETTE)])
            ax.set_title(f"Symbol {result.class_labels[i]}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Mean count")
            if len(prefixes) > 1 and i == 0:
                ax.legend(fontsize=7, ncol=2, frameon=False)
        fig.tight_layout()
        return _fig_to_b64(fig)


def plot_confusion(result: ExperimentResult) -> str:
    import matplotlib.pyplot as plt
    import seaborn as sns

    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            result.confusion, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=result.class_labels, yticklabels=result.class_labels,
            cbar=True, annot_kws={"size": 8},
        )
        ax.grid(False)  # heatmap cells shouldn't show the shared axes grid
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------
_CSS = """
:root {
  --accent: #2c6fb5; --ink: #1b2733; --muted: #6b7785;
  --line: #e3e8ee; --bg: #f5f7fa; --card: #ffffff;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--ink); background: var(--bg); margin: 0; padding: 0 1rem 4rem;
  line-height: 1.5;
}
.wrap { max-width: 1080px; margin: 0 auto; }
header.report {
  padding: 2.2rem 0 1.4rem; border-bottom: 1px solid var(--line); margin-bottom: 1.6rem;
}
header.report .eyebrow {
  text-transform: uppercase; letter-spacing: .12em; font-size: .72rem;
  color: var(--accent); font-weight: 700; margin: 0 0 .35rem;
}
header.report h1 { margin: 0; font-size: 1.9rem; letter-spacing: -.01em; }
header.report .gen { color: var(--muted); font-size: .82rem; margin-top: .5rem; }
.panel {
  background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 1.2rem 1.4rem; margin-bottom: 1.6rem;
}
.panel h2 {
  margin: 0 0 1rem; font-size: 1.05rem; letter-spacing: -.01em;
  display: flex; align-items: center; gap: .55rem;
}
.panel h2::before {
  content: ""; width: 4px; height: 1.05rem; background: var(--accent); border-radius: 2px;
}
.meta-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: .8rem 1.4rem;
}
.meta-item .k { display: block; font-size: .72rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: .06em; }
.meta-item .v { display: block; font-size: .98rem; font-weight: 600; margin-top: .1rem; }
.badge {
  display: inline-block; font-size: .75rem; font-weight: 700; color: #fff;
  background: var(--accent); padding: .25rem .7rem; border-radius: 999px;
}
.kpi-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.1rem 0 1.4rem; }
.kpi {
  background: linear-gradient(180deg, #fbfdff, #f1f6fb);
  border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.1rem; text-align: center;
}
.kpi .val { font-size: 1.7rem; font-weight: 700; color: var(--accent); letter-spacing: -.02em; }
.kpi .lab { font-size: .74rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: .06em; margin-top: .2rem; }
figure.fig { margin: 1.4rem 0 0; }
figure.fig figcaption {
  font-size: .82rem; color: var(--muted); margin-bottom: .5rem; font-weight: 600;
}
figure.fig img {
  max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff;
}
table.cmp { width: 100%; border-collapse: collapse; font-size: .92rem; }
table.cmp th, table.cmp td { padding: .55rem .7rem; text-align: right; border-bottom: 1px solid var(--line); }
table.cmp th:first-child, table.cmp td:first-child { text-align: left; color: var(--muted); }
table.cmp thead th { font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
table.cmp .delta-pos { color: #2e7d4f; font-weight: 600; }
table.cmp .delta-neg { color: #c0392b; font-weight: 600; }
footer.report { color: var(--muted); font-size: .78rem; text-align: center;
  padding-top: 1.4rem; border-top: 1px solid var(--line); }
"""


def _kpi_row(result: ExperimentResult) -> str:
    cells = [
        (f"{result.accuracy * 100:.2f}%", "Accuracy"),
        (f"{result.mi_bits:.3f}", "MI (bits)"),
        (f"{result.nmi_sqrt:.3f}", "NMI&#8730;"),
    ]
    items = "".join(
        f'<div class="kpi"><div class="val">{v}</div><div class="lab">{l}</div></div>'
        for v, l in cells
    )
    return f'<div class="kpi-row">{items}</div>'


def _figure_block(b64: Optional[str], caption: str) -> str:
    if not b64:
        return ""
    return (
        f'<figure class="fig"><figcaption>{caption}</figcaption>'
        f'<img src="data:image/png;base64,{b64}" /></figure>'
    )


def _result_section(result: ExperimentResult, T: float) -> str:
    clean = plot_clean_messages(result, T)
    recv = plot_avg_received(result)
    cm = plot_confusion(result)
    n = len(result.class_labels)
    return f"""
  <section class="panel">
    <h2><span class="badge">{result.label}</span></h2>
    {_kpi_row(result)}
    {_figure_block(clean, "Clean messages — release schedule (encoder output, before the channel)")}
    {_figure_block(recv, "Averaged received messages — mean SSA trajectory per symbol")}
    {_figure_block(cm, f"Confusion matrix — {n}&times;{n}, decoder predictions on held-out test set")}
  </section>"""


def _comparison_panel(results: list[ExperimentResult]) -> str:
    """A before/after delta table — only shown when 2+ results are present."""
    if len(results) < 2:
        return ""
    headers = "".join(f"<th>{r.label}</th>" for r in results) + "<th>&Delta;</th>"

    def row(name: str, vals: list[float], pct: bool = False, hi_good: bool = True) -> str:
        fmt = (lambda x: f"{x * 100:.2f}%") if pct else (lambda x: f"{x:.3f}")
        delta = vals[-1] - vals[0]
        cls = "delta-pos" if (delta >= 0) == hi_good else "delta-neg"
        dtxt = (f"+{delta * 100:.2f}%" if pct else f"{delta:+.3f}")
        cells = "".join(f"<td>{fmt(v)}</td>" for v in vals)
        return f'<tr><td>{name}</td>{cells}<td class="{cls}">{dtxt}</td></tr>'

    body = (
        row("Accuracy", [r.accuracy for r in results], pct=True)
        + row("MI (bits)", [r.mi_bits for r in results])
        + row("NMI&#8730;", [r.nmi_sqrt for r in results])
    )
    return f"""
  <section class="panel">
    <h2>Before / after</h2>
    <table class="cmp"><thead><tr><th>Metric</th>{headers}</tr></thead>
    <tbody>{body}</tbody></table>
  </section>"""


def render_html(
    results: list[ExperimentResult],
    cfg: dict,
    out_html: Path,
    title: str,
) -> Path:
    channel = cfg.get("channel", {})
    T = float(channel.get("T", 1.0))
    n_classes = len(results[0].class_labels) if results else 0

    # Configuration panel items.
    meta_items: list[tuple[str, str]] = [("Symbols (N)", str(n_classes))]
    if channel:
        for key, label in [
            ("L", "Compartments (L)"), ("T", "Duration T [s]"), ("dt", "Timestep dt [s]"),
            ("D", "Diffusion D"), ("observed_compartment", "Observed compartment"),
            ("runs_per_symbol", "Runs / symbol"),
        ]:
            if key in channel:
                meta_items.append((label, str(channel[key])))
    meta_html = "".join(
        f'<div class="meta-item"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in meta_items
    )

    sections = "\n".join(_result_section(r, T) for r in results)
    comparison = _comparison_panel(results)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment Report — {title}</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header class="report">
      <p class="eyebrow">mi-race · molecular communication</p>
      <h1>{title}</h1>
      <div class="gen">Generated {generated}</div>
    </header>

    <section class="panel">
      <h2>Configuration</h2>
      <div class="meta-grid">{meta_html}</div>
    </section>
{comparison}
{sections}

    <footer class="report">Generated by <strong>mi-race</strong> · encoder &rarr; SSA channel &rarr; decoder</footer>
  </div>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


def _save_bundle(results: list[ExperimentResult], cfg: dict, out_dir: Path) -> None:
    """Persist scalar metrics + confusion matrices + codebook as JSON."""
    payload = {
        "channel": cfg.get("channel", {}),
        "results": [
            {
                "label": r.label,
                "accuracy": r.accuracy,
                "mi_bits": r.mi_bits,
                "nmi_sqrt": r.nmi_sqrt,
                "class_labels": [str(c) for c in r.class_labels],
                "confusion": r.confusion.tolist(),
                "codebook": {str(k): v for k, v in r.codebook.items()},
            }
            for r in results
        ],
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _open_in_browser(path: Path) -> None:
    """Open an HTML file in the default browser; print the path on failure."""
    import webbrowser

    uri = path.resolve().as_uri()
    try:
        if webbrowser.open(uri):
            print(f"[mi-race] report: opened {path} in your browser")
            return
    except Exception:
        pass
    print(f"[mi-race] report: open it manually → {uri}")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def run_report(args) -> None:
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"[mi-race] Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "data" not in cfg:
        raise SystemExit("[mi-race] report: config is missing a 'data' section.")

    model_name = getattr(args, "model", None) or "cnn"
    if model_name not in SUPPORTED_MODELS:
        raise SystemExit(
            f"[mi-race] report: unknown --model '{model_name}'. "
            f"Use one of: {', '.join(SUPPORTED_MODELS)}."
        )

    name = getattr(args, "name", None) or cfg_path.stem
    out_dir = Path(getattr(args, "out", None) or (Path("experiments") / name))

    print(f"[mi-race] report: training {model_name} decoder for baseline …")
    baseline = run_baseline_experiment(cfg, model_name)
    print(
        f"[mi-race] baseline  acc={baseline.accuracy * 100:.2f}%  "
        f"MI={baseline.mi_bits:.4f} bits  NMI_sqrt={baseline.nmi_sqrt:.4f}"
    )

    results = [baseline]
    out_html = out_dir / "report.html"
    render_html(results, cfg, out_html, title=name)
    _save_bundle(results, cfg, out_dir)
    print(f"[mi-race] report: wrote {out_html}")
    print(f"[mi-race] report: wrote {out_dir / 'result.json'}")

    if getattr(args, "open", False):
        _open_in_browser(out_html)
