"""report: one self-contained HTML page with everything an experiment has produced, in five tabs: Results (the
final evaluation), Codebooks (before and after the search, comparisons), Search, Data & models (the setup, the
dataset, the decoders) and the Decision log. It reads the result files only (metrics.json, search.json,
compare.json, final.json, decisions.json) and never simulates or trains: a part with no results yet says which
command makes it.
Python writes the text and the tables. The page's script (templates/report.js) draws the charts from the
numbers embedded in the page and runs the tabs and switches; the styles are templates/report.css. Both are
written into the page, so the one file works offline"""

import html
import json
import math
import warnings
from datetime import datetime
from importlib import resources
from pathlib import Path

import numpy as np

from mi_race import __version__
from mi_race.config import Config, ConfigError, load_config, parse_config, read_config_file
from mi_race.dataset import DatasetError, check_dataset, mean_signals, read_dataset
from mi_race.decisions import DecisionLogError, read_log

NAMES = {"random_forest": "Random forest", "mlp": "MLP", "cnn": "CNN"}
WORDS = {"random_forest": "random forest", "mlp": "MLP", "cnn": "CNN"}          # inside a sentence
SHORT = {"random_forest": "Forest", "mlp": "MLP", "cnn": "CNN"}                  # in a column header
SLOT = {"random_forest": "s0", "mlp": "s1", "cnn": "s2"}                         # a decoder's colour
TABS = [("results", "Results"), ("codebooks", "Codebooks"), ("search", "Search"), ("data", "Data & models"),
        ("log", "Decision log")]


# --- text and numbers -------------------------------------------------------------------------------

def text(value) -> str:
    """any value as safe HTML text"""
    return html.escape(str(value))


def signed(value: float, digits: int = 3, sign: bool = False) -> str:
    """a number with a real minus sign, and a plus when sign is asked for: '−0.012', '+0.378'"""
    return (f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}").replace("-", "−")


def mean_se(values) -> list[float]:
    """the mean of per-dataset values and its standard error"""
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return [round(float(values.mean()), 6), round(float(se), 6)]


def estimate(mean: float, se: float, strong: bool = False, sign: bool = False) -> str:
    """'1.834 ± 0.012', the mean bold when strong"""
    return (f'<span class="{"lab-strong" if strong else ""}">{signed(mean, sign=sign)}</span>'
            f'<span class="lab-se"> ± {se:.3f}</span>')


def command(words: str) -> str:
    """a mi-race command, as it is typed"""
    return f"<code>mi-race {text(words)}</code>"


def typed(vector) -> str:
    """a release vector as it is typed: '25:3 25:4'"""
    return " ".join(f"{amount}:{k}" for k, amount in enumerate(vector) if amount)


def codebook_cell(symbols) -> str:
    """a codebook on one line, as it is typed, in the page's code type: '50:0 · 50:3 · 25:5 25:6'"""
    return f'<span class="lab-codebook-cell">{" · ".join(typed(vector) for vector in symbols)}</span>'


def schedule_head(budget: int) -> str:
    return f'<div class="lab-figure-head"><span>Molecules released · common scale of 0–{budget}</span></div>'


def releases(vector, slot_dt: float) -> str:
    """a release vector as molecules @ time (s): '50 @ 0 · 20 @ 0.1'"""
    return " · ".join(f"{amount} @ {round(k * slot_dt, 6):g}" for k, amount in enumerate(vector) if amount)


def percent(share: float) -> str:
    """a share as a percentage, rounded half up as the page's script does: 0.8625 -> '86.3%'"""
    return f"{math.floor(share * 1000 + 0.5) / 10:.1f}%"


def capital(label: str) -> str:
    """'run 2' -> 'Run 2', at the start of a sentence"""
    return label[:1].upper() + label[1:]


def count(n: int, thing: str) -> str:
    """'1 session', '3 sessions'"""
    return f"{n:,} {thing}{'' if n == 1 else 's'}"


def listing(items: list) -> str:
    """'1', '1 and 2', '1, 2 and 3'"""
    items = [str(item) for item in items]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def when(stamp, layout: str = "%d %b %Y, %H:%M") -> str:
    """a saved time, readable: '2026-11-20T02:30:07' -> '20 Nov 2026, 02:30'"""
    try:
        return datetime.fromisoformat(stamp).strftime(layout)
    except (TypeError, ValueError):
        return text(stamp or "")


def read_json(path: Path) -> dict | None:
    """a result file, or None if there is none or it can't be read"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def asset(name: str) -> str:
    """a file shipped with the report: templates/report.css, templates/report.js"""
    return (resources.files("mi_race") / "templates" / name).read_text(encoding="utf-8")


# --- pieces of the page, in the design's classes ------------------------------------------------------

def note(message: str, warn: bool = False) -> str:
    """a missing result and the command that makes it, or something to be careful about (warn)"""
    return f'<p class="lab-note{" lab-warn" if warn else ""}">{message}</p>'


def section(title: str, body: str, aside: str = "", support: str = "", heading: str = "h2") -> str:
    """a titled block: the title with an aside on its right, a supporting line, then the body"""
    head = f'<div class="lab-section-head"><{heading}>{text(title)}</{heading}>{aside}</div>'
    line = f'<p class="lab-support">{support}</p>' if support else ""
    return f'<section class="lab-section">{head}{line}{body}</section>'


def aside(words: str) -> str:
    return f'<span class="lab-aside-label">{words}</span>'


def table(header: list[str], bodies, numeric=frozenset(), cls: str = "", row_class=None) -> str:
    """a table: header texts and one or more bodies of rows (rows of HTML cells, or (attributes, rows) pairs, so a
    switch can show one body at a time). numeric: the columns to right-align; row_class(i): a row's class"""
    if bodies and not isinstance(bodies[0], tuple):
        bodies = [("", bodies)]

    def attribute(name: str, value: str) -> str:
        return f' {name}="{value}"' if value else ""

    def cell(tag: str, value: str, j: int) -> str:
        classes = " ".join(c for c in ("lab-num" if j in numeric else "",
                                       "lab-table-name" if j == 0 and tag == "td" else "") if c)
        return f"<{tag}{attribute('scope', 'col' if tag == 'th' else '')}{attribute('class', classes)}>{value}</{tag}>"

    head = "".join(cell("th", text(h), j) for j, h in enumerate(header))
    parts = []
    for attributes, rows in bodies:
        lines = [f"<tr{attribute('class', row_class(i) if row_class else '')}>"
                 + "".join(cell("td", value, j) for j, value in enumerate(row)) + "</tr>" for i, row in enumerate(rows)]
        parts.append(f"<tbody{attributes}>{''.join(lines)}</tbody>")
    return (f'<div class="lab-table-scroll"><table class="{cls or "lab-plain"}"><thead><tr>{head}</tr></thead>'
            f'{"".join(parts)}</table></div>')


def facts(groups: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """titled groups of label-value lines, two columns of groups (the values already HTML)"""
    return '<div class="lab-facts">' + "".join(
        f'<div><h3>{text(title)}</h3><dl>' + "".join(f"<div><dt>{text(k)}</dt><dd>{v}</dd></div>" for k, v in pairs)
        + "</dl></div>" for title, pairs in groups) + "</div>"


def details(summary: str, body: str, open_: bool = False) -> str:
    return f'<details{" open" if open_ else ""}><summary>{text(summary)}</summary>{body}</details>'


def plot(key: str, label: str) -> str:
    """the place of a chart: the page's script draws data['charts'][key] into it"""
    return f'<div class="lab-plot" data-chart="{key}" aria-label="{text(label)}"></div>'


def figure(key: str, label: str, head: str, caption: str = "", legend: str = "") -> str:
    """a chart with its head line (what is plotted, and a legend) and a caption under it"""
    return (f'<figure><div class="lab-figure-head"><span>{head}</span>{legend}</div>{plot(key, label)}'
            + (f'<figcaption class="lab-caption">{caption}</figcaption>' if caption else "") + "</figure>")


def legend(items: list[tuple[str, str]]) -> str:
    """a legend: (kind, label) with kind 'dot' (the reference), 'square' (the codebook against it) or a line
    colour ('s0' … 's7', 'blue', 'mark')"""
    keys = []
    for kind, label in items:
        mark = (f'<span class="lab-{kind}" aria-hidden="true"></span>' if kind in ("dot", "square") else
                f'<span class="lab-swatch" style="background:var(--lab-{kind})" aria-hidden="true"></span>')
        keys.append(f'<span class="lab-key">{mark}{text(label)}</span>')
    return f'<div class="lab-legend">{"".join(keys)}</div>'


def matrix_block(key: str, title: str, support: str, note_text: str, options: list[tuple[str, str]],
                 label: str = "Codebook") -> str:
    """confusion matrices drawn by the page's script from data['matrices'][key]: a switch between counts and row
    percentages, and a choice between options (codebooks, pipelines) when there is more than one"""
    choose = (f'<label class="lab-select-label">{text(label)}<select data-select-for="{key}">'
              + "".join(f'<option value="{text(v)}">{text(name)}</option>' for v, name in options)
              + "</select></label>") if len(options) > 1 else ""
    switch = (f'<div class="lab-switch" role="group" aria-label="Confusion matrix values" data-unit-for="{key}">'
              '<button type="button" data-lab-unit="count" aria-pressed="true">Counts</button>'
              '<button type="button" data-lab-unit="percent" aria-pressed="false">Row %</button></div>')
    return (f'<section class="lab-section"><div class="lab-matrix-head"><div><h2>{text(title)}</h2>'
            f'<p class="lab-support">{support}</p></div><div class="lab-matrix-controls">{choose}{switch}</div></div>'
            f'<div class="lab-matrices" data-matrices="{key}" aria-live="polite"></div>'
            f'<p class="lab-matrix-note">{note_text}</p></section>')


def per_row_note(matrix) -> str:
    """'Rows are sent symbols … Each row contains 600 examples.' (the last sentence when every row has as many)"""
    rows = {int(sum(row)) for row in matrix}
    return ("Rows are sent symbols. Columns are predicted symbols."
            + (f" Each row contains {rows.pop():,} examples." if len(rows) == 1 else ""))


def line_series(points, label: str, tone: str, **more) -> dict:
    """one line of a chart, its points rounded to keep the page small"""
    return {"label": label, "tone": tone, "points": [[round(float(x), 4), round(float(y), 4)] for x, y in points],
            **more}


def symbol_tone(u: int, n: int) -> str:
    """a symbol's colour: its own while there are colours enough, else one for all"""
    return f"s{u}" if n <= 8 else "s0"


# --- what the result files hold ------------------------------------------------------------------------

def found_by(cfg: Config, config_path: Path) -> tuple[Path, Path] | None:
    """for a found config (configs/NAME-found.json, written by a search of configs/NAME.json): that config and
    its output folder, when the search there found this config's codebook as its best. None otherwise"""
    if not config_path.stem.endswith("-found"):
        return None
    source = config_path.with_name(f"{config_path.stem.removesuffix('-found')}.json")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                         # the source's own warnings are not this report's
            folder = Path(load_config(source).output.dir)
    except ConfigError:
        return None
    check = (read_json(folder / "search.json") or {}).get("check")
    if not check or check["codebooks"][check["best"]] != [list(vector) for vector in cfg.encoder.symbols]:
        return None
    return source, folder


def results_of(config_path: Path | None, symbols) -> tuple[np.ndarray | None, dict | None, list[str]]:
    """what a config holds for one codebook of a search: its dataset and its decoders' results (metrics.json),
    each None if missing, with the commands that would make them. A config whose codebook is no longer this
    one holds nothing for it"""
    if config_path is None or not config_path.exists():
        return None, None, ["no config holds this codebook"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            cfg = load_config(config_path)
        except ConfigError:
            return None, None, [f"{text(config_path)} can't be read"]
    if [list(vector) for vector in cfg.encoder.symbols] != [list(vector) for vector in symbols]:
        return None, None, [f"{text(config_path)} holds another codebook now"]
    missing, rows = [], None
    try:
        rows = read_dataset(cfg.data.path)
        check_dataset(rows, cfg)
    except DatasetError:
        rows = None
        missing.append(command(f"simulate {config_path}"))
    metrics = read_json(Path(cfg.output.dir) / "metrics.json")
    if metrics is None:
        missing.append(command(f"run {config_path}"))
    return rows, metrics and metrics["decoders"], missing


def signal_series(rows: np.ndarray) -> list[dict]:
    """each symbol's mean received signal, as the molecules absorbed so far (one line per symbol)"""
    signals = mean_signals(rows).cumsum(axis=1)
    n = len(signals)
    return [{"u": u, "tone": symbol_tone(u, n), "values": signal} for u, signal in enumerate(signals)]


# --- Results: the final evaluation --------------------------------------------------------------------

def results_tab(config_path: Path, final_folder: Path, check: dict | None, data: dict) -> tuple[str, str]:
    """the final evaluation (final.json), the result to report: the comparison plot and table, the confusion
    matrices pooled over the final datasets, and how it was measured. Returns the tab and a summary"""
    path = final_folder / "final.json"
    data["sources"]["results"] = str(path)
    final = read_json(path)
    if final is None:
        return section("Final evaluation", note(
            f"No final evaluation yet · after a search, {command(f'final {config_path}')}; for codebooks you "
            f"designed, {command(f'final {config_path} OTHER …')}. The final datasets are used once, for the "
            "result to report.")), f"no final evaluation yet · final {config_path}"
    by_hand = "reference" in final
    reference = final["reference"] if by_hand else "start"
    others = [label for label in final["codebooks"] if label != reference]
    shown = ({label: label for label in final["codebooks"]} if by_hand else
             {"start": "Starting", final["best"]: "Learned"})
    bits, datasets = final["ba_bound_bits"], final["datasets"]
    gains = final["gains"] if by_hand else {final["best"]: final["gains"]}
    decoders = list(bits[reference])
    many = len(others) > 1
    uniform = len({(gains[label][name]["higher"], gains[label][name]["datasets"]) for name in decoders
                   for label in others}) == 1
    specs, rows = [], []
    for name in decoders:
        for i, label in enumerate(others):
            ref, value, gain = mean_se(bits[reference][name]), mean_se(bits[label][name]), gains[label][name]
            specs.append({"group": NAMES[name], "label": shown[label] if many else "", "ref": ref, "value": value})
            change = estimate(gain["mean"], gain["se"], sign=True) + (
                "" if uniform else f'<span class="lab-se"> · {gain["higher"]}/{gain["datasets"]}</span>')
            rows.append([NAMES[name] if i == 0 else "", *([text(shown[label])] if many else []),
                         estimate(*ref), estimate(*value, strong=True), change])
    data["charts"]["final"] = {"type": "comparison", "reference": shown[reference], "other": shown[others[0]],
                               "rows": specs}
    header = (["Decoder", "Codebook", shown[reference], "Codebook BA", "Paired change"] if many else
              ["Decoder", shown[reference], shown[others[0]], "Paired change"])
    some = gains[others[0]][decoders[0]]
    higher_text = (f"Higher on {some['higher']}/{some['datasets']} datasets for each decoder"
                   + (" and codebook." if many else ".") if uniform else
                   "After each change, k/n: the final datasets where the codebook scores higher.")
    support = (f"{len(final['codebooks'])} codebooks designed by hand, against {text(reference)}." if by_hand else
               f"Starting codebook compared with the learned codebook from {text(final['best'])}.")
    parts = [section("Final evaluation", "".join([
        figure("final", "BA estimates of each codebook, per decoder, with one standard error",
               "BA estimate (bits) · higher is better",
               legend=legend([("dot", shown[reference]), ("square", shown[others[0]] if not many else
                                                          "codebooks against it")])),
        table(header, rows, numeric=set(range(1 + many, len(header))), cls="lab-result-table"),
        f'<div class="lab-table-foot"><span>Mean ± standard error, in bits. Error bars show ±1 SE.</span>'
        f"<span>{text(higher_text)}</span></div>",
    ]), aside=aside(f"{datasets} reserved datasets"), support=support)]
    if "scores" in final:                                       # the decisions, all final datasets together
        order = [*others, reference]
        panels = {label: [{"name": NAMES[name],
                           "matrix": np.sum([d[name]["confusion_matrix"] for d in final["scores"][label]],
                                            axis=0).tolist()} for name in decoders] for label in order}
        data["matrices"]["final"] = {"selected": others[0], "panels": panels}
        some = panels[others[0]][0]["matrix"]
        parts.append(matrix_block(
            "final", "Confusion matrices",
            f"Pooled over the {datasets} final datasets · {int(np.sum(some)):,} predictions per decoder",
            per_row_note(some), [(label, shown[label]) for label in order]))
    n_test = (final.get("scores") or {}).get(reference, [{}])[0].get(decoders[0], {}).get("n_test")
    chosen = (f"{text(capital(final['best']))} was selected on {check['datasets']} development-check datasets. These "
              f"{datasets} final datasets were reserved until after selection, and are used once."
              if not by_hand and check else
              f"The codebooks were named before the final datasets were used; the {datasets} final datasets are "
              "used once.")
    bullets = [chosen,
               (f"Each final dataset contributes {n_test} test predictions per decoder. " if n_test else "")
               + f"Confusion matrices pool the {datasets} datasets; BA estimates are averaged over datasets.",
               ("All decoders use temperature-rescaled probabilities. " if final.get("rescaled", True) else "")
               + "The change is paired within each final dataset.",
               "The BA quantity estimates a decoder-dependent lower bound on symbol information. It is not channel "
               "capacity.",
               f"Made {when(final.get('created'))} · {text(path)}"]
    parts.append(details("How these results were measured",
                         '<ul class="lab-notes">' + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"))
    summary = f"final evaluation · {count(len(final['codebooks']), 'codebook')} · {datasets} final datasets"
    return "".join(parts), summary


# --- Codebooks: before and after the search, comparisons ----------------------------------------------

def codebooks_tab(cfg: Config, config_path: Path, searched: Path, search_folder: Path,
                  data: dict) -> tuple[str, str]:
    """the codebooks: the search's start and best found one side by side (schedules, releases, signals and decoders
    from each one's own dataset and run), or this config's codebook when there is no search; then the codebooks
    compared on the check datasets (compare.json). Returns the tab and a summary"""
    record = read_json(search_folder / "search.json")
    check = (record or {}).get("check")
    compare_path = Path(cfg.output.dir) / "compare.json"
    data["sources"]["codebooks"] = (f"{search_folder / 'search.json'} · " if check else "") + str(compare_path)
    parts = []
    if check:
        best = check["best"]
        start, found = check["codebooks"]["start"], check["codebooks"][best]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            before = parse_config(record["config"])                 # the experiment as it was searched
        encoder, channel = before.encoder, before.channel
        changed = [u for u, (a, b) in enumerate(zip(start, found)) if list(a) != list(b)]
        said = (f"Symbol{'s' if len(changed) > 1 else ''} {listing(changed)} changed" if changed else
                "No symbol changed")
        data["charts"]["codebook"] = {"type": "codebook", "T": channel.T, "slot_dt": encoder.slot_dt,
                                      "budget": encoder.budget,
                                      "schedules": [{"label": "Start", "tone": "mark", "symbols": start},
                                                    {"label": "Learned", "tone": "blue", "symbols": found}]}
        rows = [[f"{u}" + ('<span class="lab-changed-note">changed</span>' if u in changed else ""),
                 f'<span class="lab-release-schedule">{releases(a, encoder.slot_dt)}</span>',
                 f'<span class="lab-release-schedule">{releases(b, encoder.slot_dt)}</span>']
                for u, (a, b) in enumerate(zip(start, found))]
        found_path = Path(record["found_config"]) if record.get("found_config") else None
        check_box = ('<label class="lab-check"><input type="checkbox" data-changes-for="codebook">'
                     "Changed symbols only</label>")
        vectors = json.dumps({"starting": start, "learned": found}, indent=1)
        parts.append(section("What changed in the encoder", "".join([
            '<figure class="lab-codebook-figure">'
            + schedule_head(encoder.budget)
            + plot("codebook", "Release schedules of the starting and learned codebooks")
            + f'<figcaption class="lab-caption">Release time in seconds. Both schedules use the same channel, '
              f'{encoder.n_slots} release slots and a {channel.T:g} s observation window.</figcaption></figure>',
            table(["Symbol", "Starting · molecules @ time (s)", "Learned · molecules @ time (s)"], rows,
                  cls="lab-codebook-table"),
            details("Exact release vectors", f"<pre>{text(vectors)}</pre>"),
        ]), aside=check_box, support=f"{said}. Every release vector still uses exactly {encoder.budget} molecules."
                                      + (f" The learned codebook is {text(found_path)}." if found_path else "")))
        rows_before, metrics_before, missing_before = results_of(searched, start)
        rows_after, metrics_after, missing_after = results_of(found_path, found)
        body = []
        if rows_before is not None and rows_after is not None:
            panels = []
            for title, rows_ in (("Start", rows_before), ("Learned", rows_after)):
                times = np.arange(rows_.shape[1] - 1) * channel.dt
                panels.append({"title": title, "series": [line_series(zip(times, s["values"]), f"symbol {s['u']}",
                                                                      s["tone"]) for s in signal_series(rows_)]})
            data["charts"]["signals"] = {"type": "lines", "height": 280, "panels": panels, "x_label": "time (s)",
                                         "y_label": "absorbed so far (mean)"}
            body.append(figure("signals", "Mean received signal of each codebook", "Mean received signal",
                               "Each codebook from its own dataset: the molecules absorbed so far, averaged over each "
                               "symbol's runs. Curves further apart are easier for a decoder to tell apart.",
                               legend([(symbol_tone(u, encoder.n_symbols), f"symbol {u}")
                                       for u in range(encoder.n_symbols)])))
        if metrics_before and metrics_after:
            rows = []
            for name, a in metrics_before.items():
                if name not in metrics_after:
                    continue
                b = metrics_after[name]
                before_ba, after_ba = a["rescaled"]["ba_bound_bits"], b["rescaled"]["ba_bound_bits"]
                rows.append([NAMES[name], f"{percent(a['plain']['accuracy'])} → {percent(b['plain']['accuracy'])}",
                             f"{signed(before_ba)} → {signed(after_ba)}",
                             f'<span class="lab-strong">{signed(after_ba - before_ba, sign=True)}</span>'])
            body.append("<h3>Decoders, start → learned</h3>"
                        + table(["Decoder", "Accuracy", "BA rescaled (bits)", "Change (bits)"], rows,
                                numeric={1, 2, 3}))
        missing = [*missing_before, *missing_after]
        if missing:
            body.append(note("For the signals and decoders of both codebooks: " + " · ".join(missing)))
        parts.append(section("Signals and decoders, before and after", "".join(body),
                             support=f"From each codebook's own dataset and run ({text(searched)}"
                                     + (f" and {text(found_path)}" if found_path else "") + "): one dataset each, a "
                                     "quick look. The final evaluation in Results is the result to report."))
        summary = f"start against {best} · {listing(changed) if changed else 'nothing'} changed"
    else:
        encoder, channel = cfg.encoder, cfg.channel
        data["charts"]["codebook"] = {"type": "codebook", "T": channel.T, "slot_dt": encoder.slot_dt,
                                      "budget": encoder.budget,
                                      "schedules": [{"label": "", "tone": "blue",
                                                     "symbols": [list(v) for v in encoder.symbols]}]}
        rows = [[str(u), f'<span class="lab-release-schedule">{releases(v, encoder.slot_dt)}</span>',
                 f'<span class="lab-codebook-cell">{typed(v)}</span>'] for u, v in enumerate(encoder.symbols)]
        parts.append(section("Codebook", "".join([
            '<figure class="lab-codebook-figure">'
            + schedule_head(encoder.budget) + plot("codebook", "Release schedule of each symbol") + "</figure>",
            table(["Symbol", "Molecules @ time (s)", "As typed"], rows, cls="lab-codebook-table"),
        ]), aside=aside("no search yet"),
            support=f"{encoder.n_symbols} symbols · {encoder.budget} molecules each · {encoder.n_slots} release "
                    f"slots, {encoder.slot_dt:g} s apart · T = {channel.T:g} s."))
        summary = "this config's codebook"
    comparison = read_json(compare_path)
    if comparison:
        reference = comparison["reference"]
        decoders = list(next(iter(comparison["ba_bound_bits"].values())))
        bits_rows = [[text(label) + (' <span class="lab-run-selected">reference</span>' if label == reference else ""),
                      codebook_cell(comparison["codebooks"][label]),
                      *[signed(float(np.mean(by[name]))) for name in decoders]]
                     for label, by in comparison["ba_bound_bits"].items()]
        gain_rows = [[text(label), *[estimate(g[name]["mean"], g[name]["se"], sign=True)
                                     + f'<span class="lab-se"> · {g[name]["higher"]}/{g[name]["datasets"]}</span>'
                                     for name in decoders]] for label, g in comparison["gains"].items()]
        parts.append(section("Compared on the check datasets", "".join([
            table(["Codebook", "As typed", *[f"{SHORT[name]} BA" for name in decoders]], bits_rows,
                  numeric=set(range(2, 2 + len(decoders)))),
            f"<h3>Paired change against {text(reference)}</h3>",
            table(["Codebook", *[NAMES[name] for name in decoders]], gain_rows,
                  numeric=set(range(1, 1 + len(decoders)))),
            f'<p class="lab-caption">Means over {comparison["datasets"]} check datasets, in bits; the change is '
            "mean ± standard error · datasets where it is higher. Development numbers, for choosing. Made "
            f"{when(comparison.get('created'))} · {text(compare_path)}</p>",
        ]), aside=aside(count(len(comparison["codebooks"]), "codebook")),
            support=f"Every rescaled decoder on the same {comparison['datasets']} check datasets, against "
                    f"{text(reference)}."))
        summary += " · 1 comparison"
    else:
        parts.append(note(f"No comparison · {command(f'compare {config_path} OTHER …')} compares this codebook with "
                          "those of other configs, on the check datasets."))
    return "".join(parts), summary


# --- Search ------------------------------------------------------------------------------------------

def search_tab(config_path: Path, search_folder: Path, searched: Path, data: dict) -> tuple[str, str]:
    """the guided search (search.json): the runs with their scores on the check datasets, the selected run, the
    settings and the noise behind the threshold, the learning curves and the best found codebook. searched is the
    config the search ran on: config_path, or for a found config the config of the search that found it.
    Returns the tab and a summary"""
    path = search_folder / "search.json"
    data["sources"]["search"] = str(path)
    record = read_json(path)
    if record is None:
        return section("Encoder search", note(f"No search yet · {command(f'search {config_path}')} looks for a "
                                              "better codebook, starting from this one.")), \
            f"no search yet · search {config_path}"
    settings, noise, runs = record["settings"], record.get("noise"), record["runs"]
    score = f"{'rescaled ' if settings['rescaled'] else ''}{WORDS[settings['decoder']]}"
    check = record.get("check")
    best = check["best"] if check else None
    decoders = list(next(iter(check["ba_bound_bits"].values()))) if check else []
    kept = [[row for row in run["history"] if row["accepted"]] for run in runs]
    notes = []
    if searched != config_path:
        notes.append(note(f"The codebook of {text(config_path)} was found by this search of {text(searched)}: the "
                          f"search, the final evaluation and the decision log here come from {text(search_folder)}."))
    if record.get("config") != read_config_file(searched):
        notes.append(note(f"{text(searched)} changed since the search · the search used the config saved in "
                          f"{text(path)}.", warn=True))
    if not check:
        notes.append(note(f"The search stopped before its check · {command(f'final {config_path}')} checks the found "
                          "codebooks, then offers the final evaluation."))

    def means(label: str) -> list[str]:
        return [signed(float(np.mean(check["ba_bound_bits"][label][name]))) for name in decoders] if check else []

    rows, labels = [], []
    if check:
        rows.append(["Start", "—", "—", "—", *means("start")])
        labels.append("start")
    for run, moves in zip(runs, kept):
        label = f"run {run['run'] + 1}"
        rows.append([f"Run {run['run'] + 1}" + ('<span class="lab-run-selected">selected</span>' if label == best
                                                else ""),
                     str(len(run["history"])), str(len(moves)), str(moves[-1]["step"] + 1) if moves else "—",
                     *(means(label) if check and label in check["ba_bound_bits"] else ["—"] * len(decoders))])
        labels.append(label)
    header = ["Run", "Steps", "Accepted", "Last accepted", *[f"{SHORT[name]} BA" for name in decoders]]
    body = [table(header, rows, numeric=set(range(1, len(header))), cls="lab-run-table",
                  row_class=lambda i: "lab-selected" if labels[i] == best else ""),
            f'<p class="lab-caption">' + (f"BA means from {check['datasets']} development-check datasets, in bits. "
                                          "The selected run is highlighted." if check else
                                          "Moves accepted by each run.") + "</p>",
            details("Codebook each run ended with", table(["Run", "Codebook, as typed"], [
                [f"Run {run['run'] + 1}", codebook_cell(run["final"])] for run in runs]))]
    parts = ["".join(notes), section("Encoder search", "".join(body),
                                    aside=aside(f"{count(len(runs), 'run')} · {score} feedback"),
                                    support=(f"{text(capital(best))} was chosen on the development checks. The final "
                                             "evaluation is reported separately." if check else
                                             f"{count(len(runs), 'run')} from the start codebook."))]
    if check:
        gains = check["gains"]
        parts.append(section(f"{capital(best)}, the best found, against the start", table(
            ["Decoder", "Paired change (bits)", "Higher on"],
            [[NAMES[name], estimate(g["mean"], g["se"], sign=True), f"{g['higher']} of {g['datasets']}"]
             for name, g in gains.items()], numeric={1, 2}),
            support=f"On the same {check['datasets']} check datasets, every decoder rescaled. Picked as the highest "
                    "mean over the decoders.", heading="h3"))
    extended = [f"Run {run['run'] + 1}: {len(run['history'])}" for run in runs
                if len(run["history"]) > settings["steps"]]
    groups = [("Moves", [("Feedback", f"{score} · 5-fold, a new dataset each step"),
                          ("Move size", f"{settings['chunk']} molecules"),
                          ("Acceptance threshold", f"{settings['threshold']:g} bits"),
                          ("Initial steps per run", str(settings["steps"])),
                          ("Extended runs", ", ".join(extended) + " steps" if extended else "none")]),
              ("Noise and files", [("Noise check", f"{len(noise['differences'])} paired datasets" if noise else "—"),
                                   ("Noise standard deviation", f"{noise['sd']:.4f} bits" if noise else "—"),
                                   ("Start codebook", codebook_cell(record["config"]["encoder"]["symbols"])),
                                   ("Found config", text(record.get("found_config") or "not written")),
                                   ("Made", when(record.get("created")))])]
    curves = all("curve" in run for run in runs)
    if curves:
        series = []
        for run in runs:
            label = f"run {run['run'] + 1}"
            chosen = label == best
            series.append(line_series([(p["steps"], p["mean"]) for p in run["curve"]],
                                      label + (" · best found" if chosen else ""), "blue" if chosen else "mark",
                                      step=True, marker=True, strong=chosen, width=2 if chosen else 1.25))
        data["charts"]["curves"] = {"type": "lines", "height": 270, "panels": [{"series": series}],
                                    "end_labels": True, "rest_label": "other runs", "x_label": "step",
                                    "y_label": "check score (bits)"}
        curve_part = figure("curves", "Learning curves of the runs", "Learning curves · check score (bits)",
                            f"Each run's score on the same check datasets after every step, with the search's own "
                            f"score ({score}, 5-fold), so the steps compare."
                            + (" The selected run is blue." if best else ""))
    else:
        curve_part = ('<p class="lab-caption">Learning curves were not saved for this search: a new search records '
                      "them.</p>")
    parts.append(section("Search settings", facts(groups) + curve_part, heading="h3"))
    summary = (f"{count(len(runs), 'run')} · {count(sum(map(len, kept)), 'move')} kept"
               + (" · checked" if check else " · no check")
               + (f" · from {searched}" if searched != config_path else ""))
    return "".join(parts), summary


# --- Data & models -----------------------------------------------------------------------------------

def data_tab(cfg: Config, config_path: Path, data: dict) -> tuple[str, str]:
    """the setup the config defines, its dataset (with the mean received signal) and its decoders as 'run' trained
    and scored them (metrics.json), plain or rescaled, with their confusion matrices and training loss.
    Returns the tab and a summary"""
    channel, encoder, settings, search = cfg.channel, cfg.encoder, cfg.data, cfg.search
    metrics_path = Path(cfg.output.dir) / "metrics.json"
    data["sources"]["data"] = f"{config_path} · {settings.path} · {metrics_path}"
    test_runs = round(settings.runs_per_symbol * settings.test_fraction) * encoder.n_symbols
    decoder_facts = []
    for name, model in cfg.decoders.items():
        if name == "random_forest":
            decoder_facts.append(("Random forest trees", str(model.n_estimators)))
        elif name == "mlp":
            decoder_facts += [("MLP hidden layers", " → ".join(map(str, model.hidden_layers))),
                              ("MLP epochs / batch / rate", f"{model.epochs} / {model.batch_size} / {model.lr:g}")]
        elif name == "cnn":
            channels = " → ".join(map(str, model.channels))
            decoder_facts += [("CNN channels / kernel", f"{channels} / {model.kernel_size}"),
                              ("CNN epochs / batch / rate", f"{model.epochs} / {model.batch_size} / {model.lr:g}")]
    parts = [section("Experiment setup", facts([
        ("Channel", [("Simulation", "SSA · absorbing receiver"),
                     ("Channel length L", f"{channel.L} ({channel.L - 1} compartments, then the receiver)"),
                     ("Compartment width", f"{channel.h:g} µm"), ("Diffusion coefficient", f"{channel.D:g} µm²/s"),
                     ("Jump rate λ", f"{channel.jump_rate:g} per second"),
                     ("Duration / snapshot interval", f"{channel.T:g} s / {channel.dt:g} s")]),
        ("Data & encoder", [("Symbols / budget", f"{encoder.n_symbols} / {encoder.budget} molecules"),
                            ("Release slots / spacing", f"{encoder.n_slots} / {encoder.slot_dt:g} s"),
                            ("Runs per symbol", str(settings.runs_per_symbol)),
                            ("Test runs", f"{test_runs} ({settings.test_fraction:g} of each symbol's)"),
                            ("Receiver values per run", str(channel.n_intervals + 1)),
                            ("Master seed", str(cfg.seed))]),
        ("Decoders", decoder_facts or [("Decoders", "none")]),
        ("Search", [("Feedback", f"{'rescaled ' if search.rescaled else ''}{WORDS[search.decoder]}"),
                    ("Move size / threshold", f"{search.chunk} molecules / {search.threshold:g} bits"),
                    ("Steps / runs", f"{search.steps} / {search.runs}")]),
    ]), support="Fixed channel and molecule budget; the encoder search changes only the release schedules.")]
    # the dataset
    path = Path(settings.path)
    summary = []
    if not path.exists():
        parts.append(section("Dataset", note(f"No dataset yet · {command(f'simulate {config_path}')} makes it.")))
        summary.append("no dataset")
    else:
        try:
            rows = read_dataset(path)
            check_dataset(rows, cfg)
        except DatasetError as problem:
            rows = None
            parts.append(section("Dataset", note(text(problem), warn=True)))
            summary.append("dataset doesn't fit")
        if rows is not None:
            absorbed = rows[:, 1:].sum(axis=1)
            stale = config_path.stat().st_mtime > path.stat().st_mtime
            means = [absorbed[rows[:, 0] == u].mean() for u in range(encoder.n_symbols)]
            table_rows = [[str(u), f"{mean:.1f}", f"{mean / encoder.budget:.0%}"] for u, mean in enumerate(means)]
            times = np.arange(rows.shape[1] - 1) * channel.dt
            series = signal_series(rows)
            data["charts"]["signal"] = {"type": "lines", "height": 270, "end_labels": encoder.n_symbols <= 4,
                                        "panels": [{"series": [line_series(zip(times, s["values"]), f"symbol {s['u']}",
                                                                           s["tone"]) for s in series]}],
                                        "x_label": "time (s)", "y_label": "absorbed so far (mean)"}
            parts.append(section("Dataset", "".join([
                note(f"{text(config_path)} was changed after this dataset was simulated · "
                     f"{command(f'simulate {config_path}')} makes it again.", warn=True) if stale else "",
                figure("signal", "Mean received signal of each symbol", "Mean received signal",
                       "The molecules absorbed so far, averaged over each symbol's runs. A later release shifts a "
                       "curve right; its end is how many molecules arrive before T.",
                       legend([(symbol_tone(u, encoder.n_symbols), f"symbol {u}") for u in range(encoder.n_symbols)])),
                "<h3>Receiver counts</h3>",
                table(["Symbol", f"Mean absorbed by {channel.T:g} s", f"Of the {encoder.budget} released"], table_rows,
                      numeric={1, 2}),
            ]), aside=aside(f"simulated {when(datetime.fromtimestamp(path.stat().st_mtime).isoformat())}"),
                support=f"{len(rows):,} simulated runs · {settings.runs_per_symbol} per symbol · "
                        f"{channel.n_intervals + 1} receiver values each · {text(path)}"))
            summary.append(f"{len(rows):,} runs" + (" · changed config: simulate again" if stale else ""))
    # the decoders
    results = read_json(metrics_path)
    if results is None:
        parts.append(section("Decoders · one dataset", note(f"Not run yet · {command(f'run {config_path}')} trains "
                                                            "and scores the decoders.")))
        summary.append("decoders not run")
    else:
        decoders = results["decoders"]
        stale = results.get("config") != read_config_file(config_path)
        bodies = []
        for kind in ("plain", "rescaled"):
            best = max(d[kind]["ba_bound_bits"] for d in decoders.values())
            body_rows = [[NAMES[name], percent(d[kind]["accuracy"]), signed(d[kind]["macro_f1"]),
                          signed(d[kind]["hard_decision_mi_bits"]),
                          f'<span class="{"lab-strong" if d[kind]["ba_bound_bits"] == best else ""}">'
                          f'{signed(d[kind]["ba_bound_bits"])}</span>',
                          signed(d[kind]["temperature"], 2) if "temperature" in d[kind] else "—"]
                         for name, d in decoders.items()]
            bodies.append((f' data-show-for="decoders" data-value="{kind}"' + (" hidden" if kind != "plain" else ""),
                           body_rows))
        some = next(iter(decoders.values()))["plain"]
        choose = ('<label class="lab-select-label">Pipeline<select data-select-for="decoders">'
                  '<option value="plain">Plain</option><option value="rescaled">Temperature-rescaled</option>'
                  "</select></label>")
        loss = {name: d.get("training_loss") for name, d in decoders.items() if d.get("training_loss")}
        body = [note(f"These results were made with an earlier version of {text(config_path)} · "
                     f"{command(f'run {config_path}')} makes them again.", warn=True) if stale else "",
                table(["Decoder", "Accuracy", "Macro-F1", "Hard MI (bits)", "BA (bits)", "Temperature"], bodies,
                      numeric={1, 2, 3, 4, 5}, cls="lab-baseline-table"),
                f'<p class="lab-caption">These are separate fitted pipelines. The rescaled pipeline holds back '
                f"calibration rows, so its accuracy may differ from the plain one. Hard MI reads about "
                f"{some['mi_bias_bits']:.3f} bits high on {some['n_test']} test rows; the best BA is in bold. "
                f"Made {when(results.get('created'))} · {text(metrics_path)}</p>"]
        parts.append(section("Decoders · one dataset", "".join(body), aside=choose,
                             support=f"This config's codebook · {results['n_train'] + results['n_test']:,} "
                                     f"simulated runs · one {1 - settings.test_fraction:.0%}/"
                                     f"{settings.test_fraction:.0%} split · {results['n_test']} test rows"))
        data["matrices"]["decoders"] = {"selected": "plain", "panels": {
            kind: [{"name": NAMES[name], "matrix": d[kind]["confusion_matrix"]} for name, d in decoders.items()]
            for kind in ("plain", "rescaled")}}
        parts.append(matrix_block("decoders", "Confusion matrices · one dataset",
                                  f"{some['n_test']} test rows per decoder · the pipeline chosen above",
                                  per_row_note(some["confusion_matrix"]), []))
        if loss:
            series = [line_series(enumerate(values, start=1), NAMES[name], SLOT[name]) for name, values in loss.items()]
            data["charts"]["loss"] = {"type": "lines", "height": 250, "log": True, "end_labels": True, "x_min": 1,
                                      "panels": [{"series": series}], "x_label": "epoch",
                                      "y_label": "training loss (nats)"}
            parts.append(section("Training loss", figure(
                "loss", "Training loss of the network decoders",
                "Mean cross-entropy over the training runs · log scale",
                legend=legend([(SLOT[name], NAMES[name]) for name in loss])), heading="h3"))
        summary.append(f"{count(len(decoders), 'decoder')}" + (" · changed config: run again" if stale else ""))
    configuration = json.dumps(read_config_file(config_path), indent=2)
    parts.append(details("Full configuration", f"<pre>{text(configuration)}</pre>"))
    return "".join(parts), " · ".join(summary)


# --- Decision log --------------------------------------------------------------------------------------

def log_tab(folders: list[Path], data: dict) -> tuple[str, str]:
    """every question the guided commands asked (decisions.json in each folder, oldest session first), with its
    answer and how it was given. The evidence behind each one stays in the files. Returns the tab and a summary"""
    data["sources"]["log"] = " · ".join(str(folder / "decisions.json") for folder in folders)
    try:
        sessions = sorted((s for folder in folders for s in read_log(folder / "decisions.json")),
                          key=lambda s: s["started"])
    except DecisionLogError as problem:
        return section("Decision log", note(text(problem), warn=True)), "can't be read"
    if not sessions:
        return section("Decision log", note("No decisions yet · search and final record every question they ask "
                                            "here.")), "none yet"
    blocks = []
    for index, session in enumerate(sessions):
        rows = "".join(f'<div class="lab-decision-row"><span class="lab-decision-time">{when(d["time"], "%H:%M")}'
                       f'</span><span>{text(d["question"])}</span><span class="lab-decision-answer">'
                       f'{"Yes" if d["answer"] else "No"}<span class="lab-decision-how">{text(d["how"])}</span></span>'
                       "</div>" for d in session["decisions"])
        started = when(session["started"], "%d %b %Y · %H:%M")
        blocks.append(f'<details class="lab-session"{" open" if index == len(sessions) - 1 else ""}><summary>'
                      f'{text(session["command"])}<span class="lab-session-time">{started}</span></summary>'
                      f"<div>{rows}</div></details>")
    decisions = sum(len(s["decisions"]) for s in sessions)
    page = (f'<div class="lab-section-head"><h2>Decision log</h2>{aside(count(len(sessions), "command session"))}</div>'
            '<p class="lab-support">Questions and answers recorded by the guided commands. The evidence shown with '
            "each question is saved with it.</p>" + "".join(blocks))
    return page, f"{count(len(sessions), 'session')} · {count(decisions, 'decision')}"


# --- the page ------------------------------------------------------------------------------------------

def build_report(cfg: Config, config_path: str) -> tuple[str, list[tuple[str, str]]]:
    """the whole report of a config, as one HTML page, and a summary of each tab for the terminal.
    A found config with no search of its own shows the search that found it, with that search's final evaluation
    and decisions (found_by); its codebook, dataset and decoders are its own"""
    path = Path(config_path)
    name = path.stem
    own = Path(cfg.output.dir)
    origin = None if (own / "search.json").exists() else found_by(cfg, path)
    searched, search_folder = origin or (path, own)
    final_folder = search_folder if origin and not (own / "final.json").exists() else own
    check = (read_json(search_folder / "search.json") or {}).get("check")
    data = {"charts": {}, "matrices": {}, "sources": {}}
    built = {"results": results_tab(path, final_folder, check, data),
             "codebooks": codebooks_tab(cfg, path, searched, search_folder, data),
             "search": search_tab(path, search_folder, searched, data),
             "data": data_tab(cfg, path, data),
             "log": log_tab(list(dict.fromkeys([own, search_folder])), data)}
    final = read_json(final_folder / "final.json")
    made = datetime.now().strftime("%d %b %Y")
    meta = made + (f" · final evaluation at {when(final.get('created'), '%H:%M')}" if final else "")
    nav = "".join(f'<button id="lab-tab-{key}" type="button" role="tab" aria-selected="{str(i == 0).lower()}" '
                  f'aria-controls="lab-{key}" data-lab-tab="{key}">{text(title)}</button>'
                  for i, (key, title) in enumerate(TABS))
    panels = "".join(f'<div id="lab-{key}" class="lab-page" role="tabpanel" aria-labelledby="lab-tab-{key}"'
                     f'{"" if i == 0 else " hidden"}>{built[key][0]}</div>' for i, (key, _) in enumerate(TABS))
    folders = " · ".join(str(folder) for folder in dict.fromkeys([own, search_folder, final_folder]))
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{text(name)} · mi-race report</title><style>{asset('report.css')}</style></head><body>"
            f'<div id="mi-lab-report"><header><div class="lab-top"><span class="lab-brand">mi-race</span>'
            f'<span class="lab-top-sep">/</span><span class="lab-top-label">Experiment report</span>'
            f'<span class="lab-version">{__version__}</span></div><div class="lab-inner"><div class="lab-title-row">'
            f'<div><h1>{text(name)}</h1><p class="lab-path">{text(config_path)}</p></div>'
            f'<div class="lab-title-meta"><p>{meta}</p><p>{count(cfg.encoder.n_symbols, "symbol")} · '
            f"{cfg.encoder.budget} molecules per symbol</p></div></div>"
            f'<nav class="lab-tabs" role="tablist" aria-label="Report sections">{nav}</nav></div></header>'
            f'<main class="lab-inner">{panels}<footer class="lab-footer"><span>mi-race · {text(name)}</span>'
            f'<span id="lab-source-label">Every number comes from the result files in {text(folders)}</span></footer>'
            f'</main></div><script type="application/json" id="mi-lab-report-data">{payload}</script>'
            f"<script>{asset('report.js')}</script></body></html>\n")
    return page, [(key, built[key][1]) for key, _ in TABS]
