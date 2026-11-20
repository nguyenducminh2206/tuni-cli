"""report: one self-contained HTML page with everything an experiment has produced, read from its result
files: the setup from the config, the dataset, the decoders (metrics.json), the search (search.json), a
comparison (compare.json), the final evaluation (final.json) and the decision log (decisions.json).
It never simulates or trains: a section with no results yet says which command makes them.
The styles sit in STYLE and the figures are inline SVG (figures.py), so the page needs nothing else"""

import html
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from mi_race import __version__, figures
from mi_race.config import Config, ConfigError, load_config, read_config_file
from mi_race.dataset import DatasetError, check_dataset, read_dataset
from mi_race.decisions import DecisionLogError, read_log

NAMES = figures.DECODERS                    # 'random forest', 'MLP', 'CNN'

STYLE = """
:root { color-scheme: light; --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --secondary: #52514e;
        --muted: #898781; --hairline: #e1e0d9; --accent: #2a78d6; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 820px; margin: 0 auto; padding: 48px 24px 80px; }
header { border-bottom: 1px solid var(--hairline); padding-bottom: 20px; margin-bottom: 8px; }
.kicker { color: var(--muted); font-size: 13px; letter-spacing: .04em; text-transform: uppercase; margin: 0; }
h1 { font-size: 30px; line-height: 1.2; margin: 6px 0 8px; font-weight: 600; }
.meta { color: var(--secondary); font-size: 13px; margin: 0; }
nav { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; padding: 12px 0 0; }
nav a { color: var(--secondary); text-decoration: none; }
nav a:hover { color: var(--accent); }
section { margin-top: 44px; }
h2 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.lead { color: var(--secondary); margin: 0 0 18px; }
h3 { font-size: 15px; font-weight: 600; margin: 26px 0 8px; }
dl.facts { display: grid; grid-template-columns: max-content 1fr; gap: 4px 20px; margin: 0 0 16px;
           font-size: 14px; }
dl.facts dt { color: var(--muted); }
dl.facts dd { margin: 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 4px 0 16px;
        font-variant-numeric: tabular-nums; }
th { text-align: left; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--hairline);
     padding: 6px 10px 6px 0; }
td { border-bottom: 1px solid var(--hairline); padding: 6px 10px 6px 0; }
th, td { white-space: nowrap; }
th.num, td.num { text-align: right; }
td.wrap { white-space: normal; }
td.best { font-weight: 600; }
code { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }
figure { margin: 18px 0 24px; background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px;
         padding: 14px 14px 10px; }
figure svg { display: block; width: 100%; height: auto; }
figcaption { color: var(--secondary); font-size: 13px; margin-top: 8px; }
.note { color: var(--secondary); background: var(--surface); border: 1px solid var(--hairline);
        border-left: 3px solid var(--muted); border-radius: 6px; padding: 10px 14px; margin: 8px 0 16px;
        font-size: 14px; }
.note.warning { border-left-color: #eda100; }
footer { margin-top: 56px; padding-top: 14px; border-top: 1px solid var(--hairline); color: var(--muted);
         font-size: 12px; }
@media print { body { background: #fff; } main { padding: 0; } nav { display: none; }
               figure, table { break-inside: avoid; } }
"""


# --- small pieces of HTML --------------------------------------------------------------------------

def text(value) -> str:
    """any value as safe HTML text"""
    return html.escape(str(value))


def facts(pairs: list[tuple[str, str]]) -> str:
    """label and value lines, the values already HTML"""
    return '<dl class="facts">' + "".join(f"<dt>{text(label)}</dt><dd>{value}</dd>" for label, value in pairs) \
        + "</dl>"


def table(header: list[str], rows: list[list[str]], numeric: set[int] = frozenset(), bold: set = frozenset(),
          wrap: set[int] = frozenset()) -> str:
    """a table: header texts, rows of HTML cells; numeric: the columns to right-align; bold: (row, column)
    cells to set in bold; wrap: the columns whose cells may wrap (long codebooks, questions), the rest stay
    on one line"""
    def cell(tag: str, value: str, classes: list[str]) -> str:
        names = " ".join(name for name in classes if name)
        return f'<{tag} class="{names}">{value}</{tag}>' if names else f"<{tag}>{value}</{tag}>"

    head = "".join(cell("th", text(h), ["num" if j in numeric else ""]) for j, h in enumerate(header))
    body = "".join("<tr>" + "".join(cell("td", value, ["num" if j in numeric else "", "wrap" if j in wrap else "",
                                                       "best" if (i, j) in bold else ""])
                                    for j, value in enumerate(row)) + "</tr>" for i, row in enumerate(rows))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def figure(svg: str, caption: str) -> str:
    """a figure with its caption (the caption already HTML)"""
    return f"<figure>{svg}<figcaption>{caption}</figcaption></figure>"


def note(message: str, warning: bool = False) -> str:
    """a boxed note: a missing result and the command that makes it, or something to be careful about"""
    return f'<p class="note{" warning" if warning else ""}">{message}</p>'


def command(words: str) -> str:
    """a mi-race command, as it is typed"""
    return f"<code>mi-race {text(words)}</code>"


def typed(vector) -> str:
    """a release vector as it is typed: '25:3 25:4'"""
    return " ".join(f"{amount}:{k}" for k, amount in enumerate(vector) if amount)


def count(n: int, thing: str) -> str:
    """'1 session', '3 sessions'"""
    return f"{n} {thing}{'' if n == 1 else 's'}"


def codebook_line(symbols) -> str:
    """a codebook on one line, as it is typed: '50:0 · 50:3 · 25:5 25:6 · 50:9'"""
    return " · ".join(typed(vector) for vector in symbols)


def when(stamp: str) -> str:
    """a saved time, readable: '2026-11-20T02:30:07' -> '20 Nov 2026, 02:30'"""
    try:
        return datetime.fromisoformat(stamp).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return text(stamp)


def mean_se(values) -> str:
    """the mean of some per-dataset values ± its standard error"""
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else float("nan")
    return f"{values.mean():.3f} ± {se:.3f}"


def bits_table(bits: dict, with_se: bool = False) -> str:
    """the BA bound of every codebook under every decoder (bits: codebook -> decoder -> one value per dataset):
    the mean over the datasets (± se), and the mean over the decoders"""
    decoders = list(next(iter(bits.values())))
    rows = [[text(label), *[mean_se(by[name]) if with_se else f"{np.mean(by[name]):.3f}" for name in decoders],
             f"{np.mean([np.mean(by[name]) for name in decoders]):.3f}"] for label, by in bits.items()]
    return table(["codebook", *[NAMES[name] for name in decoders], "mean"], rows,
                 numeric=set(range(1, len(decoders) + 2)))


def gains_table(gains: dict) -> str:
    """the paired gain of each codebook over the reference (gains: codebook -> decoder -> gain): mean ± se,
    and on how many datasets it is higher"""
    decoders = list(next(iter(gains.values())))
    rows = [[text(label), *[f"{g[name]['mean']:+.3f} ± {g[name]['se']:.3f} · {g[name]['higher']} of "
                            f"{g[name]['datasets']}" for name in decoders]] for label, g in gains.items()]
    return table(["codebook", *[NAMES[name] for name in decoders]], rows, numeric=set(range(1, len(decoders) + 1)))


def codebooks_table(codebooks: dict, reference: str) -> str:
    """the codebooks by name, as they are typed; the reference marked"""
    return table(["codebook", "typed"], [[text(label) + (" · reference" if label == reference else ""),
                                          f"<code>{codebook_line(symbols)}</code>"]
                                         for label, symbols in codebooks.items()], wrap={1})


def read_json(path: Path) -> dict | None:
    """a result file, or None if there is none or it can't be read"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- the sections ----------------------------------------------------------------------------------

def setup_section(cfg: Config) -> tuple[str, str]:
    """the experiment as the config defines it: channel, time, encoder, data, decoders, seed; the codebook.
    Returns the section and a summary of it, for the terminal"""
    channel, encoder, data, search = cfg.channel, cfg.encoder, cfg.data, cfg.search
    test_runs = round(data.runs_per_symbol * data.test_fraction) * encoder.n_symbols
    rows = [[f"symbol {u}", f"<code>{typed(vector)}</code>",
             " · ".join(f"{amount} at {k * encoder.slot_dt:g} s" for k, amount in enumerate(vector) if amount)]
            for u, vector in enumerate(encoder.symbols)]
    page = "".join([
        '<section id="setup"><h2>Setup</h2>',
        '<p class="lead">The channel, how a symbol is sent, and the data the decoders learn from.</p>',
        facts([
            ("channel", f"L {channel.L} · h {channel.h:g} µm · D {channel.D:g} µm²/s · "
                        f"λ {channel.jump_rate:g}/s"),
            ("time", f"T {channel.T:g} s · Δt {channel.dt:g} s · {channel.n_intervals + 1} values per run"),
            ("encoder", f"{encoder.n_symbols} symbols · {encoder.n_slots} slots · Δs {encoder.slot_dt:g} s · "
                        f"budget {encoder.budget} molecules"),
            ("data", f"{data.runs_per_symbol} runs per symbol · {test_runs} test runs · {text(data.path)}"),
            ("decoders", " · ".join(NAMES[name] for name in cfg.decoders) or "none"),
            ("search", f"{'rescaled ' if search.rescaled else ''}{NAMES[search.decoder]} · chunk {search.chunk} · "
                       f"θ {search.threshold:g} bits · {search.steps} steps · {search.runs} runs"),
            ("seed", str(cfg.seed)),
        ]),
        "<h3>Codebook</h3>",
        table(["symbol", "typed", "releases"], rows, wrap={2}),
        figure(figures.release_schedules(cfg),
               "Release schedules: a bar for each release, at its time and as tall as the molecules it releases. "
               f"The axis runs to T = {channel.T:g} s, the end of a run."),
        "</section>",
    ])
    return page, f"the config · {encoder.n_symbols} symbols · 1 figure"


def dataset_section(cfg: Config, config_path: Path) -> tuple[str, str]:
    """the dataset at data.path: its size, how many molecules reach the receiver, and the mean signal; or what
    is wrong with it. Returns the section and a summary of it"""
    path = Path(cfg.data.path)
    head = '<section id="dataset"><h2>Dataset</h2>'
    if not path.exists():
        return (head + note(f"No dataset yet · {command(f'simulate {config_path}')} makes it.") + "</section>",
                f"not simulated yet · simulate {config_path}")
    try:
        rows = read_dataset(path)
        check_dataset(rows, cfg)
    except DatasetError as problem:
        return head + note(f"{text(problem)}", warning=True) + "</section>", f"doesn't fit · simulate {config_path}"
    encoder = cfg.encoder
    absorbed = rows[:, 1:].sum(axis=1)
    stale = config_path.stat().st_mtime > path.stat().st_mtime
    table_rows = [[f"symbol {u}", f"{absorbed[rows[:, 0] == u].mean():.1f}",
                   f"{absorbed[rows[:, 0] == u].mean() / encoder.budget:.0%}"] for u in range(encoder.n_symbols)]
    page = "".join([
        head,
        f'<p class="lead">{len(rows)} simulated runs: {cfg.data.runs_per_symbol} of each symbol, '
        f'{cfg.channel.n_intervals + 1} receiver values each.</p>',
        note(f"{text(config_path)} was changed after this dataset was simulated · "
             f"{command(f'simulate {config_path}')} makes it again.", warning=True) if stale else "",
        facts([("file", text(path)),
               ("simulated", datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y, %H:%M"))]),
        "<h3>Absorbed by the receiver</h3>",
        table(["symbol", "mean molecules", f"of {encoder.budget} released"], table_rows, numeric={1, 2}),
        figure(figures.mean_signal(cfg, rows),
               "Mean received signal: the molecules absorbed so far, averaged over each symbol's runs. A later "
               "release shifts a curve right; its end is how many molecules arrive before T."),
        "</section>",
    ])
    return page, f"{len(rows)} runs · 1 figure" + (" · changed config: simulate again" if stale else "")


def decoders_section(cfg: Config, config_path: Path) -> tuple[str, str]:
    """the decoders as 'run' trained and scored them (metrics.json): a table of every decoder, the confusion
    matrices and the training loss. Returns the section and a summary of it"""
    path = Path(cfg.output.dir) / "metrics.json"
    head = '<section id="decoders"><h2>Decoders</h2>'
    results = read_json(path)
    if results is None:
        return (head + note(f"Not run yet · {command(f'run {config_path}')} trains and scores the decoders.")
                + "</section>", f"not run yet · run {config_path}")
    decoders = results["decoders"]
    stale = results.get("config") != read_config_file(config_path)
    loss = figures.training_loss({name: result.get("training_loss") for name, result in decoders.items()})
    best = max(result["rescaled"]["ba_bound_bits"] for result in decoders.values())
    rows, bold = [], set()
    for i, (name, result) in enumerate(decoders.items()):
        plain, rescaled = result["plain"], result["rescaled"]
        rows.append([NAMES[name], f"{plain['accuracy']:.1%}", f"{plain['hard_decision_mi_bits']:.3f}",
                     f"{plain['ba_bound_bits']:.3f}", f"{rescaled['temperature']:.2f}",
                     f"{rescaled['ba_bound_bits']:.3f}"])
        if rescaled["ba_bound_bits"] == best:
            bold.add((i, 5))
    some = next(iter(decoders.values()))["plain"]
    page = "".join([
        head,
        f'<p class="lead">Every decoder trained on {results["n_train"]} runs and scored on the other '
        f'{results["n_test"]}, as it is (plain) and with its probabilities rescaled by a temperature T fitted on '
        'held-back training runs.</p>',
        note(f"These results were made with an earlier version of {text(config_path)} · "
             f"{command(f'run {config_path}')} makes them again.", warning=True) if stale else "",
        table(["decoder", "accuracy", "hard MI (bits)", "BA bound (bits)", "T", "BA rescaled (bits)"], rows,
              numeric={1, 2, 3, 4, 5}, bold=bold),
        f'<p class="lead">Hard MI reads about {some["mi_bias_bits"]:.3f} bits high on {some["n_test"]} test runs. '
        'The BA bound is a lower bound on the mutual information; the best rescaled one is in bold.</p>',
        figure(figures.confusion_matrices({name: result["plain"]["confusion_matrix"]
                                           for name, result in decoders.items()}),
               "Confusion matrices of the plain decoders: rows are the symbol sent, columns the symbol decided. "
               "A cell is shaded by its share of the row, so a dark diagonal means right decisions."),
        figure(loss, "Training loss of the network decoders, epoch by epoch (log scale): the mean "
                     "cross-entropy over the training runs.") if loss else "",
        facts([("results", text(path)), ("made", text(results.get("created", "")).replace("T", ", ")),
               ("mi-race", text(results.get("mi_race", "")))]),
        "</section>",
    ])
    return page, (f"{len(decoders)} decoders · {2 if loss else 1} figures"
                  + (" · changed config: run again" if stale else ""))


def search_section(config_path: Path, folder: Path, searched: Path) -> tuple[str, str]:
    """the guided search (folder/search.json): its settings and the noise behind its threshold, the runs, their
    learning curves, and the check of the found codebooks against the start. searched is the config the search
    ran on: config_path itself, or, for a found config, the config of the search that found it.
    Returns the section and a summary"""
    path = folder / "search.json"
    head = '<section id="search"><h2>Search</h2>'
    record = read_json(path)
    if record is None:
        return (head + note(f"No search yet · {command(f'search {config_path}')} looks for a better codebook, "
                            "from this one.") + "</section>", f"not run yet · search {config_path}")
    settings, noise, runs = record["settings"], record.get("noise"), record["runs"]
    score = f"{'rescaled ' if settings['rescaled'] else ''}{NAMES[settings['decoder']]}"
    kept = [[row for row in run["history"] if row["accepted"]] for run in runs]
    check = record.get("check")
    best = check["best"] if check else None
    rows = [[f"run {run['run'] + 1}", str(len(run["history"])), str(len(moves)),
             f"step {moves[-1]['step'] + 1}" if moves else "–", f"<code>{codebook_line(run['final'])}</code>"]
            for run, moves in zip(runs, kept)]
    parts = [
        head,
        note(f"The codebook of {text(config_path)} was found by this search of {text(searched)}: the search, "
             f"the final evaluation and the decision log here come from {text(folder)}.")
        if searched != config_path else "",
        f'<p class="lead">{len(runs)} runs from the start codebook. Each step proposes a move of '
        f'{settings["chunk"]} molecules and keeps it if the {score} scores it more than θ = '
        f'{settings["threshold"]:g} bits above the codebook before it, on a new dataset each step.</p>',
        note(f"{text(searched)} changed since the search · the search used the config saved in {text(path)}.",
             warning=True) if record.get("config") != read_config_file(searched) else "",
        facts([("start", f"<code>{codebook_line(record['config']['encoder']['symbols'])}</code>"),
               ("feedback", f"{score} · BA bound, 5-fold, on a new dataset each step"),
               ("noise", f"sd {noise['sd']:.4f} bits on {len(noise['differences'])} paired datasets · "
                         f"2 × sd ≈ {noise['threshold']:g} bits" if noise else "not measured"),
               ("made", when(record.get("created")))]),
        table(["run", "steps", "kept", "last kept", "codebook"], rows, numeric={1, 2}, wrap={4}),
    ]
    if not all("curve" in run for run in runs):
        parts.append(note("No learning curves: this search was made before mi-race recorded them · a new search "
                          "records them."))
    else:
        curves = {f"run {run['run'] + 1}": run["curve"] for run in runs}
        parts.append(figure(figures.learning_curves(curves, best),
                            f"Learning curves: each run's score on the same check datasets after every step, with "
                            f"the search's own score ({score}, 5-fold), so the steps compare. "
                            + ("The run with the best found codebook is blue." if best else "")))
    if check:
        gains = check["gains"]
        parts += [
            "<h3>Check</h3>",
            f'<p class="lead">The start and the codebooks the runs found, under every rescaled decoder, on the '
            f'same {check["datasets"]} check datasets. These are development numbers, for choosing; the final '
            'evaluation below is the result to report.</p>',
            bits_table(check["ba_bound_bits"]),
            facts([("best found", f"{text(best)} · <code>{codebook_line(check['codebooks'][best])}</code>"),
                   *[(NAMES[name], f"{g['mean']:+.3f} ± {g['se']:.3f} bits over the start · higher on "
                                   f"{g['higher']} of {g['datasets']}") for name, g in gains.items()],
                   ("config", text(record["found_config"]) if record.get("found_config") else "not written")]),
        ]
    else:
        parts.append(note(f"The search stopped before its check · {command(f'final {config_path}')} checks the "
                          "found codebooks, then offers the final evaluation."))
    parts.append("</section>")
    summary = (f"{count(len(runs), 'run')} · {count(sum(map(len, kept)), 'move')} kept"
               + (" · checked" if check else " · no check")
               + (f" · from {searched}" if searched != config_path else ""))
    return "".join(parts), summary


def comparison_section(cfg: Config, config_path: Path) -> tuple[str, str]:
    """codebooks compared with 'compare' (compare.json), on the check datasets. Returns the section and a
    summary"""
    path = Path(cfg.output.dir) / "compare.json"
    head = '<section id="comparison"><h2>Comparison</h2>'
    record = read_json(path)
    if record is None:
        return (head + note(f"No comparison · {command(f'compare {config_path} OTHER …')} compares this "
                            "codebook with those of other configs.") + "</section>", "none")
    reference = record["reference"]
    page = "".join([
        head,
        f'<p class="lead">{len(record["codebooks"])} codebooks under every rescaled decoder, on the same '
        f'{record["datasets"]} check datasets, against {text(reference)}. Development numbers, for choosing.</p>',
        codebooks_table(record["codebooks"], reference),
        bits_table(record["ba_bound_bits"]),
        f"<h3>Gain over {text(reference)}</h3>",
        gains_table(record["gains"]),
        facts([("made", when(record.get("created"))), ("results", text(path))]),
        "</section>",
    ])
    return page, f"{len(record['codebooks'])} codebooks"


def final_section(config_path: Path, folder: Path) -> tuple[str, str]:
    """the final evaluation (folder/final.json), the result to report: a search's start and best found codebook,
    or codebooks designed by hand, on the final datasets, used once. Returns the section and a summary"""
    path = folder / "final.json"
    head = '<section id="final"><h2>Final evaluation</h2>'
    record = read_json(path)
    if record is None:
        return (head + note(f"No final evaluation · after a search, {command(f'final {config_path}')}; for "
                            f"codebooks you designed, {command(f'final {config_path} OTHER …')}.") + "</section>",
                "none")
    by_hand = "reference" in record
    reference = record["reference"] if by_hand else "start"
    gains = record["gains"] if by_hand else {record["best"]: record["gains"]}
    parts = [
        head,
        note(f"The result to report: {record['datasets']} final datasets, never used for any choice, and used "
             "once."),
        f'<p class="lead">' + (f"{len(record['codebooks'])} codebooks designed by hand, against "
                               f"{text(reference)}." if by_hand else
                               f"The search's start against its best found codebook, {text(record['best'])}, "
                               "chosen on the check datasets.") + "</p>",
        codebooks_table(record["codebooks"], reference),
        "<h3>BA bound (bits) · mean ± se over the datasets</h3>",
        bits_table(record["ba_bound_bits"], with_se=True),
        f"<h3>Gain over {text(reference)} · paired by dataset</h3>",
        gains_table(gains),
    ]
    if not by_hand and "scores" in record:                  # the found codebook's decisions, all datasets together
        best = record["best"]
        matrices = {name: np.sum([dataset[name]["confusion_matrix"] for dataset in record["scores"][best]], axis=0)
                    for name in record["scores"][best][0]}
        parts.append(figure(figures.confusion_matrices(matrices),
                            f"Confusion matrices of {text(best)}, the found codebook, summed over the "
                            f"{record['datasets']} final datasets (rescaled decoders)."))
    parts += [facts([("made", when(record.get("created"))), ("results", text(path))]), "</section>"]
    return "".join(parts), f"{len(record['codebooks'])} codebooks · {record['datasets']} final datasets"


def decisions_section(folders: list[Path]) -> tuple[str, str]:
    """every question the guided commands asked (decisions.json in each folder, oldest session first): the step,
    the question, the answer, how it was given and when. The evidence behind each one stays in the files.
    Returns the section and a summary"""
    head = '<section id="decisions"><h2>Decision log</h2>'
    try:
        sessions = sorted((session for folder in folders for session in read_log(folder / "decisions.json")),
                          key=lambda session: session["started"])
    except DecisionLogError as problem:
        return head + note(text(problem), warning=True) + "</section>", "can't be read"
    if not sessions:
        return (head + note("No decisions yet · search and final record every question they ask here.")
                + "</section>", "none")
    files = " and ".join(text(folder / "decisions.json") for folder in folders)
    parts = [head, '<p class="lead">Every question the guided commands asked, with its answer. The evidence '
                   f'shown with each question is saved in {files}.</p>']
    for session in sessions:
        rows = [[text(d["step"]), text(d["question"]),
                 ("yes" if d["answer"] else "no") + (" · default" if d["how"] != "typed" else ""),
                 text(d["how"]), when(d["time"]).split(", ")[-1]] for d in session["decisions"]]
        parts += [f"<h3>{text(session['command'])} · {when(session['started'])}</h3>",
                  table(["step", "question", "answer", "how", "time"], rows, wrap={1})]
    parts.append("</section>")
    decisions = sum(len(session["decisions"]) for session in sessions)
    return "".join(parts), f"{count(len(sessions), 'session')} · {count(decisions, 'decision')}"


# --- the page --------------------------------------------------------------------------------------

SECTIONS = [("setup", "Setup"), ("dataset", "Dataset"), ("decoders", "Decoders"), ("search", "Search"),
            ("comparison", "Comparison"), ("final", "Final evaluation"), ("decisions", "Decision log")]


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


def build_report(cfg: Config, config_path: str) -> tuple[str, list[tuple[str, str]]]:
    """the whole report of a config, as one HTML page, and a summary of each section for the terminal.
    A found config with no search of its own shows the search that found it, with that search's final
    evaluation and decisions (found_by); its setup, dataset and decoders are its own"""
    path = Path(config_path)
    name = path.stem
    own = Path(cfg.output.dir)
    origin = None if (own / "search.json").exists() else found_by(cfg, path)
    searched, search_folder = origin or (path, own)
    final_folder = search_folder if origin and not (own / "final.json").exists() else own
    built = [setup_section(cfg), dataset_section(cfg, path), decoders_section(cfg, path),
             search_section(path, search_folder, searched), comparison_section(cfg, path),
             final_section(path, final_folder), decisions_section(list(dict.fromkeys([own, search_folder])))]
    sections = [page for page, _ in built]
    summary = [(anchor, line) for (anchor, _), (_, line) in zip(SECTIONS, built)]
    nav = "".join(f'<a href="#{anchor}">{title}</a>' for anchor, title in SECTIONS)
    made = datetime.now().strftime("%d %b %Y, %H:%M")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{text(name)} · mi-race report</title><style>{STYLE}</style></head><body><main>"
            f'<header><p class="kicker">mi-race report</p><h1>{text(name)}</h1>'
            f'<p class="meta">{text(config_path)} · made {made} · mi-race {__version__}</p><nav>{nav}</nav></header>'
            + "".join(sections)
            + "<footer>Made by mi-race from the config and the result files in "
              f"{' and '.join(text(folder) for folder in dict.fromkeys([own, search_folder, final_folder]))}. "
              "Every number here comes from those files.</footer></main></body></html>\n",
            summary)
