"""cli: the mi-race command. Typed alone, `mi-race` opens the mi-race shell, where commands are typed one
after another; `mi-race COMMAND ...` runs one command, for scripts and batch jobs.
Each command is a function run_<name>(args) that returns the exit code. Warnings appear as
'[mi-race] ! warning' lines as they happen; a broken config, dataset or file becomes one
'[mi-race] ✗ error' line"""

import argparse
import atexit
import copy
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, replace
from functools import partial
from datetime import datetime
from importlib import resources
from pathlib import Path

import numpy as np

from mi_race import __version__, ui
from mi_race.compare import measure_noise
from mi_race.config import Config, ConfigError, load_config, parse_config, read_config_file, write_config_file
from mi_race.dataset import (DatasetError, check_dataset, read_dataset, simulate_dataset, split_dataset,
                             write_dataset)
from mi_race.decisions import DecisionLog, DecisionLogError
from mi_race.decoders.rescaling import train_rescaled
from mi_race.metrics import evaluate
from mi_race.search import TRAINERS, FOLDS, score_codebook, search
from mi_race.streams import random_stream
from mi_race.symbols import SymbolError, ask_symbols, parse_vector, write_symbols

DECODER_NAMES = {"random_forest": "random forest", "mlp": "MLP", "cnn": "CNN"}
UNITS = {"random_forest": "trees", "mlp": "epochs", "cnn": "epochs"}       # what a decoder's progress counts


def color() -> bool:
    """styles for normal output, only on a terminal"""
    return ui.use_color(sys.stdout)


def describe(cfg: Config) -> str:
    """what a config sets up: channel, time, encoder, data, decoders, seed and output, as facts"""
    channel, encoder, data = cfg.channel, cfg.encoder, cfg.data
    test_runs = round(data.runs_per_symbol * data.test_fraction) * encoder.n_symbols
    return ui.facts([
        ("channel", f"L {channel.L} · h {channel.h:g} µm · D {channel.D:g} µm²/s · λ {channel.jump_rate:g}/s"),
        ("time", f"T {channel.T:g} s · Δt {channel.dt:g} s · {channel.n_intervals + 1} values per run"),
        ("encoder", f"{encoder.n_symbols} symbols · {encoder.n_slots} slots · Δs {encoder.slot_dt:g} s · "
                    f"budget {encoder.budget}"),
        ("data", f"{data.runs_per_symbol} runs per symbol · {test_runs} test runs · {data.path}"),
        ("decoders", " · ".join(DECODER_NAMES[name] for name in cfg.decoders) or "none"),
        ("search", f"{'rescaled ' if cfg.search.rescaled else ''}{DECODER_NAMES[cfg.search.decoder]} · "
            f"chunk {cfg.search.chunk} · θ {cfg.search.threshold:g} bits · {cfg.search.steps} steps · "
            f"{cfg.search.runs} runs"),
        ("seed", str(cfg.seed)),
        ("output", cfg.output.dir),
    ], color())


def describe_symbols(cfg: Config, symbols=None, typed: bool = False) -> str:
    """each symbol's releases: '50 at 0.3 s', or several for a split burst (the config's own symbols,
    or others for the same slots). typed: first the symbol as it is typed, amount:slot ('50:3')"""
    encoder = cfg.encoder
    symbols = encoder.symbols if symbols is None else symbols
    pairs = [" ".join(f"{amount}:{k}" for k, amount in enumerate(vector) if amount) for vector in symbols]
    width = max(len(text) for text in pairs)
    return ui.facts([(f"symbol {u}", (ui.paint(pairs[u].ljust(width), ui.BOLD, color()) + "  " if typed else "")
                      + " · ".join(f"{amount} at {k * encoder.slot_dt:g} s" for k, amount in enumerate(vector) if amount))
                     for u, vector in enumerate(symbols)], color())


def template() -> dict:
    """the example config shipped inside the package: the start of every new experiment"""
    return json.loads((resources.files("mi_race") / "templates" / "example.json").read_text(encoding="utf-8"))


def show_config(cfg: Config) -> None:
    """what a valid config sets up, its codebook, and the ✓ line"""
    print(describe(cfg))
    print()
    print(ui.section("codebook", f"{cfg.encoder.n_symbols} symbols · budget {cfg.encoder.budget}",
                     describe_symbols(cfg), color()))
    print()
    print(ui.done("config is valid", "every rule passed", color=color()))


def interactive() -> bool:
    """someone at a terminal, who can answer questions and use an editor"""
    return ui.is_terminal(sys.stdin) and ui.is_terminal(sys.stdout)


def editor() -> list[str]:
    """the editor that opens configs: $VISUAL or $EDITOR if set (e.g. 'code -w'), else nano, vi or notepad"""
    chosen = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if chosen:
        return shlex.split(chosen)
    for name in ("nano", "vi", "notepad"):
        if shutil.which(name):
            return [name]
    raise OSError("no text editor found: set one, e.g. export EDITOR=nano")


def edit_until_valid(path: Path) -> Config | None:
    """open the config in the editor, and check it when the editor closes. If only the symbols broke
    (n_slots or the budget changed), the codebook fitted to them is offered (fit_after_edit); otherwise a
    broken config can be opened again to fix it. Returns the valid config, or None"""
    while True:
        command = editor()
        print(ui.info(f"opening {path} in {Path(command[0]).name} · save and close it to come back", color()))
        subprocess.run([*command, str(path)])
        print()
        try:
            return load_config(path)
        except ConfigError as problem:
            complain(str(problem))
            print()
            fitted = fit_after_edit(path, str(problem))
            if fitted is not None:
                return fitted
            if not ui.ask("Open it again to fix it?"):
                return None
            print()


# --- the questions of a new config ---------------------------------------------------------------

QUESTIONS = [                               # what 'new' asks, in this order: (section, key, what it is · unit)
    ("channel", "L", "channel length · L-1 compartments, then the receiver"),
    ("channel", "h", "compartment width · µm"),
    ("channel", "D", "diffusion coefficient · µm²/s"),
    ("channel", "T", "recording time · s"),
    ("channel", "dt", "time between snapshots · s"),
    ("encoder", "n_slots", "release slots"),
    ("encoder", "slot_dt", "time between slots · s"),
    ("encoder", "budget", "molecules per symbol"),
    ("encoder", "symbols", "how many different symbols"),    # a count; then the codebook step (ask_codebook)
    ("data", "runs_per_symbol", "simulated runs per symbol"),
]
SEARCH_QUESTIONS = [                        # what 'search' asks before it starts
    ("search", "runs", "runs, each from the start codebook"),
    ("search", "steps", "steps per run · more while a run still improves"),
    ("search", "chunk", "molecules a move carries · at most the budget"),
]
SECTION_NOTES = {"channel": "where the molecules diffuse", "encoder": "when a symbol releases its molecules",
                 "data": "the simulated dataset", "search": "how the codebook is searched"}


def typed_value(text: str, default):
    """a typed answer, read as JSON reads a number ('30' → 30, '0.05' → 0.05); a whole number for a setting
    with decimals gets them (2 → 2.0); anything else stays text, for the config's own checks to name"""
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return float(value) if isinstance(default, float) and type(value) is int else value


def is_number(*values) -> bool:
    """every value a finite number (not true or false)"""
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def round_below(limit: float) -> float:
    """the largest of 1, 2 and 5 × 10^k below limit: a round time step that fits (0.069 → 0.05)"""
    power = 10.0 ** math.floor(math.log10(limit))
    for value in (5 * power, 2 * power, power, power / 2):
        if value < limit * (1 - 1e-9):
            return float(f"{value:.12g}")


def fit_timing(raw: dict) -> str:
    """lower encoder.slot_dt to a round value if the last slot would release at or after channel.T.
    Returns what changed, or '' (nothing to do, or a value the config's checks will name)"""
    encoder, T = raw["encoder"], raw["channel"]["T"]
    n_slots, slot_dt = encoder["n_slots"], encoder["slot_dt"]
    if not is_number(n_slots, slot_dt, T) or n_slots < 2 or T <= 0 or (n_slots - 1) * slot_dt < T * (1 - 1e-9):
        return ""
    encoder["slot_dt"] = round_below(T / (n_slots - 1))
    last = n_slots - 1
    return (f"slot_dt lowered to {encoder['slot_dt']:g} s: slot {last} releases at {last * encoder['slot_dt']:g} s, "
            "before T")


def fit_unasked(raw: dict, key: str) -> list[str]:
    """settings 'new' does not ask about, lowered where the answer to key leaves them out of bounds:
    search.chunk to a smaller budget (a search move carries at most the budget), and the CNN's kernel_size
    to the values per run after T or dt. Returns what changed"""
    notes, channel, encoder = [], raw["channel"], raw["encoder"]
    search, cnn = raw.get("search", {}), raw.get("decoders", {}).get("cnn")
    budget, chunk = encoder["budget"], search.get("chunk")
    if key == "budget" and type(budget) is int and type(chunk) is int and 1 <= budget < chunk:
        search["chunk"] = budget
        notes.append(f"search.chunk lowered to {budget}: a move carries at most the budget")
    T, dt = channel["T"], channel["dt"]
    if key in ("T", "dt") and isinstance(cnn, dict) and is_number(T, dt) and T > 0 and dt > 0 \
            and type(cnn.get("kernel_size")) is int:
        n_values = round(T / dt) + 1                                # rx_0 ... rx_M
        kernel = n_values if n_values % 2 else n_values - 1         # the kernel must be odd
        if cnn["kernel_size"] > n_values >= 2:
            cnn["kernel_size"] = kernel
            notes.append(f"decoders.cnn.kernel_size lowered to {kernel}: at most the {n_values} values per run")
    return notes


def spread_codebook(n_symbols: int, n_slots: int, budget: int) -> list[list[int]]:
    """a codebook for any slots and budget: one burst per symbol, spread evenly from the first slot to the
    last (4 symbols on 10 slots: slots 0, 3, 6 and 9, as in the example). With fewer slots than symbols,
    the budget is split between the first slot and the last, in even steps"""
    if n_slots < 2:
        raise ConfigError("with 1 slot every symbol is the same: use 2 slots or more")
    if n_slots < n_symbols and budget < n_symbols - 1:
        raise ConfigError(f"{n_symbols} symbols are too many to spread over {n_slots} slots with a budget of "
                          f"{budget}: use at most {max(n_slots, budget + 1)} symbols, more slots or a larger budget")
    codebook = [[0] * n_slots for _ in range(n_symbols)]
    for u in range(n_symbols):
        if n_slots >= n_symbols:                        # u × (n_slots - 1) / (n_symbols - 1), rounded half up
            codebook[u][(2 * u * (n_slots - 1) + n_symbols - 1) // (2 * (n_symbols - 1))] = budget
        else:
            late = (2 * u * budget + n_symbols - 1) // (2 * (n_symbols - 1))
            codebook[u][0], codebook[u][-1] = budget - late, late
    return codebook


def spread_note(n_symbols: int, n_slots: int) -> str:
    """how spread_codebook made a codebook, for the note above it"""
    return "a burst each, spread over the slots" if n_slots >= n_symbols else "split between the first and last slot"


def fit_codebook(raw: dict) -> None:
    """spread the codebook over the slots (spread_codebook) if the symbols no longer fit n_slots or the
    budget. Nothing is said here: the codebook step shows the codebook after the encoder questions"""
    encoder = raw["encoder"]
    n_slots, budget, symbols = encoder["n_slots"], encoder["budget"], encoder["symbols"]
    if not (type(n_slots) is int and type(budget) is int and n_slots >= 1 and budget >= 1):
        return                                          # the config's checks name the value
    if any(len(vector) != n_slots or sum(vector) != budget for vector in symbols):
        encoder["symbols"] = spread_codebook(len(symbols), n_slots, budget)


def try_answer(raw: dict, section: str, key: str, value) -> tuple[dict, Config, list[str], list[str]]:
    """the config with one answer in it, slot_dt, the settings not asked about (fit_unasked) and the
    codebook fitted to it where needed, and checked by every rule. The config was valid before, so a broken
    rule is the answer's: ConfigError. The answer to 'symbols' is a count: a new count gets a new codebook,
    spread over the slots.
    Returns the new raw config, the checked one, what was fitted, and any warnings"""
    candidate = copy.deepcopy(raw)
    encoder = candidate["encoder"]
    if key == "symbols":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"the number of symbols must be a whole number, got {value!r}")
        if value < 2:
            raise ConfigError(f"a codebook needs at least 2 symbols, got {value}")
        if value != len(encoder["symbols"]):
            encoder["symbols"] = spread_codebook(value, encoder["n_slots"], encoder["budget"])
    else:
        candidate.setdefault(section, {})[key] = value          # 'search' may be left out of a config
    notes = [fit_timing(candidate) if key != "slot_dt" else "", *fit_unasked(candidate, key)]
    fit_codebook(candidate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = parse_config(candidate)
    return candidate, cfg, [note for note in notes if note], [str(warning.message) for warning in caught]


def ask_values(raw: dict, questions: list = QUESTIONS) -> tuple[dict, Config]:
    """questions about a config, one setting at a time: those of a new config (QUESTIONS), or of the search
    (SEARCH_QUESTIONS). Enter keeps the value in brackets; a typed value is checked at once with every rule
    of the config, and asked again, with the reason, if it breaks one. After the number of symbols comes the
    codebook step (ask_codebook). Returns the answered raw config and the checked one"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = parse_config(raw)
    current, shown = None, {str(warning.message) for warning in caught}    # warnings from before: not again
    for section, key, meaning in questions:
        if section != current:
            print()
            print(ui.section(section, SECTION_NOTES[section], color=color()))
            print()
            current = section
            width = max(len(other) for part, other, _ in questions if part == section)    # the keys line up
        default = cfg.encoder.n_symbols if key == "symbols" else getattr(getattr(cfg, section), key)
        while True:
            try:
                text = input(ui.value_prompt(key, meaning, f"{default:g}", width, color())).strip()
            except EOFError:                                        # Ctrl-D: stop, nothing is written
                raise KeyboardInterrupt from None
            value = typed_value(text, default) if text else default
            try:
                raw, cfg, notes, caught = try_answer(raw, section, key, value)
                break
            except ConfigError as problem:
                complain(str(problem))
        for note in notes:
            print(ui.info(note, color()))
        for message in caught:
            if message not in shown:                                # each warning once, under its answer
                shown.add(message)
                sys.stdout.flush()
                print(ui.warning(message, ui.use_color(sys.stderr)), file=sys.stderr)
        if key == "symbols":
            raw, cfg = ask_codebook(raw, cfg)
    return raw, cfg


# --- the codebook step: in 'new', and after an edit that broke the symbols ----------------------

def codebook_section(cfg: Config, how: str) -> str:
    """a proposed codebook: each symbol as it is typed (amount:slot) and when it releases, with how it
    was made in the note"""
    encoder = cfg.encoder
    return ui.section("codebook", f"{encoder.n_symbols} symbols · budget {encoder.budget} · {how}",
                      describe_symbols(cfg, typed=True), color())


def typing_help(encoder) -> str:
    """how a symbol is typed, and when each slot releases"""
    last, budget = encoder.n_slots - 1, encoder.budget
    slot = min(3, last - 1)                                         # a slot for the examples, and the next one
    split = f" · {budget // 2}:{slot} {budget - budget // 2}:{slot + 1} splits a burst" if budget > 1 else ""
    return ui.facts([("type", f"amount:slot adding up to {budget} · e.g. {budget}:{slot}{split}"),
                     ("slots", f"0 … {last} · slot k releases at k × {encoder.slot_dt:g} s "
                               f"(slot {last} at {last * encoder.slot_dt:g} s)")], color())


def type_symbols(encoder, count: int) -> tuple:
    """the codebook of the 'symbols' command, typed one symbol at a time as amount:slot, each line checked
    as it is typed (ask_symbols). Ctrl-D stops, as Ctrl-C does"""
    print(typing_help(encoder))
    print()

    def read(prompt: str) -> str:
        try:
            return input(f"{ui.INDENT}{prompt.rstrip(': ')} › ")
        except EOFError:                                            # Ctrl-D: stop, nothing is written
            raise KeyboardInterrupt from None

    def show(text: str) -> None:                                    # ask_symbols shows its rules, then any mistake
        if text.startswith(" "):
            complain(text.strip())

    return ask_symbols(count, encoder.n_slots, encoder.budget, ask=read, show=show)


def ask_codebook(raw: dict, cfg: Config) -> tuple[dict, Config]:
    """the codebook step of 'new': each symbol typed as amount:slot, checked as it is typed (parse_vector,
    and no two symbols the same). Enter keeps the symbol in brackets: the example's, or one burst per symbol
    spread over the slots once n_slots, the budget or the count changed.
    Returns the raw config with the typed codebook in it, and the checked one"""
    encoder = cfg.encoder
    print()
    print(ui.section("codebook", f"{encoder.n_symbols} symbols · budget {encoder.budget} · "
                                 "Enter keeps the one in brackets", color=color()))
    print()
    print(typing_help(encoder))
    print()
    width = len(f"symbol {encoder.n_symbols - 1}")
    symbols = []
    for u, proposed in enumerate(encoder.symbols):
        default = " ".join(f"{amount}:{k}" for k, amount in enumerate(proposed) if amount)
        while True:
            try:
                text = input(ui.value_prompt(f"symbol {u}", "", default, width, color())).strip()
            except EOFError:                                        # Ctrl-D: stop, nothing is written
                raise KeyboardInterrupt from None
            try:
                vector = parse_vector(text or default, encoder.n_slots, encoder.budget)
                if vector in symbols:
                    raise SymbolError(f"symbol {u} is the same as symbol {symbols.index(vector)}: "
                                      "no decoder could tell them apart")
                break
            except SymbolError as problem:
                complain(str(problem))
        symbols.append(vector)
    raw = copy.deepcopy(raw)
    raw["encoder"]["symbols"] = [list(vector) for vector in symbols]
    return raw, replace(cfg, encoder=replace(encoder, symbols=tuple(symbols)))


def scaled(vector: list[int], budget: int) -> list[int]:
    """amounts scaled to a new budget, adding up to it exactly: each amount gets its whole part, and the
    molecules left over go to the largest remainders (25:3 25:4, budget 60 → 30:3 30:4)"""
    total = sum(vector)
    amounts = [amount * budget // total for amount in vector]
    left = budget - sum(amounts)
    for k in sorted(range(len(vector)), key=lambda k: -(vector[k] * budget % total))[:left]:
        amounts[k] += 1
    return amounts


def fitted_symbols(symbols: list, n_slots: int, budget: int) -> tuple[list, str]:
    """a codebook fitted to new slots and a new budget, changed as little as possible: empty slots added at
    the end (or removed, if they are empty), and amounts scaled to the budget. If that can't be done, or
    makes two symbols the same, one burst per symbol spread over the slots. Returns the codebook and how"""
    fitted, changes = [], []
    for vector in symbols:
        if any(vector[n_slots:]) or not any(vector):                # molecules in slots that go, or none at all
            break
        if len(vector) != n_slots:
            changes.append("empty slots added" if len(vector) < n_slots else "empty slots removed")
            vector = vector[:n_slots] + [0] * (n_slots - len(vector))
        if sum(vector) != budget:
            changes.append("amounts scaled")
            vector = scaled(vector, budget)
        fitted.append(vector)
    else:
        if len({tuple(vector) for vector in fitted}) == len(fitted):
            return fitted, ", ".join(dict.fromkeys(changes))       # each change once, in order
    return spread_codebook(len(symbols), n_slots, budget), spread_note(len(symbols), n_slots)


def fit_after_edit(path: Path, shown: str) -> Config | None:
    """after an edit that broke only the symbols (n_slots or the budget changed): the codebook fitted to
    them (fitted_symbols), shown, and written after a yes. If fitting shows another problem, which the
    symbols hid, that is shown too (shown is the problem already on the screen).
    Returns the valid config, or None: another problem, or a no"""
    try:
        raw = read_config_file(path)
    except ConfigError:
        return None
    encoder = raw.get("encoder") if isinstance(raw, dict) else None
    if not isinstance(encoder, dict):
        return None
    n_slots, budget, symbols = encoder.get("n_slots"), encoder.get("budget"), encoder.get("symbols")

    def whole(value, least: int) -> bool:                           # a whole number (not true or false)
        return type(value) is int and value >= least

    if not (whole(n_slots, 1) and whole(budget, 1) and isinstance(symbols, list) and len(symbols) >= 2
            and all(isinstance(vector, list) and all(whole(a, 0) for a in vector) for vector in symbols)):
        return None                                                 # the error on the screen names it
    if all(len(vector) == n_slots and sum(vector) == budget for vector in symbols):
        return None                                                 # the symbols fit: the problem is elsewhere
    candidate = copy.deepcopy(raw)
    try:
        candidate["encoder"]["symbols"], how = fitted_symbols(symbols, n_slots, budget)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                         # shown when the written config is loaded
            cfg = parse_config(candidate)
    except ConfigError as problem:
        if f"{path}: {problem}" != shown:
            complain(f"{path}: {problem}")
            print()
        return None
    print(codebook_section(cfg, how))
    print()
    if not ui.ask(f"Write this codebook to {path}?"):
        print()
        return None
    write_config_file(path, candidate)
    print()
    print(ui.done("codebook written", str(path), color=color()))
    print()
    return load_config(path)


# --- commands ------------------------------------------------------------------------------------

def run_new(args) -> int:
    """mi-race new NAME: a config for a new experiment, from the example, with its own data file
    (data/NAME.csv) and output folder (outputs/NAME). At a terminal it asks for the settings that define the
    experiment, each checked as it is typed, shows the config and writes it after a yes; without one it
    copies the example. An existing file is never overwritten"""
    path = Path(args.folder) / f"{args.name}.json"
    print(ui.heading("new", str(path), color()))
    print()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        raise ConfigError(f"'{args.name}' can't be a config name: use letters, digits, - and _")
    if path.exists():
        raise ConfigError(f"{path} already exists: pick another name, or edit that file")
    raw = template()
    raw["data"]["path"] = f"data/{args.name}.csv"
    raw["output"]["dir"] = f"outputs/{args.name}"
    if not interactive():                           # a batch job: the example, checked before it is written
        cfg = parse_config(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_config_file(path, raw)
        print(describe(cfg))
        print()
        print(ui.done("config created", str(path), color=color()))
        print()
        print(ui.facts([("next", f"edit {path}"), ("", f"then: check {path}")], color()))
        return 0
    print(ui.info("Enter keeps the value in brackets · Ctrl-C stops without writing", color()))
    raw, cfg = ask_values(raw)
    print()
    show_config(cfg)
    print()
    if not ui.ask(f"Write {path}?"):
        print(ui.info(f"nothing written · {path} was not created", color()))
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_config_file(path, raw)
    print()
    print(ui.done("config created", str(path), color=color()))
    print()
    if ui.ask(f"Open {path} in {Path(editor()[0]).name} to check it?", yes=False):
        print()
        edited = edit_until_valid(path)
        if edited is None:
            return 1
        show_config(edited)
        return 0
    print()
    print(ui.facts([("next", f"simulate {path}"), ("", f"or: edit {path} · for the decoders and search")], color()))
    return 0


def run_edit(args) -> int:
    """mi-race edit CONFIG: open a config in the editor, then check it"""
    path = Path(args.config)
    print(ui.heading("edit", str(path), color()))
    print()
    if not path.exists():
        raise ConfigError(f"{path} does not exist · 'new NAME' creates a config")
    if not interactive():
        raise OSError("edit needs a terminal to open the editor in")
    edited = edit_until_valid(path)
    if edited is None:
        return 1
    show_config(edited)
    return 0


def run_symbols(args) -> int:
    """mi-race symbols CONFIG: type a new codebook, one release vector per symbol, checked as it is typed;
    the config is only changed after a yes, and only if the whole config is still valid"""
    path = Path(args.config)
    print(ui.heading("symbols", str(path), color()))
    print()
    cfg = load_config(path)
    if not interactive():
        raise OSError("symbols needs a terminal to type the symbols in")
    encoder = cfg.encoder
    count = args.count or encoder.n_symbols
    if count < 2:
        raise ConfigError(f"a codebook needs at least 2 symbols, got {count}")
    print(ui.section("current codebook", f"{encoder.n_symbols} symbols · budget {encoder.budget}",
                     describe_symbols(cfg, typed=True), color()))
    print()
    symbols = type_symbols(encoder, count)
    print()
    print(ui.section("new codebook", f"{count} symbols · budget {encoder.budget}",
                     describe_symbols(cfg, symbols, typed=True), color()))
    print()
    if not ui.ask(f"Write these symbols to {path}?"):
        print(ui.info(f"nothing written · {path} is unchanged", color()))
        return 0
    write_symbols(path, symbols)
    print()
    print(ui.done("symbols written", f"{count} symbols", str(path), color=color()))
    return 0


def size(path: Path) -> str:
    """a file's size, readable: '1.2 MB'"""
    count = path.stat().st_size
    for unit in ("bytes", "kB", "MB"):
        if count < 1000 or unit == "MB":
            return f"{count:.0f} {unit}" if unit == "bytes" else f"{count:.1f} {unit}"
        count /= 1000


def simulate_and_write(cfg: Config) -> np.ndarray:
    """simulate the dataset with a progress bar, then save it to data.path"""
    with ui.Progress("simulating", "dataset simulated", unit="runs") as bar:
        rows = simulate_dataset(cfg, bar)
    write_dataset(cfg.data.path, rows)
    return rows


def dataset_for(cfg: Config, config_path: str) -> np.ndarray:
    """the dataset of a config: read from data.path, or simulated after a yes when it is missing,
    or when the config was changed after the dataset was simulated"""
    path = Path(cfg.data.path)
    if not path.exists():
        if not ui.ask(f"{path} does not exist yet. Simulate it now?"):
            raise DatasetError(f"no dataset · 'simulate {config_path}' makes it")
        print()
        return simulate_and_write(cfg)
    if Path(config_path).stat().st_mtime > path.stat().st_mtime:
        if ui.ask(f"{config_path} was changed after {path} was simulated. Simulate it again?"):
            print()
            return simulate_and_write(cfg)
        print()
    rows = read_dataset(path)
    check_dataset(rows, cfg)
    print(ui.done("dataset loaded", f"{len(rows)} runs", str(path), color=color()))
    return rows


def comparison(results: dict) -> str:
    """every decoder in one table: plain accuracy, hard MI and BA bound, then T and the rescaled BA bound;
    the best rescaled bound in bold"""
    header = ["decoder", "accuracy", "hard MI", "BA bound", "T", "BA rescaled"]
    rows = [[DECODER_NAMES[name], f"{r['plain']['accuracy']:.1%}", f"{r['plain']['hard_decision_mi_bits']:.3f}",
             f"{r['plain']['ba_bound_bits']:.3f}", f"{r['rescaled']['temperature']:.2f}",
             f"{r['rescaled']['ba_bound_bits']:.3f}"] for name, r in results.items()]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    best = max(r["rescaled"]["ba_bound_bits"] for r in results.values())

    def line(cells: list[str], style: str = "", bold_last: bool = False) -> str:
        text = [cells[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(cells[1:], widths[1:])]
        if bold_last:
            text[-1] = ui.paint(text[-1], ui.BOLD, color())
        return ui.INDENT + (ui.paint("   ".join(text), style, color()) if style else "   ".join(text))

    return "\n".join([line(header, ui.DIM)] + [line(row, bold_last=row[-1] == f"{best:.3f}") for row in rows])


def run_simulate(args) -> int:
    """mi-race simulate CONFIG: run the channel for every symbol, as many times as the config says,
    and save the receiver signals as the dataset (CSV). The same config always gives the same dataset"""
    print(ui.heading("simulate", args.config, color()))
    print()
    cfg = load_config(args.config)
    encoder, data = cfg.encoder, cfg.data
    path = Path(data.path)
    print(ui.facts([("runs", f"{data.runs_per_symbol} per symbol · {encoder.n_symbols} symbols · "
                             f"{data.runs_per_symbol * encoder.n_symbols} in all"),
                    ("signal", f"{cfg.channel.n_intervals + 1} values per run · one every {cfg.channel.dt:g} s"),
                    ("seed", str(cfg.seed)),
                    ("file", str(path))], color()))
    print()
    if path.exists():
        if not ui.ask(f"{path} already exists. Simulate it again and replace it?"):
            print(ui.info(f"nothing simulated · {path} is unchanged", color()))
            return 0
        print()
    rows = simulate_and_write(cfg)
    absorbed = rows[:, 1:].sum(axis=1)
    print()
    print(ui.section("absorbed by the receiver", f"mean over {data.runs_per_symbol} runs, of {encoder.budget} released",
                     ui.facts([(f"symbol {u}", f"{absorbed[rows[:, 0] == u].mean():.1f}  "
                                               f"({absorbed[rows[:, 0] == u].mean() / encoder.budget:.0%})")
                               for u in range(encoder.n_symbols)], color()), color()))
    print()
    print(ui.done("dataset written", f"{len(rows)} runs", size(path), color=color()))
    print(ui.facts([("file", str(path.resolve()))], color()))
    return 0


def run_run(args) -> int:
    """mi-race run CONFIG: train every decoder of the config on the training runs, plain and rescaled
    (T fitted on a fifth of the training runs), and score them on the test runs. The results, with the
    config, seed and version, go to output.dir/metrics.json"""
    print(ui.heading("run", args.config, color()))
    print()
    cfg = load_config(args.config)
    if not cfg.decoders:
        raise ConfigError(f"{args.config} has no decoders: add one to its 'decoders' section")
    data = cfg.data
    n_test = round(data.runs_per_symbol * data.test_fraction) * cfg.encoder.n_symbols
    output = Path(cfg.output.dir) / "metrics.json"
    print(ui.facts([("dataset", data.path),
                    ("split", f"{data.runs_per_symbol * cfg.encoder.n_symbols - n_test} training runs · "
                              f"{n_test} test runs · seed {cfg.seed}"),
                    ("decoders", " · ".join(DECODER_NAMES[name] for name in cfg.decoders) + " · plain and rescaled"),
                    ("results", str(output))], color()))
    print()
    rows = dataset_for(cfg, args.config)
    train, test = split_dataset(rows, data.test_fraction, random_stream(cfg.seed, "split"))
    X, y, X_test, y_test = train[:, 1:], train[:, 0], test[:, 1:], test[:, 0]
    results = {}
    for name, settings in cfg.decoders.items():
        title = DECODER_NAMES[name]
        with ui.Progress(title, f"{title} trained", unit=UNITS[name]) as bar:
            model = TRAINERS[name](X, y, settings, random_stream(cfg.seed, name), progress=bar)
        with ui.Progress("rescaled", f"{title} rescaled", unit=UNITS[name]) as bar:
            rescaled = train_rescaled(TRAINERS[name], X, y, settings, random_stream(cfg.seed, name), progress=bar)
        results[name] = {"settings": asdict(settings),
                         "plain": evaluate(y_test, model.predict_proba(X_test)),
                         "rescaled": evaluate(y_test, rescaled.predict_proba(X_test)) | {"temperature": rescaled.temperature},
                         "training_loss": getattr(model, "losses", None)}
    for name, result in results.items():
        plain, rescaled = result["plain"], result["rescaled"]
        print()
        print(ui.section(DECODER_NAMES[name], f"{plain['n_test']} test runs", ui.confusion_table(plain, color()), color()))
        print()
        print(ui.facts([("hard MI", f"{plain['hard_decision_mi_bits']:.3f} bits · reads about "
                                    f"{plain['mi_bias_bits']:.3f} high on {plain['n_test']} runs"),
                        ("rescaled", f"BA bound {rescaled['ba_bound_bits']:.3f} bits · T {rescaled['temperature']:.2f} · "
                                     f"accuracy {rescaled['accuracy']:.1%}")], color()))
    print()
    print(ui.section("all decoders", "bits · T fitted on held-back training runs",
                     comparison(results), color()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"mi_race": __version__, "created": datetime.now().isoformat(timespec="seconds"),
                                  "config_file": args.config, "config": read_config_file(args.config),
                                  "seed": cfg.seed, "dataset": data.path, "n_train": len(train), "n_test": len(test),
                                  "decoders": results}, indent=2) + "\n", encoding="utf-8")
    print()
    print(ui.done("results saved", f"{len(results)} decoders", size(output), color=color()))
    print(ui.facts([("file", str(output.resolve()))], color()))
    return 0


def run_check(args) -> int:
    """mi-race check CONFIG: every rule of the config, then what it sets up"""
    print(ui.heading("check", args.config, color()))
    print()
    show_config(load_config(args.config))          # a warning about the config appears here, under the heading
    return 0


# --- the guided search ---

def codebook_text(symbols) -> str:
    """a codebook on one line, as symbols are typed: '50:0 · 50:3 · 25:5 25:6 · 50:9'"""
    return " · ".join(" ".join(f"{amount}:{k}" for k, amount in enumerate(vector) if amount) for vector in symbols)


def wrap_codebook(text: str, room: int) -> list[str]:
    """a codebook_text cut into lines of at most room columns, only between symbols"""
    lines = [""]
    for symbol in text.split(" · "):
        joined = f"{lines[-1]} · {symbol}" if lines[-1] else symbol
        if len(joined) <= room or not lines[-1]:
            lines[-1] = joined
        else:
            lines.append(symbol)
    return lines


def write_json(path: Path, data: dict) -> None:
    """a result file: indented, each list of plain numbers on one line, numpy values as plain ones"""
    text = json.dumps(data, indent=2, ensure_ascii=False, default=lambda value: value.tolist())
    text = re.sub(r"\[\n\s+([-\d.,\s]+?)\n\s+\]", lambda m: "[" + " ".join(m[1].split()) + "]", text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def feedback(cfg: Config):
    """the score of the search: the BA bound of the config's search decoder, out of fold"""
    return partial(score_codebook, decoder=cfg.search.decoder, rescaled=cfg.search.rescaled)


def move_text(parent, moved) -> str:
    """a move in words: 'symbol 1 · 10 molecules from slot 3 to slot 4'"""
    u = next(u for u, (a, b) in enumerate(zip(parent, moved)) if tuple(a) != tuple(b))
    before, after = parent[u], moved[u]
    source = next(k for k in range(len(before)) if after[k] < before[k])
    target = next(k for k in range(len(before)) if after[k] > before[k])
    return f"symbol {u} · {before[source] - after[source]} molecules from slot {source} to slot {target}"


def set_threshold(cfg: Config, config_path: str, log: DecisionLog) -> tuple[Config, dict]:
    """step 1: measure the noise of the search's feedback and propose θ = 2 × sd, written to the config
    after a yes. Returns the config the search runs with, and the noise measurement"""
    current = cfg.search.threshold
    with ui.Progress("noise", "noise measured", unit="pairs") as bar:
        noise = measure_noise(cfg, feedback(cfg), progress=bar)
    proposed = noise["threshold"]
    print()
    print(ui.section("noise of the feedback", "a nearly neutral move against its parent", ui.facts([
        ("move", move_text(noise["parent"], noise["moved"])),
        ("noise", f"sd {noise['sd']:.4f} bits on {len(noise['differences'])} paired datasets · "
                  f"mean {noise['mean']:+.4f} bits"),
        ("threshold", f"{current:g} bits now · 2 × sd ≈ {proposed:g} bits")], color()), color()))
    print()
    if proposed == current:
        print(ui.done("threshold kept", f"{current:g} bits is 2 × sd", color=color()))
        return cfg, noise
    evidence = {"move": move_text(noise["parent"], noise["moved"]), "pairs": len(noise["differences"]),
                "sd_bits": noise["sd"], "mean_bits": noise["mean"], "threshold_now": current}
    if not log.ask("threshold", f"Set search.threshold in {config_path} to {proposed:g} bits?",
                   evidence=evidence, proposal={"search.threshold": proposed}):
        print(ui.info(f"threshold kept · {current:g} bits", color()))
        return cfg, noise
    raw = read_config_file(config_path)
    raw.setdefault("search", {})["threshold"] = proposed
    write_config_file(config_path, raw)
    print()
    print(ui.done("threshold set", f"{proposed:g} bits", config_path, color=color()))
    return replace(cfg, search=replace(cfg.search, threshold=proposed)), noise


def search_run(cfg: Config, run: dict, steps: int, label: str, finished: str) -> None:
    """one run searched up to 'steps' steps in all, on from where its history ends, with a progress bar;
    the run's record gets the codebook it ends with, the history, the steps and the time"""
    settings = cfg.search
    began = time.monotonic()
    with ui.Progress(label, finished, unit="steps") as bar:
        final, history = search(cfg, cfg.encoder.symbols, steps, run=run["run"], score=feedback(cfg),
                                chunk=settings.chunk, threshold=settings.threshold, progress=bar,
                                history=run["history"])
    run.update(steps=steps, final=final, history=history,
               seconds=round(run["seconds"] + time.monotonic() - began, 1))


def search_runs(cfg: Config, output: Path, record: dict) -> list:
    """step 2: the search runs from the config's codebook, one after another; search.json is saved
    after each, so a search stopped early keeps the runs that ended"""
    settings = cfg.search
    for r in range(settings.runs):
        run = {"run": r, "steps": 0, "final": cfg.encoder.symbols, "history": [], "seconds": 0.0}
        label = f"run {r + 1} of {settings.runs}"
        search_run(cfg, run, settings.steps, label, label)
        record["runs"].append(run)
        write_json(output, record)
    return record["runs"]


def kept_steps(run: dict) -> list:
    """the steps of a run that kept their move"""
    return [row["step"] for row in run["history"] if row["accepted"]]


def still_improving(run: dict, window: int) -> bool:
    """a run that kept a move in its last 'window' steps, and did all its steps (a legal move was left)"""
    kept = kept_steps(run)
    return len(run["history"]) == run["steps"] and bool(kept) and kept[-1] >= run["steps"] - window


def run_names(runs: list) -> str:
    """'run 2', 'runs 2 and 4', 'runs 1, 2 and 4': the runs as the table numbers them"""
    numbers = [str(run["run"] + 1) for run in runs]
    if len(numbers) == 1:
        return f"run {numbers[0]}"
    return f"runs {', '.join(numbers[:-1])} and {numbers[-1]}"


def more_steps(cfg: Config, config_path: str, output: Path, record: dict, log: DecisionLog) -> list:
    """step 3: a run that kept a move in the last quarter of search.steps is still improving. While runs are,
    propose to search on with them, by half of search.steps, up to twice search.steps in all"""
    settings = cfg.search
    window, more, most = max(1, settings.steps // 4), max(1, settings.steps // 2), 2 * settings.steps
    last = f"last {window} steps" if window > 1 else "last step"
    while True:
        runs = record["runs"]
        print(ui.section("runs", f"θ {settings.threshold:g} bits · still improving: a move kept in the {last}",
                         runs_table(runs), color()))
        print()
        improving = [run for run in runs if still_improving(run, window)]
        if not improving:
            print(ui.done("runs settled", f"no run kept a move in its {last}", color=color()))
            return runs
        steps = improving[0]["steps"]                   # every run still improving has gone on as far
        if steps >= most:
            print(ui.warning(f"{run_names(improving)} still improving at {steps} steps, twice search.steps: "
                             f"to search longer, raise search.steps in {config_path}", color()))
            return runs
        to = min(steps + more, most)
        evidence = {"window": window, "runs": [{"run": run["run"], "steps": run["steps"], "kept": len(kept_steps(run)),
                                                "last_kept": kept_steps(run)[-1]} for run in improving]}
        if not log.ask("steps", f"{run_names(improving).capitalize()} kept a move in the {last}: "
                                f"go on to {to} steps?", evidence=evidence,
                       proposal={"runs": [run["run"] for run in improving], "steps": to}):
            print(ui.info(f"{run_names(improving)} stopped at {steps} steps", color()))
            return runs
        print()
        for run in improving:
            label = f"run {run['run'] + 1} of {settings.runs}"
            search_run(cfg, run, to, label, f"{label} continued")
            write_json(output, record)
        print()


def runs_table(runs: list) -> str:
    """one line per run: its steps, the moves it kept, the step of the last one, and the codebook it ended with"""
    header = ["run", "steps", "kept", "last kept", "codebook"]
    rows = []
    for run in runs:
        kept = kept_steps(run)
        rows.append([str(run["run"] + 1), str(len(run["history"])), str(len(kept)),
                     f"step {kept[-1] + 1}" if kept else "–", codebook_text(run["final"])])
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(4)]
    left = len(ui.INDENT) + sum(widths) + 4 * 3                     # where the codebook starts
    lines = [ui.INDENT + ui.paint("   ".join([h.rjust(w) for h, w in zip(header, widths)] + [header[4]]),
                                  ui.DIM, color())]
    for row in rows:
        codebook = wrap_codebook(row[4], ui.WIDTH - left)
        lines.append(ui.INDENT + "   ".join([cell.rjust(w) for cell, w in zip(row, widths)] + [codebook[0]]))
        lines += [" " * left + more for more in codebook[1:]]
    return "\n".join(lines)


def ask_search_settings(cfg: Config, config_path: str) -> Config:
    """step 0: the search settings (SEARCH_QUESTIONS), each checked as it is typed. Changed ones are saved
    to the config, so it always describes the search that ran. Returns the config to search with"""
    print(ui.info("Enter keeps the value in brackets · Ctrl-C stops without searching", color()))
    raw, answered = ask_values(read_config_file(config_path), SEARCH_QUESTIONS)
    print()
    changed = [f"{key} {getattr(answered.search, key)}" for _, key, _ in SEARCH_QUESTIONS
               if getattr(answered.search, key) != getattr(cfg.search, key)]
    if changed:
        write_config_file(config_path, raw)
        print(ui.done("search settings saved", *changed, config_path, color=color()))
        print()
    return answered


def run_search(args) -> int:
    """mi-race search CONFIG: the guided encoder search. The tool proposes every change, shows the evidence
    and asks yes/no; a no keeps what is there. The questions, evidence and answers go to
    output.dir/decisions.json, the runs to output.dir/search.json.
    Step 0: the settings, typed (at a terminal, without --yes). Step 1: the threshold, from the noise of the
    feedback. Step 2: the runs. Step 3: more steps for the runs still improving"""
    print(ui.heading("search", args.config, color()))
    print()
    cfg = load_config(args.config)
    output = Path(cfg.output.dir) / "search.json"
    log = DecisionLog(cfg.output.dir, "search", args.config, accept_defaults=args.yes)
    if output.exists():
        if not log.ask("replace", f"{output} already holds a search. Search again and replace it?", yes=False):
            print(ui.info(f"nothing searched · {output} is unchanged", color()))
            return 0
        print()
    if interactive() and not args.yes:
        cfg = ask_search_settings(cfg, args.config)
    settings = cfg.search
    start = wrap_codebook(codebook_text(cfg.encoder.symbols), ui.WIDTH - len(ui.INDENT) - len("feedback  "))
    print(ui.facts([("feedback", f"{'rescaled ' if settings.rescaled else ''}{DECODER_NAMES[settings.decoder]} · "
                                 f"BA bound on a new dataset each step · {FOLDS}-fold"),
                    ("moves", f"{settings.chunk} molecules · kept if the gain is above θ {settings.threshold:g} bits"),
                    ("runs", f"{settings.runs} · {settings.steps} steps each, "
                             f"up to {2 * settings.steps} while still improving"),
                    ("start", start[0]), *[("", line) for line in start[1:]],
                    ("results", f"{output} · {log.path}")], color()))
    print()
    cfg, noise = set_threshold(cfg, args.config, log)
    print()
    record = {"mi_race": __version__, "created": datetime.now().isoformat(timespec="seconds"),
              "config_file": args.config, "config": read_config_file(args.config), "seed": cfg.seed,
              "settings": asdict(cfg.search), "noise": noise, "runs": []}
    write_json(output, record)
    search_runs(cfg, output, record)
    print()
    runs = more_steps(cfg, args.config, output, record, log)
    print()
    print(ui.done("search saved", f"{len(runs)} runs", size(output), color=color()))
    print(ui.facts([("file", str(output.resolve()))], color()))
    return 0


# --- the command line ---

class CommandLineError(Exception):
    """a command typed wrong: unknown command, a missing or an extra argument"""


class Parser(argparse.ArgumentParser):
    """argparse, but a typing mistake becomes a CommandLineError instead of ending the program,
    so the shell can show it in our style and wait for the next command"""

    def error(self, message):
        raise CommandLineError(message)


def build_parser() -> Parser:
    """mi-race COMMAND ...: one subcommand per task"""
    parser = Parser(prog="mi-race", description=ui.DESCRIPTION)
    parser.add_argument("--version", action="version", version=f"mi-race {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    new = commands.add_parser("new", help="create a config for a new experiment, from the example")
    new.add_argument("name", metavar="NAME", help="the experiment's name: letters, digits, - and _")
    new.add_argument("--folder", default="configs", help="where the config goes (default: configs)")
    new.set_defaults(run=run_new)

    edit = commands.add_parser("edit", help="open a config in the editor (nano, or $EDITOR), then check it")
    edit.add_argument("config", metavar="CONFIG", help="the config file, e.g. configs/my-channel.json")
    edit.set_defaults(run=run_edit)

    symbols = commands.add_parser("symbols", help="type a new codebook into a config, symbol by symbol")
    symbols.add_argument("config", metavar="CONFIG", help="the config file, e.g. configs/my-channel.json")
    symbols.add_argument("--count", type=int, help="how many symbols (default: as many as the config has)")
    symbols.set_defaults(run=run_symbols)

    simulate = commands.add_parser("simulate", help="simulate the dataset: every symbol through the channel")
    simulate.add_argument("config", metavar="CONFIG", help="the config file, e.g. configs/my-channel.json")
    simulate.set_defaults(run=run_simulate)

    run = commands.add_parser("run", help="train and score every decoder, plain and rescaled; save metrics.json")
    run.add_argument("config", metavar="CONFIG", help="the config file, e.g. configs/my-channel.json")
    run.set_defaults(run=run_run)

    guided = commands.add_parser("search", help="the guided encoder search: it proposes, you answer yes or no")
    guided.add_argument("config", metavar="CONFIG", help="the config file, e.g. configs/my-channel.json")
    guided.add_argument("--yes", action="store_true", help="take every default answer, e.g. in a batch job")
    guided.set_defaults(run=run_search)

    check = commands.add_parser("check", help="check a config against every rule and show what it sets up")
    check.add_argument("config", metavar="CONFIG", help="the config file, e.g. configs/example.json")
    check.set_defaults(run=run_check)
    return parser


def subcommands(parser: Parser) -> argparse._SubParsersAction:
    """the commands of the parser (check, ...)"""
    return next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))


def complain(text: str) -> None:
    """one '✗ error' line on stderr, after everything printed so far"""
    sys.stdout.flush()
    print(ui.error(text, ui.use_color(sys.stderr)), file=sys.stderr)


def command_list(parser: Parser) -> str:
    """every command with its arguments, in bold, and what it does, faint"""
    commands = subcommands(parser)
    rows = []
    for choice in commands._choices_actions:
        arguments = [action.metavar or action.dest.upper() for action in commands.choices[choice.dest]._actions
                     if not action.option_strings]
        rows.append((" ".join([choice.dest, *arguments]), choice.help))
    rows += [("help", "show these commands"), ("exit", "leave mi-race")]
    width = max(len(command) for command, _ in rows)
    return "\n".join(ui.INDENT + ui.paint(command.ljust(width), ui.BOLD, color()) + "  " + ui.paint(text, ui.DIM, color())
                     for command, text in rows)


def run_command(parser: Parser, words: list[str]) -> int:
    """parse one command and run it: a problem becomes one '✗ error' line. Returns the exit code"""
    try:
        if not words[0].startswith("-") and words[0] not in subcommands(parser).choices:
            raise CommandLineError(f"unknown command '{words[0]}'")
        args = parser.parse_args(words)
        if args.command is None:
            raise CommandLineError("type a command, e.g. 'check configs/example.json'")
        return args.run(args)
    except CommandLineError as problem:
        complain(f"{problem} · 'help' lists the commands")
        return 2
    except (ConfigError, DatasetError, SymbolError, DecisionLogError, OSError) as problem:
        complain(str(problem))
        return 1
    except KeyboardInterrupt:
        print()
        complain("stopped")
        return 130
    except SystemExit as stop:                                      # --help and --version end here, not the shell
        return stop.code if isinstance(stop.code, int) else 0  


def line_editing() -> str | None:
    """turn on line editing: ↑ recalls earlier commands, also from earlier sessions (~/.mi_race_history).
    Returns the backend: 'readline' (GNU), 'editline' (macOS) or None (no line editing, e.g. plain Windows)"""
    try:
        import readline
    except ImportError:
        return None
    history = Path.home() / ".mi_race_history"
    try:
        readline.read_history_file(history)
    except OSError:
        pass
    readline.set_history_length(1000)
    atexit.register(lambda: readline.write_history_file(history))
    return getattr(readline, "backend", "editline" if "libedit" in (readline.__doc__ or "") else "readline")


def prompt(backend: str | None) -> str:
    """'  [mi-race] › ' with [mi-race] in bold cyan: GNU readline needs the style codes marked with
    \\001 \\002 so they take no room; editline (macOS) and plain input() take them as they are"""
    if not color():
        return f"{ui.INDENT}{ui.PREFIX} › "
    if backend == "readline":
        return f"{ui.INDENT}\001{ui.BRAND}\002{ui.PREFIX}\001{ui.RESET}\002 › "
    return f"{ui.INDENT}{ui.paint(ui.PREFIX, ui.BRAND, True)} › "


def shell(parser: Parser) -> int:
    """the mi-race shell: type commands one after another until 'exit' (or Ctrl-D)"""
    ask = prompt(line_editing())
    print(ui.banner(color=color()))
    print()
    print(ui.section("commands", "type one below · ↑ recalls earlier ones", command_list(parser), color()))
    print()
    while True:
        try:
            line = input(ask).strip()
        except EOFError:                                            # Ctrl-D
            print()
            break
        except KeyboardInterrupt:                                   # Ctrl-C: a new prompt
            print()
            continue
        words = shlex.split(line) if line.count('"') % 2 == 0 and line.count("'") % 2 == 0 else line.split()
        if words[:1] == ["mi-race"]:                                # 'mi-race check …' works too
            words = words[1:]
        if not words:
            continue
        if words[0] in ("exit", "quit"):
            break
        if words[0] in ("help", "?"):
            print(command_list(parser))
        else:
            run_command(parser, words)
        print()
    print(ui.info("bye", color()))
    return 0


def main(argv: list[str] | None = None) -> int:
    """the mi-race command. With no arguments it opens the mi-race shell; with a command
    (mi-race check CONFIG) it runs just that, for scripts and batch jobs.
    Returns the exit code: 0 fine, 1 a problem, 2 a command typed wrong, 130 stopped with Ctrl-C"""
    words = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    errors = ui.use_color(sys.stderr)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = lambda message, *rest: (sys.stdout.flush(),
                                                       print(ui.warning(str(message), errors) + "\n", file=sys.stderr))
        if not words:
            return shell(parser)
        if ui.is_terminal(sys.stdout) and words[0] not in ("-h", "--help", "--version"):
            print(ui.banner(next((word for word in words[1:] if word.endswith(".json")), None), color=color()))
            print()
        return run_command(parser, words)