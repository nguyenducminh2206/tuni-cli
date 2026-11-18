"""cli: the mi-race command. Typed alone, `mi-race` opens the mi-race shell, where commands are typed one
after another; `mi-race COMMAND ...` runs one command, for scripts and batch jobs.
Each command is a function run_<name>(args) that returns the exit code. Warnings appear as
'[mi-race] ! warning' lines as they happen; a broken config, dataset or file becomes one
'[mi-race] ✗ error' line"""

import argparse
import atexit
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import warnings
from importlib import resources
from pathlib import Path

from mi_race import __version__, ui
from mi_race.config import Config, ConfigError, load_config, parse_config, write_config_file
from mi_race.dataset import DatasetError
from mi_race.symbols import SymbolError

DECODER_NAMES = {"random_forest": "random forest", "mlp": "MLP", "cnn": "CNN"}


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
        ("seed", str(cfg.seed)),
        ("output", cfg.output.dir),
    ], color())


def describe_symbols(cfg: Config) -> str:
    """each symbol's releases: '50 at 0.3 s', or several for a split burst"""
    encoder = cfg.encoder
    return ui.facts([(f"symbol {u}", " · ".join(f"{amount} at {k * encoder.slot_dt:g} s"
                                                for k, amount in enumerate(vector) if amount))
                     for u, vector in enumerate(encoder.symbols)], color())


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
    """open the config in the editor, and check it when the editor closes;
    a broken config can be opened again to fix it. Returns the valid config, or None"""
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
            if not ui.ask("Open it again to fix it?"):
                return None
            print()


# --- commands ------------------------------------------------------------------------------------

def run_new(args) -> int:
    """mi-race new NAME: a config for a new experiment, copied from the example, with its own data file
    (data/NAME.csv) and output folder (outputs/NAME). It is checked before it is written, and an existing
    file is never overwritten"""
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
    cfg = parse_config(raw)                         # every rule passes before anything is written
    path.parent.mkdir(parents=True, exist_ok=True)
    write_config_file(path, raw)
    print(describe(cfg))
    print()
    print(ui.done("config created", str(path), color=color()))
    print()
    if interactive() and ui.ask(f"Open {path} in {Path(editor()[0]).name} to set the channel, symbols and decoders?"):
        print()
        edited = edit_until_valid(path)
        if edited is None:
            return 1
        show_config(edited)
        return 0
    print(ui.facts([("next", f"edit {path}"), ("", f"then: check {path}")], color()))
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


def run_check(args) -> int:
    """mi-race check CONFIG: every rule of the config, then what it sets up"""
    print(ui.heading("check", args.config, color()))
    print()
    show_config(load_config(args.config))          # a warning about the config appears here, under the heading
    return 0


# --- the command line ----------------------------------------------------------------------------

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
    except (ConfigError, DatasetError, SymbolError, OSError) as problem:
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