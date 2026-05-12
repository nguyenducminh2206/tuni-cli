import argparse
import os
import shlex
import subprocess
import sys
from typing import List, Optional, Sequence
from importlib.metadata import version, PackageNotFoundError

from mi_race.train.orchestrator import run_cmd
from mi_race.train.registry import SUPPORTED_MODELS
from mi_race.data_preprocessing.concat import run_concat_csv_files
from mi_race.data_preprocessing.filter_csv import run_filter_csv
from mi_race.reporting.compare_models import run_compare
from mi_race.encoder.dataset_gen import run_generate_data
from mi_race.cli.ui import render_box


def _pkg_version() -> str:
    for name in ("mi-race", "mi_race"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "0.0.0"


def _banner_suppressed(args: argparse.Namespace) -> bool:
    if getattr(args, "no_banner", False):
        return True
    return os.environ.get("MI_RACE_NO_LOGO", "").lower() in ("1", "true", "yes")


def _print_banner(args: argparse.Namespace) -> None:
    """Print ASCII art header and a version box unless suppressed."""
    if _banner_suppressed(args):
        return

    ascii_header = r"""


    ███╗   ███╗██╗      ██████╗  █████╗  ██████╗███████╗
    ████╗ ████║██║      ██╔══██╗██╔══██╗██╔════╝██╔════╝
    ██╔████╔██║██║█████╗██████╔╝███████║██║     █████╗
    ██║╚██╔╝██║██║╚════╝██╔══██╗██╔══██║██║     ██╔══╝
    ██║ ╚═╝ ██║██║      ██║  ██║██║  ██║╚██████╗███████╗
    ╚═╝     ╚═╝╚═╝      ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝

""".strip("\n")

    version_str = _pkg_version()
    lines: List[str] = [
        f"Version  v{version_str}",
        f"Config   {getattr(args, 'config', 'config.json')}",
    ]
    box = render_box(lines, title="mi-race", min_width=34)

    print(ascii_header)
    print(box)


def _open_editor(path: str) -> None:
    """Open ``path`` in the user's preferred editor.

    Lookup order: ``$VISUAL`` → ``$EDITOR`` → platform default (``notepad`` on
    Windows, ``nano`` elsewhere). The env vars may contain arguments, e.g.
    ``EDITOR="code -w"``.
    """
    abs_path = os.path.abspath(path)
    editor_cmd = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor_cmd:
        cmd = shlex.split(editor_cmd) + [abs_path]
    elif os.name == "nt":
        cmd = ["notepad", abs_path]
    else:
        cmd = ["nano", abs_path]
    subprocess.run(cmd, check=False)


def _rewrite_legacy_argv(argv: List[str]) -> List[str]:
    """Translate deprecated command spellings to the current form.

    Currently handles ``concat csv-files ...`` → ``concat ...`` with a warning.
    """
    out = list(argv)
    try:
        i = out.index("concat")
    except ValueError:
        return out
    if i + 1 < len(out) and out[i + 1] == "csv-files":
        print(
            "[mi-race][WARN] 'concat csv-files' is deprecated; use 'concat' directly.",
            file=sys.stderr,
        )
        del out[i + 1]
    return out


def _run_shell(parser: argparse.ArgumentParser, *, startup_args: argparse.Namespace) -> None:
    """Interactive shell that reuses the existing argparse parser and handlers."""
    print("Hello from mi-race! Pick a command below.\n")
    parser.print_help()
    print("\nType 'help' to see this again, 'exit' to quit.")
    print("Type 'edit' or 'edit config.json' to edit your config.\n")
    print("Type 'filter' to filter rows from your concat CSV.\n")

    while True:
        try:
            line = input("\033[32mmi-race> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[mi-race] Bye.")
            return

        if not line:
            continue

        low = line.lower()
        if low in {"exit", "quit"}:
            print("[mi-race] Bye.")
            return
        if low in {"help", "?"}:
            parser.print_help()
            continue

        if low == "edit":
            _open_editor("config.json")
            continue
        if low.startswith("edit "):
            target = line.split(" ", 1)[1].strip().strip('"')
            _open_editor(target or "config.json")
            continue

        if low.startswith("mi-race "):
            line = line[len("mi-race ") :].lstrip()

        try:
            tokens = shlex.split(line, posix=(os.name != "nt"))
        except ValueError as e:
            print(f"[mi-race][shell] Parse error: {e}")
            continue

        tokens = _rewrite_legacy_argv(tokens)

        try:
            args = parser.parse_args(tokens)
        except SystemExit:
            continue

        if getattr(args, "cmd", None) is None:
            print("[mi-race][shell] Please enter a command (e.g., 'run', 'run-all', 'compare', 'concat').")
            continue

        try:
            args.func(args)
        except SystemExit as e:
            if str(e):
                print(str(e))
        except Exception as e:
            print(f"[mi-race][shell][ERROR] {type(e).__name__}: {e}")


def _run_all(args: argparse.Namespace) -> None:
    """Iterate through supported models and invoke orchestrator per model.

    Failures in one model are caught and reported; remaining models still run.
    A summary table is printed at the end.
    """
    results: List[tuple[str, str, str]] = []  # (model, "PASS"|"FAIL", message)
    for m in SUPPORTED_MODELS:
        ns = argparse.Namespace(**{**vars(args), "model": m})
        print(f"\n[mi-race] ===== Running all: {m} =====")
        try:
            run_cmd(ns)
        except SystemExit as e:
            msg = str(e) or "SystemExit"
            print(f"[mi-race][run-all][ERROR] {m}: {msg}")
            results.append((m, "FAIL", msg))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[mi-race][run-all][ERROR] {m}: {msg}")
            results.append((m, "FAIL", msg))
        else:
            results.append((m, "PASS", ""))

    print("\n" + ("=" * 20) + " run-all summary " + ("=" * 20))
    name_w = max((len(m) for m, _, _ in results), default=0)
    for m, status, msg in results:
        line = f"  {m.ljust(name_w)}  {status}"
        if msg:
            line += f"  {msg}"
        print(line)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser. Extracted so tests can introspect it."""
    parser = argparse.ArgumentParser(
        prog="mi-race",
        description="mi-race CLI (machine learning for science).",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"mi-race {_pkg_version()}"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="suppress the startup ASCII banner (same as MI_RACE_NO_LOGO=1)",
    )

    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="train/evaluate a model from json config")
    p_run.add_argument(
        "--model",
        choices=list(SUPPORTED_MODELS),
        help=f"model selection ({', '.join(repr(m) for m in SUPPORTED_MODELS)}).",
    )
    p_run.add_argument("-c", "--config", default="config.json", help="config json path")
    p_run.set_defaults(func=run_cmd)

    p_run_all = sub.add_parser(
        "run-all", help="train/evaluate all supported models using the same config"
    )
    p_run_all.add_argument("-c", "--config", default="config.json", help="config json path")
    p_run_all.set_defaults(func=_run_all)

    p_cmp = sub.add_parser(
        "compare",
        help="plot overall accuracy and accuracy vs noise from outputs/summary_models.csv",
    )
    p_cmp.add_argument(
        "--split", help="restrict compare to a single split prefix (e.g., 'noise' or 'kcross')"
    )
    p_cmp.set_defaults(func=run_compare)

    p_concat = sub.add_parser(
        "concat",
        help="if data.path is a folder: concat CSVs (auto-detects nested subfolders); if it's a CSV file: pass through; writes <name>_concat.csv",
    )
    p_concat.add_argument("-c", "--config", default="config.json", help="config json path")
    p_concat.add_argument("--out", help="output csv path override")
    p_concat.add_argument("--pattern", default="*.csv", help="glob pattern (default: *.csv)")
    p_concat.add_argument("--recursive", action="store_true", help="search subfolders")
    p_concat.set_defaults(func=run_concat_csv_files)

    p_filter = sub.add_parser(
        "filter",
        help="filter rows from a concatenated CSV and write a new CSV next to it",
    )
    p_filter.add_argument("-c", "--config", default="config.json", help="config json path")
    p_filter.add_argument("--file", help="input csv path override (e.g., OU/OU_concat.csv)")
    p_filter.add_argument("--out", help="output csv path override")
    p_filter.add_argument("--column", help="column name to filter (skips prompt if used with --value)")
    p_filter.add_argument("--value", help="value to match (skips prompt if used with --column)")
    p_filter.set_defaults(func=run_filter_csv)

    p_gen = sub.add_parser(
        "generate-data",
        help="run the SSA channel over a codebook and write a labeled CSV",
    )
    p_gen.add_argument(
        "-c", "--config", default="config.json",
        help="config json path (must include 'channel' block)",
    )
    p_gen.add_argument("--out", help="output csv path override")
    p_gen.set_defaults(func=run_generate_data)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)
    rewritten = _rewrite_legacy_argv(raw_argv)

    parser = _build_parser()
    args = parser.parse_args(rewritten)

    _print_banner(args)

    if getattr(args, "cmd", None) is None:
        _run_shell(parser, startup_args=args)
        return

    args.func(args)
