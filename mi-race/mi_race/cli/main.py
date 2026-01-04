import argparse
import os
from typing import List, Optional
from importlib.metadata import version, PackageNotFoundError
from mi_race.train.orchestrator import run_cmd
from mi_race.reporting.compare_models import run_compare


def _pkg_version() -> str:
    for name in ("mi-race", "mi_race"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "0.0.0"


def _build_box(lines: List[str], title: Optional[str] = None, *, min_width: int = 34) -> str:
    """Build a Unicode box using box-drawing chars with optional centered title."""
    # Compute inner width based on longest content line
    inner_width = max((len(line) for line in lines), default=0)
    # Ensure minimum width to make the title look good
    min_width = max(inner_width, len(title or ""))
    inner_width = max(min_width, 34)

    # Top border with optional centered title
    if title:
        # We add spaces around title: " title "
        t = f" {title} "
        # Remaining dashes to fill
        remaining = inner_width - len(t)
        left = remaining // 2
        right = remaining - left
        top = f"╭{'─' * left}{t}{'─' * right}╮"
    else:
        top = f"╭{'─' * inner_width}╮"

    # Middle lines padded
    middle = [f"│ {line.ljust(inner_width - 2)} │" for line in lines]

    # Bottom border
    bottom = f"╰{'─' * inner_width}╯"

    return "\n".join([top, *middle, bottom])


def _print_banner(args: argparse.Namespace) -> None:
    """Print ASCII art header and a version box unless suppressed."""
    env_no_logo = os.environ.get("MI_RACE_NO_LOGO", "").lower() in ("1", "true", "yes")
    if env_no_logo:
        return

    # User-provided ASCII header
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
    box = _build_box(lines, title="mi-race", min_width=72)

    print(ascii_header)
    print(box)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mi-race",
        description="mi-race CLI (machine learning for science).",
    )
    parser.add_argument("-v", "--version", action="version",
                        version=f"mi-race {_pkg_version()}")
    # Note: banner suppression is controlled by MI_RACE_NO_LOGO env var only

    # Subcommands (not required so we can show banner + usage when none provided)
    sub = parser.add_subparsers(dest="cmd")

    # mi-race run <model> [-c mi-race.json]
    p_run = sub.add_parser("run", help="train/evaluate a model from json config")
    p_run.add_argument(
        "--model",
        choices=["mlp", "cnn", "rnn", "rf"],
        help="model selection ('mlp', 'cnn', 'rnn', or 'rf'). Run the command once per model.",
    )
    p_run.add_argument("-c", "--config", default="config.json", help="config json path")
    p_run.set_defaults(func=run_cmd)

    # mi-race run-all [-c mi-race.json]
    def _run_all(args):
        # Iterate through supported models and invoke orchestrator per model
        for m in ("mlp", "cnn", "rnn", "rf"):
            ns = argparse.Namespace(**{**vars(args), "model": m})
            print(f"\n[mi-race] ===== Running all: {m} =====")
            run_cmd(ns)

    p_run_all = sub.add_parser("run-all", help="train/evaluate all supported models using the same config")
    p_run_all.add_argument("-c", "--config", default="config.json", help="config json path")
    p_run_all.set_defaults(func=_run_all)

    # mi-race compare
    p_cmp = sub.add_parser("compare", help="plot overall accuracy and accuracy vs noise from outputs/summary_models.csv")
    p_cmp.add_argument("--split", help="restrict compare to a single split prefix (e.g., 'noise' or 'kcross')")
    p_cmp.set_defaults(func=run_compare)

    args = parser.parse_args()

    # Print banner
    _print_banner(args)

    # If no subcommand provided, show a friendly greeting and usage
    if getattr(args, "cmd", None) is None:
        print("Hello from mi-race! Pick a command below.\n")
        parser.print_help()
        return

    # Dispatch to subcommand handler
    args.func(args)