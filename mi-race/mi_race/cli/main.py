import argparse
from importlib.metadata import version, PackageNotFoundError
from mi_race.data.preview import preview_csv
from mi_race.train.runner import run_cmd


def _pkg_version() -> str:
    for name in ("mi-race", "mi_race"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "0.0.0"


def _load_cmd(args):
    preview_csv(args.path, label=args.label)  


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mi-race",
        description="mi-race CLI (machine learning for science).",
    )
    parser.add_argument("-v", "--version", action="version",
                        version=f"mi-race {_pkg_version()}")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # mi-race load <path/to/file.csv> [--label y]
    p_load = sub.add_parser("load", help="preview a CSV file (no changes made)")
    p_load.add_argument("path", help="path to CSV file")
    p_load.add_argument("--label", help="label column name to count classes", default=None)
    p_load.set_defaults(func=_load_cmd)

    # mi-race run <model> [-c mi-race.json]
    p_run = sub.add_parser("run", help="train/evaluate a model from json config")
    p_run.add_argument("--model", choices=["mlp", "cnn"], help="model selection")
    p_run.add_argument("-c", "--config", default="config.json", help="config json path")
    p_run.set_defaults(func=run_cmd)

    args = parser.parse_args()
    args.func(args)