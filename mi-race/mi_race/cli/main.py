import argparse
from importlib.metadata import version, PackageNotFoundError
from mi_race.train.orchestrator import run_cmd


def _pkg_version() -> str:
    for name in ("mi-race", "mi_race"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "0.0.0"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mi-race",
        description="mi-race CLI (machine learning for science).",
    )
    parser.add_argument("-v", "--version", action="version",
                        version=f"mi-race {_pkg_version()}")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # mi-race run <model> [-c mi-race.json]
    p_run = sub.add_parser("run", help="train/evaluate a model from json config")
    p_run.add_argument(
        "--model",
        choices=["mlp", "cnn"],
        help="model selection ('mlp' or 'cnn'). Run the command once per model.",
    )
    p_run.add_argument("-c", "--config", default="config.json", help="config json path")
    p_run.set_defaults(func=run_cmd)

    args = parser.parse_args()
    args.func(args)