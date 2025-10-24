import argparse
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
        choices=["mlp", "cnn", "rnn", "rf"],
        help="model selection ('mlp', 'cnn', 'rnn', or 'rf'). Run the command once per model.",
    )
    p_run.add_argument("-c", "--config", default="config.json", help="config json path")
    p_run.set_defaults(func=run_cmd)

    # mi-race compare
    p_cmp = sub.add_parser("compare", help="plot overall accuracy and accuracy vs noise from outputs/summary_models.csv")
    p_cmp.set_defaults(func=run_compare)

    args = parser.parse_args()
    args.func(args)