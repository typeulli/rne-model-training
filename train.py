"""Train one of the models under ``models/``.

The first argument names the model; everything after it is handed, untouched, to
that model's own ``train.main(argv)``. Each model therefore owns its
hyperparameters instead of sharing a lowest-common-denominator parser here --
including ``--help``, which is why the model name is split off before argparse
ever sees the command line.

Examples::

    python train.py --list
    python train.py gpidon_old --iterations 20000 --lr 1e-3
    python train.py gpidon_old --help
"""

from __future__ import annotations

import argparse
import sys

from models import available_models, train_module


def _usage() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "model", nargs="?", help=f"one of: {', '.join(available_models())}"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the available models and exit"
    )
    return parser


def main() -> None:
    argv = sys.argv[1:]

    if argv and not argv[0].startswith("-"):
        try:
            module = train_module(argv[0])
        except ValueError as error:
            _usage().error(str(error))
        module.main(argv[1:])
        return

    parser = _usage()
    args = parser.parse_args(argv)
    if not args.list:
        parser.error("a model name is required")
    for name in available_models():
        print(name)


if __name__ == "__main__":
    main()
