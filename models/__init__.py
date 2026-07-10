"""Registry of the trainable models.

A model is a package under ``models/`` holding, by convention:

``model.py``    the architecture
``dataset.py``  how it draws batches from :mod:`dataset`
``loss.py``     its objective and the physical constants that objective needs
``train.py``    exposing ``main(argv: list[str]) -> None``, called by ``../train.py``
``agent.py``    exposing ``build_agent(...) -> BaseAgent``, called by the plotting scripts

Adding a directory that follows it is enough to make ``python train.py <name>``
and ``python visualize.py --model <name>`` work; nothing here needs editing.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

MODELS_DIR = Path(__file__).resolve().parent


def available_models() -> list[str]:
    """Every subdirectory of ``models/`` that looks like a model package."""
    return sorted(
        path.name
        for path in MODELS_DIR.iterdir()
        if path.is_dir()
        and not path.name.startswith(("_", "."))
        and (path / "train.py").exists()
    )


def _import(name: str, module: str) -> ModuleType:
    known = available_models()
    if name not in known:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(known)}")
    return importlib.import_module(f"models.{name}.{module}")


def train_module(name: str) -> ModuleType:
    """The model's ``train`` module, which must expose ``main(argv)``."""
    return _import(name, "train")


def build_agent(name: str, *args, **kwargs):
    """Construct the model's agent; arguments are forwarded to its ``build_agent``."""
    return _import(name, "agent").build_agent(*args, **kwargs)
