"""Training utilities shared by every model under ``models/``.

Nothing here knows about a particular architecture: run naming, best-checkpoint
tracking, device resolution and parameter counting are the same whether the
network is an operator net or a convolutional surrogate.
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "runs"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"


def resolve_device(name: str | None = None) -> torch.device:
    """``name``, or CUDA when it is available and ``name`` is None."""
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


# ---------------------------------------------------------------------------
# Run naming
# ---------------------------------------------------------------------------


def make_run_name(model_name: str, tag: str | None = None) -> str:
    """``<model>[-<tag>]-<YYYYmmdd-HHMMSS>``, unique enough to sort by time."""
    parts = [model_name]
    if tag:
        parts.append(tag)
    parts.append(time.strftime("%Y%m%d-%H%M%S"))
    return "-".join(parts)


def unique_path(path: Path) -> Path:
    """``path``, or ``path`` with ``-1``, ``-2``, ... appended until it is free.

    Two runs started within the same second would otherwise share a TensorBoard
    directory and interleave their scalars.
    """
    if not path.exists():
        return path
    for suffix in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free name near {path}")


def resolve_run_dir(log_dir: Path, run_name: str) -> Path:
    """The TensorBoard directory for this run, created and guaranteed unused."""
    directory = unique_path(Path(log_dir) / run_name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_checkpoint_path(
    checkpoint: Path | None, model_name: str, run_name: str
) -> Path:
    """``checkpoint`` if given, else ``checkpoints/<model>/<run_name>.pt``."""
    path = (
        Path(checkpoint)
        if checkpoint is not None
        else DEFAULT_CHECKPOINT_DIR / model_name / f"{run_name}.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


_RUN_TIMESTAMP = re.compile(r"\d{8}-\d{6}")


def checkpoint_stamp(path: Path) -> str:
    """The ``YYYYmmdd-HHMMSS`` a checkpoint was created at.

    :func:`make_run_name` bakes this into the run name, and
    :func:`resolve_checkpoint_path` bakes the run name into the checkpoint's
    filename, so it can be read back out of the stem. Falls back to the file's
    mtime for a checkpoint named by hand, so the figures this stamps stay
    sorted with the checkpoint they came from either way.
    """
    match = _RUN_TIMESTAMP.search(Path(path).stem)
    if match:
        return match.group(0)
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(Path(path).stat().st_mtime))


def resolve_figure_path(out: Path | None, *parts: str, stamp: str | None = None) -> Path:
    """``out`` if given, else an auto-named path under ``figures/``.

    A bare filename in ``out`` (no directory component) still lands under
    ``figures/``, so ``--out plot.png`` and omitting ``--out`` both keep
    figures in one place; a path with a directory in it is used as-is.
    ``parts`` -- e.g. ``(model, f"P{power:g}", plane)`` -- are joined with
    ``_`` and suffixed with ``stamp`` (see :func:`checkpoint_stamp`; defaults
    to the current time) when ``out`` is omitted entirely.
    """
    if out is None:
        stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        name = "_".join(parts) + f"_{stamp}.png"
        path = DEFAULT_FIGURES_DIR / name
    elif out.parent == Path("."):
        path = DEFAULT_FIGURES_DIR / out
    else:
        path = out
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically.

    A long run interrupted mid-``torch.save`` would otherwise leave a truncated
    file where the best checkpoint used to be, so the new one is written beside
    it and moved into place only once it is complete.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, map_location: torch.device | str = "cpu"
) -> dict[str, Any]:
    """Read a checkpoint written by :class:`BestCheckpoint`."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    for key in ("model", "architecture"):
        if key not in checkpoint:
            raise KeyError(f"{path}: checkpoint is missing the `{key}` key")
    return checkpoint


class BestCheckpoint:
    """Keeps the single checkpoint with the best value of a monitored metric.

    ``mode='min'`` for losses and errors, ``mode='max'`` for scores. The payload
    is supplied by the caller so each model can store whatever ``agent.py`` needs
    to rebuild itself -- but ``model`` (the state dict) and ``architecture`` (the
    keyword arguments of its constructor) are mandatory, since the loaders
    depend on them.
    """

    def __init__(self, path: Path, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.path = Path(path)
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.step: int | None = None

    def is_better(self, value: float) -> bool:
        return value < self.best if self.mode == "min" else value > self.best

    def update(self, value: float, payload: dict[str, Any], step: int | None = None) -> bool:
        """Save ``payload`` if ``value`` improves on the best seen. Returns whether it did."""
        if not self.is_better(value):
            return False

        self.best = value
        self.step = step
        save_checkpoint(self.path, {**payload, "metric": value, "step": step})
        return True
