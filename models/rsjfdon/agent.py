"""RSJFDoN behind the shared :class:`~agent.BaseAgent` inference contract.

Identical to :mod:`models.jfdon`'s agent except for the frame the gate is drawn
in: held over dark travel and fading there, and lagging through corners. All four
numbers that decide it -- ``retouched``, ``dark_tau``, ``slewed``, ``slew_tau`` --
are restored from the checkpoint, so an agent can never be built in a frame the
weights were not fitted in.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .._pathdon import Toolpath
from .._pathdon import build_agent as _build_agent

MODEL_NAME = "rsjfdon"


def build_agent(
    checkpoint: Path,
    toolpath: Path | Toolpath | None = None,
    shape: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
):
    """Rebuild RSJFDoN, its time grid, and the frame it was fitted in."""
    return _build_agent(checkpoint, MODEL_NAME, toolpath=toolpath, shape=shape, device=device)
