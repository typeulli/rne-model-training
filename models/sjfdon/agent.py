"""SJFDoN behind the shared :class:`~agent.BaseAgent` inference contract.

Identical to :mod:`models.jfdon`'s agent except that the heading the gate is
drawn with lags through a corner -- restored from the checkpoint's ``slewed`` flag
and its ``slew_tau``, so an agent can never be built in a frame the weights were
not fitted in.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .._pathdon import Toolpath
from .._pathdon import build_agent as _build_agent

MODEL_NAME = "sjfdon"


def build_agent(
    checkpoint: Path,
    toolpath: Path | Toolpath | None = None,
    shape: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
):
    """Rebuild SJFDoN, its time grid, and the heading it was fitted with."""
    return _build_agent(checkpoint, MODEL_NAME, toolpath=toolpath, shape=shape, device=device)
