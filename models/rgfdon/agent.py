"""RGFDoN behind the shared :class:`~agent.BaseAgent` inference contract.

Identical to :mod:`models.gfdon`'s agent except that the beam frame it asks for
is the held one -- restored from the checkpoint's ``retouched`` flag, so an agent
can never be built in a frame the weights were not fitted in.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .._pathdon import Toolpath
from .._pathdon import build_agent as _build_agent

MODEL_NAME = "rgfdon"


def build_agent(
    checkpoint: Path,
    toolpath: Path | Toolpath | None = None,
    shape: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
):
    """Rebuild RGFDoN, its time grid, and the frame it was fitted in."""
    return _build_agent(checkpoint, MODEL_NAME, toolpath=toolpath, shape=shape, device=device)
