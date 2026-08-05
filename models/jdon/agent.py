"""JDoN behind the shared :class:`~agent.BaseAgent` inference contract.

The network is a pointwise map, so ``predict_at`` is a forward pass and the
volumetric ``predict_of`` inherited from the base class is the same map on a
regular grid. The one thing that is not in the ``[B, 5]`` signature is *which
toolpath*, so it is fixed when the agent is built --
:class:`~models._pathdon.PathAgent` explains why that is the right place for it.

    from models.jdon import build_agent
    agent = build_agent(checkpoint, toolpath="data/toolpath/spiral/toolpath.json")
"""

from __future__ import annotations

from pathlib import Path

import torch

from .._pathdon import PathAgent, Toolpath
from .._pathdon import build_agent as _build_agent

MODEL_NAME = "jdon"


def build_agent(
    checkpoint: Path,
    toolpath: Path | Toolpath | None = None,
    shape: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
) -> PathAgent:
    """Rebuild JDoN from the architecture stored alongside its weights."""
    return _build_agent(checkpoint, MODEL_NAME, toolpath=toolpath, shape=shape, device=device)
