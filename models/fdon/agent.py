"""FDoN behind the shared :class:`~agent.BaseAgent` inference contract.

The network answers on a fixed grid of times, and the contract asks for one
number at one ``t``. :class:`~models._pathdon.SequenceAgent` bridges that by
evaluating the history once and reading ``t`` off it, linearly between the two
nearest stored times. The toolpath is fixed when the agent is built, for the
reason :class:`~models._pathdon.PathAgent` gives.

    from models.fdon import build_agent
    agent = build_agent(checkpoint, toolpath="data/toolpath/spiral/toolpath.json")
"""

from __future__ import annotations

from pathlib import Path

import torch

from .._pathdon import SequenceAgent, Toolpath
from .._pathdon import build_agent as _build_agent

MODEL_NAME = "fdon"


def build_agent(
    checkpoint: Path,
    toolpath: Path | Toolpath | None = None,
    shape: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
) -> SequenceAgent:
    """Rebuild FDoN and the time grid its answer lives on."""
    return _build_agent(checkpoint, MODEL_NAME, toolpath=toolpath, shape=shape, device=device)
