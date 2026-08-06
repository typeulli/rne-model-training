"""RPKFDoN behind the shared :class:`~agent.BaseAgent` inference contract.

Identical to :mod:`models.pkfdon`'s agent except that the beam frame it asks for
is the held one -- restored from the checkpoint's ``retouched`` flag, so an agent
can never be built in a frame the weights were not fitted in.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .._pathdon import Toolpath
from .._pathdon import build_agent as _build_agent

MODEL_NAME = "rpkfdon"


def build_agent(
    checkpoint: Path,
    toolpath: Path | Toolpath | None = None,
    shape: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
):
    """Rebuild RPKFDoN, its time grid, and the frame it was fitted in."""
    return _build_agent(checkpoint, MODEL_NAME, toolpath=toolpath, shape=shape, device=device)


def peak_corrected(
    base,
    checkpoint: Path | str | None = None,
    device: torch.device | str | None = None,
):
    """Wrap ``base`` so the melt pool is overwritten by this patch.

    Both halves must hold the beam the same way; ``GenericPeakCorrectedAgent``
    refuses the pair otherwise, because a held window and a tracking one are in
    different places for every instant the laser is off.
    """
    from .._pathdon import GenericPeakCorrectedAgent

    directory = Path("checkpoints", MODEL_NAME)
    path = Path(checkpoint) if checkpoint else max(
        directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None
    )
    if path is None:
        raise FileNotFoundError(
            f"no checkpoints under {directory}; train one with `python train.py {MODEL_NAME}`"
        )
    patch = build_agent(path, toolpath=base.path, shape=base.shape,
                        device=device or base.device)
    shape = (
        f"{2e3 * patch.radius:g} mm disc"
        if patch.window == "circle"
        else f"{2e3 * patch.radius:g} x {2e3 * patch.radius:g} mm square"
    )
    print(f"[load] {MODEL_NAME}: {path.name}, pasted over a {shape} held with the pool")
    return GenericPeakCorrectedAgent(base, patch)
