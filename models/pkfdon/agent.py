"""PKFDoN behind the shared :class:`~agent.BaseAgent` inference contract.

``predict_at`` takes plate coordinates and does both conversions on the way in --
into the beam frame, and then out of the history at the query time. Reach it
through :func:`peak_corrected`, which decides which points are the patch's.

    from models.pkfdon import peak_corrected
    from models.gfdon import build_agent as build_base

    base = build_base(gfdon_ckpt, toolpath=path, shape=shape)
    agent = peak_corrected(base, pkfdon_ckpt)
"""

from __future__ import annotations

from pathlib import Path

import torch

from .._pathdon import (
    GenericPeakCorrectedAgent,
    SequenceAgent,
    SequencePatchAgent,
    Toolpath,
)
from .._pathdon import build_agent as _build_agent

MODEL_NAME = "pkfdon"


def build_agent(
    checkpoint: Path,
    toolpath: Path | Toolpath | None = None,
    shape: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
) -> SequencePatchAgent:
    """Rebuild PKFDoN, its window, and the time grid its answer lives on."""
    return _build_agent(checkpoint, MODEL_NAME, toolpath=toolpath, shape=shape, device=device)


def peak_corrected(
    base: SequenceAgent,
    checkpoint: Path | str | None = None,
    device: torch.device | str | None = None,
) -> GenericPeakCorrectedAgent:
    """Wrap ``base`` so the melt pool is overwritten by this patch.

    The patch is pointed at ``base``'s own toolpath, so the two cannot describe
    different scans. ``checkpoint`` defaults to the most recent under
    ``checkpoints/pkfdon/`` and the resolved path is printed either way.
    """
    directory = Path("checkpoints", MODEL_NAME)
    path = Path(checkpoint) if checkpoint else max(
        directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None
    )
    if path is None:
        raise FileNotFoundError(
            f"no checkpoints under {directory}; train one with `python train.py {MODEL_NAME}`"
        )

    patch = build_agent(
        path, toolpath=base.path, shape=base.shape, device=device or base.device
    )
    shape = (
        f"{2e3 * patch.radius:g} mm disc"
        if patch.window == "circle"
        else f"{2e3 * patch.radius:g} x {2e3 * patch.radius:g} mm square"
    )
    print(f"[load] {MODEL_NAME}: {path.name}, pasted over a {shape} riding with the beam")
    return GenericPeakCorrectedAgent(base, patch)
