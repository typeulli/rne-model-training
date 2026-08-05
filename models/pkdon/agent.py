"""PKDoN behind the shared :class:`~agent.BaseAgent` inference contract.

``predict_at`` keeps the ``[B, 5]`` of ``(x, y, z, t, P)`` every script is written
against and converts to the beam frame on the way in -- unlike
``models/cpkmlp/agent.py``, which takes offsets already computed and leaves that
conversion to its caller. It can, because the toolpath is in hand.

This is still not a surrogate for the plate: ask it for a point 10 mm from the
beam and it will answer, but the answer means nothing. Reach it through
:func:`peak_corrected`, which is what decides where the patch applies.

    from models.pkdon import peak_corrected
    from models.jdon import build_agent as build_base

    base = build_base(jdon_ckpt, toolpath=path, shape=shape)
    agent = peak_corrected(base, pkdon_ckpt)
"""

from __future__ import annotations

from pathlib import Path

import torch

from .._pathdon import (
    PatchAgent,
    PathAgent,
    GenericPeakCorrectedAgent,
    Toolpath,
)
from .._pathdon import build_agent as _build_agent

MODEL_NAME = "pkdon"


def build_agent(
    checkpoint: Path,
    toolpath: Path | Toolpath | None = None,
    shape: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
) -> PatchAgent:
    """Rebuild PKDoN and the window it is only meaningful inside."""
    return _build_agent(checkpoint, MODEL_NAME, toolpath=toolpath, shape=shape, device=device)


def peak_corrected(
    base: PathAgent,
    checkpoint: Path | str | None = None,
    device: torch.device | str | None = None,
) -> GenericPeakCorrectedAgent:
    """Wrap ``base`` so the melt pool is overwritten by this patch.

    The patch is pointed at ``base``'s own toolpath -- not at a default -- so the
    two cannot be describing different scans. ``checkpoint`` defaults to the most
    recent under ``checkpoints/pkdon/``, and the resolved path is printed either
    way: which patch a figure was corrected by is not a thing to guess at later.
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
    print(f"[load] {MODEL_NAME}: {path.name}, pasted over a "
          f"{2e3 * patch.radius:g} x {2e3 * patch.radius:g} mm square "
          f"riding with the beam")
    return GenericPeakCorrectedAgent(base, patch)
