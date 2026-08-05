"""Drop from a base model's corpus exactly the window ``cpkmlp`` will paste over.

``models/cpkmlp`` is fitted to the melt pool alone and its answer *replaces* the
base's inside a window that rides with the beam. Until now the base was fitted to
that window as well, so part of its capacity went to an answer
:class:`~agent.PeakCorrectedAgent` then discards. On the toolpath corpus that
window is 6.3% of the samples and carries 54% of the base's squared error; ceding
it there moved the pasted result from 7.49 K to 6.39 K.

**Why this takes a checkpoint and not a radius.** The obvious knob is
``--cede-radius 2.5e-3``, matching the patch's own ``--radius``. It is wrong, and
measurably so: ceding a symmetric 2.5 mm square drops 80 000 rows while the paste
covers 68 500, leaving **11 500 rows that neither model ever learns**. Two reasons,
both invisible from the flag:

* the paste is bounded by the patch's ``bounds``, which are the min and max of the
  *node offsets it actually kept* -- ``[-2.368, +2.382] mm``, not ``+/-2.5``, because
  the beam sits between nodes;
* those bounds carry a time range too, and a patch trained with ``--exclude 1``
  starts at ``t = 0.4 s``, so ceding by radius alone hands over every row at
  ``t = 0``.

So the base is given the patch's checkpoint and cedes the box the patch reports.
There is no second number to keep in step, and :class:`~agent.PeakCorrectedAgent`
re-checks the pairing at paste time against the same box.

The straight-pass beam only ever travels ``+x``, so the window is axis-aligned and
needs no rotation -- which is why ``models/cpkmlp/dataset.py`` does not rotate
either. The sequence family's ``--cede-radius`` is a different case and stays a
radius: ``pkfdon`` pastes by its radius rather than by data-derived bounds, and it
covers every stored time, so there the two agree exactly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from dataset import SimulationDataset
from models.cpkmlp.dataset import NODE_TOLERANCE
from models.cpkmlp.laser import BeamPath
from utils import load_checkpoint

__all__ = ["add_cede_argument", "cede_patch_window", "window_of", "windows_agree"]


def add_cede_argument(parser) -> None:
    """Register ``--cede-from`` on a base model's parser."""
    parser.add_argument(
        "--cede-from",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="a cpkmlp checkpoint. Drops from the corpus exactly the window that "
        "patch will be pasted over, so the base does not spend capacity on an "
        "answer that is discarded. Omit to keep the whole plate (default)",
    )


def window_of(payload: dict) -> tuple[Tensor, dict]:
    """``(bounds[4, 2], anchor)`` -- the box a patch owns and the frame it is in."""
    if "bounds" not in payload:
        raise KeyError("patch checkpoint has no `bounds`; it cannot say what it covers")
    if "anchor" not in payload:
        raise KeyError(
            "patch checkpoint has no `anchor`: it was fitted around the peak rather "
            "than the beam, and its offsets mean something else. Retrain it"
        )
    return torch.as_tensor(payload["bounds"], dtype=torch.float64), payload["anchor"]


def windows_agree(a: Tensor, b: Tensor, tolerance: float = NODE_TOLERANCE) -> bool:
    """Whether two ``[4, 2]`` windows are the same box to within a node tolerance."""
    return bool(torch.allclose(
        torch.as_tensor(a, dtype=torch.float64),
        torch.as_tensor(b, dtype=torch.float64),
        atol=tolerance, rtol=0.0,
    ))


def cede_patch_window(
    corpus: SimulationDataset,
    checkpoint: Path | str | None,
    verbose: bool = True,
) -> tuple[SimulationDataset, list[list[float]] | None, dict | None]:
    """``(corpus without the patch's window, that window, its anchor)``.

    ``checkpoint`` of None returns the corpus untouched and no window, so a caller
    can pass the flag through unconditionally. The window and anchor are returned
    so the base's own checkpoint can record what it gave up -- without that,
    nothing at paste time can tell a ceded base from a whole-plate one.
    """
    if checkpoint is None:
        return corpus, None, None

    bounds, anchor = window_of(load_checkpoint(Path(checkpoint), map_location="cpu"))
    beam = BeamPath(**anchor)

    x, y, z, t = (corpus.coords[:, i] for i in range(4))
    offset_x = x - (beam.start_x + beam.speed * t)
    offset_y = y - beam.y

    lower = bounds[:, 0] - NODE_TOLERANCE
    upper = bounds[:, 1] + NODE_TOLERANCE
    inside = (
        (offset_x >= lower[0]) & (offset_x <= upper[0])
        & (offset_y >= lower[1]) & (offset_y <= upper[1])
        & (z >= lower[2]) & (z <= upper[2])
        & (t >= lower[3]) & (t <= upper[3])
    )
    keep = ~inside

    if verbose:
        window = bounds * 1e3
        print(
            f"[cede] {Path(checkpoint).name}: dropping {int(inside.sum())} of "
            f"{len(corpus)} rows ({100.0 * float(inside.sum()) / len(corpus):.2f}%) "
            f"in dx [{window[0, 0]:.3f}, {window[0, 1]:.3f}] x "
            f"dy [{window[1, 0]:.3f}, {window[1, 1]:.3f}] mm, "
            f"t [{bounds[3, 0]:g}, {bounds[3, 1]:g}] s"
        )
    if not bool(keep.any()):
        raise ValueError("the ceded window covers the whole corpus; nothing to train on")

    return (
        SimulationDataset(
            corpus.coords[keep], corpus.power[keep], corpus.temperature[keep]
        ),
        bounds.tolist(),
        dict(anchor),
    )
