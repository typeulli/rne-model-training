"""The inference contract every model exposes through its ``agent.py``.

A trained network is only useful to the plotting and analysis scripts through
two questions, so those are the only two an agent has to answer:

``predict_at(inputs)``
    ``inputs`` is ``[B, 5]`` of ``(x, y, z, t, P)`` and the result is ``[B, 1]``
    of temperature in Kelvin. Arbitrary points, no structure assumed -- this is
    what a slice, a line cut or a scattered probe needs.

``predict_of(inputs)``
    ``inputs`` is ``[B, 2]`` of ``(t, P)`` and the result is ``[B, 1, D, H, W]``
    of temperature in Kelvin: the whole three-dimensional field, one volume per
    row. The trailing axes follow the ``Conv3d`` convention ``D = z``, ``H = y``,
    ``W = x``, so ``field[b, 0, k, j, i]`` is ``T(x[i], y[j], z[k])`` at that
    row's ``(t, P)``.

Only :meth:`BaseAgent.predict_at` is abstract. ``predict_of`` is derived from it
here by evaluating the reconstruction grid, so a new model gets the volumetric
view for free and the two can never disagree.

Spatial and temporal bounds come from the checkpoint, not from the dataset, so
plotting a saved model never touches the 13 GB corpus.

:class:`PeakCorrectedAgent` is an agent built out of two others rather than out of
a network: it answers with one model everywhere and with a second one over the
melt pool. It satisfies the same contract, so it is a drop-in for whatever it
wraps, and :func:`peak_corrected` is the one-line way to build it.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from dataset import DEFAULT_FIELD_SHAPE
from utils import DEFAULT_CHECKPOINT_DIR

ArrayLike = Tensor | np.ndarray | Sequence

# The window a patch is pasted over is decided by `|x - x_peak| <= radius`, and
# both sides are floating point sums of the grid spacing, so the node sitting
# exactly on the boundary must not be lost to the last bit. Matches
# `models.cpkmlp.dataset.NODE_TOLERANCE`, which decides the same thing for the
# rows the patch was fitted to.
NODE_TOLERANCE = 1e-9


class BaseAgent(abc.ABC):
    """Wraps a trained network as a temperature field ``T(x, y, z, t; P)``.

    ``bounds`` is ``[4, 2]`` of ``(lower, upper)`` for ``(x, y, z, t)`` in SI
    units and defines the box :meth:`predict_of` reconstructs on. ``shape`` is
    ``(D, H, W) = (nz, ny, nx)``, defaulting to the resolution of the simulation
    the model was fitted to.
    """

    def __init__(
        self,
        bounds: Tensor,
        shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        chunk: int = 65536,
    ) -> None:
        bounds = torch.as_tensor(bounds, dtype=dtype, device=device)
        if bounds.shape != (4, 2):
            raise ValueError(f"bounds must be [4, 2], got {tuple(bounds.shape)}")
        if len(shape) != 3 or any(int(size) < 1 for size in shape):
            raise ValueError(f"shape must be three positive ints, got {shape}")

        self.bounds = bounds
        self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        self.device = torch.device(device)
        self.dtype = dtype
        self.chunk = chunk
        self._grid: Tensor | None = None

    # -- the grid ``predict_of`` reconstructs on -----------------------------

    @property
    def axes(self) -> tuple[Tensor, Tensor, Tensor]:
        """``(z, y, x)`` node positions in metres, matching ``(D, H, W)``."""
        depth, height, width = self.shape
        return tuple(  # type: ignore[return-value]
            torch.linspace(
                float(self.bounds[axis, 0]),
                float(self.bounds[axis, 1]),
                count,
                dtype=self.dtype,
                device=self.device,
            )
            for axis, count in ((2, depth), (1, height), (0, width))
        )

    @property
    def grid(self) -> Tensor:
        """``[D*H*W, 3]`` of ``(x, y, z)``, in the row order of a ``[D, H, W]`` volume."""
        if self._grid is None:
            z, y, x = self.axes
            grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
            self._grid = torch.stack(
                (grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)), dim=-1
            )
        return self._grid

    # -- inference -----------------------------------------------------------

    @abc.abstractmethod
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        """``[B, 5]`` of ``(x, y, z, t, P)`` to ``[B, 1]`` of Kelvin."""

    @torch.no_grad()
    def predict_of(self, inputs: ArrayLike) -> Tensor:
        """``[B, 2]`` of ``(t, P)`` to ``[B, 1, D, H, W]`` of Kelvin."""
        inputs = self._as_tensor(inputs, columns=2, name="predict_of")
        depth, height, width = self.shape
        grid = self.grid
        nodes = grid.size(0)

        volumes = []
        for row in inputs:
            time, power = row[0], row[1]
            query = torch.empty((nodes, 5), dtype=self.dtype, device=self.device)
            query[:, :3] = grid
            query[:, 3] = time
            query[:, 4] = power
            volumes.append(self.predict_at(query).reshape(depth, height, width))

        return torch.stack(volumes).unsqueeze(1)

    # -- helpers -------------------------------------------------------------

    def _as_tensor(self, inputs: ArrayLike, columns: int, name: str) -> Tensor:
        tensor = torch.as_tensor(inputs, dtype=self.dtype, device=self.device)
        if tensor.dim() != 2 or tensor.size(1) != columns:
            raise ValueError(
                f"{name} expects [B, {columns}], got {tuple(tensor.shape)}"
            )
        return tensor


class PeakCorrectedAgent(BaseAgent):
    """``base``'s field with a ``cpkmlp`` patch pasted over the melt pool.

    The models that fit the whole plate get the pool wrong in a particular way: it
    is a few millimetres of a 40 mm block, so it contributes almost nothing to a
    mean squared error and the fit blunts it. ``cpkmlp`` is fitted to that
    neighbourhood alone, in coordinates that ride with the beam -- but for the
    same reason it knows nothing about the rest of the plate. Each model supplies
    what the other lacks:

    * ``base`` answers everywhere;
    * ``patch`` answers inside its own window about the beam, and its answer
      replaces ``base``'s outright.

    The temperatures pasted in are ``patch``'s own, in Kelvin, unblended -- only
    the *window* moves with the beam, never the values -- so the seam at the
    window's edge is a genuine discontinuity and reads as exactly what it is: the
    disagreement between the two models at that point.

    Where the window is centred is the one thing that must not be guessed, and
    this is the second answer to that question. The first was the hottest node of
    ``base``'s own top face, and it failed for a reason worth recording. The
    solver's maximum trails the beam by 0.24 mm and so lands 0.003 mm past the
    midpoint between two nodes whose temperatures differ by 0.46 K in a 1353 K
    rise; the beam advances exactly 16 nodes per stored step, so that near-tie
    repeats identically at every instant rather than averaging out. Every model in
    ``models/`` places its continuous maximum within 0.06 mm of the solver's -- a
    fifth of a node -- and still falls on the other side of that midpoint, so the
    argmax picked the node *behind*, always, and the patch landed 0.25 mm off. An
    argmax reads the difference between two nodes, and here that difference is
    thirty times smaller than the error a good fit makes on either one.

    The beam is not estimated from a field at all. ``patch.centre(t)`` is three
    calibrated numbers and a clock, restored from the checkpoint the patch was
    fitted with, so the frame it was fitted in and the frame it is pasted in are
    the same frame exactly rather than to within a node -- and no evaluation of
    ``base`` is needed to find it.

    The window itself is ``patch.bounds``: the paste covers exactly the offsets,
    depths and times the patch has data for, and outside them ``base`` is left
    alone rather than being overwritten by an extrapolation.
    """

    def __init__(self, base: BaseAgent, patch: BaseAgent) -> None:
        super().__init__(
            base.bounds,
            shape=base.shape,
            device=base.device,
            dtype=base.dtype,
            chunk=base.chunk,
        )
        if not callable(getattr(patch, "centre", None)):
            raise TypeError(
                f"{type(patch).__name__} has no `centre(time)`; the patch must say "
                "what frame its offsets are measured in"
            )
        # A base trained with `--cede-from` has a hole in it the exact shape of the
        # window it was told a patch would fill. Pasting a patch that covers a
        # different box leaves either a ring neither model ever learned or ground
        # the base did learn being overwritten -- and neither shows up as anything
        # but a worse number. The two boxes are compared rather than trusted.
        ceded = getattr(base, "cede_window", None)
        if ceded is not None:
            from models._ceded import windows_agree

            if not windows_agree(ceded, patch.bounds):
                raise ValueError(
                    f"{type(base).__name__} ceded {torch.as_tensor(ceded).tolist()} "
                    f"but {type(patch).__name__} covers {patch.bounds.tolist()}; "
                    "they must be the same window"
                )

        self.base = base
        self.patch = patch
        self.window = patch.bounds.to(self.device)

        # `benchmark.py` counts an agent's parameters through `.model`, and what a
        # corrected prediction costs is both networks, so both are exposed.
        self.model = torch.nn.ModuleList([base.model, patch.model])

    # -- what it overwrites --------------------------------------------------

    def patch_box(
        self, time: float
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
        """``((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))`` overwritten at ``time``, in metres.

        None when ``time`` falls outside the clock the patch was fitted over, which
        is exactly when :meth:`predict_at` leaves ``base`` alone. Figures draw this
        so a reader can see which part of a panel is whose.

        ``time`` is put through this agent's dtype before it is tested, because
        that is what happens to it on its way into :meth:`predict_at` and the two
        must agree. ``bounds`` is stored in that dtype too, so the first stored
        time comes back as 0.4000000059604645; a caller's float64 0.4 sits 6e-9 s
        below it, and untouched it would report no patch at an instant where the
        paste had in fact happened.
        """
        window = self.window
        time = float(torch.as_tensor(float(time), dtype=self.dtype))
        if not float(window[3, 0]) <= time <= float(window[3, 1]):
            return None
        centre_x, centre_y = self.patch.centre(float(time))
        return (
            (centre_x + float(window[0, 0]), centre_x + float(window[0, 1])),
            (centre_y + float(window[1, 0]), centre_y + float(window[1, 1])),
            (float(window[2, 0]), float(window[2, 1])),
        )

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        """``[B, 5]`` of ``(x, y, z, t, P)`` to ``[B, 1]`` of Kelvin.

        Rows are grouped by their ``(t, P)`` first, so the window is placed once
        per instant rather than once per point.
        """
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")
        outputs = self.base.predict_at(inputs).clone()

        lower = self.window[:, 0] - NODE_TOLERANCE
        upper = self.window[:, 1] + NODE_TOLERANCE

        instants, row_of = torch.unique(inputs[:, 3:5], dim=0, return_inverse=True)
        for index, instant in enumerate(instants):
            time = float(instant[0])
            if not lower[3] <= time <= upper[3]:
                continue  # outside the clock the patch was fitted over

            centre_x, centre_y = self.patch.centre(time)
            offset_x = inputs[:, 0] - centre_x
            offset_y = inputs[:, 1] - centre_y
            inside = (
                (row_of == index)
                & (offset_x >= lower[0])
                & (offset_x <= upper[0])
                & (offset_y >= lower[1])
                & (offset_y <= upper[1])
                & (inputs[:, 2] >= lower[2])
                & (inputs[:, 2] <= upper[2])
            )
            if not bool(inside.any()):
                continue

            # The patch takes the same five columns, with the first two read as
            # offsets from the beam rather than positions on the plate.
            query = inputs[inside].clone()
            query[:, 0] = offset_x[inside]
            query[:, 1] = offset_y[inside]
            outputs[inside] = self.patch.predict_at(query)
        return outputs


def peak_corrected(
    base: BaseAgent,
    checkpoint: Path | str | None = None,
    model: str = "cpkmlp",
    device: torch.device | str | None = None,
) -> PeakCorrectedAgent:
    """Wrap ``base`` so the melt pool is overwritten by ``model``'s patch.

    ``checkpoint`` defaults to the most recently written one under
    ``checkpoints/<model>/``, so ``--pkcorrect`` can be given on its own; the
    resolved path is printed either way, since which patch a figure was corrected
    by is not something to have to guess at afterwards.
    """
    # Imported here rather than at module scope: every `models/*/agent.py` imports
    # this module, so the package cannot be imported while it is still executing.
    from models import build_agent

    path = Path(checkpoint) if checkpoint else _latest_checkpoint(model)
    patch = build_agent(model, path, device=device or base.device)
    corrected = PeakCorrectedAgent(base, patch)
    window = corrected.window * 1e3  # mm, and seconds in the last row
    print(
        f"[load] {model}: {path.name}, pasted about the beam over "
        f"dx [{window[0, 0]:.3f}, {window[0, 1]:.3f}] x "
        f"dy [{window[1, 0]:.3f}, {window[1, 1]:.3f}] mm, "
        f"t [{window[3, 0] / 1e3:g}, {window[3, 1] / 1e3:g}] s"
    )
    return corrected


def _latest_checkpoint(model: str) -> Path:
    """The most recently written ``checkpoints/<model>/*.pt``."""
    directory = DEFAULT_CHECKPOINT_DIR / model
    candidates = sorted(directory.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoints under {directory}; train one with "
            f"`python train.py {model}`, or name one explicitly"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)
