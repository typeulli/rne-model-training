"""``fmlp`` behind the shared :class:`~agent.BaseAgent` contract, with the two
questions answered in the opposite order.

:class:`~agent.BaseAgent` makes ``predict_at`` abstract and derives ``predict_of``
from it by evaluating the network once per node of the reconstruction grid. Every
other model in ``models/`` is happy with that, because every other model *is*
pointwise -- asking it for 165 025 points is what it was built to do.

``fmlp`` inverts the hierarchy, exactly as ``TODO.md`` said it would have to. Its
native operation is the whole volume: two numbers in, one ``irfftn`` out. So
:meth:`FMLPAgent.predict_of` is overridden to do that directly instead of 165 025
times, and :meth:`FMLPAgent.predict_at` is the *derived* one -- it reconstructs the
volume the query's ``(t, P)`` implies and interpolates within it.

That interpolation is worth being honest about. On a grid node it is exact, so
``visualize.py`` and ``scanline.py`` -- which sample the simulation's own nodes --
see no approximation at all. Off-node it is trilinear, and the model is *not*: the
field it represents is a band-limited trigonometric interpolant, and evaluating it
between nodes would properly mean summing the modes there. Trilinear is the cheaper
answer, and at 0.25 mm spacing against a 1.7 mm beam it is a good one, but a query
deliberately placed between nodes is answered by the grid, not by the spectrum.

Rows are grouped by their ``(t, P)`` before anything is reconstructed, so a slice
of a few thousand points at one instant costs one inverse transform, not one per
point.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from agent import ArrayLike, BaseAgent
from dataset import DEFAULT_FIELD_SHAPE
from utils import load_checkpoint, resolve_device

from .model import FourierMLP


class FMLPAgent(BaseAgent):
    """Wraps a trained :class:`~models.fmlp.model.FourierMLP` for inference."""

    def __init__(
        self,
        model: FourierMLP,
        bounds: Tensor,
        shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
        device: torch.device | str = "cpu",
        chunk: int = 65536,
    ) -> None:
        dtype = next(model.parameters()).dtype
        super().__init__(bounds, shape=shape, device=device, dtype=dtype, chunk=chunk)
        self.model = model.to(self.device).eval()

    # -- the native question -------------------------------------------------

    @torch.no_grad()
    def predict_of(self, inputs: ArrayLike) -> Tensor:
        """``[B, 2]`` of ``(t, P)`` to ``[B, 1, D, H, W]`` of Kelvin.

        The contract orders the pair ``(t, P)`` and the network takes ``(P, t)``,
        so the only work before the transform is a swap of two columns.
        """
        inputs = self._as_tensor(inputs, columns=2, name="predict_of")
        volumes = self.model.field(inputs.flip(-1)).unsqueeze(1)

        # The transform reconstructs on the grid it was fitted to. A caller asking
        # for a different one (benchmark.py --shape) gets it resampled.
        if tuple(volumes.shape[2:]) != self.shape:
            volumes = F.interpolate(
                volumes, size=self.shape, mode="trilinear", align_corners=True
            )
        return volumes

    # -- and the one derived from it -----------------------------------------

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        """``[B, 5]`` of ``(x, y, z, t, P)`` to ``[B, 1]`` of Kelvin."""
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")

        instants, row_of = torch.unique(inputs[:, 3:5], dim=0, return_inverse=True)
        outputs = torch.empty(
            (inputs.size(0), 1), dtype=self.dtype, device=self.device
        )
        for index, instant in enumerate(instants):
            rows = row_of == index
            volume = self.model.field(instant.flip(-1).reshape(1, 2))
            outputs[rows] = self._interpolate(volume, inputs[rows, :3])
        return outputs

    def _interpolate(self, volume: Tensor, points: Tensor) -> Tensor:
        """Sample ``[1, D, H, W]`` at ``[N, 3]`` of ``(x, y, z)`` in metres.

        ``grid_sample`` indexes its last axis as ``(W, H, D) = (x, y, z)``, which is
        already the column order of the contract, and ``align_corners=True`` puts
        -1 and +1 on the outermost node centres -- so a query on a node returns that
        node, not a blend of it with its neighbour. Points outside the box clamp to
        the boundary rather than reading as absolute zero Kelvin.
        """
        lower, upper = self.bounds[:3, 0], self.bounds[:3, 1]
        unit = 2.0 * (points - lower) / (upper - lower) - 1.0
        sampled = F.grid_sample(
            volume.unsqueeze(1),
            unit.reshape(1, -1, 1, 1, 3),
            mode="bilinear",  # trilinear, for a 5-D input
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(-1, 1)


def build_agent(
    checkpoint: Path,
    shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
    device: torch.device | str | None = None,
) -> FMLPAgent:
    """Rebuild the network from the architecture stored alongside the weights.

    The architecture carries the whole transform -- which wavenumbers were kept,
    the scan speed they were de-rotated at, the statistics they were standardised
    by -- so the field a checkpoint reconstructs is the one its run was scored on.
    """
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    payload = load_checkpoint(checkpoint, map_location=device)

    if "bounds" not in payload:
        raise KeyError(f"{checkpoint} predates the `bounds` key; retrain or add it by hand")

    model = FourierMLP(**payload["architecture"])
    model.load_state_dict(payload["model"])  # the transform's buffers ride along
    return FMLPAgent(model, payload["bounds"], shape=shape, device=device)
