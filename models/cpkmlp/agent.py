"""CPKMLP behind the shared :class:`~agent.BaseAgent` inference contract.

``predict_at`` keeps the ``[B, 5]`` signature every script is written against and
drops the ``P`` column exactly as :mod:`models.cmlp.agent` does -- but its first
two columns are **offsets from the beam**, not positions on the plate. Nothing in
the contract can say so, and nothing in the network can tell the difference, so
the distinction lives here in one sentence, in :meth:`CPKMLPAgent.centre`, which
says where those offsets are measured from at a given time, and in
:attr:`~agent.BaseAgent.bounds`, which is the window they are meaningful within.

That makes this the one agent under ``models/`` that is not a surrogate for the
plate: asking it for ``(0, 0, z, t)`` returns the temperature under the beam,
wherever the beam is at ``t``, and asking it for a point 20 mm away is a question
it has no answer to. It is meant to be reached through
:func:`~agent.peak_corrected`, which turns absolute coordinates into the offsets
this takes, using :meth:`centre` -- so the frame the patch is pasted in is the one
the checkpoint says it was fitted in.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from agent import ArrayLike, BaseAgent
from dataset import DEFAULT_FIELD_SHAPE
from utils import load_checkpoint, resolve_device

from .dataset import from_contract
from .laser import BeamPath
from .model import PeakMLP


class CPKMLPAgent(BaseAgent):
    """Wraps a trained :class:`~models.cpkmlp.model.PeakMLP` for inference.

    ``beam`` is the path the offsets are measured from, restored from the
    checkpoint rather than from the module defaults, so a patch fitted against
    one calibration is never pasted against another.
    """

    def __init__(
        self,
        model: PeakMLP,
        bounds: Tensor,
        beam: BeamPath,
        shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
        device: torch.device | str = "cpu",
        chunk: int = 65536,
    ) -> None:
        dtype = next(model.parameters()).dtype
        super().__init__(bounds, shape=shape, device=device, dtype=dtype, chunk=chunk)
        self.model = model.to(self.device).eval()
        self.beam = beam

    def centre(self, time: float) -> tuple[float, float]:
        """``(x, y)`` in metres that this patch's offsets are measured from at ``time``."""
        return self.beam.centre(time)

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        """``[B, 5]`` of ``(dx, dy, z, t, P)`` to ``[B, 1]`` of Kelvin; ``P`` is ignored.

        ``dx`` and ``dy`` are offsets from :meth:`centre`; see the module docstring.
        """
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")

        outputs = []
        for start in range(0, inputs.size(0), self.chunk):
            block = inputs[start : start + self.chunk]
            outputs.append(self.model(from_contract(block)))
        return torch.cat(outputs)


def build_agent(
    checkpoint: Path,
    shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
    device: torch.device | str | None = None,
) -> CPKMLPAgent:
    """Rebuild the network, and the beam frame it was fitted in, from the checkpoint."""
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    payload = load_checkpoint(checkpoint, map_location=device)

    if "bounds" not in payload:
        raise KeyError(f"{checkpoint} predates the `bounds` key; retrain or add it by hand")
    if "anchor" not in payload:
        raise KeyError(
            f"{checkpoint} predates the `anchor` key: it was fitted around the peak "
            "rather than the beam, and its offsets mean something else. Retrain it"
        )

    model = PeakMLP(**payload["architecture"])
    model.load_state_dict(payload["model"])  # normalisation buffers ride along
    return CPKMLPAgent(
        model,
        torch.as_tensor(payload["bounds"]),
        BeamPath(**payload["anchor"]),
        shape=shape,
        device=device,
    )
