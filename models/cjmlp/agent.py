"""CJMLP behind the shared :class:`~agent.BaseAgent` inference contract.

As in :mod:`models.cgmlp.agent`, ``predict_at`` takes the ``(x, y, z, t, P)`` of
the contract and drops the ``P`` column on the way in; the gate is applied inside
the network, from the coordinate columns, so nothing here has to know that it is
a moving-source field rather than a Gaussian either.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from agent import ArrayLike, BaseAgent
from dataset import DEFAULT_FIELD_SHAPE
from utils import load_checkpoint, resolve_device

from .dataset import from_contract
from .model import ControlSourceMLP


class CJMLPAgent(BaseAgent):
    """Wraps a trained :class:`~models.cjmlp.model.ControlSourceMLP` for inference."""

    def __init__(
        self,
        model: ControlSourceMLP,
        bounds: Tensor,
        shape: tuple[int, int, int] = DEFAULT_FIELD_SHAPE,
        device: torch.device | str = "cpu",
        chunk: int = 65536,
    ) -> None:
        dtype = next(model.parameters()).dtype
        super().__init__(bounds, shape=shape, device=device, dtype=dtype, chunk=chunk)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        """``[B, 5]`` of ``(x, y, z, t, P)`` to ``[B, 1]`` of Kelvin; ``P`` is ignored."""
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
) -> CJMLPAgent:
    """Rebuild the network from the architecture stored alongside the weights.

    The gate's constants are buffers as well as constructor arguments, so the
    envelope a checkpoint was fitted under is restored from the file rather than
    from whatever :mod:`models.cjmlp.laser` currently defaults to.
    """
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    payload = load_checkpoint(checkpoint, map_location=device)

    if "bounds" not in payload:
        raise KeyError(f"{checkpoint} predates the `bounds` key; retrain or add it by hand")

    model = ControlSourceMLP(**payload["architecture"])
    model.load_state_dict(payload["model"])  # normalisation and gate buffers ride along
    return CJMLPAgent(model, payload["bounds"], shape=shape, device=device)
