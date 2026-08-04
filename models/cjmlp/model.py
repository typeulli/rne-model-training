"""The CGMLP gate replaced by the field a moving source actually makes.

:class:`~models.cgmlp.model.ControlGatedMLP` with one thing changed:

    ``T_hat = T_amb + dT * (net(t, z, y, x) * G(x, y, z, t) + b)``
    ``G = j(x, y, z, t) + p``

``j`` is :func:`~models.cjmlp.laser.moving_source` -- Rosenthal's travelling-source
solution, imaged to satisfy the plate's two faces and softened at the origin --
where ``cgmlp`` puts a Gaussian. ``p`` is the same single learnable scalar, the
floor the gate relaxes to away from the beam, and the network, the objective, the
batches and the training loop are all unchanged. So the pair ``cjmlp`` / ``cgmlp``
isolates exactly one thing: whether the prior should be the shape of the *source*
or the shape of the *field the source makes*.

There is reason to expect it matters. A Gaussian is symmetric about the beam and
the field is not: it has a steep front, a long wake behind, and a decay into the
depth that ends on a face held at ambient. All of that is in ``j`` analytically,
so the network is left to correct a shape that is already nearly right rather than
to build the asymmetry itself out of a symmetric envelope. What ``j`` does not
have is the transient at start-up, the lateral faces, and the losses from the top
surface -- see :mod:`models.cjmlp.laser` -- and those are what ``p``, the bias and
the network still have to supply.

Like ``cgmlp``, the gate is built from the beam's *position*, which is a function
of time alone, so nothing in it needs the laser power and the whole prior survives
into a ``P``-blind control intact.

The head ``Linear`` carries no bias of its own -- it would be gated along with
everything else -- and ``b`` is a separate parameter added *after* the gate.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from .laser import DIFFUSIVITY, IMAGES, SOFT_CORE, THICKNESS, moving_source

INPUT_DIM = 4  # (t, z, y, x)
INPUT_NAMES = ("t", "z", "y", "x")

# Columns of a `(t, z, y, x)` row holding `(x, y, z, t)`, which is the order the
# beam geometry in `laser.py` is written against.
COORD_COLUMNS = (3, 2, 1, 0)


def _normalisation_buffer(
    values: Iterable[float] | None, dim: int, default: float
) -> Tensor:
    """Build a ``[1, dim]`` buffer, broadcasting a scalar and validating the length."""
    if values is None:
        return torch.full((1, dim), float(default))
    tensor = torch.as_tensor(list(values), dtype=torch.get_default_dtype()).reshape(1, -1)
    if tensor.numel() == 1:
        tensor = tensor.expand(1, dim).contiguous()
    if tensor.numel() != dim:
        raise ValueError(f"expected {dim} normalisation values, got {tensor.numel()}")
    return tensor


class ControlSourceMLP(nn.Module):
    """``T_hat(t, z, y, x)`` as a dense stack gated by a moving-source field, blind to ``P``.

    ``input_mean`` and ``input_scale`` are ``[4]`` in the input column order;
    ``temperature_offset`` and ``temperature_scale`` map the network's O(1) output
    back to Kelvin. Those, and the three constants the gate's shape depends on, are
    buffers -- so the envelope a checkpoint was trained under travels with it and
    is not silently re-read from this module's defaults.
    """

    input_mean: Tensor
    input_scale: Tensor
    temperature_offset: Tensor
    temperature_scale: Tensor
    diffusivity: Tensor
    soft_core: Tensor
    thickness: Tensor

    def __init__(
        self,
        hidden_layers: Sequence[int] = (256, 256, 256, 256),
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        use_bias: bool = True,
        input_mean: Sequence[float] | None = None,
        input_scale: Sequence[float] | None = None,
        temperature_offset: float = 0.0,
        temperature_scale: float = 1.0,
        diffusivity: float = DIFFUSIVITY,
        soft_core: float = SOFT_CORE,
        thickness: float = THICKNESS,
        images: int = IMAGES,
        gate_offset: float = 0.5,
    ) -> None:
        """``diffusivity``, ``soft_core`` and ``thickness`` are the ``alpha``,
        ``sigma`` and ``d`` of :func:`~models.cjmlp.laser.moving_source`. Together
        with the scan speed they are the entire shape of the gate: ``v / (2 alpha)``
        sets how far the wake reaches, ``sigma`` how sharp the peak is, ``d`` where
        the field is pinned to ambient underneath. ``images`` is how many terms of
        the image series are summed either side of the real source.

        ``gate_offset`` is only the *initial* value of the learnable ``p`` in
        ``G = j + p``; training moves it, and the trained value is what rides in
        the state dict. It is a constructor argument rather than a buffer for that
        reason -- the number worth carrying is the one learned, not the one guessed.
        """
        super().__init__()
        if temperature_scale == 0.0:
            raise ValueError("temperature_scale must be non-zero")
        if soft_core <= 0.0:
            raise ValueError(f"soft_core must be positive, got {soft_core}")
        if diffusivity <= 0.0:
            raise ValueError(f"diffusivity must be positive, got {diffusivity}")
        if thickness <= 0.0:
            raise ValueError(f"thickness must be positive, got {thickness}")

        layers: list[nn.Module] = []
        current = INPUT_DIM
        for width in hidden_layers:
            layers.append(nn.Linear(current, width))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = width
        # No bias on the head: it would be multiplied by the gate along with the
        # rest of the output, and the gate-independent offset is `self.bias`.
        layers.append(nn.Linear(current, 1, bias=False))
        self.network = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None
        # The `p` of `G = j + p`: the floor the gate relaxes to away from the beam.
        self.gate_offset = nn.Parameter(torch.tensor(float(gate_offset)))
        # Structural rather than physical: the number of terms, not a length.
        self.images = int(images)

        self.register_buffer("input_mean", _normalisation_buffer(input_mean, INPUT_DIM, 0.0))
        self.register_buffer("input_scale", _normalisation_buffer(input_scale, INPUT_DIM, 1.0))
        self.register_buffer("temperature_offset", torch.tensor(float(temperature_offset)))
        self.register_buffer("temperature_scale", torch.tensor(float(temperature_scale)))
        self.register_buffer("diffusivity", torch.tensor(float(diffusivity)))
        self.register_buffer("soft_core", torch.tensor(float(soft_core)))
        self.register_buffer("thickness", torch.tensor(float(thickness)))

    def forward(self, inputs: Tensor) -> Tensor:
        """``[batch, 4]`` of ``(t, z, y, x)`` in SI units to ``[batch, 1]`` in Kelvin."""
        if inputs.dim() != 2 or inputs.size(-1) != INPUT_DIM:
            raise ValueError(
                f"inputs must have shape [batch, {INPUT_DIM}] for {INPUT_NAMES}, "
                f"got {tuple(inputs.shape)}"
            )

        normalised = (inputs - self.input_mean) / self.input_scale
        # The gate is built from the physical columns, so it stays in metres and
        # seconds while the dense stack sees O(1) values.
        gate = (
            moving_source(
                inputs[:, COORD_COLUMNS],
                diffusivity=self.diffusivity,
                soft_core=self.soft_core,
                thickness=self.thickness,
                images=self.images,
            )
            + self.gate_offset
        )

        latent = self.network(normalised) * gate
        if self.bias is not None:
            latent = latent + self.bias
        return self.temperature_offset + self.temperature_scale * latent
