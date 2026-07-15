"""A dense stack that predicts a spectrum instead of a temperature.

Every other model in ``models/`` is pointwise: it maps one ``(x, y, z, t; P)`` to
one ``T``, so a full volume costs ``25 x 41 x 161 = 165 025`` forward passes. This
one maps

    ``(P, t) -> every stored Fourier coefficient``

and recovers the whole field with a single inverse transform. Two numbers in, one
volume out.

The network itself is three hidden layers and nothing else -- no gate, no operator
structure, no physics. That is deliberate: what is being tested here is the
*representation*, not the architecture. The spatial dependence is already carried
by the basis :mod:`~models.fmlp.dataset` built, so all that is left to learn is how
the coefficients move as the power and the clock change, and a map from ``R^2``
does not need depth to be expressive -- it needs a well-conditioned target.

And that is the claim ``TODO.md`` set out to test. Real-space temperature is a bad
regression target: near-ambient almost everywhere, violently peaked over a
beam-radius, so the MSE is dominated by the cold desert and the peak is a rare
event the loss barely notices. The Fourier transform inverts exactly this. A sharp
isolated peak does not concentrate its energy in a few wavenumbers, it *spreads*
it -- a narrow Gaussian of radius ``r_b`` transforms to a broad one of width
``~1/r_b`` -- so the target becomes many coefficients of comparable size rather
than one spike in a sea of zeros. Sharper in space is flatter in ``k``, and flat is
what a network can fit.

The output layer is over 99% of the weights, which sounds alarming and is not. The
targets are 56 snapshots of a field that varies smoothly in two parameters, so they
span a subspace of at most 56 dimensions no matter how many coefficients are
stored; the last hidden layer has room to spare. The bottleneck was never capacity.

The activation is ``ReLU``, not the ``SiLU`` the other dense models use. They need
a smooth one because a PDE residual differentiates them twice and a piecewise-linear
activation would make the Laplacian identically zero; nothing here is differentiated
with respect to anything, so the reason does not apply.

:meth:`FourierMLP.forward` returns the *normalised coefficient vector*, because
that is what the loss is written against and it is not a physical observable.
:meth:`FourierMLP.field` returns Kelvin on the grid, and is what the agent, the
validation metric and every figure actually use. The two are one denormalisation
and one ``irfftn`` apart, and both are differentiable, so a real-space objective
could be attached to ``field`` without changing anything here -- which is the door
``TODO.md`` leaves open for the physics-informed variant.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn

INPUT_DIM = 2  # (P, t)
INPUT_NAMES = ("P", "t")


def retrend_x(field: Tensor, ramp: Tensor) -> Tensor:
    """Put back the linear ``x`` ramp :func:`~models.fmlp.dataset.detrend_x` removed.

    ``field`` is ``[B, nx, ny, nz]`` and ``ramp`` is ``[B, ny, nz]``. The
    coefficients carry the pool that travels with the laser; the ramp carries the
    wrap mismatch that does not. Keeping them apart is the whole point of the
    split, and this is where they are put back together.
    """
    fraction = torch.arange(
        field.size(1), dtype=field.dtype, device=field.device
    ) / field.size(1)
    return field + ramp[:, None] * fraction[None, :, None, None]


class FourierMLP(nn.Module):
    """``(P, t) -> {T_hat(kappa)}``, and :meth:`field` to turn that into Kelvin.

    Everything needed to invert the transform is a buffer or a constructor
    argument, so a checkpoint reconstructs the field exactly as the run that wrote
    it did: which wavenumbers were kept, whether they were de-rotated and at what
    scan speed, whether an ``x`` ramp was split off, and the statistics they were
    standardised by.
    """

    input_scale: Tensor
    coefficient_mean: Tensor
    coefficient_scale: Tensor
    temperature_offset: Tensor
    mx: Tensor
    my: Tensor

    def __init__(
        self,
        grid_shape: Sequence[int],
        kx: int,
        ky: int,
        hidden_layers: Sequence[int] = (128, 128, 128),
        activation: type[nn.Module] = nn.ReLU,
        detrend: bool = False,
        derotate: bool = True,
        scan_speed: float = 0.01,
        length_x: float = 0.04025,
        input_scale: Sequence[float] = (250.0, 2.8),
        coefficient_mean: Sequence[float] | float = 0.0,
        coefficient_scale: Sequence[float] | float = 1.0,
        temperature_offset: float = 298.0,
    ) -> None:
        super().__init__()

        depth, height, width = (int(size) for size in grid_shape)
        self.grid_shape = (depth, height, width)
        self.kx, self.ky = int(kx), int(ky)
        self.detrend = bool(detrend)
        self.derotate = bool(derotate)
        self.scan_speed = float(scan_speed)
        self.length_x = float(length_x)

        # 0..kx, -kx..-1 -- the signed wavenumbers, in the order the FFT lays them
        # out. y keeps only its non-negative half; the rest are conjugates.
        mx = torch.cat(
            (torch.arange(self.kx + 1), torch.arange(-self.kx, 0))
        ).to(torch.long)
        my = torch.arange(self.ky + 1, dtype=torch.long)

        self.n_coefficient = 2 * mx.numel() * my.numel() * depth
        self.n_ramp = height * depth if self.detrend else 0
        n_out = self.n_coefficient + self.n_ramp

        layers: list[nn.Module] = []
        current = INPUT_DIM
        for size in hidden_layers:
            layers.append(nn.Linear(current, size))
            layers.append(activation())
            current = size
        layers.append(nn.Linear(current, n_out))
        self.network = nn.Sequential(*layers)

        self.register_buffer("mx", mx)
        self.register_buffer("my", my)
        self.register_buffer(
            "input_scale", torch.as_tensor(list(input_scale), dtype=torch.get_default_dtype()).reshape(1, INPUT_DIM)
        )
        self.register_buffer(
            "coefficient_mean",
            torch.as_tensor(coefficient_mean, dtype=torch.get_default_dtype()).reshape(-1),
        )
        self.register_buffer(
            "coefficient_scale",
            torch.as_tensor(coefficient_scale, dtype=torch.get_default_dtype()).reshape(-1),
        )
        self.register_buffer(
            "temperature_offset", torch.tensor(float(temperature_offset))
        )

    # -- the network ---------------------------------------------------------

    def forward(self, inputs: Tensor) -> Tensor:
        """``[B, 2]`` of ``(P, t)`` in SI units to ``[B, n_out]`` normalised coefficients.

        ``P`` and ``t`` are both scaled from zero rather than centred: a laser power
        of zero means no heating and ``t = 0`` is the initial condition, so both
        ends are physically meaningful and their midpoints are an artefact of
        whichever sweep happens to be on disk.
        """
        if inputs.dim() != 2 or inputs.size(-1) != INPUT_DIM:
            raise ValueError(
                f"inputs must have shape [batch, {INPUT_DIM}] for {INPUT_NAMES}, "
                f"got {tuple(inputs.shape)}"
            )
        return self.network(inputs / self.input_scale)

    # -- the field it stands for ---------------------------------------------

    def _lab_frame(self, coefficient: Tensor, time: Tensor) -> Tensor:
        """Undo the de-rotation: spin the coefficients back into the lab frame.

        The dataset divided out ``exp(-2i pi kx v t)``, the phase a field of the
        form ``f(x - v t)`` puts on every mode. This multiplies it back in. It is
        evaluated at whatever ``t`` is asked for, not at a snapshot time, so the
        model is continuous in time even though it was fitted to eight instants.
        """
        wavenumber = self.mx.to(time.dtype) / self.length_x
        angle = -2.0 * math.pi * wavenumber[None, :] * self.scan_speed * time[:, None]
        phase = torch.polar(torch.ones_like(angle), angle)
        return coefficient * phase[:, :, None, None]

    def field(self, inputs: Tensor) -> Tensor:
        """``[B, 2]`` of ``(P, t)`` to ``[B, nz, ny, nx]`` of Kelvin.

        One forward pass and one ``irfftn`` per volume. This is the operation the
        model exists for; ``predict_at`` is the derived one, which is the exact
        reverse of every other agent in the repo.
        """
        depth, height, width = self.grid_shape
        flat = self.coefficient_mean + self.coefficient_scale * self.forward(inputs)

        packed = flat[:, : self.n_coefficient].reshape(
            -1, self.mx.numel(), self.my.numel(), depth, 2
        )
        coefficient = torch.complex(packed[..., 0], packed[..., 1])
        if self.derotate:
            coefficient = self._lab_frame(coefficient, inputs[:, 1])

        spectrum = torch.zeros(
            (inputs.size(0), width, height // 2 + 1, depth),
            dtype=coefficient.dtype,
            device=coefficient.device,
        )
        spectrum[:, self.mx[:, None] % width, self.my[None, :], :] = coefficient

        rise = torch.fft.irfftn(
            spectrum, s=(width, height), dim=(1, 2), norm="ortho"
        )  # [B, nx, ny, nz]
        if self.detrend:
            ramp = flat[:, self.n_coefficient :].reshape(-1, height, depth)
            rise = retrend_x(rise, ramp)

        # (B, nx, ny, nz) -> (B, nz, ny, nx), the (D, H, W) of the shared contract
        return (rise + self.temperature_offset).permute(0, 3, 2, 1)
