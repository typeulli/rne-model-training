"""The moving-source field CJMLP gates its output with.

``models/cgmlp/laser.py`` puts a Gaussian on the beam. A Gaussian is the shape of
the *source*, not of the field the source makes: it is symmetric, it has no wake,
it does not know the plate has a bottom, and it does not know the beam is moving.
This module replaces it with a closed-form solution of the problem the beam is
actually solving.

In the frame that travels with the beam, ``xi = x - x_l(t)``, a point source of
strength ``Q`` moving at ``v`` through a medium of conductivity ``k`` and
diffusivity ``alpha`` leaves the steady field

    ``T - T_inf = Q / (2 pi k) * exp[-v (r + xi) / (2 alpha)] / r``

-- Rosenthal's solution. The exponent is what makes it a *moving* source: ahead of
the beam ``r ~ +xi`` and the field dies within ``alpha/v``; behind it ``r ~ -xi``,
so ``r + xi -> 0``, the exponential stops decaying and what is left is the ``1/r``
tail. That asymmetry is the whole difference between this and a Gaussian, and it
is the trailing wake every figure in this repo shows the models struggling with.

Two corrections turn the point source into this plate:

**The two faces.** The plate is ``d`` thick, insulated on top (the beam's own
face) and held at ``T_inf`` underneath. Both conditions are met exactly by the
method of images: copies of the source at ``z = 2 n d`` carrying sign ``(-1)^n``.
Pairs either side of the top face share a sign, so ``dT/dz`` vanishes there; pairs
either side of the bottom face have opposite signs, so ``T`` vanishes there. The
sum over ``n`` is the field of all of them.

**The beam is not a point.** A true point source is infinite at ``r = 0``, and a
gate must be finite everywhere. ``r`` is softened to
``sqrt(xi^2 + y^2 + z^2 + sigma^2)``, which is finite at the origin and reduces to
the point source far from it.

``sigma`` is the beam's own width, ``BEAM_RADIUS``, and how that value was
arrived at is worth recording, because the obvious way of choosing it is wrong.

Fit ``sigma`` by making the gate *look like* the field -- least squares against
the solver's normalised temperature, which is what a soft core is nominally for --
and it comes out at 0.45 mm, the diffusion length ``2 alpha / v = 0.479 mm`` to
within 7%. That gate is the better *picture* by every measure: it tracks the wake,
it decays into the depth five times more accurately than a Gaussian, and its
maximum sits 0.28 mm behind the beam where the solver's sits 0.24 mm behind. Train
on it and the model is the worst of the three:

    sigma [mm]            0.45    0.90    1.70
    shape RMSE, line     0.092   0.181   0.397     <- lower is a better picture
    val RMSE [K]         2.149   1.799   1.204     <- lower is a better model
    val max  [K]         192.1    77.7    29.5

(64x3 on the 175 W split, ``--exclude 1``; ``cgmlp``'s Gaussian scores 1.532 K on
the same run and ``cmlp``, ungated, 1.689 K.)

The ranking inverts because a gate is not a prediction, it is a factor. The
network has to supply ``T / G``, so what a good ``G`` has to be is *smooth* and
roughly right, not pointwise accurate. At 0.45 mm the gate falls from 1.00 to 0.55
across a single 0.25 mm cell at the beam and to 0.005 a millimetre ahead of it,
where the field is still at a third of its peak: the quotient the network is left
with is worse conditioned than the field it started from, and the product of two
steep functions overshoots the melt pool by 220 K. At the beam radius the gate is
a broad envelope that carries the asymmetry and nothing sharp, and the network
does the rest.

So the shape fit is not what selects ``sigma``; it is only what shows why the
selection is not obvious. Anything between the diffusion length and the beam
radius is defensible on the physics, and the training runs above are what decide
between them.

What the solution assumes, and this problem does not satisfy exactly: no
convection or radiation from the top face (it is taken as insulated), an
infinitely wide plate (no lateral faces), and a steady state in the beam frame (no
start-up transient). None of that disqualifies it as a *gate* -- it is a prior the
network multiplies, not a prediction -- but it is why ``cjmlp`` is not simply this
formula with a fitted amplitude.

``Q`` and ``k`` do not appear below. They set only the amplitude, and the gate is
normalised to 1 under the beam, exactly as ``cgmlp``'s Gaussian is: the amplitude
is what the network and its ``dT`` scale already carry. What survives the
normalisation is the shape, and the shape is set by ``v / (2 alpha)``, ``sigma``
and ``d`` alone.

Constants are the ones ``python calibrate.py`` recovered from the data; every
quantity is SI.
"""

from __future__ import annotations

import torch
from torch import Tensor

MM = 1.0e-3

# ---------------------------------------------------------------------------
# Fitted jointly across all seven powers by `python calibrate.py`.
# ---------------------------------------------------------------------------
BEAM_RADIUS = 1.6971 * MM  # r_b [m], the 1/e^2 spot -- the `sigma` of the soft core
LASER_START_X = 4.8683 * MM
LASER_Y = 4.9929 * MM
SCAN_SPEED = 10.0000 * MM  # v [m s^-1]

# alpha = k / (rho c_p), fitted at correlation 0.979 against the interior
# Laplacian; see models/cpimlp/loss.py, where the same number sets `conductivity`.
DIFFUSIVITY = 2.3935e-6  # alpha [m^2 s^-1]

# The soft core is the beam's own width. The module docstring has the runs that
# settle it, and why the shape fit that disagrees is the wrong question to ask.
SOFT_CORE = BEAM_RADIUS  # sigma [m]

# Not fitted: the geometry of the exported domain. The beam scans z = z_max and
# the solver holds z = 0 at ambient, so the plate is `THICKNESS` thick and its
# heated face is at `LASER_Z`.
LASER_Z = 6.0000 * MM
THICKNESS = 6.0000 * MM  # d [m]

# Terms kept either side of n = 0. The images sit 2d = 12 mm apart and the field
# decays over alpha/v = 0.24 mm, so they are individually negligible -- but they
# are what makes T vanish on the bottom face, where the real source is just as
# negligible, so the cancellation there is exact regardless of how small both are.
IMAGES = 3


def beam_centre_x(t: Tensor | float) -> Tensor | float:
    """``x_l(t) = LASER_START_X + SCAN_SPEED * t``, the beam centre at time ``t``."""
    return LASER_START_X + SCAN_SPEED * t


def _series(
    xi: Tensor,
    y: Tensor,
    depth: Tensor,
    speed: Tensor | float,
    diffusivity: Tensor | float,
    soft_core: Tensor | float,
    thickness: Tensor | float,
    images: int,
) -> Tensor:
    """``sum_n (-1)^n exp[-v (r_n + xi) / (2 alpha)] / r_n``, in ``m^-1``.

    ``depth`` is measured from the heated face, so it runs from 0 down to ``-d``.
    The exponent is never positive -- ``r_n >= |xi|`` by construction -- so no term
    can overflow, and terms far ahead of the beam underflow to zero, which is the
    right answer for them.
    """
    order = torch.arange(-images, images + 1, device=xi.device)
    sign = (1 - 2 * (order % 2)).to(xi.dtype)  # +1, -1, +1, ... about n = 0
    offset = depth - 2.0 * thickness * order.to(xi.dtype)  # [batch, 2*images+1]

    radius = torch.sqrt(xi**2 + y**2 + offset**2 + soft_core**2)
    return (
        sign * torch.exp(-speed * (radius + xi) / (2.0 * diffusivity)) / radius
    ).sum(dim=-1, keepdim=True)


_PEAK: dict[tuple[float, float, float, int], float] = {}


def _peak(
    diffusivity: Tensor | float,
    soft_core: Tensor | float,
    thickness: Tensor | float,
    images: int,
) -> float:
    """The series' maximum, which normalises the gate to a unit peak.

    It has no closed form -- the maximum sits behind the beam, at an offset that
    depends on all three constants -- so it is found once by a 5 um sweep of the
    surface centre line in float64 and cached against the constants that produced
    it. Once, not per call: the constants are buffers, so every call after the
    first is a dict lookup.

    Being a plain float, it does not carry a gradient. Nothing needs one today --
    the constants are fixed, not learned -- but a version that fitted ``sigma``
    would have to differentiate through this, and a sweep is the wrong shape for
    that.
    """
    key = (float(diffusivity), float(soft_core), float(thickness), int(images))
    if key not in _PEAK:
        # Wide enough to hold the maximum for any soft core up to a few mm; the
        # far end of that range puts it 3.2 mm back, well inside 20 mm.
        offsets = torch.linspace(-0.020, 0.002, 4401, dtype=torch.float64).reshape(-1, 1)
        zeros = torch.zeros_like(offsets)
        # `key`, not the arguments: the constants arrive as buffers, which live on
        # the training device, and this sweep is a CPU one.
        values = _series(offsets, zeros, zeros, SCAN_SPEED, *key[:3], images)
        _PEAK[key] = float(values.max())
    return _PEAK[key]


def moving_source(
    coords: Tensor,
    diffusivity: Tensor | float = DIFFUSIVITY,
    soft_core: Tensor | float = SOFT_CORE,
    thickness: Tensor | float = THICKNESS,
    images: int = IMAGES,
) -> Tensor:
    """``[batch, 1]``: the moving-source field, normalised to a unit peak.

    ``coords`` is ``[batch, 4]`` of ``(x, y, z, t)`` in metres and seconds. The
    result is a dimensionless envelope in the same role, and on the same scale, as
    ``cgmlp``'s unit-peak Gaussian -- 1 at its maximum and falling to 0 away from
    the beam -- so ``G = j + p`` means exactly what it means there.

    Where the two differ is that the Gaussian's peak is *on* the beam by
    construction and this one's is a little behind it, by the same lag the solver
    has. That lag is not put in; it comes out of the exponent.
    """
    x, y, z, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3], coords[:, 3:4]
    shifted = x - beam_centre_x(t)
    depth = z - LASER_Z

    field = _series(
        shifted,
        y - LASER_Y,
        depth,
        SCAN_SPEED,
        diffusivity,
        soft_core,
        thickness,
        images,
    )
    return field / _peak(diffusivity, soft_core, thickness, images)
