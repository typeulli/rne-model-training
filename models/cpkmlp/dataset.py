"""Cut the corpus down to the beam's neighbourhood, in coordinates that ride with it.

Every other model draws its rows from the whole plate. This one keeps, from each
stored snapshot, only the nodes within ``radius`` of where the beam is at that
instant -- the full depth in ``z``, since the pool reaches down from the surface
and the beam has no depth to centre on -- and rewrites their ``x`` and ``y`` as
offsets from the beam:

    ``(x, y, z, t) -> (x - x_l(t), y - y_l, z, t)``     ``x_l(t) = x_0 + v t``

which is the melt pool seen from the beam frame. At 0.25 mm spacing and the
default 2.5 mm radius that is ``20 x 20 x 25 = 10 000`` nodes per snapshot.

**Why the beam and not the peak.** The obvious anchor is the hottest node, and it
was the first one tried. It does not survive contact with the grid. The solver's
maximum trails the beam by 0.24 mm and therefore lands 0.003 mm past the midpoint
between two nodes whose temperatures differ by 0.46 K in a 1353 K rise -- and
because the beam advances exactly 16 nodes per stored step, that near-tie repeats
identically at every instant instead of averaging out. Every model in ``models/``
places its continuous maximum within 0.06 mm of the solver's, a fifth of a node,
and still lands on the other side of that midpoint, so an argmax over predicted
temperatures picks the node *behind* the solver's, always. The patch was then
pasted 0.25 mm off, which on a flank falling 1350 K over a beam radius cost a
couple of hundred Kelvin -- far more than the fit error it was correcting.

The beam has none of that. It is not estimated from a field at all: ``x_l(t)`` is
three calibrated numbers (:mod:`models.cpkmlp.laser`) and a clock, so the frame
the patch is *fitted* in and the frame it is *pasted* in are the same frame
exactly, not to within a node. It is also the physically honest choice -- a
quasi-steady melt pool is stationary in the beam frame, which is precisely what
makes it a function this network can learn.

The offsets are not multiples of the spacing, since the beam sits between nodes,
and they are not symmetric about zero, since a symmetric window in metres clips
to an asymmetric set of nodes. Neither matters: the network is continuous in its
inputs, and the same 16-node advance that ruined the argmax means every snapshot
contributes the *same* offsets, so the window is one consistent set of points
rather than a different one at every step.

Batching, normalisation and the input column order are :mod:`models.cmlp.dataset`'s,
re-exported here: the coordinates arriving in :class:`~dataset.SimulationDataset`
are already relative, so ``(x, y, z, t) -> (t, z, y, x)`` is exactly the
transposition this model needs as well.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dataset import Grid, SimulationDataset, load_grid
from models.cmlp.dataset import CMLPDataset, from_contract, to_inputs

from .laser import BeamPath

__all__ = [
    "BeamPath",
    "CMLPDataset",
    "DEFAULT_RADIUS",
    "NODE_TOLERANCE",
    "build",
    "from_contract",
    "patch_rows",
    "to_inputs",
]

# Half-width of the window kept around the beam, in metres, in x and in y.
DEFAULT_RADIUS = 2.5e-3

# `|x - x_l(t)| <= radius` decides which nodes are kept, and both sides are
# floating point sums of a 0.25 mm spacing, so a node sitting exactly on the
# boundary must not be lost to the last bit. A nanometre is far below the grid
# and far above the error.
NODE_TOLERANCE = 1e-9


def patch_rows(
    grid: Grid, radius: float = DEFAULT_RADIUS, beam: BeamPath = BeamPath()
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One file's beam neighbourhoods as ``(coords[N,4], power[N,1], temperature[N,1])``.

    ``coords`` is ``(dx, dy, z, t)`` -- the column order
    :class:`~dataset.SimulationDataset` takes, with the first two measured from
    the beam of their own snapshot.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")

    coord_blocks, temperature_blocks = [], []
    for index, time in enumerate(grid.t):
        centre_x, centre_y = beam.centre(float(time))
        keep_y = np.abs(grid.y - centre_y) <= radius + NODE_TOLERANCE
        keep_x = np.abs(grid.x - centre_x) <= radius + NODE_TOLERANCE
        if not keep_x.any() or not keep_y.any():
            raise ValueError(
                f"{grid.name}: the beam is off the plate at t = {float(time)} s "
                f"(x_l = {centre_x * 1e3:.3f} mm); nothing to keep"
            )

        block = grid.temperature[index][:, keep_y][:, :, keep_x]  # [nz, ny_p, nx_p]
        mesh_z, mesh_dy, mesh_dx = np.meshgrid(
            grid.z,
            grid.y[keep_y] - centre_y,
            grid.x[keep_x] - centre_x,
            indexing="ij",
        )
        coord_blocks.append(
            np.stack(
                [
                    mesh_dx.reshape(-1),
                    mesh_dy.reshape(-1),
                    mesh_z.reshape(-1),
                    np.full(mesh_dx.size, float(time)),
                ],
                axis=1,
            )
        )
        temperature_blocks.append(block.reshape(-1, 1))

    coords = np.concatenate(coord_blocks)
    temperature = np.concatenate(temperature_blocks)
    power = np.full((coords.shape[0], 1), grid.power)
    return coords, power, temperature


def build(
    paths: list[Path],
    *,
    radius: float = DEFAULT_RADIUS,
    beam: BeamPath = BeamPath(),
    exclude: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> SimulationDataset:
    """Load ``paths`` and keep only the beam neighbourhood of each snapshot.

    The counterpart of :meth:`~dataset.SimulationDataset.from_files`, and it
    returns the same class, so everything downstream -- the split, the sampler,
    the normalisation read off the domain -- is unchanged. It goes through
    :func:`~dataset.load_grid` rather than the flat loader because the window is
    a property of a *snapshot*, which only the structured view exposes.

    ``exclude`` drops the first that many time steps of every file. Unlike the
    peak-anchored corpus this replaced, ``t = 0`` is perfectly well defined here
    -- the beam has a position before it has heated anything -- so the flag is a
    choice about the initial transient rather than a repair.

    No sub-ambient clipping is offered here, unlike the flat loader: the solver
    artefact it exists for lives far from the beam, and nothing far from the beam
    is kept.
    """
    if not paths:
        raise FileNotFoundError("no .npy files given")

    coord_blocks, power_blocks, temperature_blocks = [], [], []
    for path in paths:
        grid = load_grid(path).exclude_initial_steps(exclude)
        coords, power, temperature = patch_rows(grid, radius, beam)
        if verbose:
            print(
                f"[data] {grid.name}: {coords.shape[0]} rows from {grid.t.size} "
                f"snapshots around the beam (+/-{radius * 1e3:g} mm), "
                f"P={grid.power:g} W, peak {temperature.max():.1f} K"
            )
        coord_blocks.append(coords)
        power_blocks.append(power)
        temperature_blocks.append(temperature)

    def stack(blocks: list[np.ndarray]) -> torch.Tensor:
        return torch.as_tensor(np.concatenate(blocks), dtype=dtype, device=device)

    return SimulationDataset(
        stack(coord_blocks), stack(power_blocks), stack(temperature_blocks)
    )
