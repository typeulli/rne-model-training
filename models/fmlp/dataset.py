"""Turn the shared corpus into the spatial Fourier coefficients ``fmlp`` regresses.

Every other model draws rows of ``(x, y, z, t, P) -> T`` from
:class:`~dataset.SimulationDataset`. This one does not draw rows at all: its
sample is a whole snapshot. The corpus holds seven powers of eight snapshots
each, so the training set is **56 pairs** of ``(P, t)`` against a coefficient
vector -- and the entire thing fits in a few megabytes, which is why it is built
in memory at startup rather than cached to disk.

Three decisions are made here, and the first two are forced by the data.

**Only ``x`` and ``y`` are transformed.** ``z`` runs from a Dirichlet substrate at
the bottom (exactly 298 K, the solver holds it there) to the laser-heated top
face, 1917 K above ambient at 250 W. A DFT wraps those two faces onto each other,
so it would see a 1917 K cliff that does not exist, and the coefficients would
decay like 1/m and ring under truncation. So ``z`` is left on the grid --
``rfftn`` over ``(x, y)`` only, ``z`` kept whole. It costs a factor of ``nz`` in
stored numbers and buys back an exact bottom face: a reconstruction from these
coefficients reproduces ``z = 0`` to 0.00 K, for free and by construction.

**Only the non-negative ``y`` wavenumbers are stored.** The field is real, so
``C(-kx, -ky) = conj(C(kx, ky))`` and half the plane is redundant. ``rfftn``
drops it; :meth:`~models.fmlp.model.FourierMLP.field` puts it back.

**The box is chosen by energy, not by hand.** :func:`energy_box` finds the
cheapest ``|mx| <= kx, |my| <= ky`` that retains ``--energy-target`` of *every*
power's energy -- every power, not the pooled total, because energy scales like
``P^2`` and a budget fitted to the sum would be free to discard 100 W outright.
On the shipped sweep:

    target      kx    ky    outputs    floor RMSE    peak
    0.9999      19     5     11 700      0.53 K     -0.68%
    0.99999     38     6     26 950      0.20 K     -0.04%

-- quoting the worst power of the seven, which is always 250 W, since the error
scales with the field. The floor is what the kept modes reconstruct *exactly*; no
network fitted to them can do better, so :mod:`~models.fmlp.train` reports it
alongside the model's own error, and the gap between the two is the only part the
network is answerable for.

The jump from ``kx = 19`` to ``38`` for one more nine is not the melt pool getting
sharper. It is the **x-wrap ridge**: the 15.7 K step the DFT sees between the far
end of the plate and the near one, an artefact of a periodicity the plate does not
have. ``--detrend`` takes that ramp out before transforming and stores it
alongside as ``ny * nz`` extra numbers, which collapses the tail (``kx = 38 -> 24``
at 0.99999) -- and it matters for more than cost, because that ridge is pinned to
the domain rather than travelling with the laser, which is exactly what de-rotation
below cannot handle.

De-rotation is the third decision, and it is what makes the target a function of
time rather than a list of eight numbers. A source moving at ``v`` makes the field
roughly ``f(x - v t)``, whose transform is ``g(kx) * exp(-2i pi kx v t)``: every
coefficient spins in the complex plane, the faster the mode the faster it spins.
Mode ``mx = 19`` turns 13 times over the 2.8 s run -- and the run is sampled 8
times, every 0.4 s, which resolves the spin only out to ``|mx| = 5``. Everything
above that is aliased. :func:`spin` is that phase, known analytically, so it is
divided out here and multiplied back in at reconstruction: an exact, invertible
multiplication, so nothing is lost.

That the snapshots alias it is not an objection, and this is the crux. The phase is
evaluated at the exact times, not inferred from them -- aliasing stops you
*inferring* a frequency from samples, and here the frequency is known.

What it costs to skip it is worth stating precisely, because the headline metric
almost hides it. Trained with ``--no-derotate`` the model still scores 0.421 K on
the held-out power against 0.353 K with (see ``train.py``), which looks like a minor
ablation -- but only because validation holds out a *power*, and the spin does not
depend on power. Ask the two models for a time they were not shown, and they part
company completely: between snapshots the raw model's peak collapses from 1230 K to
about 770 K and oscillates from one to the other, while the de-rotated one holds a
flat 1230 K, which is what a quasi-steady melt pool actually does. The raw model has
memorised eight aliased samples; the de-rotated one has learned a function of ``t``.
Only the second is a surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset import Grid, load_grid

# The solver holds the substrate at ambient and heats from the top face, so this
# is both the initial condition and the bottom Dirichlet value. Subtracting it
# gives dT, which is what actually gets transformed: it decays to zero away from
# the beam, and a transform of a field sitting on a 298 K pedestal would spend its
# entire DC coefficient saying so.
AMBIENT_TEMPERATURE = 298.0

# The one thing de-rotation needs to know. Recovered from the data by
# `python calibrate.py`, and restated here rather than imported from
# models/gmlp/laser.py for the reason that module gives for being a copy itself:
# so an experiment on one model cannot shift another underneath it.
SCAN_SPEED = 10.0e-3  # [m s^-1]


# ---------------------------------------------------------------------------
# The transform, and the ramp it needs taken out first
# ---------------------------------------------------------------------------


def detrend_x(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove the linear ramp that makes ``x`` periodic. ``(residual, ramp)``.

    ``field`` is ``[nt, nx, ny, nz]``. The DFT glues ``x = x_max`` back onto
    ``x = 0``, and they do not agree; subtracting a ramp that carries the whole
    mismatch leaves a residual whose ends do, so the transform stops spending high
    wavenumbers on a discontinuity that is an artefact of its own periodicity.

    ``ramp`` is the ``[nt, ny, nz]`` end-face mismatch, kept so
    :func:`~models.fmlp.model.retrend_x` can put it back.
    """
    mismatch = field[:, -1] - field[:, 0]
    fraction = np.arange(field.shape[1]) / field.shape[1]
    return field - mismatch[:, None] * fraction[None, :, None, None], mismatch


def spin(mx: np.ndarray, times: np.ndarray, length_x: float, speed: float) -> np.ndarray:
    """``exp(+2i pi kx v t)``, the phase to divide out. ``[nt, len(mx)]``.

    ``length_x`` is ``nx * dx`` -- the period the DFT assumes, which is one cell
    wider than the ``x`` extent of the plate and is the length the wavenumbers are
    actually referred to.
    """
    wavenumber = mx / length_x
    return np.exp(2j * np.pi * wavenumber[None, :] * speed * times[:, None])


def temperature_rise(grid: Grid) -> np.ndarray:
    """``dT`` as ``[nt, nx, ny, nz]``.

    The corpus stores ``[nt, nz, ny, nx]`` -- the ``(D, H, W)`` of a ``Conv3d``.
    The transform is written over ``(x, y)`` with ``z`` trailing and untouched, so
    the axes are reversed once here rather than at every call site.
    """
    return np.transpose(grid.temperature - AMBIENT_TEMPERATURE, (0, 3, 2, 1))


# ---------------------------------------------------------------------------
# The mode budget
# ---------------------------------------------------------------------------


def _fold(power: np.ndarray, axis: int) -> np.ndarray:
    """Sum ``|C|^2`` over ``+m`` and ``-m``, leaving ``|m| = 0 .. N//2`` on ``axis``.

    ``nx`` and ``ny`` are both odd here, so there is no Nyquist bin to double
    count: index 0 is ``m = 0``, ``1 .. N//2`` are ``+1 .. +N//2``, and the
    reversed tail is ``-1 .. -N//2``.
    """
    power = np.moveaxis(power, axis, 0)
    half = power.shape[0] // 2
    folded = np.empty((half + 1, *power.shape[1:]))
    folded[0] = power[0]
    folded[1:] = power[1 : half + 1] + power[: half : -1]
    return np.moveaxis(folded, 0, axis)


def energy_box(fields: list[np.ndarray], target: float) -> tuple[int, int]:
    """The cheapest ``(kx, ky)`` retaining ``target`` of *every* field's energy.

    Cheapest means fewest stored numbers, and since ``z`` is kept whole every box
    costs ``(2kx + 1)(ky + 1) * nz`` complex coefficients -- so the cost is a
    product over the two transformed axes and the search is a plain argmin over
    the grid of boxes that clear the target.
    """
    spectra, totals = [], []
    for field in fields:
        squared = np.abs(np.fft.fftn(field, axes=(1, 2), norm="ortho")) ** 2
        energy = _fold(_fold(squared.sum(axis=0), 0), 1).sum(axis=2)  # -> (|mx|, |my|)
        spectra.append(energy)
        totals.append(energy.sum())

    retained = np.stack(
        [energy.cumsum(0).cumsum(1) / total for energy, total in zip(spectra, totals)]
    )
    worst = retained.min(axis=0)

    grid_x, grid_y = np.meshgrid(
        np.arange(worst.shape[0]), np.arange(worst.shape[1]), indexing="ij"
    )
    cost = (2 * grid_x + 1) * (grid_y + 1)

    clears = worst >= target
    if not clears.any():
        raise ValueError(
            f"no box retains {target} of every power's energy; the most any box "
            f"retains of the worst power is {worst.max():.8f}"
        )
    cheapest = int(
        np.where(clears.ravel(), cost.ravel(), np.iinfo(np.int64).max).argmin()
    )
    return int(grid_x.ravel()[cheapest]), int(grid_y.ravel()[cheapest])


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


@dataclass
class FourierCorpus:
    """``(P, t) -> coefficients`` for every snapshot of every power given.

    ``targets`` are already normalised and packed the way the network emits them:
    ``[Re, Im]`` interleaved over the box, followed by the flattened ramp when
    ``detrend`` is on. ``floor`` is the field these very coefficients reconstruct
    to -- the truncation error, and the bound on what any network fitted to them
    can achieve.
    """

    inputs: np.ndarray  # [n, 2] of (P, t), in SI units
    targets: np.ndarray  # [n, n_out], normalised
    truth: np.ndarray  # [n, nz, ny, nx] in Kelvin, the solver's own field
    floor: np.ndarray  # [n, nz, ny, nx] in Kelvin, the box reconstructed exactly
    powers: np.ndarray  # [n], the power each row belongs to
    mx: np.ndarray  # the x wavenumbers kept, signed: 0..kx, -kx..-1
    my: np.ndarray  # the y wavenumbers kept, 0..ky (the rest are conjugates)
    mean: np.ndarray
    scale: np.ndarray
    n_coefficient: int  # where the coefficients end and the ramp begins

    def __len__(self) -> int:
        return self.inputs.shape[0]

    @property
    def box(self) -> tuple[int, int]:
        return int(self.mx.max()), int(self.my.max())


def _pack(
    field: np.ndarray,
    times: np.ndarray,
    mx: np.ndarray,
    my: np.ndarray,
    length_x: float,
    speed: float,
    detrend: bool,
    derotate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """One power's snapshots to ``([nt, n_out] raw targets, [nt, nx, ny, nz] floor)``."""
    work, ramp = detrend_x(field) if detrend else (field, None)

    spectrum = np.fft.rfftn(work, axes=(1, 2), norm="ortho")
    kept = spectrum[:, mx][:, :, my]

    # De-rotation is a phase, so it leaves |C| alone and the floor with it -- which
    # is why the floor below is built from `kept` and not from what the network is
    # asked to learn.
    turned = kept * spin(mx, times, length_x, speed)[:, :, None, None] if derotate else kept
    packed = np.stack([turned.real, turned.imag], -1).reshape(len(times), -1)
    if ramp is not None:
        packed = np.concatenate([packed, ramp.reshape(len(times), -1)], axis=1)

    nt, nx, ny, nz = field.shape
    full = np.zeros((nt, nx, ny // 2 + 1, nz), dtype=np.complex128)
    full[:, mx[:, None], my[None, :], :] = kept
    floor = np.fft.irfftn(full, s=(nx, ny), axes=(1, 2), norm="ortho")
    if ramp is not None:
        fraction = np.arange(nx) / nx
        floor = floor + ramp[:, None] * fraction[None, :, None, None]

    return packed, floor


def build(
    paths: list[Path],
    *,
    box: tuple[int, int] | None = None,
    energy_target: float = 0.9999,
    detrend: bool = False,
    derotate: bool = True,
    speed: float = SCAN_SPEED,
    normalisation: str = "global",
    exclude: int = 0,
    reference: FourierCorpus | None = None,
    verbose: bool = True,
) -> tuple[FourierCorpus, dict]:
    """Load ``paths`` and transform them. Returns ``(corpus, architecture keywords)``.

    ``reference`` reuses another corpus's box and normalisation, which is what the
    held-out power must do: a validation set that chose its own box, or standardised
    itself by its own statistics, would not be measuring the same model.

    ``exclude`` drops the first that many snapshots of every power. A sample here
    is a whole snapshot, so this removes rows outright rather than thinning them;
    the box is then chosen -- and the statistics taken -- on what is left.
    """
    grids = [load_grid(path).exclude_initial_steps(exclude) for path in paths]
    fields = [temperature_rise(grid) for grid in grids]
    nt, nx, ny, nz = fields[0].shape
    spacing = grids[0].spacing[0]
    length_x = nx * spacing  # the period the DFT assumes, not the plate's extent

    if reference is not None:
        box = reference.box
    if box is None:
        # Choose the box on the field that is actually transformed. Under --detrend
        # that is the residual, whose tail is far shorter than the raw field's --
        # picking the box from the raw spectrum would spend modes on a ramp that is
        # no longer there, which is the entire cost the flag exists to avoid.
        box = energy_box(
            [detrend_x(field)[0] for field in fields] if detrend else fields,
            energy_target,
        )
    kx, ky = box

    mx = np.r_[np.arange(kx + 1), np.arange(-kx, 0)]  # 0..kx, -kx..-1
    my = np.arange(ky + 1)

    packed, floors, truths, powers, inputs = [], [], [], [], []
    for grid, field in zip(grids, fields):
        rows, floor = _pack(
            field, grid.t, mx, my, length_x, speed, detrend, derotate
        )
        packed.append(rows)
        floors.append(np.transpose(floor, (0, 3, 2, 1)) + AMBIENT_TEMPERATURE)
        truths.append(grid.temperature)
        powers.append(np.full(len(grid.t), grid.power))
        inputs.append(np.stack([np.full(len(grid.t), grid.power), grid.t], axis=1))
        if verbose:
            print(
                f"[fmlp] {grid.name}: {len(grid.t)} snapshots, P={grid.power:g} W, "
                f"peak dT {field.max():.1f} K"
            )

    raw = np.concatenate(packed).astype(np.float64)

    if reference is not None:
        mean, scale = reference.mean, reference.scale
    elif normalisation == "global":
        # One scale for every coefficient keeps their relative sizes, so by
        # Parseval the MSE on them *is* the field's L2 error, up to that constant.
        mean, scale = np.zeros(1), np.full(1, raw.std())
    elif normalisation == "per-coef":
        mean, scale = raw.mean(0), raw.std(0) + 1e-12
    else:
        raise ValueError(f"unknown normalisation {normalisation!r}")

    corpus = FourierCorpus(
        inputs=np.concatenate(inputs),
        targets=(raw - mean) / scale,
        truth=np.concatenate(truths),
        floor=np.concatenate(floors),
        powers=np.concatenate(powers),
        mx=mx,
        my=my,
        mean=mean,
        scale=scale,
        n_coefficient=2 * len(mx) * len(my) * nz,
    )

    architecture = dict(
        grid_shape=(nz, ny, nx),
        kx=kx,
        ky=ky,
        detrend=detrend,
        derotate=derotate,
        scan_speed=speed,
        length_x=length_x,
        coefficient_mean=mean,
        coefficient_scale=scale,
        temperature_offset=AMBIENT_TEMPERATURE,
    )
    return corpus, architecture


def bounds_of(path: Path, exclude: int = 0) -> np.ndarray:
    """``[4, 2]`` of ``(lower, upper)`` for ``(x, y, z, t)``, from one file's axes.

    The agent needs the box it reconstructs on, and the grid axes settle it
    exactly -- so a checkpoint can be plotted without touching the corpus.
    ``exclude`` matches :func:`build`, so the ``t`` bound recorded in a checkpoint
    is the range the model was actually fitted over.
    """
    grid = load_grid(path).exclude_initial_steps(exclude)
    return np.array(
        [
            [grid.x[0], grid.x[-1]],
            [grid.y[0], grid.y[-1]],
            [grid.z[0], grid.z[-1]],
            [grid.t[0], grid.t[-1]],
        ]
    )
