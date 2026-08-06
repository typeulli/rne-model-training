"""Shared machinery for the toolpath-conditioned DeepONets: ``don``, ``gdon``, ``jdon``.

The models under ``models/`` learn ``(x, y, z, t; P) -> T`` for *one* scan: a
straight pass along ``x``, hardcoded into each one's ``laser.py``. Change the
scan and every one of them is wrong, because the only thing their branch network
ever saw was a laser power.

These three put the **scan itself** in the branch. The operator learned is

    G: (toolpath, P)  ->  (T: (x, y, z, t) -> K)

so a single trained network answers for a serpentine and for a spiral, and the
question "does the branch actually read the path" becomes measurable rather than
assumed.

The three differ in exactly one thing -- what multiplies the branch/trunk inner
product before the bias:

    don    nothing.               T = T_amb + dT * (<b, u> + c)
    gdon   a Gaussian on the beam. T = T_amb + dT * (<b, u> * G + c),  G = g + p
    jdon   Rosenthal's moving source, in the beam's own travel frame.

which is the same don/gdon/jdon distinction ``mlp``/``gmlp`` and
``cgmlp``/``cjmlp`` already draw, carried over to an operator net and generalised
off the straight pass.

Everything here is SI (metres, seconds, Kelvin, watts). The ``.npy`` files store
millimetres and are converted on load, exactly as :mod:`dataset` does for the
straight-pass corpus.

Why one shared module and not three copies
------------------------------------------
``models/cjmlp/laser.py`` is a deliberate copy of ``models/cgmlp/laser.py`` so an
experiment on one cannot move the other. That reasoning applies to a *gate* --
the thing being varied. It does not apply to the corpus loader, which is 300
lines that must be identical for the three numbers to be comparable at all: three
copies of it would be three chances for the comparison to be silently invalid.
So the gates are separate (below, and selected per model) and the loader is not.
The leading underscore keeps this file out of ``available_models()``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import ArrayLike, BaseAgent
from dataset import DATA_ROOT, MM, Domain
from utils import (
    DEFAULT_LOG_DIR,
    BestCheckpoint,
    add_optimizer_args,
    build_optimizer,
    count_parameters,
    load_checkpoint,
    make_run_name,
    resolve_checkpoint_path,
    resolve_device,
    resolve_lr,
    resolve_run_dir,
    seed_everything,
)

# `toolpath.py` is the solver's, and it is the definition of what a path is --
# reimplementing `position()` here would be a second definition that could drift
# from the one that actually made the data.
SIMULATION = DATA_ROOT.parent / "rne-kaist" / "simulation"
if str(SIMULATION) not in sys.path:
    sys.path.insert(0, str(SIMULATION))

from toolpath import Toolpath  # noqa: E402

DEFAULT_TOOLPATH_DIR = DATA_ROOT / "toolpath"
AMBIENT_TEMPERATURE = 298.0

# The branch reads the path at this many fixed times. See `branch_vector`.
DEFAULT_SENSORS = 64

# `utils.DEFAULT_LR` is 1e-3 for adam, and every other model here is four layers
# wide. These are six, and at 1e-3 all three diverge -- not gently: iteration
# 8000 of a `don` run went to a 17000 K validation error, with the gradients
# already clipped to a norm of 1.0. Measured over 15000 iterations, 3e-4 falls
# monotonically with no spike at all. An explicit `--lr` still wins.
DEFAULT_LR = 3e-4


# ---------------------------------------------------------------------------
# The path, as the branch sees it and as the gates need it
# ---------------------------------------------------------------------------


def beam_state(path: Toolpath, times: np.ndarray) -> np.ndarray:
    """``[len(times), 5]`` of ``(x_l, y_l, lit, v_x, v_y)`` in SI at each time.

    :meth:`Toolpath.position` gives the first three. The velocity is the fourth
    and fifth, and it is here because ``jdon`` needs it: Rosenthal's field is
    written in the frame that travels *with* the source, so the wake points along
    the direction of travel and a path that turns a corner turns its wake with it.
    A gate that assumed ``+x``, as every ``laser.py`` in this repository does,
    would put the trail in the wrong place for three of the four patterns.

    Taken from the segment the time lands in rather than differenced from the
    positions, so it is exact and it does not smear across a corner.
    """
    t = np.asarray(times, dtype=float)
    held = np.clip(t, 0.0, path.duration)
    index = np.clip(
        np.searchsorted(path.t_nodes, held, side="right") - 1, 0, len(path.on) - 1
    )

    x, y, lit = path.position(t)
    step = path.nodes[index + 1] - path.nodes[index]
    length = np.hypot(step[:, 0], step[:, 1])  # `_Poly` drops zero-length moves
    speed = path.speed[index]

    return np.stack(
        [
            x * MM,
            y * MM,
            lit.astype(float),
            step[:, 0] / length * speed * MM,
            step[:, 1] / length * speed * MM,
        ],
        axis=1,
    )


def retouched_beam_state(path: Toolpath, times: np.ndarray) -> np.ndarray:
    """:func:`beam_state`, but the beam does not move while the laser is off.

    Over a dark segment the head is travelling to the next track at 30 mm/s and
    depositing nothing. The plain beam state follows it, which makes two things
    wrong at once for the models built on it:

    * a window that rides with the head is somewhere with no melt pool in it,
      while the pool it *should* be describing sits cooling where the laser was
      last on;
    * the gates multiply by ``lit``, so they go to zero and the models have no
      shape at all for that pool -- 22% of ``raster``'s run and 16% of
      ``nested_l``'s is spent this way, and all of it is currently a blind spot.

    Here the position, the heading and the speed are all held at their last lit
    values, and ``lit`` is held at 1. The frame stops tracking the head and starts
    tracking the pool, which is the thing every model downstream is actually
    trying to describe. Nothing is frozen *before* the first lit instant or after
    the scan ends: there is no pool yet in the first case, and the run is over in
    the second.

    This is the only difference between the ``r*`` family and the rest.
    """
    state = beam_state(path, times)
    t = np.asarray(times, dtype=float)

    # Where the pool is must be a property of the *path* and the query time, not
    # of which other times happen to be in the same array. Forward-filling over
    # the samples looks equivalent and is not: ask for one dark instant on its
    # own and there is no earlier lit sample in the array to fill from, so the
    # answer silently falls back to the tracking beam. That is exactly how the
    # agents query it -- one instant per call, or one time repeated over a whole
    # slice -- so the models trained in a held frame were being evaluated and
    # plotted in a tracking one.
    #
    # So the fill is done over the path's own segments instead, once, and any
    # query time then reads it directly.
    on = np.asarray(path.on, dtype=bool)
    if on.all() or not on.any():
        return state  # a path that never lifts, or never fires

    # `previous[i]` is the most recent lit segment at or before segment i.
    previous = np.where(on, np.arange(len(on)), -1)
    np.maximum.accumulate(previous, out=previous)

    held = np.clip(t, 0.0, path.duration)
    segment = np.clip(
        np.searchsorted(path.t_nodes, held, side="right") - 1, 0, len(on) - 1
    )
    source = previous[segment]

    # A dark segment with a lit one before it: hold the end of that lit segment,
    # travelling as it was. Everything else -- lit now, or dark before the laser
    # has ever fired -- keeps the plain state, and so does anything past the end
    # of the scan, where the run is simply over.
    hold = ~on[segment] & (source >= 0) & (t <= path.duration)
    if not hold.any():
        return state

    out = state.copy()
    i = source[hold]
    step = path.nodes[i + 1] - path.nodes[i]
    length = np.hypot(step[:, 0], step[:, 1])
    speed = path.speed[i]

    out[hold, 0] = path.nodes[i + 1, 0] * MM
    out[hold, 1] = path.nodes[i + 1, 1] * MM
    out[hold, 2] = 1.0  # the pool is there whether or not the laser is
    out[hold, 3] = step[:, 0] / length * speed * MM
    out[hold, 4] = step[:, 1] / length * speed * MM
    return out


def beam_of(retouched: bool):
    """The beam-state function a model wants: held over dark travel, or not."""
    return retouched_beam_state if retouched else beam_state


def branch_vector(path: Toolpath, sensors: np.ndarray, retouched: bool = False) -> np.ndarray:
    """``[3 * len(sensors)]`` -- the path read at fixed times, the DeepONet way.

    A branch network takes the input function sampled at a fixed set of sensors.
    Here the input function is the trajectory ``u(tau) = (x_l, y_l, lit)`` and the
    sensors are times, shared by every path in the corpus, so the same coordinate
    of the vector means the same instant for all of them and the network can
    compare one path against another.

    The times are absolute, not fractions of each path's duration. That matters:
    the four patterns take between 9.20 s and 10.78 s, and normalising the clock
    away would hide exactly that difference. Past its own end a path is held at
    its last node with ``lit = 0`` (:meth:`Toolpath.position`), so "this scan
    finished a second before that one" is something the branch can read off.
    """
    state = beam_of(retouched)(path, sensors)
    return state[:, 0:3].reshape(-1)


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


class GaussianGate(nn.Module):
    """``[batch, 1]`` unit-peak Gaussian riding on the beam, wherever the beam is.

    ``coords`` is ``[batch, 4]`` of ``(x, y, z, t)`` and ``beam`` is ``[batch, 5]``
    of ``(x_l, y_l, lit, v_x, v_y)`` at that row's time on that row's path -- the
    generalisation of ``models/gmlp/laser.py``, which could only ask
    ``LASER_START_X + SCAN_SPEED * t``.

    Multiplied by ``lit``, so a segment travelled dark contributes no gate at all.
    A Gaussian is a picture of the *source*; where there is no source there is
    nothing for it to be a picture of, and the heat still on the plate is the
    trail, which this shape does not describe. The learnable floor ``p`` the
    network adds is what covers that.
    """

    radius: Tensor
    surface_z: Tensor
    exponent_scale: Tensor

    def __init__(self, radius: float, surface_z: float, exponent_scale: float = 1.0):
        super().__init__()
        self.register_buffer("radius", torch.tensor(float(radius)))
        self.register_buffer("surface_z", torch.tensor(float(surface_z)))
        self.register_buffer("exponent_scale", torch.tensor(float(exponent_scale)))

    def forward(self, coords: Tensor, beam: Tensor) -> Tensor:
        squared = (
            (coords[:, 0:1] - beam[:, 0:1]) ** 2
            + (coords[:, 1:2] - beam[:, 1:2]) ** 2
            + (coords[:, 2:3] - self.surface_z) ** 2
        )
        return beam[:, 2:3] * torch.exp(
            -2.0 * self.exponent_scale * squared / self.radius**2
        )

    def sequence(self, xyz: Tensor, beam: Tensor) -> Tensor:
        """``[B, Nt]`` -- the same gate at every stored time, for a fixed point.

        ``xyz`` is ``[B, 3]`` and ``beam`` is ``[Nt, 5]``: one point, the whole
        clock. A model that returns a temperature history needs a gate that is
        also a history, and the beam has moved between every entry of it.
        """
        squared = (
            (xyz[:, 0:1] - beam[None, :, 0]) ** 2
            + (xyz[:, 1:2] - beam[None, :, 1]) ** 2
            + (xyz[:, 2:3] - self.surface_z) ** 2
        )
        return beam[None, :, 2] * torch.exp(
            -2.0 * self.exponent_scale * squared / self.radius**2
        )


def _rosenthal_series(
    along: Tensor,
    across: Tensor,
    depth: Tensor,
    speed: Tensor,
    diffusivity: float,
    soft_core: float,
    thickness: float,
    images: int,
) -> Tensor:
    """``sum_n (-1)^n exp[-v (r_n + xi) / (2 alpha)] / r_n``, in ``m^-1``.

    Lifted from ``models/cjmlp/laser.py`` with one thing changed: ``xi`` is the
    displacement resolved *along the direction of travel* rather than along ``x``.
    Everything else -- the image sum that meets the insulated top face and the
    ambient bottom one, the soft core that keeps a point source finite -- is that
    module's, and its docstring is the reference for why each is there.

    The exponent is never positive (``r_n >= |xi|``), so nothing overflows.
    """
    order = torch.arange(-images, images + 1, device=along.device)
    sign = (1 - 2 * (order % 2)).to(along.dtype)
    offset = depth - 2.0 * thickness * order.to(along.dtype)

    radius = torch.sqrt(along**2 + across**2 + offset**2 + soft_core**2)
    return (
        sign * torch.exp(-speed * (radius + along) / (2.0 * diffusivity)) / radius
    ).sum(dim=-1, keepdim=True)


class MovingSourceGate(nn.Module):
    """Rosenthal's travelling source, oriented by the beam's own velocity.

    A Gaussian is symmetric and has no wake. This does: ahead of the beam the
    field dies within ``alpha / v``, behind it the exponential stops decaying and
    a ``1/r`` tail is left. That asymmetry is the trailing heat every figure in
    this repository shows the models struggling with -- and on a toolpath it is
    also *directional*, so the wake has to turn when the path turns. It does,
    because ``xi`` is the displacement projected on ``(v_x, v_y) / |v|`` rather
    than on ``x``.

    Normalised to a unit peak so ``G = j + p`` sits on the same scale as the
    Gaussian's ``G = g + p`` and the two models differ in shape alone.
    """

    def __init__(
        self,
        diffusivity: float,
        soft_core: float,
        thickness: float,
        surface_z: float,
        scan_speed: float,
        images: int = 3,
    ) -> None:
        super().__init__()
        self.diffusivity = float(diffusivity)
        self.soft_core = float(soft_core)
        self.thickness = float(thickness)
        self.images = int(images)
        self.register_buffer("surface_z", torch.tensor(float(surface_z)))
        self.register_buffer("peak", torch.tensor(self._peak(float(scan_speed))))

    def _peak(self, speed: float) -> float:
        """The series' maximum, swept once on the CPU in float64.

        No closed form -- the maximum sits behind the beam by an offset that
        depends on every constant at once -- so it is found by sweeping the centre
        line. Every lit segment in this corpus travels at the same speed, so one
        sweep serves the whole run; a dark segment has ``lit = 0`` and never
        reaches here.
        """
        offsets = torch.linspace(-0.020, 0.002, 4401, dtype=torch.float64).reshape(-1, 1)
        zeros = torch.zeros_like(offsets)
        values = _rosenthal_series(
            offsets,
            zeros,
            zeros,
            torch.tensor(speed, dtype=torch.float64),
            self.diffusivity,
            self.soft_core,
            self.thickness,
            self.images,
        )
        return float(values.max())

    def forward(self, coords: Tensor, beam: Tensor) -> Tensor:
        delta_x = coords[:, 0:1] - beam[:, 0:1]
        delta_y = coords[:, 1:2] - beam[:, 1:2]

        speed = torch.linalg.vector_norm(beam[:, 3:5], dim=1, keepdim=True)
        safe = speed.clamp_min(1e-12)
        unit_x, unit_y = beam[:, 3:4] / safe, beam[:, 4:5] / safe

        along = delta_x * unit_x + delta_y * unit_y  # xi, down the direction of travel
        across = -delta_x * unit_y + delta_y * unit_x  # the perpendicular in-plane axis

        field = _rosenthal_series(
            along,
            across,
            coords[:, 2:3] - self.surface_z,
            speed,
            self.diffusivity,
            self.soft_core,
            self.thickness,
            self.images,
        )
        return beam[:, 2:3] * field / self.peak

    def sequence(self, xyz: Tensor, beam: Tensor) -> Tensor:
        """``[B, Nt]`` -- the wake at a fixed point, at every stored time.

        The beam's heading changes between entries, so ``along`` and ``across``
        are resolved per time as well as per point: the same node sits behind the
        source at one instant and beside it at the next, and that is the whole
        content of a moving-source field.
        """
        speed = torch.linalg.vector_norm(beam[:, 3:5], dim=1).clamp_min(1e-12)
        unit_x, unit_y = beam[:, 3] / speed, beam[:, 4] / speed

        delta_x = xyz[:, 0:1] - beam[None, :, 0]  # [B, Nt]
        delta_y = xyz[:, 1:2] - beam[None, :, 1]
        along = delta_x * unit_x[None] + delta_y * unit_y[None]
        across = -delta_x * unit_y[None] + delta_y * unit_x[None]

        field = _rosenthal_series(
            along.unsqueeze(-1),
            across.unsqueeze(-1),
            (xyz[:, 2:3] - self.surface_z).unsqueeze(-1),
            speed[None, :, None],
            self.diffusivity,
            self.soft_core,
            self.thickness,
            self.images,
        ).squeeze(-1)
        return beam[None, :, 2] * field / self.peak


GATES: dict[str, str] = {
    "don": "none",
    "gdon": "gaussian",
    "jdon": "moving_source",
    # `pkdon` is the patch: same network as `don`, no gate, but fitted to the
    # melt pool alone in beam-frame coordinates. The gate is not what makes it
    # different -- the corpus is -- which is exactly how `cpkmlp` relates to
    # `cmlp`. See `PatchCorpus`.
    "pkdon": "none",
    # The sequence family: the trunk drops `t` and the output gains it. Same
    # three gates, so `fdon` : `gfdon` : `jfdon` is the same contrast as
    # `don` : `gdon` : `jdon`, drawn on a different output shape.
    "fdon": "none",
    "gfdon": "gaussian",
    "jfdon": "moving_source",
    # The sequence family's patch: `pkdon` is to `don` as this is to `fdon`.
    "pkfdon": "none",
    # The retouched family: identical to the four above except that the beam
    # frame stops following the head while the laser is off and holds the pool
    # it last made. See `retouched_beam_state`.
    "rfdon": "none",
    "rgfdon": "gaussian",
    "rjfdon": "moving_source",
    "rpkfdon": "none",
}

# Models whose beam frame is held over dark travel rather than tracking the head.
RETOUCHED_MODELS = {"rfdon", "rgfdon", "rjfdon", "rpkfdon"}

# Models whose trunk is `(x, y, z)` and whose output is `[B, Nt]`, trained on
# `SequenceCorpus`.
SEQUENCE_MODELS = {"fdon", "gfdon", "jfdon", "pkfdon",
                   "rfdon", "rgfdon", "rjfdon", "rpkfdon"}

# Models whose trunk takes `(along, across, z, t)` about the beam rather than
# `(x, y, z, t)` on the plate, and which are therefore trained on `PatchCorpus`.
PATCH_MODELS = {"pkdon", "pkfdon", "rpkfdon"}


def build_gate(kind: str, physics: dict, exponent_scale: float) -> nn.Module | None:
    """The gate a model name asks for, built from the constants the data carries."""
    if kind == "none":
        return None
    if kind == "gaussian":
        return GaussianGate(
            physics["beam_radius"], physics["surface_z"], exponent_scale
        )
    if kind == "moving_source":
        return MovingSourceGate(
            physics["diffusivity"],
            physics["beam_radius"],
            physics["thickness"],
            physics["surface_z"],
            physics["scan_speed"],
        )
    raise ValueError(f"unknown gate {kind!r}; have {', '.join(sorted(set(GATES.values())))}")


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


@dataclass
class Condition:
    """One ``(toolpath, power)`` pair: the field it produced and how to ask for it."""

    pattern: str
    power: float
    path: Toolpath
    field: Tensor  # [nt * nx * ny * nz] K, flattened in the file's own order
    times: Tensor  # [nt] s, this run's own snapshot times
    beam: Tensor  # [nt, 5] SI, the beam state at each stored snapshot
    branch: Tensor  # [3 * sensors] the path as the branch reads it
    nt: int

    @property
    def count(self) -> int:
        return self.field.numel()


class PathCorpus:
    """Every ``(toolpath, power)`` run under ``data/toolpath``, on the GPU.

    The rows are *not* materialised. A run is a uniform grid, so ``(x, y, z, t)``
    is an arithmetic function of a flat index and storing it would cost six times
    what the temperature does for no information at all: 115 M points would be
    3.7 GB of coordinates that ``torch.arange`` reproduces exactly. Only ``T``
    is held, and a batch decodes its own coordinates from the indices it drew.

    The train/validation split is over *points*, drawn once per condition and kept
    as index arrays, so a batch is a gather rather than a rejection loop. The
    held-out power is not in either -- it is a separate set of conditions and is
    never sampled, only scored.
    """

    def __init__(
        self,
        directory: Path,
        holdout_power: float,
        sensors: int,
        val_fraction: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        patterns: Sequence[str] | None = None,
    ) -> None:
        runs = sorted(p for p in directory.iterdir() if (p / "config.json").exists())
        if patterns is not None:
            wanted = set(patterns)
            runs = [p for p in runs if p.name in wanted]
        if not runs:
            raise FileNotFoundError(f"no solver runs under {directory}")

        self.device, self.dtype = device, dtype
        self.train: list[Condition] = []
        self.holdout: list[Condition] = []
        self.physics = self._physics(runs[0])

        # One sensor grid for the whole corpus, spanning the longest path, so the
        # same coordinate of every branch vector is the same instant.
        span = 0.0
        paths: dict[str, Toolpath] = {}
        for run in runs:
            paths[run.name] = Toolpath.load(run / "toolpath.json")
            span = max(span, paths[run.name].duration)
        self.sensors = np.linspace(0.0, span, sensors)
        self.path_span = span

        t_max = 0.0
        for run in runs:
            path = paths[run.name]
            branch = torch.as_tensor(
                branch_vector(path, self.sensors), device=device, dtype=dtype
            )
            for file in sorted(run.glob("data_*W.npy")):
                rows = np.load(file, mmap_mode="r")
                power = float(rows[0, 4])

                axes = self._axes(rows)
                nt = axes["t"].size
                t_max = max(t_max, float(axes["t"][-1]))
                if not hasattr(self, "shape"):
                    self.shape = (axes["x"].size, axes["y"].size, axes["z"].size)
                    self.origin = np.array([axes["x"][0], axes["y"][0], axes["z"][0]])
                    self.spacing = float(axes["x"][1] - axes["x"][0])
                    self.extent = np.array([axes["x"][-1], axes["y"][-1], axes["z"][-1]])
                elif self.shape != (axes["x"].size, axes["y"].size, axes["z"].size):
                    raise ValueError(
                        f"{file} is {axes['x'].size}x{axes['y'].size}x{axes['z'].size}, "
                        f"but the corpus is {self.shape}; one grid per corpus"
                    )

                field = torch.as_tensor(
                    np.ascontiguousarray(rows[:, 5], dtype=np.float32),
                    device=device,
                ).to(dtype)
                condition = Condition(
                    pattern=run.name,
                    power=power,
                    path=path,
                    field=field,
                    times=torch.as_tensor(axes["t"], device=device, dtype=dtype),
                    beam=torch.as_tensor(
                        beam_state(path, axes["t"]), device=device, dtype=dtype
                    ),
                    branch=branch,
                    nt=nt,
                )
                bucket = (
                    self.holdout
                    if abs(power - holdout_power) < 1e-9
                    else self.train
                )
                bucket.append(condition)
                print(
                    f"[data] {run.name:11s} {power:5.0f} W  {nt:3d} x "
                    f"{'x'.join(str(n) for n in self.shape)} = {condition.count:>9d} pts"
                    f"{'   (holdout)' if bucket is self.holdout else ''}"
                )

        if not self.train:
            raise ValueError(f"holding out {holdout_power} W left nothing to train on")
        self.t_max = t_max

        # Split each condition's points once. int32 is enough -- the largest
        # condition here is 4.4 M points -- and halves what the indices cost.
        self.train_index: list[Tensor] = []
        self.val_index: list[Tensor] = []
        for condition in self.train:
            order = torch.randperm(condition.count, generator=generator, device=device)
            cut = int(round(condition.count * (1.0 - val_fraction)))
            self.train_index.append(order[:cut].to(torch.int32))
            self.val_index.append(order[cut:].to(torch.int32))

        self.branch_dim = 3 * sensors + 1
        self.max_power = max(c.power for c in self.train + self.holdout)

    # -- provenance -------------------------------------------------------

    @staticmethod
    def _physics(run: Path) -> dict:
        """The constants the gates need, read off the dataset rather than guessed.

        ``calibrate.py`` exists because the straight-pass ``.npy`` files carry no
        material properties and the numbers had to be recovered by fitting. These
        runs carry a ``config.json``, so there is nothing to recover: the beam
        radius, the diffusivity and the plate thickness are the values the solver
        was actually given. Fitting them again could only add error.
        """
        config = json.loads((run / "config.json").read_text())
        material = config["material"]
        domain = config["domain_mm"]
        path = config["beam"]["toolpath"]
        speeds = [s for s, on in zip(path["speed_mm_s"], path["on"]) if on]
        return {
            "beam_radius": config["beam"]["radius_mm"] * MM,
            "diffusivity": material["k_W_mmK"]
            / (material["rho_g_mm3"] * material["Cp_J_gK"])
            * MM**2,
            "thickness": domain[2] * MM,
            "surface_z": domain[2] * MM,
            "scan_speed": max(speeds) * MM,
        }

    @staticmethod
    def _axes(rows: np.ndarray) -> dict[str, np.ndarray]:
        """The four axes of a run, recovered from its own coordinate columns.

        The layout is documented (``t`` outermost, then ``meshgrid(x, y, z,
        indexing='ij')``) but reading it off the file is what makes the reshape
        below safe -- if the writer ever changes, this raises instead of quietly
        transposing the field.
        """
        t = np.unique(rows[:, 3])
        block = rows[: rows.shape[0] // t.size]
        x, y, z = (np.unique(block[:, i]) for i in range(3))
        if x.size * y.size * z.size * t.size != rows.shape[0]:
            raise ValueError(
                f"{rows.shape[0]} rows is not {x.size} x {y.size} x {z.size} x {t.size}"
            )
        return {"x": x, "y": y, "z": z, "t": t}

    # -- geometry ---------------------------------------------------------

    @property
    def domain(self) -> Domain:
        """The space-time box, in SI, that the trunk is normalised against."""
        lower = [0.0, 0.0, 0.0, 0.0]
        upper = [
            float(self.extent[0]) * MM,
            float(self.extent[1]) * MM,
            float(self.extent[2]) * MM,
            self.t_max,
        ]
        return Domain(torch.tensor([[lo, hi] for lo, hi in zip(lower, upper)]))

    def _decode(self, condition: Condition, flat: Tensor) -> tuple[Tensor, Tensor]:
        """``(coords[B, 4], beam[B, 5])`` for flat indices into one condition.

        The file is written ``t`` outermost then ``x, y, z``, so the index divides
        as ``((it * nx + ix) * ny + iy) * nz + iz``.

        ``t`` comes from the condition's own snapshot times rather than from a
        step shared across the corpus. The four patterns run for different
        lengths, so they store different numbers of snapshots, and a shared step
        would only be right for as long as every run happened to use the same
        ``--snap_dt``.
        """
        nx, ny, nz = self.shape
        flat = flat.long()
        iz = flat % nz
        rest = flat // nz
        iy = rest % ny
        rest = rest // ny
        ix = rest % nx
        it = rest // nx

        step = self.spacing * MM
        coords = torch.stack(
            [
                ix.to(self.dtype) * step,
                iy.to(self.dtype) * step,
                iz.to(self.dtype) * step,
                condition.times[it],
            ],
            dim=1,
        )
        return coords, condition.beam[it]

    def _branch(self, condition: Condition, count: int) -> Tensor:
        power = torch.full(
            (count, 1), condition.power, device=self.device, dtype=self.dtype
        )
        return torch.cat([condition.branch.expand(count, -1), power], dim=1)

    # -- batching ---------------------------------------------------------

    def batch(self, per_condition: int, generator: torch.Generator) -> tuple[Tensor, ...]:
        """``(branch, coords, beam, T)`` with ``per_condition`` points from each run.

        Equal counts per condition rather than uniform over all points: the runs
        differ in length by up to 12%, and a batch that mirrored that would weight
        ``raster`` above ``spiral`` for a reason that is an artefact of how long
        its jumps take. Equal counts make the four patterns and the six powers
        contribute alike, which is what the branch is being asked to separate.
        """
        branch, coords, beam, temperature = [], [], [], []
        for condition, index in zip(self.train, self.train_index):
            draw = torch.randint(
                index.numel(), (per_condition,), generator=generator, device=self.device
            )
            flat = index[draw]
            block_coords, block_beam = self._decode(condition, flat)
            branch.append(self._branch(condition, per_condition))
            coords.append(block_coords)
            beam.append(block_beam)
            temperature.append(condition.field[flat.long()].unsqueeze(1))
        return (
            torch.cat(branch),
            torch.cat(coords),
            torch.cat(beam),
            torch.cat(temperature),
        )

    @torch.no_grad()
    def score(
        self,
        model: nn.Module,
        which: str,
        limit: int = 400_000,
        chunk: int = 65536,
        generator: torch.Generator | None = None,
    ) -> tuple[float, float]:
        """``(rmse, max_abs_error)`` in Kelvin over ``'val'`` or ``'holdout'``.

        Scored on a fixed-size sample per condition rather than every point: the
        validation half alone is 9.9 M points and the cadence here is every few
        hundred iterations. ``limit`` points per condition, drawn with the same
        seeded generator each call, keeps the number comparable across calls while
        costing a fraction of a second.
        """
        if which == "val":
            pairs = list(zip(self.train, self.val_index))
        elif which == "holdout":
            pairs = [
                (
                    condition,
                    torch.arange(condition.count, device=self.device, dtype=torch.int32),
                )
                for condition in self.holdout
            ]
        else:
            raise ValueError(f"which must be 'val' or 'holdout', got {which!r}")
        if not pairs:
            return float("nan"), float("nan")

        squared, worst, seen = 0.0, 0.0, 0
        for condition, index in pairs:
            if index.numel() > limit:
                draw = torch.randint(
                    index.numel(), (limit,), generator=generator, device=self.device
                )
                index = index[draw]
            for start in range(0, index.numel(), chunk):
                flat = index[start : start + chunk]
                coords, beam = self._decode(condition, flat)
                branch = self._branch(condition, flat.numel())
                error = (
                    model(branch, coords, beam)
                    - condition.field[flat.long()].unsqueeze(1)
                )
                squared += float(error.pow(2).sum())
                worst = max(worst, float(error.abs().max()))
                seen += flat.numel()
        return math.sqrt(squared / seen), worst


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


def _mlp(
    input_dim: int,
    hidden: Sequence[int],
    output_dim: int,
    activation: type[nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for size in hidden:
        layers += [nn.Linear(width, size), activation()]
        width = size
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class PathDeepONet(nn.Module):
    """A DeepONet whose branch reads a toolpath and a power.

    ``T_hat = T_amb + dT * ( <branch(u), trunk(x, y, z, t)> * G + c )``

    with ``u`` the trajectory sampled at the shared sensors, concatenated with
    ``P``. ``G`` is ``gate(coords, beam) + p`` when a gate is given and ``1``
    when it is not, and ``p`` is learnable -- the floor the gate relaxes to away
    from the beam.

    ``p`` is what stops a gate from being a straitjacket. With ``G = g`` alone the
    prediction is pinned to ``T_amb + dT*c`` everywhere the gate has died, which
    on these paths is most of the plate most of the time and includes the whole
    trail the beam has already laid. With ``G = g + p`` the operator gets the
    beam's shape for free near the source and is free to describe the rest. The
    trained ``p`` reads directly as how far the fit had to back away from the
    prior. This is the same ``G = g + p`` as ``models/gdon_old``, for the same
    reason.

    Normalisation happens inside :meth:`forward`, so callers pass SI throughout
    and the gate -- which is built from those same physical coordinates -- keeps
    its metres.
    """

    coord_mean: Tensor
    coord_scale: Tensor
    branch_mean: Tensor
    branch_scale: Tensor
    temperature_offset: Tensor
    temperature_scale: Tensor

    def __init__(
        self,
        branch_input_dim: int,
        gate: str = "none",
        physics: dict | None = None,
        hidden_layers: Sequence[int] = (128, 128, 128, 128),
        latent_dim: int = 128,
        activation: type[nn.Module] = nn.Tanh,
        coord_mean: Sequence[float] | None = None,
        coord_scale: Sequence[float] | None = None,
        branch_mean: Sequence[float] | None = None,
        branch_scale: Sequence[float] | None = None,
        temperature_offset: float = AMBIENT_TEMPERATURE,
        temperature_scale: float = 1.0,
        gaussian_exponent_scale: float = 1.0,
        gate_offset: float = 0.5,
    ) -> None:
        super().__init__()
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = 4
        self.latent_dim = latent_dim
        self.gate_kind = gate
        self.physics = physics or {}

        hidden = [int(h) for h in hidden_layers]
        self.branch_net = _mlp(branch_input_dim, hidden, latent_dim, activation)
        self.trunk_net = _mlp(self.trunk_input_dim, hidden, latent_dim, activation)
        self.bias = nn.Parameter(torch.zeros(1))

        self.gate = build_gate(gate, self.physics, gaussian_exponent_scale)
        self.gate_offset = (
            nn.Parameter(torch.tensor(float(gate_offset)))
            if self.gate is not None
            else None
        )

        def buffer(values, dim, default):
            if values is None:
                return torch.full((1, dim), float(default))
            return torch.as_tensor(
                list(values), dtype=torch.get_default_dtype()
            ).reshape(1, -1)

        self.register_buffer("coord_mean", buffer(coord_mean, 4, 0.0))
        self.register_buffer("coord_scale", buffer(coord_scale, 4, 1.0))
        self.register_buffer("branch_mean", buffer(branch_mean, branch_input_dim, 0.0))
        self.register_buffer("branch_scale", buffer(branch_scale, branch_input_dim, 1.0))
        self.register_buffer(
            "temperature_offset", torch.tensor(float(temperature_offset))
        )
        self.register_buffer("temperature_scale", torch.tensor(float(temperature_scale)))

    def forward(self, branch: Tensor, coords: Tensor, beam: Tensor) -> Tensor:
        """``branch [B, 3m+1]``, ``coords [B, 4]`` SI, ``beam [B, 5]`` SI -> ``[B, 1]`` K."""
        latent = (
            self.branch_net((branch - self.branch_mean) / self.branch_scale)
            * self.trunk_net((coords - self.coord_mean) / self.coord_scale)
        ).sum(dim=-1, keepdim=True)

        if self.gate is not None:
            latent = latent * (self.gate(coords, beam) + self.gate_offset)
        return self.temperature_offset + self.temperature_scale * (latent + self.bias)


class SequenceDeepONet(nn.Module):
    """A DeepONet that returns a temperature *history* at a point, ``[B, Nt]``.

    ``T_hat(x, y, z) = T_amb + dT * ( <b(u), tau_j(x, y, z)> * G_j + c )``  for each
    stored time ``j``, with ``u`` the toolpath sampled at the shared sensors and
    concatenated with ``P``.

    The trunk takes ``(x, y, z)`` and no clock. Time has moved from being a
    coordinate the network is queried at to being the axis its answer lives on --
    which is the shape the problem already had on the input side, since the branch
    reads the path as a time series. Both subnetworks emit ``latent x Nt`` and the
    inner product is taken per time, the multi-output form
    ``models/gdon_old/model.py`` already uses for its ``output_dim``.

    What it buys: one forward pass per point instead of ``Nt`` of them, and a
    history that cannot be self-inconsistent because it comes out of one latent.

    What it costs: the answer exists only on the corpus's own grid. ``t`` is no
    longer something the network is continuous in -- :class:`SequenceAgent`
    interpolates between stored times, and that interpolation is linear and is the
    agent's, not the model's.
    """

    coord_mean: Tensor
    coord_scale: Tensor
    branch_mean: Tensor
    branch_scale: Tensor
    temperature_offset: Tensor
    temperature_scale: Tensor

    def __init__(
        self,
        branch_input_dim: int,
        steps: int,
        gate: str = "none",
        physics: dict | None = None,
        hidden_layers: Sequence[int] = (128, 128, 128, 128),
        latent_dim: int = 128,
        activation: type[nn.Module] = nn.Tanh,
        coord_mean: Sequence[float] | None = None,
        coord_scale: Sequence[float] | None = None,
        branch_mean: Sequence[float] | None = None,
        branch_scale: Sequence[float] | None = None,
        temperature_offset: float = AMBIENT_TEMPERATURE,
        temperature_scale: float = 1.0,
        gaussian_exponent_scale: float = 1.0,
        gate_offset: float = 0.5,
    ) -> None:
        super().__init__()
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = 3
        self.steps = int(steps)
        self.latent_dim = latent_dim
        self.gate_kind = gate
        self.physics = physics or {}

        hidden = [int(h) for h in hidden_layers]
        width = latent_dim * self.steps
        self.branch_net = _mlp(branch_input_dim, hidden, width, activation)
        self.trunk_net = _mlp(self.trunk_input_dim, hidden, width, activation)
        self.bias = nn.Parameter(torch.zeros(1))

        self.gate = build_gate(gate, self.physics, gaussian_exponent_scale)
        self.gate_offset = (
            nn.Parameter(torch.tensor(float(gate_offset)))
            if self.gate is not None
            else None
        )

        def buffer(values, dim, default):
            if values is None:
                return torch.full((1, dim), float(default))
            return torch.as_tensor(
                list(values), dtype=torch.get_default_dtype()
            ).reshape(1, -1)

        self.register_buffer("coord_mean", buffer(coord_mean, 3, 0.0))
        self.register_buffer("coord_scale", buffer(coord_scale, 3, 1.0))
        self.register_buffer("branch_mean", buffer(branch_mean, branch_input_dim, 0.0))
        self.register_buffer("branch_scale", buffer(branch_scale, branch_input_dim, 1.0))
        self.register_buffer(
            "temperature_offset", torch.tensor(float(temperature_offset))
        )
        self.register_buffer("temperature_scale", torch.tensor(float(temperature_scale)))

    def forward(self, branch: Tensor, xyz: Tensor, beam: Tensor) -> Tensor:
        """``branch [B, 3m+1]``, ``xyz [B, 3]`` SI, ``beam [Nt, 5]`` or ``[C, Nt, 5]``.

        A batch is built as ``C`` equal blocks, one per condition, so ``beam`` may
        arrive as one table per block rather than one per row -- the beam is a
        property of the run, not of the point.
        """
        b = self.branch_net((branch - self.branch_mean) / self.branch_scale)
        u = self.trunk_net((xyz - self.coord_mean) / self.coord_scale)
        latent = (
            b.view(-1, self.steps, self.latent_dim)
            * u.view(-1, self.steps, self.latent_dim)
        ).sum(dim=-1)  # [B, Nt]

        if self.gate is not None:
            if beam.dim() == 3:
                block = xyz.size(0) // beam.size(0)
                gate = torch.cat(
                    [
                        self.gate.sequence(xyz[i * block : (i + 1) * block], beam[i])
                        for i in range(beam.size(0))
                    ]
                )
            else:
                gate = self.gate.sequence(xyz, beam)
            latent = latent * (gate + self.gate_offset)
        return self.temperature_offset + self.temperature_scale * (latent + self.bias)


class MaskedScaledMSELoss(nn.Module):
    """``ScaledMSELoss`` over the entries a run actually has.

    The shortest scan here stores 24 snapshots and the longest 27, so three of the
    27 columns are padding for ``spiral`` and none are for ``raster``. Averaging
    over them would score the network on times the solver never computed.
    """

    def __init__(self, scale: float) -> None:
        super().__init__()
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = scale

    def forward(self, predicted: Tensor, measured: Tensor, mask: Tensor) -> Tensor:
        squared = ((predicted - measured) / self.scale).pow(2) * mask
        return squared.sum() / mask.sum().clamp_min(1.0)


class ScaledMSELoss(nn.Module):
    """``mean(((T_hat - T) / scale)^2)`` -- the objective, dimensionless in ``dT``.

    The same normalisation ``models/gdon_old/loss.py`` uses, so a number from
    these three is on the scale numbers from that one are on.
    """

    def __init__(self, scale: float) -> None:
        super().__init__()
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = scale

    def forward(self, predicted: Tensor, measured: Tensor) -> Tensor:
        return ((predicted - measured) / self.scale).pow(2).mean()


# ---------------------------------------------------------------------------
# The sequence corpus: one row per *point*, labelled with its whole history
# ---------------------------------------------------------------------------


class SequenceCorpus:
    """The same runs, cut by space instead of by space-time.

    ``PathCorpus`` hands out ``((x, y, z, t), T)`` pairs, so the network is asked
    for one number at a time and ``t`` is a trunk coordinate like any other. Here
    a row is a *point*: ``(x, y, z)`` with the whole temperature history at it,
    ``[Nt]``. The operator becomes

        G: (toolpath, P)  ->  ( (x, y, z) -> T(t) )

    which lines the output's clock up with the input's -- the branch already reads
    the path as a time series, and now the thing it predicts is one too.

    **The common grid.** Every run stores snapshots 0.4 s apart from ``t = 0``, but
    they stop at different places: 24 for ``spiral``, 27 for ``raster``, because
    the scans take 9.20 s and 10.78 s. The output is therefore ``Nt = 27`` on the
    grid ``0.4k``, and a run with fewer snapshots is a *prefix* of it, carrying a
    mask so the loss never scores a time that run has no answer for. Padding with
    zeros and scoring them would teach the network that ``raster`` is cold at
    10.4 s and ``spiral`` is 298 K there, which is not a fact about either.

    The check that the grid really is common is done on load rather than assumed.
    """

    def __init__(
        self,
        directory: Path,
        holdout_power: float,
        sensors: int,
        val_fraction: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        patterns: Sequence[str] | None = None,
        cede_radius: float = 0.0,
        retouched: bool = False,
    ) -> None:
        """``cede_radius > 0`` hides the beam's neighbourhood from the loss.

        When a run is going to be pasted over by ``pkfdon``, the window the patch
        owns is 6.3% of the node-times and carries 54% of the base's squared
        error -- capacity spent on an answer that is then discarded. Ceding it
        lets the base spend that capacity on the 94% it is actually asked for.

        The cost is the seam: the base no longer sees the field it has to be
        continuous with at the window's edge, so just outside it is extrapolating
        rather than interpolating. Whether that trade is worth it is measured,
        not assumed -- the flag exists so it can be.
        """
        runs = sorted(p for p in directory.iterdir() if (p / "config.json").exists())
        if patterns is not None:
            runs = [p for p in runs if p.name in set(patterns)]
        if not runs:
            raise FileNotFoundError(f"no solver runs under {directory}")

        self.device, self.dtype = device, dtype
        self.cede_radius = float(cede_radius)
        self.retouched = bool(retouched)
        beam_at = beam_of(self.retouched)
        self.physics = PathCorpus._physics(runs[0])
        self.train: list[dict] = []
        self.holdout: list[dict] = []

        span = 0.0
        paths = {}
        for run in runs:
            paths[run.name] = Toolpath.load(run / "toolpath.json")
            span = max(span, paths[run.name].duration)
        self.sensors = np.linspace(0.0, span, sensors)
        self.path_span = span

        loaded, step, t_max = [], None, 0.0
        for run in runs:
            path = paths[run.name]
            branch = torch.as_tensor(
                branch_vector(path, self.sensors, self.retouched),
                device=device, dtype=dtype,
            )
            for file in sorted(run.glob("data_*W.npy")):
                rows = np.load(file, mmap_mode="r")
                power = float(rows[0, 4])
                axes = PathCorpus._axes(rows)
                nx, ny, nz = axes["x"].size, axes["y"].size, axes["z"].size
                nt, times = axes["t"].size, axes["t"]

                # One grid, or the prefix story below is false.
                spacing = float(np.diff(times).mean())
                if step is None:
                    step = spacing
                    self.shape = (nx, ny, nz)
                    self.spacing = float(axes["x"][1] - axes["x"][0])
                    self.plate = (float(axes["x"][-1]) * MM, float(axes["y"][-1]) * MM)
                    self.extent = np.array(
                        [axes["x"][-1], axes["y"][-1], axes["z"][-1]]
                    )
                if not np.allclose(times, np.arange(nt) * step, atol=1e-9):
                    raise ValueError(
                        f"{file}: times are not {step:g} s apart from 0; a sequence "
                        f"model needs one grid across the corpus"
                    )
                if self.shape != (nx, ny, nz):
                    raise ValueError(f"{file} is {(nx, ny, nz)}, corpus is {self.shape}")

                t_max = max(t_max, float(times[-1]))
                loaded.append((run.name, power, path, branch, nt, file))

        self.time_step = float(step)
        self.nt = int(round(t_max / self.time_step)) + 1
        self.times = torch.arange(self.nt, device=device, dtype=dtype) * self.time_step
        self.points = int(np.prod(self.shape))
        self.t_max = t_max

        for pattern, power, path, branch, nt, file in loaded:
            rows = np.load(file, mmap_mode="r")
            # [nt, S]: the file is written `t` outermost, so a point's history is
            # one strided column and needs no gather.
            field = torch.as_tensor(
                np.ascontiguousarray(rows[:, 5], dtype=np.float32).reshape(nt, self.points),
                device=device,
            ).to(dtype)
            condition = {
                "pattern": pattern, "power": power, "path": path, "branch": branch,
                "field": field, "nt": nt,
                "beam": torch.as_tensor(
                    beam_at(path, np.arange(self.nt) * self.time_step),
                    device=device, dtype=dtype,
                ),
            }
            bucket = self.holdout if abs(power - holdout_power) < 1e-9 else self.train
            bucket.append(condition)
            print(
                f"[data] {pattern:11s} {power:5.0f} W  {self.points:>7d} points x "
                f"{nt:2d}/{self.nt} times"
                f"{'   (holdout)' if bucket is self.holdout else ''}"
            )

        if not self.train:
            raise ValueError(f"holding out {holdout_power} W left nothing to train on")

        # The split is over *points* now, not over space-time samples: a point and
        # its whole history go to one side or the other together. Splitting inside
        # a history would put a node's t = 4.0 s in training and its t = 4.4 s in
        # validation, which is not a held-out anything.
        order = torch.randperm(self.points, generator=generator, device=device)
        cut = int(round(self.points * (1.0 - val_fraction)))
        self.train_index = order[:cut].to(torch.int32)
        self.val_index = order[cut:].to(torch.int32)

        self.branch_dim = 3 * sensors + 1
        self.max_power = max(c["power"] for c in self.train + self.holdout)

    @property
    def domain(self) -> Domain:
        return Domain(
            torch.tensor(
                [
                    [0.0, float(self.extent[0]) * MM],
                    [0.0, float(self.extent[1]) * MM],
                    [0.0, float(self.extent[2]) * MM],
                    [0.0, self.t_max],
                ]
            )
        )

    def _xyz(self, flat: Tensor) -> Tensor:
        """``[B, 3]`` in metres from flat spatial indices into ``meshgrid(x, y, z)``."""
        nx, ny, nz = self.shape
        flat = flat.long()
        iz = flat % nz
        rest = flat // nz
        iy = rest % ny
        ix = rest // ny
        step = self.spacing * MM
        return torch.stack(
            [ix.to(self.dtype) * step, iy.to(self.dtype) * step, iz.to(self.dtype) * step],
            dim=1,
        )

    def _branch(self, condition: dict, count: int) -> Tensor:
        power = torch.full(
            (count, 1), condition["power"], device=self.device, dtype=self.dtype
        )
        return torch.cat([condition["branch"].expand(count, -1), power], dim=1)

    def _mask(self, condition: dict, count: int, xyz: Tensor | None = None) -> Tensor:
        mask = torch.zeros((count, self.nt), device=self.device, dtype=self.dtype)
        mask[:, : condition["nt"]] = 1.0
        if self.cede_radius > 0.0 and xyz is not None:
            # Zero wherever the patch will answer instead: |along| and |across|
            # within the window, resolved per stored time because it turns.
            beam = condition["beam"]
            speed = torch.linalg.vector_norm(beam[:, 3:5], dim=1).clamp_min(1e-12)
            ux, uy = beam[:, 3] / speed, beam[:, 4] / speed
            dx = xyz[:, 0:1] - beam[None, :, 0]
            dy = xyz[:, 1:2] - beam[None, :, 1]
            along = dx * ux[None] + dy * uy[None]
            across = -dx * uy[None] + dy * ux[None]
            ceded = (along.abs() <= self.cede_radius) & (across.abs() <= self.cede_radius)
            mask = mask * (~ceded).to(mask.dtype)
        return mask

    def _target(self, condition: dict, flat: Tensor) -> Tensor:
        """``[B, Nt]``, zero-padded past this run's own end (the mask hides it)."""
        out = torch.zeros((flat.numel(), self.nt), device=self.device, dtype=self.dtype)
        out[:, : condition["nt"]] = condition["field"][:, flat.long()].T
        return out

    def batch(self, per_condition: int, generator: torch.Generator):
        """``(branch, xyz, beam, target, mask)`` -- beam is ``[C, Nt, 5]``, one per block."""
        branch, xyz, beams, target, mask = [], [], [], [], []
        for condition in self.train:
            draw = torch.randint(
                self.train_index.numel(), (per_condition,),
                generator=generator, device=self.device,
            )
            flat = self.train_index[draw]
            branch.append(self._branch(condition, per_condition))
            block = self._xyz(flat)
            xyz.append(block)
            beams.append(condition["beam"])
            target.append(self._target(condition, flat))
            mask.append(self._mask(condition, per_condition, block))
        return (
            torch.cat(branch), torch.cat(xyz), torch.stack(beams),
            torch.cat(target), torch.cat(mask),
        )

    @torch.no_grad()
    def score(self, model, which, limit=60_000, chunk=8192, generator=None):
        """``(rmse, max_abs_error)`` in Kelvin over the scored times only."""
        if which == "val":
            pairs = [(c, self.val_index) for c in self.train]
        elif which == "holdout":
            pairs = [
                (c, torch.arange(self.points, device=self.device, dtype=torch.int32))
                for c in self.holdout
            ]
        else:
            raise ValueError(f"which must be 'val' or 'holdout', got {which!r}")
        if not pairs:
            return float("nan"), float("nan")

        squared, worst, seen = 0.0, 0.0, 0
        for condition, index in pairs:
            if index.numel() > limit:
                draw = torch.randint(
                    index.numel(), (limit,), generator=generator, device=self.device
                )
                index = index[draw]
            for start in range(0, index.numel(), chunk):
                flat = index[start : start + chunk]
                xyz = self._xyz(flat)
                predicted = model(
                    self._branch(condition, flat.numel()), xyz, condition["beam"]
                )
                # Scored on what the model is answerable for. With `--cede-radius`
                # that is the plate minus the patch's window: selecting a
                # checkpoint by its accuracy in a region it was not trained on
                # would be choosing on noise.
                mask = self._mask(condition, flat.numel(), xyz)
                error = (predicted - self._target(condition, flat)) * mask
                squared += float(error.pow(2).sum())
                worst = max(worst, float(error.abs().max()))
                seen += int(mask.sum())
        return math.sqrt(squared / max(seen, 1)), worst


# ---------------------------------------------------------------------------
# The patch corpus: the melt pool alone, in the frame that rides with the beam
# ---------------------------------------------------------------------------

# Half-width of the window kept about the beam, in metres, along and across.
DEFAULT_RADIUS = 2.5e-3

# `|along| <= radius` decides which nodes are kept, and both sides are floating
# point sums over a 0.25 mm spacing, so a node sitting exactly on the boundary
# must not be lost to the last bit. A nanometre is far below the grid.
NODE_TOLERANCE = 1e-9


def beam_frame(x: np.ndarray, y: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plate coordinates to ``(along, across)`` in the beam's own travel frame.

    ``state`` is one row of :func:`beam_state`. ``along`` is the displacement
    resolved down the direction of travel -- negative behind the beam, where the
    trail is -- and ``across`` is the in-plane perpendicular.

    ``models/cpkmlp/dataset.py`` does not rotate, and is right not to: its beam
    only ever travels ``+x``, so ``dx`` *is* the along-track axis. Here the beam
    turns corners, and an unrotated window would put the wake at a different
    angle in every snapshot -- the network would have to infer the heading from
    the clock, which is the one thing the frame exists to remove. Rotated, the
    quasi-steady pool is stationary in these coordinates whatever the path is
    doing, which is what makes it a function this network can learn.
    """
    speed = float(np.hypot(state[3], state[4]))
    if speed <= 0.0:
        unit_x, unit_y = 1.0, 0.0
    else:
        unit_x, unit_y = state[3] / speed, state[4] / speed
    dx, dy = x - state[0], y - state[1]
    return dx * unit_x + dy * unit_y, -dx * unit_y + dy * unit_x


class PatchCorpus:
    """Every run's melt pool, in beam-frame rows, ready to be batched.

    The whole-plate corpus is 115 M points and the pool is a few millimetres of a
    20 mm square, so a mean squared error over the plate barely notices it and the
    fit blunts it. This keeps, from each stored snapshot, only the nodes inside
    ``radius`` of the beam -- the full depth in ``z``, since the pool reaches down
    from the surface and the beam has no depth to centre on -- and rewrites their
    position as ``(along, across)`` in the travel frame.

    The window is a square *in that frame*, not in the plate's, so the region the
    patch is fitted over and the region it is pasted over are the same square
    whatever direction the beam is heading. Rows are materialised here rather than
    decoded from indices as :class:`PathCorpus` does: at 2.5 mm this is 7 M rows,
    not 115 M, and which nodes fall in the window changes every snapshot, so
    there is no index arithmetic to exploit.

    Snapshots where the laser is dark are kept. There is no pool at the beam then
    -- for ``raster`` that is a fifth of the run -- but the paste needs an answer
    at every instant it covers, and refusing to fit the ones where the honest
    answer is "a cooling plate" would leave the patch extrapolating into them.
    """

    def __init__(
        self,
        directory: Path,
        holdout_power: float,
        sensors: int,
        val_fraction: float,
        radius: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        patterns: Sequence[str] | None = None,
    ) -> None:
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")

        runs = sorted(p for p in directory.iterdir() if (p / "config.json").exists())
        if patterns is not None:
            runs = [p for p in runs if p.name in set(patterns)]
        if not runs:
            raise FileNotFoundError(f"no solver runs under {directory}")

        self.device, self.dtype, self.radius = device, dtype, radius
        self.physics = PathCorpus._physics(runs[0])
        self.train: list[dict] = []
        self.holdout: list[dict] = []

        span = 0.0
        paths = {}
        for run in runs:
            paths[run.name] = Toolpath.load(run / "toolpath.json")
            span = max(span, paths[run.name].duration)
        self.sensors = np.linspace(0.0, span, sensors)
        self.path_span = span

        t_max = z_max = 0.0
        for run in runs:
            path = paths[run.name]
            branch = torch.as_tensor(
                branch_vector(path, self.sensors), device=device, dtype=dtype
            )
            for file in sorted(run.glob("data_*W.npy")):
                rows = np.load(file, mmap_mode="r")
                power = float(rows[0, 4])
                axes = PathCorpus._axes(rows)
                nx, ny, nz = axes["x"].size, axes["y"].size, axes["z"].size
                field = np.asarray(rows[:, 5], dtype=np.float32).reshape(-1, nx, ny, nz)
                # Kept for the checkpoint and for the branch's normalisation: the
                # trunk here is the window, but the branch still reads positions
                # on the plate and has to be scaled against the plate.
                self.shape = (nx, ny, nz)
                self.spacing = float(axes["x"][1] - axes["x"][0])
                self.plate = (float(axes["x"][-1]) * MM, float(axes["y"][-1]) * MM)
                states = beam_state(path, axes["t"])
                t_max = max(t_max, float(axes["t"][-1]))
                z_max = max(z_max, float(axes["z"][-1]) * MM)

                coords, temps = self._window(axes, field, states, radius)
                condition = {
                    "pattern": run.name, "power": power, "path": path,
                    "branch": branch,
                    "coords": torch.as_tensor(coords, device=device, dtype=dtype),
                    "field": torch.as_tensor(temps, device=device, dtype=dtype),
                }
                bucket = (
                    self.holdout if abs(power - holdout_power) < 1e-9 else self.train
                )
                bucket.append(condition)
                print(
                    f"[data] {run.name:11s} {power:5.0f} W  {coords.shape[0]:>8d} rows "
                    f"within {radius * 1e3:g} mm of the beam, peak {temps.max():7.1f} K"
                    f"{'   (holdout)' if bucket is self.holdout else ''}"
                )

        if not self.train:
            raise ValueError(f"holding out {holdout_power} W left nothing to train on")
        self.t_max, self.z_max = t_max, z_max

        self.train_index, self.val_index = [], []
        for condition in self.train:
            count = condition["field"].numel()
            order = torch.randperm(count, generator=generator, device=device)
            cut = int(round(count * (1.0 - val_fraction)))
            self.train_index.append(order[:cut].to(torch.int32))
            self.val_index.append(order[cut:].to(torch.int32))

        self.branch_dim = 3 * sensors + 1
        self.max_power = max(c["power"] for c in self.train + self.holdout)

    @staticmethod
    def _window(axes, field, states, radius):
        """``(coords[N,4] of (along, across, z, t), T[N])`` over every snapshot."""
        x_mm, y_mm, z_mm, times = axes["x"], axes["y"], axes["z"], axes["t"]
        x, y, z = x_mm * MM, y_mm * MM, z_mm * MM
        # A square in the rotated frame is contained in a plate-aligned box of
        # half-width `radius * sqrt(2)`; crop to that first so the rotation runs
        # over a few hundred nodes instead of the whole face.
        reach = radius * np.sqrt(2.0) + NODE_TOLERANCE

        coord_blocks, temperature_blocks = [], []
        for index, state in enumerate(states):
            keep_x = np.abs(x - state[0]) <= reach
            keep_y = np.abs(y - state[1]) <= reach
            if not keep_x.any() or not keep_y.any():
                raise ValueError(
                    f"the beam is off the plate at t = {times[index]} s; nothing to keep"
                )

            mesh_x, mesh_y, mesh_z = np.meshgrid(
                x[keep_x], y[keep_y], z, indexing="ij"
            )
            along, across = beam_frame(mesh_x, mesh_y, state)
            inside = (np.abs(along) <= radius + NODE_TOLERANCE) & (
                np.abs(across) <= radius + NODE_TOLERANCE
            )

            block = field[index][np.ix_(keep_x, keep_y, np.ones(z.size, bool))]
            coord_blocks.append(
                np.stack(
                    [
                        along[inside],
                        across[inside],
                        mesh_z[inside],
                        np.full(int(inside.sum()), float(times[index])),
                    ],
                    axis=1,
                ).astype(np.float32)
            )
            temperature_blocks.append(block[inside].astype(np.float32))

        return np.concatenate(coord_blocks), np.concatenate(temperature_blocks)

    @property
    def domain(self) -> Domain:
        """``(along, across, z, t)`` bounds -- the window, not the plate."""
        return Domain(
            torch.tensor(
                [
                    [-self.radius, self.radius],
                    [-self.radius, self.radius],
                    [0.0, self.z_max],
                    [0.0, self.t_max],
                ]
            )
        )

    def _branch(self, condition: dict, count: int) -> Tensor:
        power = torch.full(
            (count, 1), condition["power"], device=self.device, dtype=self.dtype
        )
        return torch.cat([condition["branch"].expand(count, -1), power], dim=1)

    def batch(self, per_condition: int, generator: torch.Generator) -> tuple[Tensor, ...]:
        branch, coords, temperature = [], [], []
        for condition, index in zip(self.train, self.train_index):
            draw = torch.randint(
                index.numel(), (per_condition,), generator=generator, device=self.device
            )
            flat = index[draw].long()
            branch.append(self._branch(condition, per_condition))
            coords.append(condition["coords"][flat])
            temperature.append(condition["field"][flat].unsqueeze(1))
        return torch.cat(branch), torch.cat(coords), torch.cat(temperature)

    @torch.no_grad()
    def score(self, model, which, limit=400_000, chunk=65536, generator=None):
        if which == "val":
            pairs = list(zip(self.train, self.val_index))
        elif which == "holdout":
            pairs = [
                (c, torch.arange(c["field"].numel(), device=self.device, dtype=torch.int32))
                for c in self.holdout
            ]
        else:
            raise ValueError(f"which must be 'val' or 'holdout', got {which!r}")
        if not pairs:
            return float("nan"), float("nan")

        squared, worst, seen = 0.0, 0.0, 0
        for condition, index in pairs:
            if index.numel() > limit:
                draw = torch.randint(
                    index.numel(), (limit,), generator=generator, device=self.device
                )
                index = index[draw]
            for start in range(0, index.numel(), chunk):
                flat = index[start : start + chunk].long()
                error = model(
                    self._branch(condition, flat.numel()),
                    condition["coords"][flat],
                    None,
                ) - condition["field"][flat].unsqueeze(1)
                squared += float(error.pow(2).sum())
                worst = max(worst, float(error.abs().max()))
                seen += flat.numel()
        return math.sqrt(squared / seen), worst


class SequencePatchCorpus:
    """The melt pool as a history: a fixed offset from the beam, over the whole scan.

    ``PatchCorpus`` keeps, per snapshot, whichever *nodes* fall inside the window,
    so the offsets it hands the network are a slightly different set at every
    instant. That is fine when each row is scored on its own. It does not give a
    sequence: to ask "what does the temperature at 1 mm behind the beam do over
    the scan" the offset has to be the *same* offset at every time, and no node is.

    So the lattice is defined in the beam frame -- ``(along, across)`` on the grid
    spacing, ``|.| <= radius``, at the plate's own depths -- and the solver field
    is **interpolated** onto it. That is the one place in this repository where a
    target is not a raw solver node, and it is worth being exact about the cost:
    bilinear interpolation on this 0.25 mm grid is worth about 11 K at worst and
    0.2 K in RMS, measured by coarsening the field to 0.5 mm and interpolating the
    skipped nodes back. In the melt pool itself it is nearer 5 K. Against a patch
    whose errors are hundreds of Kelvin that is noise, and it touches training
    targets only -- the paste is still scored against raw nodes.

    Interpolation is bilinear in ``(x, y)`` and exact in ``z``: the lattice uses
    the plate's own depths, so there is nothing to interpolate along that axis and
    no error introduced on it.

    A lattice point can fall off the plate -- the window's corner reaches
    ``radius * sqrt(2) = 3.54 mm`` and the scan is inset only 3 mm -- so the mask
    carries that as well as the ragged clock.
    """

    def __init__(
        self,
        directory: Path,
        holdout_power: float,
        sensors: int,
        val_fraction: float,
        radius: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        patterns: Sequence[str] | None = None,
        retouched: bool = False,
    ) -> None:
        runs = sorted(p for p in directory.iterdir() if (p / "config.json").exists())
        if patterns is not None:
            runs = [p for p in runs if p.name in set(patterns)]
        if not runs:
            raise FileNotFoundError(f"no solver runs under {directory}")

        self.device, self.dtype, self.radius = device, dtype, radius
        self.retouched = bool(retouched)
        self.physics = PathCorpus._physics(runs[0])
        self.train: list[dict] = []
        self.holdout: list[dict] = []

        span = 0.0
        paths = {}
        for run in runs:
            paths[run.name] = Toolpath.load(run / "toolpath.json")
            span = max(span, paths[run.name].duration)
        self.sensors = np.linspace(0.0, span, sensors)
        self.path_span = span

        loaded, step, t_max = [], None, 0.0
        for run in runs:
            for file in sorted(run.glob("data_*W.npy")):
                rows = np.load(file, mmap_mode="r")
                axes = PathCorpus._axes(rows)
                if step is None:
                    step = float(np.diff(axes["t"]).mean())
                    self.axes = axes
                    self.spacing = float(axes["x"][1] - axes["x"][0])
                t_max = max(t_max, float(axes["t"][-1]))
                loaded.append((run.name, float(rows[0, 4]), axes["t"].size, file, run.name))

        self.time_step = float(step)
        self.nt = int(round(t_max / self.time_step)) + 1
        self.times = torch.arange(self.nt, device=device, dtype=dtype) * self.time_step
        self.t_max = t_max

        # The lattice, in metres, in the beam's frame. `z` is the plate's own.
        self.shape = (
            self.axes["x"].size, self.axes["y"].size, self.axes["z"].size
        )
        n = int(round(2 * radius / (self.spacing * MM))) + 1
        offs = np.linspace(-radius, radius, n)
        depth = self.axes["z"] * MM
        gz, ga, gs = np.meshgrid(depth, offs, offs, indexing="ij")
        self.lattice = np.stack([ga.ravel(), gs.ravel(), gz.ravel()], axis=1)  # (M,3)
        self.points = self.lattice.shape[0]
        self.plate = (float(self.axes["x"][-1]) * MM, float(self.axes["y"][-1]) * MM)

        for pattern, power, nt, file, run_name in loaded:
            path = paths[run_name]
            rows = np.load(file, mmap_mode="r")
            nx, ny, nz = self.axes["x"].size, self.axes["y"].size, self.axes["z"].size
            field = np.asarray(rows[:, 5], dtype=np.float32).reshape(nt, nx, ny, nz)
            target, mask = self._sample(field, path, nt)

            condition = {
                "pattern": pattern, "power": power, "path": path, "nt": nt,
                "branch": torch.as_tensor(
                    branch_vector(path, self.sensors, self.retouched),
                    device=device, dtype=dtype,
                ),
                "target": torch.as_tensor(target, device=device).to(dtype),
                "mask": torch.as_tensor(mask, device=device).to(dtype),
            }
            bucket = self.holdout if abs(power - holdout_power) < 1e-9 else self.train
            bucket.append(condition)
            print(
                f"[data] {pattern:11s} {power:5.0f} W  {self.points:>6d} offsets x "
                f"{nt:2d}/{self.nt} times, {100 * mask.mean():.1f}% on the plate"
                f"{'   (holdout)' if bucket is self.holdout else ''}"
            )

        if not self.train:
            raise ValueError(f"holding out {holdout_power} W left nothing to train on")

        order = torch.randperm(self.points, generator=generator, device=device)
        cut = int(round(self.points * (1.0 - val_fraction)))
        self.train_index = order[:cut].to(torch.int32)
        self.val_index = order[cut:].to(torch.int32)

        self.coords = torch.as_tensor(self.lattice, device=device, dtype=dtype)
        self.branch_dim = 3 * sensors + 1
        self.max_power = max(c["power"] for c in self.train + self.holdout)

    def _sample(self, field: np.ndarray, path: Toolpath, nt: int):
        """``(target[M, Nt], mask[M, Nt])`` -- the lattice, interpolated, per time."""
        x = self.axes["x"] * MM
        y = self.axes["y"] * MM
        dx = float(x[1] - x[0])
        target = np.zeros((self.points, self.nt), dtype=np.float32)
        mask = np.zeros((self.points, self.nt), dtype=np.float32)

        nz = self.axes["z"].size
        along, across = self.lattice[:, 0], self.lattice[:, 1]
        iz = np.rint(self.lattice[:, 2] / (self.axes["z"][1] - self.axes["z"][0]) / MM)
        iz = iz.astype(int)

        states = beam_of(self.retouched)(path, np.arange(nt) * self.time_step)
        for k, state in enumerate(states):
            speed = float(np.hypot(state[3], state[4]))
            ux, uy = (1.0, 0.0) if speed <= 0 else (state[3] / speed, state[4] / speed)
            px = state[0] + along * ux - across * uy
            py = state[1] + along * uy + across * ux

            inside = (px >= x[0]) & (px <= x[-1]) & (py >= y[0]) & (py <= y[-1])
            fx = np.clip((px - x[0]) / dx, 0, len(x) - 1.000001)
            fy = np.clip((py - y[0]) / dx, 0, len(y) - 1.000001)
            i0, j0 = fx.astype(int), fy.astype(int)
            a, b = fx - i0, fy - j0

            f = field[k]
            target[:, k] = (
                f[i0, j0, iz] * (1 - a) * (1 - b)
                + f[i0 + 1, j0, iz] * a * (1 - b)
                + f[i0, j0 + 1, iz] * (1 - a) * b
                + f[i0 + 1, j0 + 1, iz] * a * b
            )
            mask[:, k] = inside
        return target, mask

    @property
    def domain(self) -> Domain:
        z_max = float(self.axes["z"][-1]) * MM
        return Domain(
            torch.tensor(
                [
                    [-self.radius, self.radius],
                    [-self.radius, self.radius],
                    [0.0, z_max],
                    [0.0, self.t_max],
                ]
            )
        )

    def _branch(self, condition, count):
        power = torch.full(
            (count, 1), condition["power"], device=self.device, dtype=self.dtype
        )
        return torch.cat([condition["branch"].expand(count, -1), power], dim=1)

    def batch(self, per_condition, generator):
        branch, coords, target, mask = [], [], [], []
        for condition in self.train:
            draw = torch.randint(
                self.train_index.numel(), (per_condition,),
                generator=generator, device=self.device,
            )
            flat = self.train_index[draw].long()
            branch.append(self._branch(condition, per_condition))
            coords.append(self.coords[flat])
            target.append(condition["target"][flat])
            mask.append(condition["mask"][flat])
        return torch.cat(branch), torch.cat(coords), None, torch.cat(target), torch.cat(mask)

    @torch.no_grad()
    def score(self, model, which, limit=None, chunk=4096, generator=None):
        if which == "val":
            pairs = [(c, self.val_index) for c in self.train]
        elif which == "holdout":
            pairs = [
                (c, torch.arange(self.points, device=self.device, dtype=torch.int32))
                for c in self.holdout
            ]
        else:
            raise ValueError(f"which must be 'val' or 'holdout', got {which!r}")
        if not pairs:
            return float("nan"), float("nan")

        squared, worst, seen = 0.0, 0.0, 0
        for condition, index in pairs:
            for start in range(0, index.numel(), chunk):
                flat = index[start : start + chunk].long()
                m = condition["mask"][flat]
                error = (
                    model(self._branch(condition, flat.numel()), self.coords[flat], None)
                    - condition["target"][flat]
                ) * m
                squared += float(error.pow(2).sum())
                worst = max(worst, float(error.abs().max()))
                seen += int(m.sum())
        return math.sqrt(squared / max(seen, 1)), worst


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class PathAgent(BaseAgent):
    """One trained network *plus one toolpath*, behind the shared agent contract.

    ``predict_at`` takes ``[B, 5]`` of ``(x, y, z, t, P)`` and nothing else, which
    is what every downstream script in this repository is written against. A
    toolpath-conditioned model needs a sixth thing -- which path -- and there is
    no room for it in that signature, so it is fixed when the agent is built.
    An agent is therefore a model *at* a scan pattern, and comparing two patterns
    means building two agents from the one checkpoint.

    That is not a workaround. The branch is constant over a whole field: for one
    path and one power it is evaluated once, not once per query point, and fixing
    it here is what lets ``predict_of`` reconstruct a volume in a single pass.
    """

    def __init__(
        self,
        model: "PathDeepONet",
        path: Toolpath,
        sensors: np.ndarray,
        bounds: Tensor,
        shape: tuple[int, int, int],
        device: torch.device | str = "cpu",
        chunk: int = 65536,
    ) -> None:
        dtype = next(model.parameters()).dtype
        super().__init__(bounds, shape=shape, device=device, dtype=dtype, chunk=chunk)
        self.model = model.to(self.device).eval()
        self.path = path
        self.branch_path = torch.as_tensor(
            branch_vector(path, sensors), device=self.device, dtype=dtype
        )

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")

        outputs = []
        for start in range(0, inputs.size(0), self.chunk):
            block = inputs[start : start + self.chunk]
            beam = torch.as_tensor(
                beam_state(self.path, block[:, 3].detach().cpu().numpy()),
                device=self.device,
                dtype=self.dtype,
            )
            branch = torch.cat(
                [self.branch_path.expand(block.size(0), -1), block[:, 4:5]], dim=1
            )
            outputs.append(self.model(branch, block[:, 0:4], beam))
        return torch.cat(outputs)


class SequenceAgent(BaseAgent):
    """A history-predicting model behind the pointwise ``[B, 5]`` contract.

    Every script downstream asks for ``(x, y, z, t, P)`` and expects one number
    back. This model answers on a fixed grid of times instead, so the agent
    evaluates the whole history at each point and reads ``t`` off it -- linearly
    between the two nearest stored times, and held at the ends.

    That interpolation is the agent's, not the model's, and it is worth being
    plain about: ``don`` and its siblings are continuous in ``t`` because ``t``
    goes into the trunk, and these are not. At the solver's own 0.4 s cadence the
    beam moves 4 mm between stored times, so a query landing between them is
    genuinely being guessed at. Asking for a stored time -- which is what
    ``visualize_don.py`` does -- costs nothing.
    """

    def __init__(
        self,
        model: "SequenceDeepONet",
        path: Toolpath,
        sensors: np.ndarray,
        times: np.ndarray,
        bounds: Tensor,
        shape: tuple[int, int, int],
        device: torch.device | str = "cpu",
        chunk: int = 16384,
        retouched: bool = False,
    ) -> None:
        dtype = next(model.parameters()).dtype
        super().__init__(bounds, shape=shape, device=device, dtype=dtype, chunk=chunk)
        self.model = model.to(self.device).eval()
        self.path = path
        self.retouched = bool(retouched)
        self.times = torch.as_tensor(times, device=self.device, dtype=dtype)
        self.branch_path = torch.as_tensor(
            branch_vector(path, sensors, self.retouched), device=self.device, dtype=dtype
        )
        self.beam = torch.as_tensor(
            beam_of(self.retouched)(path, np.asarray(times, dtype=float)),
            device=self.device, dtype=dtype,
        )

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")

        outputs = []
        for start in range(0, inputs.size(0), self.chunk):
            block = inputs[start : start + self.chunk]
            branch = torch.cat(
                [self.branch_path.expand(block.size(0), -1), block[:, 4:5]], dim=1
            )
            history = self.model(branch, block[:, 0:3], self.beam)  # [B, Nt]
            outputs.append(self._at_time(history, block[:, 3]))
        return torch.cat(outputs)

    def _at_time(self, history: Tensor, when: Tensor) -> Tensor:
        """``[B, 1]`` -- ``history`` read at ``when``, linearly between stored times."""
        t = when.clamp(float(self.times[0]), float(self.times[-1]))
        upper = torch.searchsorted(self.times, t.contiguous()).clamp(1, len(self.times) - 1)
        lower = upper - 1
        span = (self.times[upper] - self.times[lower]).clamp_min(1e-12)
        weight = ((t - self.times[lower]) / span).unsqueeze(1)
        low = history.gather(1, lower.unsqueeze(1))
        high = history.gather(1, upper.unsqueeze(1))
        return low + weight * (high - low)


class SequencePatchAgent(SequenceAgent):
    """``pkfdon``: the melt pool's history, asked for in plate coordinates.

    Two conversions on the way in, both of which the agent can do because it holds
    the toolpath: the query point is resolved into the beam's travel frame at the
    query time, and the history that comes back is read at that same time. What a
    caller passes and gets is still ``(x, y, z, t, P) -> K``.
    """

    def __init__(self, *args, radius: float = DEFAULT_RADIUS, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.radius = float(radius)

    def frame(self, block: Tensor) -> tuple[Tensor, Tensor]:
        """``(offsets[B, 3], inside[B])`` for ``[B, 5]`` plate rows."""
        state = torch.as_tensor(
            beam_of(self.retouched)(self.path, block[:, 3].detach().cpu().numpy()),
            device=self.device, dtype=self.dtype,
        )
        speed = torch.linalg.vector_norm(state[:, 3:5], dim=1, keepdim=True).clamp_min(1e-12)
        unit_x, unit_y = state[:, 3:4] / speed, state[:, 4:5] / speed
        dx = block[:, 0:1] - state[:, 0:1]
        dy = block[:, 1:2] - state[:, 1:2]

        along = dx * unit_x + dy * unit_y
        across = -dx * unit_y + dy * unit_x
        inside = (along.abs() <= self.radius + NODE_TOLERANCE) & (
            across.abs() <= self.radius + NODE_TOLERANCE
        )
        return torch.cat([along, across, block[:, 2:3]], dim=1), inside.squeeze(1)

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")

        outputs = []
        for start in range(0, inputs.size(0), self.chunk):
            block = inputs[start : start + self.chunk]
            offsets, _ = self.frame(block)
            branch = torch.cat(
                [self.branch_path.expand(block.size(0), -1), block[:, 4:5]], dim=1
            )
            history = self.model(branch, offsets, None)  # [B, Nt]
            outputs.append(self._at_time(history, block[:, 3]))
        return torch.cat(outputs)

    def patch_box(self, time: float) -> np.ndarray:
        """The four corners of the pasted square at ``time``, ``[4, 2]`` in metres."""
        state = beam_of(self.retouched)(self.path, np.array([float(time)]))[0]
        speed = float(np.hypot(state[3], state[4]))
        unit = (
            np.array([1.0, 0.0]) if speed <= 0.0
            else np.array([state[3], state[4]]) / speed
        )
        normal = np.array([-unit[1], unit[0]])
        centre = np.array([state[0], state[1]])
        return np.stack(
            [centre + self.radius * (a * unit + b * normal)
             for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        )


class GenericPeakCorrectedAgent(BaseAgent):
    """``base``'s field with any patch pasted over the window the patch owns.

    It asks the patch two questions rather than reaching into it -- ``frame`` for
    which rows are its business, ``predict_at`` for what it says about them -- so
    one class serves ``pkdon`` over ``don`` and ``pkfdon`` over ``fdon``, whose
    insides are nothing alike. There were two of these; there is no need for two.
    """

    def __init__(self, base: BaseAgent, patch: BaseAgent) -> None:
        super().__init__(
            base.bounds, shape=base.shape, device=base.device,
            dtype=base.dtype, chunk=base.chunk,
        )
        if patch.path.to_dict()["nodes_mm"] != base.path.to_dict()["nodes_mm"]:
            raise ValueError(
                "base and patch are pointed at different toolpaths; the paste "
                "would put one model's melt pool where the other's beam is not"
            )
        # A base trained with `--cede-radius` has a hole in it exactly the size it
        # was told the patch would fill. If the patch is a different size, the
        # paste either leaves a ring neither model ever learned (cede larger) or
        # overwrites ground the base did learn (cede smaller). Neither is
        # detectable in the output -- it just reads as a worse model.
        if bool(getattr(base, "retouched", False)) != bool(
            getattr(patch, "retouched", False)
        ):
            raise ValueError(
                "one of base and patch holds the beam over dark travel and the "
                "other follows the head; their windows are in different places "
                "whenever the laser is off"
            )
        ceded = float(getattr(base, "cede_radius", 0.0))
        if ceded > 0.0 and abs(ceded - patch.radius) > 1e-9:
            raise ValueError(
                f"base ceded a {2e3 * ceded:g} mm square but the patch covers "
                f"{2e3 * patch.radius:g} mm; they must match"
            )
        self.base, self.patch = base, patch
        self.path = base.path
        self.model = torch.nn.ModuleList([base.model, patch.model])

    def patch_box(self, time: float):
        return self.patch.patch_box(time)

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        """Each row is answered by exactly one of the two, and evaluated once.

        The base used to be run over everything and then overwritten inside the
        window, which threw away 6.3% of its work -- and, worse, read as though
        both models had a say there. They do not: inside the window the base is
        not consulted at all, and this says so.
        """
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")
        outputs = torch.empty(
            (inputs.size(0), 1), dtype=self.dtype, device=self.device
        )

        for start in range(0, inputs.size(0), self.chunk):
            block = inputs[start : start + self.chunk]
            _, inside = self.patch.frame(block)
            view = outputs[start : start + self.chunk]
            if bool(inside.any()):
                view[inside] = self.patch.predict_at(block[inside])
            outside = ~inside
            if bool(outside.any()):
                view[outside] = self.base.predict_at(block[outside])
        return outputs


class PatchAgent(PathAgent):
    """``pkdon`` alone: the melt pool, asked for in the beam's travel frame.

    The one agent in this family that is not a surrogate for the plate. Its
    ``predict_at`` takes the same ``[B, 5]`` of ``(x, y, z, t, P)`` every script
    is written against -- absolute plate coordinates -- and converts them to the
    frame it was fitted in on the way in, because it knows the toolpath and can.
    ``cpkmlp`` pushes that conversion onto its caller (its ``dx``, ``dy`` are the
    caller's job to compute); here the path is already in hand, so the agent can
    keep the contract honest instead.

    Asking it for a point 10 mm from the beam is a question it has no answer to.
    :class:`GenericPeakCorrectedAgent` is what decides which points those are.
    """

    def __init__(self, *args, radius: float = DEFAULT_RADIUS, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.radius = float(radius)

    def frame(self, block: Tensor) -> tuple[Tensor, Tensor]:
        """``(coords[B,4] in the beam frame, inside[B])`` for ``[B, 5]`` plate rows."""
        state = torch.as_tensor(
            beam_state(self.path, block[:, 3].detach().cpu().numpy()),
            device=self.device, dtype=self.dtype,
        )
        speed = torch.linalg.vector_norm(state[:, 3:5], dim=1, keepdim=True).clamp_min(1e-12)
        unit_x, unit_y = state[:, 3:4] / speed, state[:, 4:5] / speed
        dx = block[:, 0:1] - state[:, 0:1]
        dy = block[:, 1:2] - state[:, 1:2]

        along = dx * unit_x + dy * unit_y
        across = -dx * unit_y + dy * unit_x
        coords = torch.cat([along, across, block[:, 2:3], block[:, 3:4]], dim=1)
        inside = (along.abs() <= self.radius + NODE_TOLERANCE) & (
            across.abs() <= self.radius + NODE_TOLERANCE
        )
        return coords, inside.squeeze(1)

    @torch.no_grad()
    def predict_at(self, inputs: ArrayLike) -> Tensor:
        inputs = self._as_tensor(inputs, columns=5, name="predict_at")
        outputs = []
        for start in range(0, inputs.size(0), self.chunk):
            block = inputs[start : start + self.chunk]
            coords, _ = self.frame(block)
            branch = torch.cat(
                [self.branch_path.expand(block.size(0), -1), block[:, 4:5]], dim=1
            )
            outputs.append(self.model(branch, coords, None))
        return torch.cat(outputs)

    def patch_box(self, time: float) -> np.ndarray:
        """The four corners of the pasted square at ``time``, ``[4, 2]`` in metres.

        A polygon rather than a pair of intervals: this window is rotated into the
        beam's travel frame and turns with it, so on a corner it sits at an angle
        to the plate and a bounding box would claim ground the paste never touched.
        """
        state = beam_state(self.path, np.array([float(time)]))[0]
        speed = float(np.hypot(state[3], state[4]))
        unit = (
            np.array([1.0, 0.0]) if speed <= 0.0
            else np.array([state[3], state[4]]) / speed
        )
        normal = np.array([-unit[1], unit[0]])
        centre = np.array([state[0], state[1]])
        return np.stack(
            [centre + self.radius * (a * unit + b * normal)
             for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        )


def build_agent(
    checkpoint: Path,
    model_name: str,
    toolpath: Path | Toolpath | None = None,
    shape: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
) -> PathAgent:
    """Rebuild a trained network and point it at a toolpath.

    ``toolpath`` defaults to the first run under ``data/toolpath`` so that
    ``build_agent(checkpoint)`` works for a quick look, but it warns when it does.
    The default is a real trap: ``visualize.py`` calls ``build_agent(model,
    checkpoint, device=...)`` with no room to pass a path, and picks the field it
    compares against with ``--data-dir``. Nothing makes those two agree, and a
    figure of ``spiral``'s temperatures under ``nested_l``'s beam is a wrong
    figure that looks like a right one. Anything comparing patterns must pass
    ``toolpath`` explicitly.

    The sensor grid and the reconstruction shape ride in the checkpoint, so the
    branch is read at the times it was trained with rather than at times a caller
    guessed.
    """
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    payload = load_checkpoint(checkpoint, map_location=device)
    for key in ("architecture", "bounds", "sensors", "field_shape"):
        if key not in payload:
            raise KeyError(f"{checkpoint} has no {key!r}; it predates {model_name}")

    if toolpath is None:
        toolpath = sorted(DEFAULT_TOOLPATH_DIR.iterdir())[0] / "toolpath.json"
        warnings.warn(
            f"{model_name}: no toolpath given, defaulting to {toolpath.parent.name}. "
            f"This agent will answer as if every query were on that scan; if the "
            f"field you are comparing against came from another one, the result is "
            f"meaningless. Pass toolpath= explicitly.",
            stacklevel=2,
        )
    path = toolpath if isinstance(toolpath, Toolpath) else Toolpath.load(Path(toolpath))

    if "times" in payload:
        model = SequenceDeepONet(**payload["architecture"])
        model.load_state_dict(payload["model"])
        retouched = bool(payload.get("retouched", False))
        if "radius" in payload:
            return SequencePatchAgent(
                model, path, np.asarray(payload["sensors"], dtype=float),
                np.asarray(payload["times"], dtype=float), payload["bounds"],
                shape=tuple(shape or payload["field_shape"]), device=device,
                radius=float(payload["radius"]), retouched=retouched,
            )
        agent = SequenceAgent(
            model, path, np.asarray(payload["sensors"], dtype=float),
            np.asarray(payload["times"], dtype=float), payload["bounds"],
            shape=tuple(shape or payload["field_shape"]), device=device,
            retouched=retouched,
        )
        agent.cede_radius = float(payload.get("cede_radius", 0.0))
        return agent

    model = PathDeepONet(**payload["architecture"])
    model.load_state_dict(payload["model"])

    # `pkdon` answers in the beam frame and only inside its own window, so it
    # gets the agent that knows both facts. `bounds` for it is that window, not
    # the plate, which is why the two cannot share one class.
    if "radius" in payload:
        return PatchAgent(
            model, path, np.asarray(payload["sensors"], dtype=float),
            payload["bounds"], shape=tuple(shape or payload["field_shape"]),
            device=device, radius=float(payload["radius"]),
        )
    return PathAgent(
        model,
        path,
        np.asarray(payload["sensors"], dtype=float),
        payload["bounds"],
        shape=tuple(shape or payload["field_shape"]),
        device=device,
    )


# ---------------------------------------------------------------------------
# The training loop, shared by the three
# ---------------------------------------------------------------------------


def parse_args(model_name: str, doc: str, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f"train.py {model_name}", description=doc)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TOOLPATH_DIR,
                        help="directory of solver runs, one subdirectory each")
    parser.add_argument("--patterns", type=str, nargs="+", default=None,
                        help="restrict to these run directories (default: all of them)")
    parser.add_argument("--holdout", type=float, default=175.0,
                        help="the power held out of training entirely, in W")
    parser.add_argument("--cede-radius", type=float, default=0.0,
                        help="sequence bases only: drop the beam's neighbourhood "
                             "from the loss, on the grounds that pkfdon will be "
                             "pasted over it. 0 keeps the whole plate (default)")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                        help="patch models only: half-width in metres of the "
                             "window kept about the beam, along and across its "
                             f"travel (default: {DEFAULT_RADIUS})")
    parser.add_argument("--sensors", type=int, default=DEFAULT_SENSORS,
                        help="times the branch reads the toolpath at")
    parser.add_argument("--iterations", type=int, default=40000)
    add_optimizer_args(parser)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128, 128],
                        help="hidden widths of the branch and trunk subnetworks alike")
    parser.add_argument("--latent-dim", type=int, default=128,
                        help="width of the inner product the branch and trunk meet in")
    parser.add_argument("--batch-data", type=int, default=4096,
                        help="target batch size; rounded to a multiple of the "
                             "condition count so every run contributes equally")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=500,
                        help="validation and console cadence")
    parser.add_argument("--scalar-every", type=int, default=25,
                        help="TensorBoard loss cadence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--double", action="store_true", help="run in float64")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help=f"defaults to checkpoints/{model_name}/<run-name>.pt")
    parser.add_argument("--logdir", type=Path, default=DEFAULT_LOG_DIR,
                        help="TensorBoard output")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="label folded into the run name")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--gaussian-exponent-scale", type=float, default=1.0,
                        help="tightens (>1) or widens (<1) the gaussian gate; "
                             "ignored by don and jdon")
    parser.add_argument("--gate-offset", type=float, default=0.5,
                        help="initial value of the learnable p in G = g + p; "
                             "ignored by don")
    return parser.parse_args(argv)


def run(model_name: str, doc: str, argv: list[str] | None = None) -> None:
    """Load the corpus, fit ``model_name``'s network to it, and checkpoint the best."""
    args = parse_args(model_name, doc, argv)
    dtype = torch.float64 if args.double else torch.float32
    torch.set_default_dtype(dtype)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    patching = model_name in PATCH_MODELS
    sequence = model_name in SEQUENCE_MODELS
    retouched = model_name in RETOUCHED_MODELS
    common = dict(
        holdout_power=args.holdout, sensors=args.sensors,
        val_fraction=args.val_fraction, generator=generator,
        device=device, dtype=dtype, patterns=args.patterns,
    )
    if patching and sequence:
        corpus = SequencePatchCorpus(
            args.data_dir, radius=args.radius, retouched=retouched, **common
        )
    elif patching:
        corpus = PatchCorpus(args.data_dir, radius=args.radius, **common)
    elif sequence:
        corpus = SequenceCorpus(
            args.data_dir, cede_radius=args.cede_radius, retouched=retouched, **common
        )
    else:
        corpus = PathCorpus(args.data_dir, **common)
    domain = corpus.domain

    # The branch vector is (x_l, y_l, lit) repeated, then P. `lit` is already 0
    # or 1 and the power scales against the sweep's maximum, so every input the
    # branch sees is O(1) without a statistics pass over the corpus.
    half = domain.half_width.tolist()
    centre = domain.center.tolist()

    # The beam positions in that vector are on the *plate*, and for a patch model
    # the trunk's domain is the window about the beam -- a few millimetres. Scaling
    # a 20 mm coordinate by a 2.5 mm half-width would hand the branch inputs of
    # order 8 and undo the point of normalising at all, so the plate is used here
    # whatever the trunk is measured in.
    plate = corpus.plate if patching else (half[0], half[1])
    plate_centre = (
        (plate[0] / 2, plate[1] / 2) if patching else (centre[0], centre[1])
    )
    branch_mean = ([plate_centre[0], plate_centre[1], 0.5] * args.sensors) + [0.0]
    branch_scale = ([plate[0] / 2, plate[1] / 2, 0.5] * args.sensors) + [corpus.max_power]

    if patching and sequence:
        fields = [c["target"] for c in corpus.train]
    elif patching or sequence:
        fields = [c["field"] for c in corpus.train]
    else:
        fields = [c.field for c in corpus.train]
    temperature_rise = max(float(f.max()) for f in fields) - AMBIENT_TEMPERATURE

    architecture = dict(
        branch_input_dim=corpus.branch_dim,
        gate=GATES[model_name],
        physics=corpus.physics,
        hidden_layers=tuple(args.hidden),
        latent_dim=args.latent_dim,
        coord_mean=centre,
        coord_scale=half,
        branch_mean=branch_mean,
        branch_scale=branch_scale,
        temperature_offset=AMBIENT_TEMPERATURE,
        temperature_scale=temperature_rise,
        gaussian_exponent_scale=args.gaussian_exponent_scale,
        gate_offset=args.gate_offset,
    )
    if sequence:
        architecture["steps"] = corpus.nt
        architecture["coord_mean"] = centre[:3]
        architecture["coord_scale"] = half[:3]
        model = SequenceDeepONet(**architecture).to(device=device, dtype=dtype)
        criterion = MaskedScaledMSELoss(scale=temperature_rise)
    else:
        model = PathDeepONet(**architecture).to(device=device, dtype=dtype)
        criterion = ScaledMSELoss(scale=temperature_rise)

    learning_rate = (
        resolve_lr(args) if args.lr is not None or args.optimizer != "adam"
        else DEFAULT_LR
    )
    optimizer = build_optimizer(model.parameters(), args, learning_rate)
    use_lbfgs = args.optimizer == "lbfgs"
    scheduler = (
        None
        if use_lbfgs
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations)
    )

    per_condition = max(1, round(args.batch_data / len(corpus.train)))
    batch_size = per_condition * len(corpus.train)

    run_name = args.run_name or make_run_name(model_name, args.tag)
    run_dir = resolve_run_dir(args.logdir, run_name)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, model_name, run_name)
    best = BestCheckpoint(checkpoint_path, mode="min")
    writer = SummaryWriter(log_dir=str(run_dir))

    print(f"[setup] tensorboard run -> {run_dir}")
    print(f"[setup] checkpoint      -> {checkpoint_path}")
    print(f"[setup] device={device} dtype={dtype} params={count_parameters(model)}")
    print(f"[setup] gate={GATES[model_name]}  branch_dim={corpus.branch_dim} "
          f"({args.sensors} sensors over {corpus.path_span:.2f} s + P)")
    print(f"[setup] grid={corpus.shape} spacing={corpus.spacing:g} mm  "
          f"t_max={corpus.t_max:.2f}s")
    if patching and sequence:
        print(f"[setup] {corpus.points} beam-frame offsets per condition "
              f"(+/-{args.radius * 1e3:g} mm, {corpus.spacing:g} mm apart), "
              f"targets interpolated onto them")
    if sequence:
        print(f"[setup] output is a history: [B, {corpus.nt}] on a "
              f"{corpus.time_step:g} s grid to {corpus.t_max:g} s; trunk takes "
              f"(x, y, z) and no clock")
        print(f"[setup] {corpus.points} points per condition, "
              f"{corpus.train_index.numel()} of them trainable")
    if patching and not sequence:
        rows = sum(c["field"].numel() for c in corpus.train)
        print(f"[setup] patch window +/-{args.radius * 1e3:g} mm about the beam, "
              f"in its travel frame; {rows} training rows")
    print(f"[setup] train {len(corpus.train)} conditions, holdout {len(corpus.holdout)} "
          f"at {args.holdout:g} W  T_rise={temperature_rise:.1f} K")
    print(f"[setup] batch={batch_size} ({per_condition}/condition)  "
          f"{args.iterations} iterations")
    for key, value in corpus.physics.items():
        print(f"[setup] {key} = {value:.6g}")

    progress = tqdm(range(1, args.iterations + 1), desc=model_name, unit="it",
                    disable=args.no_progress, dynamic_ncols=True)
    scoring = torch.Generator(device=device).manual_seed(args.seed + 1)
    grad_norm, skipped = 0.0, 0

    for iteration in progress:
        model.train()
        mask = None
        if patching and sequence:
            branch, coords, beam, temperature, mask = corpus.batch(
                per_condition, generator
            )
        elif patching:
            branch, coords, temperature = corpus.batch(per_condition, generator)
            beam = None  # the trunk is already in the beam's frame
        elif sequence:
            branch, coords, beam, temperature, mask = corpus.batch(
                per_condition, generator
            )
        else:
            branch, coords, beam, temperature = corpus.batch(per_condition, generator)

        def closure():
            nonlocal grad_norm
            optimizer.zero_grad(set_to_none=True)
            predicted = model(branch, coords, beam)
            loss = (
                criterion(predicted, temperature, mask)
                if mask is not None
                else criterion(predicted, temperature)
            )
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            )
            return loss

        if use_lbfgs:
            loss = optimizer.step(closure)
        else:
            loss = closure()
            # A step taken on a non-finite gradient poisons Adam's moments, and
            # from there every step after it is wrong however small it is. Skip
            # it instead.
            #
            # This does *not* catch the failure that put it here. At the
            # repository's default `--lr 1e-3` this network diverged at iteration
            # 8000 -- loss 1.8e+02, a 17000 K validation error -- and spent the
            # remaining 132000 iterations climbing back out to finish worse than
            # it had been at 6000. Every one of those numbers is finite, and the
            # gradients were already clipped to a norm of 1.0, so neither this
            # check nor a larger clip would have stopped it: the step size was
            # simply too large for the curvature. The fix is `DEFAULT_LR` below,
            # measured rather than guessed. This check is for the other failure.
            if math.isfinite(grad_norm) and torch.isfinite(loss):
                optimizer.step()
            else:
                skipped += 1
            scheduler.step()

        if iteration % args.scalar_every == 0 or iteration == 1:
            value = loss.detach().item()
            writer.add_scalar("loss/data", value, iteration)
            writer.add_scalar("grad_norm", grad_norm, iteration)
            current = learning_rate if scheduler is None else scheduler.get_last_lr()[0]
            writer.add_scalar("lr", current, iteration)
            progress.set_postfix(loss=f"{value:.3e}", best=f"{best.best:.1f}K")

        if iteration % args.log_every == 0 or iteration == 1:
            model.eval()
            scoring.manual_seed(args.seed + 1)  # the same points every call
            rmse, worst = corpus.score(model, "val", generator=scoring)
            held, held_worst = corpus.score(model, "holdout", generator=scoring)
            writer.add_scalar("val/rmse", rmse, iteration)
            writer.add_scalar("val/max_error", worst, iteration)
            writer.add_scalar("holdout/rmse", held, iteration)
            writer.add_scalar("holdout/max_error", held_worst, iteration)

            note = ""
            if model.gate_offset is not None:
                offset = float(model.gate_offset.detach())
                writer.add_scalar("gate/offset", offset, iteration)
                note = f" p={offset:+7.4f}"

            progress.write(
                f"[{iteration:6d}] data={loss.detach().item():.4e} |grad|={grad_norm:8.2e} "
                f"| val={rmse:7.3f}K holdout={held:7.3f}K (max {held_worst:7.1f}K)"
                f"{note}{f'  skipped={skipped}' if skipped else ''}"
            )
            # Selection is on the in-distribution split, never on the holdout:
            # the held-out power is the number being reported, and choosing a
            # checkpoint by it would report the best of N draws as if it were one.
            best.update(
                rmse,
                {
                    "model": model.state_dict(),
                    "architecture": architecture,
                    "bounds": domain.bounds.detach().cpu(),
                    "sensors": corpus.sensors.tolist(),
                    "field_shape": (corpus.shape[2], corpus.shape[1], corpus.shape[0]),
                    "val_rmse": rmse,
                    "holdout_rmse": held,
                    "holdout_power": args.holdout,
                    **({"radius": args.radius} if patching else {}),
                    **({"times": corpus.times.tolist()} if sequence else {}),
                    **({"retouched": True} if retouched else {}),
                    # What a ceded base gave up. Without it nothing can check
                    # that the patch it is paired with covers the same square,
                    # and a mismatch leaves either a ring neither model learned
                    # or capacity spent on an answer that is discarded.
                    **({"cede_radius": args.cede_radius}
                       if (sequence and not patching and args.cede_radius > 0) else {}),
                },
                step=iteration,
            )

    progress.close()
    scoring.manual_seed(args.seed + 1)
    rmse, _ = corpus.score(model, "val", generator=scoring)
    held, held_worst = corpus.score(model, "holdout", generator=scoring)
    writer.add_hparams(
        {
            "gate": GATES[model_name],
            "optimizer": args.optimizer,
            "lr": learning_rate,
            "iterations": args.iterations,
            "hidden": str(tuple(args.hidden)),
            "latent_dim": args.latent_dim,
            "sensors": args.sensors,
            "batch": batch_size,
        },
        {"hparam/val_rmse": best.best, "hparam/holdout_rmse": held},
    )
    writer.close()
    if model.gate_offset is not None:
        print(f"[done] learned gate offset p = {float(model.gate_offset.detach()):+.4f}")
    print(f"[done] final val {rmse:.3f} K  holdout {held:.3f} K (max {held_worst:.1f} K)")
    print(f"[done] best val RMSE {best.best:.3f} K -> {checkpoint_path}")
    print(f"[done] tensorboard --logdir {args.logdir}")
