"""Compare the toolpath-conditioned DeepONets against the scan they were fitted to.

``visualize.py`` cannot plot ``don``, ``gdon`` or ``jdon`` honestly. Those models
answer a question with two parts in it -- *which scan* and *which point* -- and its
``build_agent(model, checkpoint, device=...)`` has room for only the second, so the
toolpath falls back to a default while the field being compared against is chosen
separately by ``--data-dir``. Nothing makes the two agree. Pair them wrong and the
figure still renders, the RMSE still prints, and it reads 182 K where the truth is
1.5 K: a hundredfold error that looks like a result.

Here the pairing is not the caller's to get right. ``--pattern spiral`` names one
directory under ``data/toolpath``, and both the field and the toolpath come out of
it -- ``data_<P>W.npy`` and ``toolpath.json``, written by the same solver run. They
cannot disagree, because there is no second place to take either one from.

What is drawn, per requested time: the solver's field, each model's prediction of
it, and each signed difference. Truth and predictions share a colour scale so they
can be read against each other; the errors use a symmetric diverging scale about
zero, so blue is under-prediction and red is over. The beam is marked where it was
at that instant, with the lit part of its trail behind it -- on these paths "where
the beam is" is the whole question, and a panel without it cannot be read.

The slice conventions, the colour rules and the metric line are ``visualize.py``'s,
imported rather than copied so the two figures stay the same figure.

Examples::

    python visualize_don.py                              # 3 models x 4 patterns at 175 W
    python visualize_don.py --pattern spiral --model jdon
    python visualize_don.py --pattern raster --power 250 --times 2 6 10
    python visualize_don.py --pattern serpentine --plane depth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from dataset import MM, Grid, find_grid
from models import build_agent
from models._pathdon import (
    DEFAULT_TOOLPATH_DIR, GATES, PATCH_MODELS, RETOUCHED_MODELS, SEQUENCE_MODELS
)
import models.pkdon
import models.pkfdon
import models.rpkfdon
from utils import load_checkpoint, resolve_device, resolve_figure_path
from visualize import predict, slice_coords

sys.path.insert(0, str(DEFAULT_TOOLPATH_DIR.parent.parent / "rne-kaist" / "simulation"))
from toolpath import Toolpath  # noqa: E402

# The two families a figure can be drawn from. `--model` may mix them, but the
# default is the pointwise trio -- drawing all six puts thirteen columns in a row.
POINTWISE = [m for m in GATES if m not in PATCH_MODELS and m not in SEQUENCE_MODELS]
SEQUENCE = [m for m in GATES if m in SEQUENCE_MODELS and m not in PATCH_MODELS]
MODELS = POINTWISE + SEQUENCE
INK = "#52514e"


def run_dir(pattern: str) -> Path:
    """The solver run ``pattern`` names, checked before anything is loaded from it."""
    directory = DEFAULT_TOOLPATH_DIR / pattern
    if not (directory / "toolpath.json").exists():
        available = sorted(
            p.name for p in DEFAULT_TOOLPATH_DIR.iterdir() if (p / "toolpath.json").exists()
        )
        raise SystemExit(f"no run {pattern!r} under {DEFAULT_TOOLPATH_DIR}; have {available}")
    return directory


def latest_checkpoint(model: str, tag: str | None = None) -> Path:
    """The newest checkpoint for ``model``, restricted to ``tag`` when one is given.

    Newest by modification time, not by name. Once two network sizes are trained
    the names sort the wrong way -- ``don-256x6-...`` orders *before*
    ``don-64x3-...`` because ``2`` precedes ``6`` -- so taking the last name would
    quietly hand a figure the other size's weights and label it with this one's.

    ``--tag`` narrows first, so ``--tag 256x6`` draws 256x6 checkpoints and
    nothing else. That is the whole reason the tag is worth having: a figure and
    the weights behind it should not be able to disagree about which run they
    came from.
    """
    directory = Path("checkpoints", model)
    # The tag must sit immediately before the timestamp, or `64x3` also matches
    # `gfdon-64x3-ceded-...` -- a different experiment that happens to share a
    # prefix, and the newer of the two at that, so it would silently win.
    found = sorted(directory.glob(f"*-{tag}-[0-9]*.pt" if tag else "*.pt"))
    if not found:
        wanted = f" tagged {tag!r}" if tag else ""
        have = sorted(p.name for p in directory.glob("*.pt"))
        raise SystemExit(
            f"no checkpoint{wanted} under {directory}; have {have or 'nothing'}"
        )
    return max(found, key=lambda p: p.stat().st_mtime)


def default_times(grid: Grid, count: int) -> list[float]:
    """``count`` stored snapshot times spread over the run, ``t = 0`` excluded.

    Snapped to times the solver actually stored, so ``slice_coords`` compares
    against a real snapshot rather than the nearest one to a number nobody chose.
    The straight-pass default of 0.5/1.5/2.5 s is meaningless on a 9-11 s scan.

    The first snapshot is dropped because it is the initial condition: the plate
    is uniformly at ambient, every model reproduces it to within a few Kelvin,
    and a panel of one flat colour spends a row of the figure saying so. Ask for
    it with ``--times 0`` if you want it -- :func:`draw` keeps it readable.
    """
    picks = np.linspace(1, grid.t.size - 1, count).round().astype(int)
    return [float(grid.t[i]) for i in dict.fromkeys(picks)]


def mark_patch(axis, corners: np.ndarray, colour: str) -> None:
    """Outline the square ``pkdon`` overwrote, in millimetres.

    ``visualize.py`` draws this as an axis-aligned rectangle because its patch
    window is one. This window rides in the beam's travel frame and turns with
    it, so it is a polygon: on a corner of a spiral it sits at 45 degrees to the
    plate, and a bounding box would claim ground the paste never touched.
    """
    closed = np.vstack([corners, corners[:1]]) / MM
    axis.plot(closed[:, 0], closed[:, 1], color=colour, linewidth=1.1,
              linestyle=(0, (4, 2)))


def mark_beam(axis, path: Toolpath, time: float, colour: str) -> None:
    """Ring the beam and reveal the lit trail up to ``time``, in millimetres.

    Dark stretches are held as NaN so the trail breaks over them instead of
    drawing a line the laser never wrote -- the same treatment
    ``simulation/animate_field.py`` gives them, for the same reason.
    """
    steps = max(2, int(round(path.duration / 0.01)) + 1)
    sample = np.linspace(0.0, path.duration, steps)
    x, y, lit = path.position(sample)
    x, y = np.where(lit, x, np.nan), np.where(lit, y, np.nan)

    axis.plot(x, y, color=colour, alpha=0.18, linewidth=0.9)
    shown = sample <= min(time, path.duration)
    axis.plot(x[shown], y[shown], color=colour, alpha=0.55, linewidth=1.2)

    bx, by, firing = path.position(time)
    over = time > path.duration
    axis.plot([float(bx)], [float(by)], "o", mfc="none", ms=11, mew=1.6,
              mec=colour, alpha=1.0 if (firing and not over) else 0.5)


def draw(
    grid: Grid,
    path: Toolpath,
    agents: dict,
    times: list[float],
    plane: str,
    cut_y: float,
    pattern: str,
    provenance: str = "",
    patch_models: set[str] | None = None,
) -> Figure:
    """Rows are times; columns are the truth, then a prediction and error per model.

    ``provenance`` names the checkpoints behind the panels and is drawn along the
    bottom. A figure that cannot say which weights made it has to be identified by
    its filename, and a filename is a string somebody typed: this session found
    two sets of figures and one checkpoint whose names disagreed with what they
    actually were. ``visualize.py`` at least stamps its filenames with the
    checkpoint's own timestamp; that cannot work here, where a figure holds three
    models and a patch.
    """
    _, _, probe, _ = slice_coords(grid, plane, times[0], cut_y)
    span_x, span_y = probe[1] - probe[0], probe[3] - probe[2]

    columns = 1 + 2 * len(agents)
    panel = 3.4
    figure, axes = plt.subplots(
        len(times), columns,
        figsize=(panel * columns + 1.6, (panel * span_y / span_x + 1.1) * len(times)),
        squeeze=False, constrained_layout=True,
    )
    corrected = any(hasattr(a, "patch_box") for a in agents.values())
    # Named, not assumed: the sequence family is pasted by `pkfdon` and the
    # pointwise one by `pkdon`, and a caption that says the wrong one is worse
    # than no caption.
    pasted_by = ", ".join(sorted(patch_models or {"the patch"}))
    # The outline is whatever shape the patch owns, so the caption reads it off
    # the agents rather than naming a square the window may no longer be.
    outline = "square"
    for agent in agents.values():
        window = getattr(getattr(agent, "patch", None), "window", None)
        if window is not None:
            outline = "circle" if window == "circle" else "square"
    figure.suptitle(
        f"{pattern}   P = {grid.power:.0f} W   plane = {plane}   "
        f"({path.duration:.2f} s scan, {100 * path.lit_time / path.duration:.0f}% lit)"
        + (f"   (dashed {outline}: replaced by {pasted_by})" if corrected else ""),
        fontsize=13, color=INK,
    )
    if provenance:
        # supxlabel rather than a second suptitle line: constrained_layout reserves
        # room for it, so it can never grow into the axes.
        figure.supxlabel(provenance, fontsize=7.5, color=INK, family="monospace")

    for row, time in enumerate(times):
        coords, truth, extent, labels = slice_coords(grid, plane, time, cut_y)
        predictions, errors = {}, {}
        for name, agent in agents.items():
            value = predict(agent, coords, grid.power).reshape(truth.shape)
            predictions[name] = value
            errors[name] = value - truth

        print(f"  t = {time:5.2f}s", end="")
        for name in agents:
            rmse = float(np.sqrt((errors[name] ** 2).mean()))
            print(f"   {name}: RMSE {rmse:7.3f} K  max {np.abs(errors[name]).max():7.1f} K",
                  end="")
        print()

        low, high = float(truth.min()), float(truth.max())
        # A slice can be flat -- t = 0 is the initial condition, ambient
        # everywhere -- and `vmin == vmax` saturates every panel: the truth goes
        # to one end of the map and any prediction a millikelvin below it to the
        # other, so four panels that agree to 3 K are drawn as maximally
        # different. Widen the scale to whatever the predictions actually span.
        if high - low < 1.0:
            spread = max(
                max(float(np.abs(p - low).max()) for p in predictions.values()), 0.5
            )
            low, high = low - spread, high + spread

        # One error scale across the models in a row, or a model that is twice as
        # wrong as another looks exactly as wrong -- which is the comparison.
        bound = max(max(float(np.abs(e).max()) for e in errors.values()), 1e-9)

        style = dict(origin="lower", extent=extent, aspect="equal")
        field = dict(vmin=low, vmax=high, cmap="inferno")

        truth_image = axes[row][0].imshow(truth.T, **style, **field)
        axes[row][0].set_title(f"simulation\nt = {time:.2f} s", fontsize=10)

        for i, name in enumerate(agents):
            axes[row][1 + i].imshow(predictions[name].T, **style, **field)
            rmse = float(np.sqrt((errors[name] ** 2).mean()))
            axes[row][1 + i].set_title(f"{name}\nRMSE {rmse:.2f} K", fontsize=10)

            error_image = axes[row][1 + len(agents) + i].imshow(
                errors[name].T, **style, vmin=-bound, vmax=bound, cmap="RdBu_r"
            )
            axes[row][1 + len(agents) + i].set_title(
                f"{name} - simulation\nmax |error| {np.abs(errors[name]).max():.0f} K",
                fontsize=10,
            )

        if plane == "top":
            for column in range(1 + len(agents)):
                mark_beam(axes[row][column], path, time, "white")
            for column in range(1 + len(agents), columns):
                mark_beam(axes[row][column], path, time, "#111111")

            # The window is drawn per row, because it moves and turns with the
            # beam. Colours differ by block for the same reason `visualize.py`'s
            # do: `inferno` is nearly black out there and `RdBu_r` is nearly white.
            for name, agent in agents.items():
                corners = getattr(agent, "patch_box", lambda _t: None)(time)
                if corners is None:
                    continue
                column = 1 + list(agents).index(name)
                mark_patch(axes[row][column], corners, "white")
                mark_patch(axes[row][column + len(agents)], corners, "#111111")

        for column in range(columns):
            axes[row][column].set_xlabel(labels[0], fontsize=9)
            axes[row][column].tick_params(labelsize=8, colors=INK)
        axes[row][0].set_ylabel(labels[1], fontsize=9)

        figure.colorbar(truth_image, ax=axes[row][: 1 + len(agents)].tolist(),
                        label="K", shrink=0.85)
        figure.colorbar(error_image, ax=axes[row][1 + len(agents):].tolist(),
                        label="K", shrink=0.85)

    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", nargs="+", choices=MODELS, default=POINTWISE,
                        help=f"which models to draw. Default is the pointwise trio "
                             f"{POINTWISE}; the sequence family is "
                             f"{SEQUENCE} and the two may be mixed")
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=None,
                        help="one per --model, in the same order; defaults to the "
                             "most recent under checkpoints/<model>")
    parser.add_argument("--pattern", nargs="+", default=None,
                        help="run directories under data/toolpath (default: all). "
                             "One figure each; the field and the toolpath both come "
                             "from the directory named here")
    parser.add_argument("--power", type=float, default=175.0,
                        help="which simulation to compare against (default: 175, "
                             "the power held out of training)")
    parser.add_argument("--plane", choices=("top", "depth"), default="top",
                        help="top: the z = z_max face the laser scans. depth: an "
                             "(x, z) cut at --cut-y")
    parser.add_argument("--times", type=float, nargs="+", default=None,
                        help="seconds; defaults to --frames times spread over the run")
    parser.add_argument("--frames", type=int, default=3,
                        help="how many times to draw when --times is omitted")
    parser.add_argument("--cut-y", type=float, default=None,
                        help="y in mm for --plane depth (default: the plate's middle)")
    parser.add_argument("--tag", type=str, default=None,
                        help="label folded into the figure name, e.g. the network "
                             "size the checkpoints were trained at. A figure that "
                             "does not say which run it came from is a figure that "
                             "has to be told apart by its timestamp")
    parser.add_argument(
        "--pkcorrect", nargs="?", const="", default=None, metavar="CHECKPOINT",
        help="paste the pkdon patch over each model's melt pool; takes a pkdon "
             "checkpoint, or nothing for the most recent under checkpoints/pkdon. "
             "The patch is pointed at this pattern's own toolpath, so base and "
             "patch cannot end up describing different scans",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="save path; a bare filename or omitting this lands "
                             "under figures/. Ignored when several patterns are drawn")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    patterns = args.pattern or sorted(
        p.name for p in DEFAULT_TOOLPATH_DIR.iterdir() if (p / "toolpath.json").exists()
    )
    checkpoints = args.checkpoint or [latest_checkpoint(m, args.tag) for m in args.model]
    if len(checkpoints) != len(args.model):
        raise SystemExit(
            f"{len(args.model)} models but {len(checkpoints)} checkpoints; "
            f"give one per model, in the same order"
        )

    # `visualize.py` maps 'track' to a y chosen for the straight pass. These paths
    # have no single track, so the cut is the plate's middle unless asked otherwise.
    plane = "top" if args.plane == "top" else "track"

    for pattern in patterns:
        directory = run_dir(pattern)
        grid = find_grid(directory, args.power)
        path = Toolpath.load(directory / "toolpath.json")
        shape = grid.field_shape  # (nz, ny, nx), this run's own resolution

        agents, sources, patches, patch_models = {}, [], set(), set()
        for model, checkpoint in zip(args.model, checkpoints):
            # The toolpath is passed explicitly and comes from `directory`, the
            # same place `grid` did. That is the entire point of this script.
            agents[model] = build_agent(
                model, checkpoint, toolpath=directory / "toolpath.json",
                shape=shape, device=device,
            )
            sources.append(f"{model}: {Path(checkpoint).stem}")
            payload = load_checkpoint(checkpoint)
            print(f"[load] {checkpoint.name}: step {payload['step']}, "
                  f"val RMSE {payload['val_rmse']:.3f} K, "
                  f"holdout {payload.get('holdout_rmse', float('nan')):.3f} K")

            if args.pkcorrect is not None:
                # Resolved by the same tag as the bases: a 64x3 patch pasted over
                # 256x6 predictions would be a comparison of nothing in particular,
                # and `peak_corrected`'s own default is the newest file on disk,
                # which after training a second size is the wrong one.
                # Each family has its own patch, and they are not
                # interchangeable: `pkdon` answers one instant at a time,
                # `pkfdon` answers a history, and both are only meaningful
                # pasted over a base that speaks the same way.
                # Three patches now, and they are not interchangeable: `pkdon`
                # answers one instant at a time, `pkfdon` answers a history, and
                # `rpkfdon` answers a history in a frame that stops for dark
                # travel. Each is only meaningful over a base that speaks the
                # same way, and `GenericPeakCorrectedAgent` refuses the rest.
                if model in RETOUCHED_MODELS:
                    family, paste = "rpkfdon", models.rpkfdon.peak_corrected
                elif model in SEQUENCE_MODELS:
                    family, paste = "pkfdon", models.pkfdon.peak_corrected
                else:
                    family, paste = "pkdon", models.pkdon.peak_corrected
                patch = (
                    Path(args.pkcorrect) if args.pkcorrect
                    else latest_checkpoint(family, args.tag)
                )
                agents[model] = paste(agents[model], patch, device=device)
                patches.add(Path(patch).stem)
                patch_models.add(family)

        cut_y = args.cut_y * MM if args.cut_y is not None else float(grid.y.mean())
        times = args.times or default_times(grid, args.frames)
        print(f"[load] {pattern}/{grid.name}: {grid.temperature.shape} at "
              f"P = {grid.power} W, scan {path.duration:.2f} s")

        # The patches go last and once each: two bases of the same family share
        # one, and listing it twice says something untrue about the figure.
        figure = draw(grid, path, agents, times, plane, cut_y, pattern,
                      provenance="   ".join(
                          sources + [f"patch: {s}" for s in sorted(patches)]),
                      patch_models=patch_models)
        if args.pkcorrect is not None:
            if all(m in RETOUCHED_MODELS for m in args.model):
                family = "rpkfdon"
            elif all(m in SEQUENCE_MODELS and m not in RETOUCHED_MODELS
                     for m in args.model):
                family = "pkfdon"
            elif not any(m in SEQUENCE_MODELS for m in args.model):
                family = "pkdon"
            else:
                family = "pk-mixed"
        elif list(args.model) == POINTWISE:
            family = "don"
        else:
            family = "+".join(args.model)
        parts = [family, pattern, f"P{args.power:g}", args.plane]
        if args.tag:
            parts.append(args.tag)
        out = resolve_figure_path(
            args.out if len(patterns) == 1 else None, *parts
        )
        figure.savefig(out, dpi=150)
        plt.close(figure)
        print(f"[save] {out}\n")


if __name__ == "__main__":
    main()
