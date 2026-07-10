"""Compare a trained model against the simulation it was fitted to.

Draws one row per requested time, three panels wide: the finite-element field,
the network prediction, and their signed difference. Truth and prediction share
a colour scale so they can be read against each other; the error panel uses a
symmetric diverging scale centred on zero, so blue is under-prediction and red
is over-prediction.

Two slices are available. ``--plane top`` shows the ``z = z_max`` surface the
laser scans, over ``(x, y)``. ``--plane track`` cuts along the scan line at
``y = y_c`` and shows ``(x, z)``, which is where the melt pool depth lives and
where the model is hardest to fit.

The model is reached only through its agent's ``predict_at``, so any model under
``models/`` can be plotted without changing anything here.

Examples::

    python visualize.py --model gpidon --checkpoint checkpoints/gpidon/best.pt --power 200
    python visualize.py --model gpidon --checkpoint best.pt --power 250 --plane track
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from agent import BaseAgent
from dataset import MM, DEFAULT_DATA_DIR, Grid, find_grid
from models import available_models, build_agent
from utils import checkpoint_stamp, load_checkpoint, resolve_device, resolve_figure_path


def predict(agent: BaseAgent, coords: np.ndarray, power: float) -> np.ndarray:
    """Evaluate the agent on ``[N, 4]`` physical ``(x, y, z, t)``, returning ``[N]`` Kelvin."""
    inputs = np.concatenate(
        [coords, np.full((coords.shape[0], 1), power, dtype=coords.dtype)], axis=1
    )
    return agent.predict_at(inputs).squeeze(-1).cpu().numpy()


def slice_coords(
    grid: Grid, plane: str, time: float, track_y: float
) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
    """Return ``(coords[N,4], truth[a,b], extent_mm, axis_labels)`` for one slice.

    ``truth`` is transposed out of the stored ``[t, z, y, x]`` order into the
    ``(first, second)`` axis pair the panel is drawn over.
    """
    time_index = int(np.argmin(np.abs(grid.t - time)))
    actual_time = float(grid.t[time_index])

    if plane == "top":
        first, second = grid.x, grid.y
        a, b = np.meshgrid(first, second, indexing="ij")
        c = np.full(a.shape, grid.z[-1])
        truth = grid.temperature[time_index, -1, :, :].T  # [ny, nx] -> [nx, ny]
        coords = np.stack([a, b, c, np.full(a.shape, actual_time)], axis=-1)
        labels = ("x [mm]", "y [mm]")
    elif plane == "track":
        row = int(np.argmin(np.abs(grid.y - track_y)))
        first, second = grid.x, grid.z
        a, c = np.meshgrid(first, second, indexing="ij")
        b = np.full(a.shape, grid.y[row])
        truth = grid.temperature[time_index, :, row, :].T  # [nz, nx] -> [nx, nz]
        coords = np.stack([a, b, c, np.full(a.shape, actual_time)], axis=-1)
        labels = ("x [mm]", "z [mm]")
    else:
        raise ValueError(f"unknown plane {plane!r}; expected 'top' or 'track'")

    extent = (
        float(first[0]) / MM,
        float(first[-1]) / MM,
        float(second[0]) / MM,
        float(second[-1]) / MM,
    )
    return coords.reshape(-1, 4), truth, extent, labels


def draw(
    grid: Grid,
    agent: BaseAgent,
    model_name: str,
    times: list[float],
    plane: str,
    track_y: float,
) -> Figure:
    """Render the truth / prediction / error grid and print per-slice metrics."""
    # Panels use an equal aspect, so let the slice's own shape set the row height
    # instead of leaving a band of whitespace above and below each strip.
    _, _, probe_extent, _ = slice_coords(grid, plane, times[0], track_y)
    span_x = probe_extent[1] - probe_extent[0]
    span_y = probe_extent[3] - probe_extent[2]
    panel_width = 4.3
    row_height = panel_width * (span_y / span_x) + 1.5

    figure, axes = plt.subplots(
        len(times),
        3,
        figsize=(15, row_height * len(times)),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(f"P = {grid.power:.0f} W, plane = {plane}", fontsize=13)

    for row, time in enumerate(times):
        coords, truth, extent, labels = slice_coords(grid, plane, time, track_y)
        prediction = predict(agent, coords, grid.power).reshape(truth.shape)
        error = prediction - truth

        rmse = float(np.sqrt((error**2).mean()))
        worst = float(np.abs(error).max())
        print(
            f"  t = {time:4.2f}s  RMSE = {rmse:8.3f} K   max |error| = {worst:9.3f} K"
        )

        # Truth and prediction share limits; the error scale is symmetric about zero.
        low, high = float(truth.min()), float(truth.max())
        bound = max(float(np.abs(error).max()), 1e-9)

        style = dict(origin="lower", extent=extent, aspect="equal")
        field_style = dict(vmin=low, vmax=high, cmap="inferno")

        truth_image = axes[row][0].imshow(truth.T, **style, **field_style)
        axes[row][1].imshow(prediction.T, **style, **field_style)
        error_image = axes[row][2].imshow(
            error.T, **style, vmin=-bound, vmax=bound, cmap="RdBu_r"
        )

        axes[row][0].set_title(f"simulation\nt = {time:.2f} s", fontsize=10)
        axes[row][1].set_title(f"{model_name}\nRMSE {rmse:.1f} K", fontsize=10)
        axes[row][2].set_title(
            f"prediction - simulation\nmax |error| {worst:.0f} K", fontsize=10
        )

        for column in range(3):
            axes[row][column].set_xlabel(labels[0])
        axes[row][0].set_ylabel(labels[1])

        # Truth and prediction share a scale, so one bar serves both.
        figure.colorbar(truth_image, ax=axes[row][:2].tolist(), label="K", shrink=0.9)
        figure.colorbar(error_image, ax=axes[row][2], label="K", shrink=0.9)

    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default="gpidon", choices=available_models(), help="which model to load"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--power", type=float, required=True, help="which simulation to compare against"
    )
    parser.add_argument("--plane", choices=("top", "track"), default="top")
    parser.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 2.5])
    parser.add_argument(
        "--track-y",
        type=float,
        default=4.9922,
        help="scan-line y in mm, for --plane track",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="save path; a bare filename or omitting this lands under figures/",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    grid = find_grid(args.data_dir, args.power)
    agent = build_agent(args.model, args.checkpoint, device=device)
    checkpoint = load_checkpoint(args.checkpoint)
    print(
        f"[load] {args.checkpoint.name}: step {checkpoint['step']}, "
        f"val RMSE {checkpoint['val_rmse']:.3f} K"
    )
    print(f"[load] {grid.name}: {grid.temperature.shape} at P = {grid.power} W")

    figure = draw(grid, agent, args.model, args.times, args.plane, args.track_y * MM)

    out = resolve_figure_path(
        args.out,
        args.model,
        f"P{args.power:g}",
        args.plane,
        stamp=checkpoint_stamp(args.checkpoint),
    )
    figure.savefig(out, dpi=140)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
