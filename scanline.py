"""Plot the temperature profile along the scan line: T against x at fixed y and z.

Where ``visualize.py`` shows a whole slice as an image, this cuts a single line
out of it -- the one running down the middle of the laser track -- so the melt
pool's peak height and width can be read off directly rather than inferred from
a colour. One row per requested time: the profile on the left, the signed error
on the right.

Beneath them one panel scores the *whole* plane -- not just the line -- at every
time step the solver stored. Three cuts at three instants cannot say whether the
worst error sits off-axis, or lands between the times that happened to be
plotted; this does, and the vertical rules mark which instants the rows above
cut. The plane's peak temperature, solver and model, is overlaid on a second axis
so the error can be read against the quantity it is an error in.

The default ``--y 5.0`` is the scan line and the default depth is the top
surface, which is where the laser deposits its energy and where the peak lives.

The model is reached only through its agent's ``predict_at``, so the laser and
material constants are taken as command-line inputs rather than imported from
any one model's ``loss.py``.

Examples::

    python scanline.py --model gpidon --checkpoint best.pt --power 200
    python scanline.py --model gpidon --checkpoint best.pt --power 250 --gaussian --z 5.5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from agent import BaseAgent, PeakCorrectedAgent, peak_corrected
from dataset import MM, DEFAULT_DATA_DIR, Grid, find_grid
from models import available_models, build_agent
from utils import checkpoint_stamp, load_checkpoint, resolve_device, resolve_figure_path
from visualize import predict

# Categorical slots 1 and 8 of the reference palette; worst-case CVD dE 96.7.
# Line style carries the same distinction, so colour is never the only cue.
TRUTH_COLOUR = "#2a78d6"
PREDICTION_COLOUR = "#eb6834"
INK_SECONDARY = "#52514e"
# The cpkmlp window is an annotation, not a series, so it takes a neutral ink and
# a low alpha rather than a categorical slot of its own.
PATCH_COLOUR = "#6b6a66"

# Defaults matching the calibrated constants in models/gpidon/loss.py.
AMBIENT_TEMPERATURE = 298.0  # [K]
BEAM_RADIUS = 1.6971  # [mm], 1/e^2


@dataclass
class GaussianFit:
    """A ``T_amb + amplitude * exp(-2 (x - centre)^2 / width^2)`` fit of one profile.

    The ``1/e^2`` convention matches the fitted beam radius, so ``width`` is
    directly comparable to the beam that deposited the energy.
    """

    amplitude: float  # [K] above ambient
    centre: float  # [m]
    width: float  # [m], 1/e^2 radius
    r_squared: float
    ambient: float  # [K]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        exponent = -2.0 * (x - self.centre) ** 2 / self.width**2
        return self.ambient + self.amplitude * np.exp(exponent)


def fit_gaussian(
    x: np.ndarray,
    temperature: np.ndarray,
    window: float,
    ambient: float,
    beam_radius: float,
) -> GaussianFit | None:
    """Least-squares fit around the profile's own peak, or None if it does not converge.

    Only points within ``window`` of the peak are used. The trailing thermal wake
    is not Gaussian, so fitting the whole line would let the tail set the width;
    restricting to a symmetric window measures the peak itself. Truth and
    prediction get the same treatment, so the comparison between them is fair.
    """
    from scipy.optimize import curve_fit

    def model(coord: np.ndarray, amplitude: float, centre: float, width: float) -> np.ndarray:
        return ambient + amplitude * np.exp(-2.0 * (coord - centre) ** 2 / width**2)

    peak = int(np.argmax(temperature))
    selected = np.abs(x - x[peak]) <= window
    if selected.sum() < 4:  # three parameters need at least four points
        return None

    guess = (temperature[peak] - ambient, x[peak], beam_radius)
    try:
        parameters, _ = curve_fit(
            model, x[selected], temperature[selected], p0=guess, maxfev=10000
        )
    except RuntimeError:
        return None

    residual = temperature[selected] - model(x[selected], *parameters)
    total = temperature[selected] - temperature[selected].mean()
    r_squared = 1.0 - float((residual**2).sum() / (total**2).sum())
    amplitude, centre, width = parameters
    return GaussianFit(float(amplitude), float(centre), abs(float(width)), r_squared, ambient)


def mark_patch(axis, span: tuple[float, float], unit: float, label: str | None):
    """Shade ``span`` -- the stretch of the axis a ``cpkmlp`` patch was pasted over.

    Kept deliberately faint. The panels it goes behind carry two curves and a
    filled error, and the point of the band is to say *where* the two models
    change hands, not to compete with what they say there. Returns the shading,
    so a caller assembling its own legend can put it in.
    """
    low, high = span[0] / unit, span[1] / unit
    shading = axis.axvspan(
        low, high, color=PATCH_COLOUR, alpha=0.10, linewidth=0, label=label
    )
    for edge in (low, high):
        axis.axvline(
            edge, color=PATCH_COLOUR, linewidth=0.9, linestyle=(0, (4, 2)), alpha=0.7
        )
    return shading


def layer_index(grid: Grid, depth: float | None) -> int:
    """The ``z`` index the profiles are cut at; the top surface when ``depth`` is None."""
    return len(grid.z) - 1 if depth is None else int(np.argmin(np.abs(grid.z - depth)))


def surface_error_history(
    grid: Grid, agent: BaseAgent, depth: float | None
) -> tuple[np.ndarray, ...]:
    """``(t, max_abs_error, rmse, peak_truth, peak_prediction)`` over the plane at ``depth``.

    The rows above cut one line out of that plane at three chosen instants; this
    scores the entire plane at *every* instant the solver stored. The worst error
    on the surface need not sit on the scan line -- it can appear off-axis or at a
    time between the sampled ones -- and a per-row maximum cannot show either.

    The two peak temperatures ride along because the error alone cannot say what
    it is an error *in*: a 150 K miss on a 1900 K melt pool and the same miss on a
    400 K tail are different failures, and the peaks put the error next to the
    quantity it is measured against.
    """
    layer = layer_index(grid, depth)
    mesh_x, mesh_y = np.meshgrid(grid.x, grid.y, indexing="ij")
    flat_x, flat_y = mesh_x.reshape(-1), mesh_y.reshape(-1)
    height = np.full(flat_x.shape, grid.z[layer])

    worst, rmse, peak_truth, peak_prediction = [], [], [], []
    for index, time in enumerate(grid.t):
        coords = np.stack(
            [flat_x, flat_y, height, np.full(flat_x.shape, time)], axis=-1
        )
        # stored as [t, z, y, x]; the meshgrid above is (x, y), hence the transpose
        truth = grid.temperature[index, layer].T.reshape(-1)
        prediction = predict(agent, coords, grid.power)
        error = prediction - truth
        worst.append(float(np.abs(error).max()))
        rmse.append(float(np.sqrt((error**2).mean())))
        peak_truth.append(float(truth.max()))
        peak_prediction.append(float(prediction.max()))
    return (
        grid.t,
        np.array(worst),
        np.array(rmse),
        np.array(peak_truth),
        np.array(peak_prediction),
    )


def limits_excluding_initial(
    times: np.ndarray, series: tuple[np.ndarray, ...], margin: float = 0.05
) -> tuple[float, float] | None:
    """``(low, high)`` covering ``series`` over ``t > 0``, or None if nothing is left.

    At ``t = 0`` the plane is uniformly at ambient and the model has had nothing
    to fit yet, so that one step can miss by more than the whole run after it.
    Letting it set the scale flattens the rest of the history into a line. The
    curves are still drawn from ``t = 0``; only the range ignores it.
    """
    keep = times > 0.0
    if not keep.any():
        return None
    values = np.concatenate([np.asarray(one)[keep] for one in series])
    low, high = float(values.min()), float(values.max())
    pad = margin * (high - low) if high > low else margin * max(abs(high), 1.0)
    return low - pad, high + pad


def line_coords(
    grid: Grid, time: float, track_y: float, depth: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return ``(x[nx], coords[nx,4], truth[nx], actual)`` along the scan line."""
    time_index = int(np.argmin(np.abs(grid.t - time)))
    row = int(np.argmin(np.abs(grid.y - track_y)))
    layer = layer_index(grid, depth)

    coords = np.stack(
        [
            grid.x,
            np.full(grid.x.shape, grid.y[row]),
            np.full(grid.x.shape, grid.z[layer]),
            np.full(grid.x.shape, grid.t[time_index]),
        ],
        axis=-1,
    )
    actual = {
        "t": float(grid.t[time_index]),
        "y": float(grid.y[row]),
        "z": float(grid.z[layer]),
    }
    return grid.x, coords, grid.temperature[time_index, layer, row, :], actual


def draw(
    grid: Grid,
    agent: BaseAgent,
    model_name: str,
    times: list[float],
    track_y: float,
    depth: float | None,
    gaussian_window: float | None = None,
    ambient: float = AMBIENT_TEMPERATURE,
    beam_radius: float = BEAM_RADIUS * MM,
) -> Figure:
    """Render one profile/error row per time, plus the surface-error history below."""
    # GridSpec rather than plt.subplots: the history row spans both columns and is
    # plotted against time, so it cannot share the x axis the rows above use.
    figure = plt.figure(figsize=(12, 3.1 * len(times) + 2.7), constrained_layout=True)
    spec = figure.add_gridspec(len(times) + 1, 2)
    axes = np.empty((len(times), 2), dtype=object)
    for row in range(len(times)):
        for column in range(2):
            axes[row, column] = figure.add_subplot(
                spec[row, column],
                sharex=None if row == 0 and column == 0 else axes[0, 0],
            )
            if row < len(times) - 1:
                axes[row, column].tick_params(labelbottom=False)
    history_axis = figure.add_subplot(spec[-1, :])

    # The cut is the same for every row, so name it once from the first probe.
    *_, placement = line_coords(grid, times[0], track_y, depth)

    for row, time in enumerate(times):
        x, coords, truth, actual = line_coords(grid, time, track_y, depth)
        prediction = predict(agent, coords, grid.power)
        error = prediction - truth

        rmse = float(np.sqrt((error**2).mean()))
        worst = float(np.abs(error).max())
        peak_gap = float(prediction.max() - truth.max())
        print(
            f"  t = {actual['t']:4.2f}s  RMSE = {rmse:8.3f} K   max |error| = {worst:9.3f} K"
            f"   peak {truth.max():7.1f} -> {prediction.max():7.1f} K ({peak_gap:+.1f})"
        )

        profile_axis, error_axis = axes[row]

        # The window travels with the beam, so it is marked per row -- and only
        # when it reaches the line this row cuts, which is a question about y and
        # z that the cut answers and the profile cannot.
        box = (
            agent.patch_box(actual["t"])
            if isinstance(agent, PeakCorrectedAgent)
            else None
        )
        if (
            box is not None
            and box[1][0] <= actual["y"] <= box[1][1]
            and box[2][0] <= actual["z"] <= box[2][1]
        ):
            mark_patch(profile_axis, box[0], MM, "cpkmlp" if row == 0 else None)
            mark_patch(error_axis, box[0], MM, None)

        profile_axis.plot(
            x / MM, truth, color=TRUTH_COLOUR, linewidth=1.8, label="simulation"
        )
        profile_axis.plot(
            x / MM,
            prediction,
            color=PREDICTION_COLOUR,
            linewidth=1.8,
            linestyle="--",
            label=model_name,
        )
        if gaussian_window is not None:
            dense = np.linspace(x[0], x[-1], 800)
            for series, colour, temperatures in (
                ("simulation", TRUTH_COLOUR, truth),
                (model_name, PREDICTION_COLOUR, prediction),
            ):
                fit = fit_gaussian(x, temperatures, gaussian_window, ambient, beam_radius)
                if fit is None:
                    print(f"    {series:14s} gaussian fit did not converge")
                    continue
                inside = np.abs(dense - fit.centre) <= gaussian_window
                profile_axis.plot(
                    dense[inside] / MM,
                    fit.evaluate(dense[inside]),
                    color=colour,
                    linewidth=1.0,
                    linestyle=":",
                    label=f"{series}, gaussian fit" if row == 0 else None,
                )
                print(
                    f"    {series:14s} gaussian: amplitude {fit.amplitude:7.1f} K"
                    f"   centre {fit.centre / MM:6.3f} mm"
                    f"   width {fit.width / MM:5.3f} mm"
                    f"   R^2 {fit.r_squared:.4f}"
                )

        profile_axis.set_ylabel("T [K]")
        profile_axis.set_title(
            f"t = {actual['t']:.2f} s   RMSE {rmse:.1f} K", fontsize=10
        )
        if row == 0:
            profile_axis.legend(frameon=False, fontsize=8)

        error_axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
        error_axis.plot(x / MM, error, color=INK_SECONDARY, linewidth=1.5)
        error_axis.fill_between(
            x / MM, error, 0.0, color=INK_SECONDARY, alpha=0.12, linewidth=0
        )
        error_axis.set_ylabel("prediction - simulation [K]")
        error_axis.set_title(f"max |error| {worst:.0f} K", fontsize=10)

        for axis in (profile_axis, error_axis):
            axis.grid(alpha=0.25, linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)

    for axis in axes[-1]:
        axis.set_xlabel("x [mm]")

    depth_label = (
        "top surface" if depth is None else f"z = {placement['z'] / MM:.2f} mm"
    )

    history = surface_error_history(grid, agent, depth)
    history_t, history_worst, history_rmse, peak_truth, peak_prediction = history
    print(f"  {depth_label}, every stored time step:")
    for values in zip(history_t, history_worst, history_rmse, peak_truth, peak_prediction):
        time, worst_value, rmse_value, truth_peak, model_peak = values
        print(
            f"    t = {time:4.2f}s  max |error| = {worst_value:9.3f} K"
            f"   RMSE = {rmse_value:8.3f} K"
            f"   peak {truth_peak:7.1f} -> {model_peak:7.1f} K ({model_peak - truth_peak:+.1f})"
        )

    # Two scales in one panel: the error is tens of K, the temperature is
    # hundreds. Errors keep the left axis, temperatures take the right, and the
    # legend is merged so the pairing stays readable.
    temperature_axis = history_axis.twinx()
    error_lines = [
        history_axis.plot(
            history_t, history_worst, color=PREDICTION_COLOUR, linewidth=1.8,
            marker="o", markersize=4, label="max |error|",
        )[0],
        history_axis.plot(
            history_t, history_rmse, color=INK_SECONDARY, linewidth=1.4,
            linestyle="--", marker="s", markersize=3, label="RMSE",
        )[0],
    ]
    # Solid is the solver and dashed the model, the same pairing the rows above
    # use, so line style carries it and colour is never the only cue.
    temperature_lines = [
        temperature_axis.plot(
            history_t, peak_truth, color=TRUTH_COLOUR, linewidth=1.5,
            label="peak T, simulation",
        )[0],
        temperature_axis.plot(
            history_t, peak_prediction, color=TRUTH_COLOUR, linewidth=1.5,
            linestyle="--", label=f"peak T, {model_name}",
        )[0],
    ]
    # Both scales are read off the history after t = 0; see the helper for why.
    error_limits = limits_excluding_initial(history_t, (history_worst, history_rmse))
    if error_limits is not None:
        history_axis.set_ylim(*error_limits)
    temperature_limits = limits_excluding_initial(
        history_t, (peak_truth, peak_prediction)
    )
    if temperature_limits is not None:
        temperature_axis.set_ylim(*temperature_limits)
    # Tie the curves back to the rows above: these are the instants they cut.
    for time in times:
        nearest = float(history_t[int(np.argmin(np.abs(history_t - time)))])
        history_axis.axvline(nearest, color=TRUTH_COLOUR, linewidth=0.9, alpha=0.45)
    history_axis.set_xlabel("t [s]")
    history_axis.set_ylabel("error over the plane [K]")
    temperature_axis.set_ylabel("peak T on the plane [K]")
    history_axis.set_title(
        f"{depth_label}: error and peak temperature over the whole plane at every "
        f"stored time (worst error {history_worst.max():.0f} K)",
        fontsize=10,
    )
    # Over time the window is a span rather than a box: the patch answers at every
    # instant of its own clock, and leaves the model alone outside it.
    patch_lines = []
    if isinstance(agent, PeakCorrectedAgent):
        clock = (float(agent.window[3, 0]), float(agent.window[3, 1]))
        patch_lines.append(mark_patch(history_axis, clock, 1.0, "cpkmlp applied"))

    # Below the axes rather than inside it: four curves on two scales leave no
    # corner reliably free, and which corner is free changes with the model.
    lines = error_lines + temperature_lines + patch_lines
    history_axis.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=len(lines),
        frameon=False,
        fontsize=8,
    )
    history_axis.grid(alpha=0.25, linewidth=0.6)
    history_axis.spines[["top"]].set_visible(False)
    temperature_axis.spines[["top"]].set_visible(False)
    figure.suptitle(
        f"P = {grid.power:.0f} W,  y = {placement['y'] / MM:.2f} mm ({depth_label})",
        fontsize=13,
    )
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
    parser.add_argument("--power", type=float, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 2.5])
    parser.add_argument("--y", type=float, default=5.0, help="scan-line y in mm")
    parser.add_argument(
        "--z", type=float, default=None, help="depth in mm; defaults to the top surface"
    )
    parser.add_argument(
        "--gaussian",
        action="store_true",
        help="overlay a 1/e^2 gaussian fitted to each peak",
    )
    parser.add_argument(
        "--beam-radius",
        type=float,
        default=BEAM_RADIUS,
        help="1/e^2 beam radius in mm, the initial guess of the gaussian fit",
    )
    parser.add_argument(
        "--ambient", type=float, default=AMBIENT_TEMPERATURE, help="T_amb in K"
    )
    parser.add_argument(
        "--fit-window",
        type=float,
        default=2.5 * BEAM_RADIUS,
        help="half-width in mm of the fit window around the peak",
    )
    parser.add_argument(
        "--pkcorrect",
        nargs="?",
        const="",
        default=None,
        metavar="CHECKPOINT",
        help="paste a cpkmlp patch over the peak of the predicted field; takes a "
        "cpkmlp checkpoint, or nothing for the most recent under checkpoints/cpkmlp/",
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

    if args.pkcorrect is not None:
        agent = peak_corrected(agent, args.pkcorrect or None, device=device)

    figure = draw(
        grid,
        agent,
        args.model,
        args.times,
        args.y * MM,
        None if args.z is None else args.z * MM,
        args.fit_window * MM if args.gaussian else None,
        ambient=args.ambient,
        beam_radius=args.beam_radius * MM,
    )

    out = resolve_figure_path(
        args.out,
        args.model,
        f"P{args.power:g}",
        "line",
        *(("pkcorrect",) if args.pkcorrect is not None else ()),
        stamp=checkpoint_stamp(args.checkpoint),
    )
    figure.savefig(out, dpi=140)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
