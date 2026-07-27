"""Visualise the weight matrices of one or more trained models on a shared scale.

Every dense layer of a network is a 2-D weight matrix; this draws each of them as
a heat-map and lays the models out as a grid -- one row per model, one column per
layer -- so the weights of different models can be read against each other.

The colour scale is diverging and centred on zero:

* a large **positive** weight is **green**,
* a large-magnitude **negative** weight is **red**,
* **zero** is **black**.

Two normalisations are offered. ``--normalize depth`` (the default) rescales each
*column* -- each network depth -- by its own largest absolute weight, so a layer
whose weights are all tiny still fills its cells rather than washing out to black
under a deeper layer's larger numbers; brightness is comparable between models at
the same depth, but not across depths, and each column's ``+/-`` scale is printed
in its cells. ``--normalize global`` instead shares one scale over every matrix of
every model, so a green of a given brightness means the same number everywhere in
the figure -- comparable across depths, at the cost of crushing the small deep
layers.

Only the ``.weight`` matrices of the ``nn.Linear`` layers are drawn; biases,
normalisation buffers and scalar parameters (the gate offset, the operator bias)
are left out, since the request is the per-layer weight *matrix*. DeepONet models
have no single stack, so their branch and trunk layers both appear, labelled
``B0, B2, ...`` and ``T0, T2, ...``.

The model is read straight from its checkpoint's ``state_dict``; the network is
never instantiated, so this works for any architecture without knowing anything
about it.

Examples::

    python modelview.py --models mlp gmlp pimlp cmlp cgmlp cpimlp
    python modelview.py --models mlp gmlp --pick best
    python modelview.py --checkpoint checkpoints/mlp/mlp-64x3-20260713-113936.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from models import available_models
from utils import DEFAULT_CHECKPOINT_DIR, DEFAULT_FIGURES_DIR, load_checkpoint

# Diverging map: red for negative, black at zero, green for positive. This is the
# colour scheme the request asks for; red/green is not colour-blind safe, but the
# sign is also given by which side of black the value sits, so black is the cue.
WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "red_black_green",
    [(0.85, 0.12, 0.12), (0.0, 0.0, 0.0), (0.15, 0.78, 0.20)],
)


def short_label(key: str) -> str:
    """A compact name for a ``.weight`` state-dict key.

    ``network.2.weight`` -> ``2``; ``operator.branch_net.network.2.weight`` ->
    ``B2`` (trunk -> ``T2``). The index is the layer's position in its
    ``nn.Sequential``, so the even numbers are the linear layers and the odd ones
    the activations that sit between them.
    """
    stem = key[: -len(".weight")]
    stem = stem.replace("operator.", "").replace("network.", "")
    stem = stem.replace("branch_net.", "B").replace("trunk_net.", "T")
    return stem


def weight_matrices(checkpoint: Path) -> list[tuple[str, np.ndarray]]:
    """Every 2-D ``.weight`` matrix in a checkpoint, in definition order."""
    payload = load_checkpoint(checkpoint)
    matrices = []
    for key, tensor in payload["model"].items():
        if key.endswith(".weight") and tensor.dim() == 2:
            matrices.append((short_label(key), tensor.detach().cpu().float().numpy()))
    if not matrices:
        raise ValueError(f"{checkpoint}: no 2-D weight matrices found")
    return matrices


def resolve_checkpoint(model: str, pick: str) -> Path:
    """Choose one checkpoint for ``model`` under ``checkpoints/<model>/``.

    ``64x3`` prefers a run whose name carries that tag and falls back to the most
    recent; ``latest`` is the most recently written; ``best`` is the lowest stored
    ``val_rmse``. The choice is printed so the figure is never a mystery.
    """
    directory = DEFAULT_CHECKPOINT_DIR / model
    candidates = sorted(directory.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints under {directory}")

    if pick == "64x3":
        tagged = [p for p in candidates if "64x3" in p.stem]
        return max(tagged or candidates, key=lambda p: p.stat().st_mtime)
    if pick == "latest":
        return max(candidates, key=lambda p: p.stat().st_mtime)
    if pick == "best":
        best, best_rmse = candidates[0], math.inf
        for path in candidates:
            rmse = load_checkpoint(path).get("val_rmse", math.inf)
            if rmse < best_rmse:
                best, best_rmse = path, rmse
        return best
    raise ValueError(f"unknown pick policy {pick!r}")


def draw(
    rows: list[tuple[str, list[tuple[str, np.ndarray]]]],
    column_scales: list[float],
    per_depth: bool,
) -> Figure:
    """Grid of weight heat-maps: one row per model, one column per layer.

    ``column_scales[c]`` is the ``+/-`` value column ``c`` is normalised by. When
    ``per_depth`` every column has its own, so each matrix is divided by its
    column's scale and drawn on a fixed ``[-1, 1]``; otherwise every entry of
    ``column_scales`` is the same global value and the matrices are drawn in their
    own physical units.
    """
    n_rows = len(rows)
    n_cols = len(column_scales)

    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.5 + 1.5, n_rows * 2.4),
        squeeze=False,
    )

    image = None
    for r, (model_label, layers) in enumerate(rows):
        for c in range(n_cols):
            axis = axes[r][c]
            if c < len(layers):
                layer_label, matrix = layers[c]
                scale = column_scales[c]
                shape = f"{matrix.shape[0]}x{matrix.shape[1]}"
                # aspect='auto' so 64x5 and 1x64 fill their cells equally rather
                # than leaving a matrix collapsed to a sliver.
                if per_depth:
                    shown = matrix / scale if scale > 0.0 else matrix
                    image = axis.imshow(
                        shown, cmap=WEIGHT_CMAP, vmin=-1.0, vmax=1.0, aspect="auto"
                    )
                    axis.set_title(f"{layer_label}  {shape}\n+/-{scale:.2g}", fontsize=8)
                else:
                    image = axis.imshow(
                        matrix, cmap=WEIGHT_CMAP, vmin=-scale, vmax=scale, aspect="auto"
                    )
                    axis.set_title(f"{layer_label}  {shape}", fontsize=8)
                axis.set_xticks([])
                axis.set_yticks([])
            else:
                axis.axis("off")
        axes[r][0].set_ylabel(model_label, fontsize=12, labelpad=8)

    if per_depth:
        title = "layer weight matrices, per-depth scale (each column to its own max)"
        cbar_label = "weight / column max"
    else:
        title = f"layer weight matrices, shared scale +/-{column_scales[0]:.3g}"
        cbar_label = "weight"
    figure.suptitle(f"{title}  (green + / black 0 / red -)", fontsize=13)
    bar = figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02, fraction=0.03
    )
    bar.set_label(cbar_label)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=available_models(),
        help="model names; each resolved to a checkpoint by --pick",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        default=None,
        help="explicit checkpoint paths, overriding --models; the row label is the "
        "file stem and the model is read straight from the state dict",
    )
    parser.add_argument(
        "--pick",
        choices=("64x3", "latest", "best"),
        default="64x3",
        help="which checkpoint to take from each model's folder (default 64x3)",
    )
    parser.add_argument(
        "--normalize",
        choices=("depth", "global"),
        default="depth",
        help="depth: rescale each column (network depth) by its own max, so small "
        "deep layers stay visible; global: one shared scale over every matrix",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="save path; a bare filename or omitting this lands under figures/",
    )
    args = parser.parse_args()
    if not args.models and not args.checkpoint:
        parser.error("give --models and/or --checkpoint")
    return args


def main() -> None:
    args = parse_args()

    # (row label, checkpoint path) in the order they will be drawn.
    entries: list[tuple[str, Path]] = []
    for model in args.models or []:
        path = resolve_checkpoint(model, args.pick)
        entries.append((model, path))
    for path in args.checkpoint or []:
        entries.append((Path(path).stem, Path(path)))

    rows = []
    for label, path in entries:
        layers = weight_matrices(path)
        largest = max(float(np.abs(matrix).max()) for _, matrix in layers)
        print(f"[load] {label:8s} {path.name}  ({len(layers)} layers, max|w|={largest:.3f})")
        rows.append((label, layers))

    n_cols = max(len(layers) for _, layers in rows)
    per_depth = args.normalize == "depth"
    if per_depth:
        # Each column's scale is the largest |w| among the models that have a
        # layer at that depth.
        column_scales = [
            max(
                float(np.abs(layers[c][1]).max())
                for _, layers in rows
                if c < len(layers)
            )
            for c in range(n_cols)
        ]
        print(
            "[scale] per-depth normalisation: "
            + ", ".join(f"col{c}=+/-{s:.3g}" for c, s in enumerate(column_scales))
        )
    else:
        peak = max(float(np.abs(m).max()) for _, layers in rows for _, m in layers)
        column_scales = [peak] * n_cols
        print(f"[scale] shared normalisation: +/-{peak:.4f}")

    if max(column_scales) == 0.0:
        raise ValueError("all weights are zero; nothing to normalise against")

    figure = draw(rows, column_scales, per_depth)

    if args.out is None:
        name = "+".join(label for label, _ in entries) + "_params.png"
        out = DEFAULT_FIGURES_DIR / name
    elif args.out.parent == Path("."):
        out = DEFAULT_FIGURES_DIR / args.out
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
