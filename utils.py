"""Training utilities shared by every model under ``models/``.

Nothing here knows about a particular architecture: run naming, best-checkpoint
tracking, device resolution and parameter counting are the same whether the
network is an operator net or a convolutional surrogate.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "runs"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"


def resolve_device(name: str | None = None) -> torch.device:
    """``name``, or CUDA when it is available and ``name`` is None."""
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed the global RNG, the one a module's parameters are initialised from.

    Every training loop already builds its own :class:`torch.Generator` from
    ``--seed``, but that generator only covers the data -- the split and the
    batches drawn from it. Weight initialisation reads the global generator
    instead, so without this two runs at the same ``--seed`` would still start
    from different weights and never be comparable.
    """
    torch.manual_seed(seed)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

OPTIMIZERS = ("adam", "lbfgs")

# Adam wants a small step; L-BFGS with a strong-Wolfe line search wants a full
# one, and the line search shortens it when that overshoots. Feeding L-BFGS
# Adam's 1e-3 would cripple it, so the default depends on which one is chosen.
DEFAULT_LR = {"adam": 1e-3, "lbfgs": 1.0}


def add_optimizer_args(parser: Any) -> None:
    """Register ``--optimizer`` and its L-BFGS knobs on a model's parser.

    ``--lr`` defaults to ``None`` and is resolved by :func:`resolve_lr`, so that
    each optimizer gets the step size that suits it while an explicit ``--lr``
    still wins.
    """
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZERS,
        default="adam",
        help="adam (stochastic, resampled batches) or lbfgs (quasi-Newton, fixed batch)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"step size; defaults per optimizer: {DEFAULT_LR}",
    )
    parser.add_argument(
        "--lbfgs-batch",
        type=int,
        default=65536,
        help="rows in the ONE batch L-BFGS is fitted against (supervised-MSE models "
        "only; the PINN models freeze their own --batch-data/-physics/-boundary "
        "point sets instead)",
    )
    parser.add_argument(
        "--freeze-batch",
        action="store_true",
        help="draw the batch ONCE and reuse it, even with adam. L-BFGS always does "
        "this; the flag exists so the batch and the optimizer can be varied "
        "independently, which is the only way to tell which one an effect belongs to",
    )
    parser.add_argument("--lbfgs-history", type=int, default=100)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)


def resolve_lr(args: Any) -> float:
    """``args.lr`` if given, else the default for ``args.optimizer``."""
    if getattr(args, "lr", None) is not None:
        return float(args.lr)
    return DEFAULT_LR[getattr(args, "optimizer", "adam")]


def build_optimizer(parameters: Any, args: Any, lr: float) -> torch.optim.Optimizer:
    """Adam, or L-BFGS with a strong-Wolfe line search.

    L-BFGS is a quasi-Newton method: it approximates the curvature of the loss
    from the gradients of the *previous* steps. That approximation is only
    meaningful if every step differs from the last by the parameters alone --
    if the batch is resampled each step, consecutive gradients come from
    different functions and the curvature history becomes noise. Every training
    loop here therefore draws a single fixed batch of ``--lbfgs-batch`` rows and
    reuses it, so the objective L-BFGS descends is one fixed function.

    The price is that the fixed batch is the only data the model sees, so it can
    overfit to it; the validation split is untouched by the choice and remains
    the honest score.
    """
    if args.optimizer == "lbfgs":
        return torch.optim.LBFGS(
            parameters,
            lr=lr,
            history_size=args.lbfgs_history,
            max_iter=args.lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )
    return torch.optim.Adam(parameters, lr=lr)


# ---------------------------------------------------------------------------
# Gradient diagnostics
# ---------------------------------------------------------------------------


# every key :func:`_secant_metrics` can produce, by prefix
_SECANT_PREFIXES = ("curv_", "cos_s_y_", "cos_y_", "dot_y_", "ynorm_", "snorm")


def alignment_group(name: str) -> str:
    """The TensorBoard group a :func:`gradient_alignment` key belongs in.

    The first-order geometry stays under ``align/`` and everything the secant
    family adds goes under ``align_curvature/``, so a run with
    ``--align-curvature`` on presents them as two sections rather than one
    48-item list. The prefixes cannot collide with the first-order names: no loss
    term is called ``y``, so ``cos_y_``/``dot_y_`` are unambiguous, and
    ``gnorm_`` is distinct from ``ynorm_``.
    """
    return "align_curvature" if name.startswith(_SECANT_PREFIXES) else "align"


def _secant_metrics(
    gradients: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
    weights: Any,
    previous: dict[str, Any],
) -> dict[str, float]:
    """How each term's gradient *changed* since the last call, against the step taken.

    This is the second-order information L-BFGS itself runs on. It builds its
    curvature estimate from pairs ``(s, y)`` -- the parameter step ``s = θ_k -
    θ_{k-1}`` and the gradient change ``y = g_k - g_{k-1}`` -- and the update is
    only guaranteed positive definite while ``sᵀy > 0``. Splitting ``y`` by loss
    term says which term supplies that curvature and which one fights it.

    ``previous`` is the caller's state dict, read and then overwritten in place;
    the first call therefore returns nothing but the state for the second.

    ``curv_<t>`` is ``sᵀy_t``, the per-term curvature condition, and
    ``cos_s_y_<t>`` the same sign normalised into [-1, 1]. ``curv_total`` is the
    weighted sum -- the quantity L-BFGS actually tests. ``cos_y_<a>_<b>`` and
    ``dot_y_<a>_<b>`` are the pairwise geometry of the gradient *changes*, the
    second-order counterpart of the first-order cosines.

    Metrics whose denominator is zero are omitted rather than reported as 0.0: a
    step that moved nothing leaves the cosines genuinely undefined, and a gap in
    the series says that more honestly than a spurious "orthogonal".
    """
    current = nn.utils.parameters_to_vector(parameters).detach().clone()
    last_gradients = previous.get("gradients")
    last_parameters = previous.get("parameters")

    metrics: dict[str, float] = {}
    if last_gradients is not None and last_parameters is not None:
        step = current - last_parameters
        step_norm = float(step.norm())
        metrics["snorm"] = step_norm

        changes = {
            term: gradient - last_gradients[term]
            for term, gradient in gradients.items()
            if term in last_gradients
        }

        for term, change in changes.items():
            change_norm = float(change.norm())
            curvature = float(torch.dot(step, change))
            metrics[f"ynorm_{term}"] = change_norm
            metrics[f"curv_{term}"] = curvature
            if step_norm > 0.0 and change_norm > 0.0:
                metrics[f"cos_s_y_{term}"] = curvature / (step_norm * change_norm)

        for first, second in itertools.combinations(changes, 2):
            a, b = changes[first], changes[second]
            inner = float(torch.dot(a, b))
            metrics[f"dot_y_{first}_{second}"] = inner
            scale = float(a.norm()) * float(b.norm())
            if scale > 0.0:
                metrics[f"cos_y_{first}_{second}"] = inner / scale

        if changes:
            total = sum(
                float(getattr(weights, term)) * change
                for term, change in changes.items()
            )
            metrics["curv_total"] = float(torch.dot(step, total))

    previous["gradients"] = {
        term: gradient.detach().clone() for term, gradient in gradients.items()
    }
    previous["parameters"] = current
    return metrics


def gradient_alignment(
    components: dict[str, torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
    weights: Any,
    terms: tuple[str, ...] = ("data", "pde", "bc", "ic"),
    previous: dict[str, Any] | None = None,
) -> dict[str, float]:
    """How a PINN's loss terms pull against each other in parameter space.

    A PINN's objective is a sum of terms that need not agree. The direction the
    data term wants to move the parameters and the direction the PDE residual
    wants can oppose each other, and the sum is then a compromise neither asked
    for. The loss values cannot show that -- two terms both falling says nothing
    about whether they are fighting -- so each one is differentiated separately
    here and the geometry of the results is reported.

    ``components`` is the dict :meth:`PINNLoss.forward` returns alongside the
    total; all four PINN models key it identically, which is why nothing here
    needs to know which architecture produced it. ``weights`` is the run's
    ``LossWeights``, read by attribute name for each of ``terms``.

    Three families come out:

    ``cos_<a>_<b>`` / ``dot_<a>_<b>``
        The pairwise geometry, from the *unweighted* gradients. A negative
        cosine is the interesting case: the two terms disagree about where to
        go. Cosine is invariant to positive rescaling, so this is the terms'
        own geometry and no choice of ``--w-*`` can change it.
    ``gnorm_<t>`` / ``gnorm_weighted_<t>``
        How big each term's gradient is on its own, and after its weight -- the
        second being what the optimizer actually steps along. A term can be
        negligible in the loss and still dominate the update.
    ``cos_data_physics`` / ``norm_ratio_physics_data``
        The data term against the sum of all the others, both weighted: one
        number for "is the physics, as configured, opposing the fit?"

    Pass a dict as ``previous`` -- the same one every call, initially empty -- to
    add the second-order family from :func:`_secant_metrics`: how each term's
    gradient changed since the last call, measured against the step the
    parameters took. It costs nothing extra to compute but is only meaningful
    when calls are close together, so the cadence should be small (1 for a
    quasi-Newton run, where every step is already a line search).

    Costs one backward pass per term on top of whatever the caller is already
    doing, so it belongs behind a cadence flag rather than in every iteration.
    ``torch.autograd.grad`` does not touch ``.grad``, so calling this leaves the
    optimizer's state alone.
    """
    parameters = [p for p in parameters if p.requires_grad]

    flat: dict[str, torch.Tensor] = {}
    for term in terms:
        value = components.get(term)
        # A term whose point set was omitted is the forward's shared `zero`,
        # which carries no graph; differentiating it raises rather than yielding
        # a zero gradient, so it is dropped and its pairs are simply absent.
        if value is None or not value.requires_grad:
            continue
        pieces = torch.autograd.grad(
            value, parameters, retain_graph=True, allow_unused=True
        )
        flat[term] = torch.cat(
            [
                (torch.zeros_like(parameter) if piece is None else piece).reshape(-1)
                for parameter, piece in zip(parameters, pieces)
            ]
        )

    metrics: dict[str, float] = {}
    for term, gradient in flat.items():
        norm = float(gradient.norm())
        metrics[f"gnorm_{term}"] = norm
        metrics[f"gnorm_weighted_{term}"] = abs(float(getattr(weights, term))) * norm

    def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
        scale = float(first.norm()) * float(second.norm())
        return float(torch.dot(first, second)) / scale if scale > 0.0 else 0.0

    for first, second in itertools.combinations(flat, 2):
        metrics[f"dot_{first}_{second}"] = float(torch.dot(flat[first], flat[second]))
        metrics[f"cos_{first}_{second}"] = cosine(flat[first], flat[second])

    if "data" in flat and len(flat) > 1:
        data = float(getattr(weights, "data")) * flat["data"]
        physics = sum(
            float(getattr(weights, term)) * gradient
            for term, gradient in flat.items()
            if term != "data"
        )
        metrics["cos_data_physics"] = cosine(data, physics)
        data_norm = float(data.norm())
        metrics["norm_ratio_physics_data"] = (
            float(physics.norm()) / data_norm if data_norm > 0.0 else math.inf
        )

    if previous is not None:
        metrics.update(_secant_metrics(flat, parameters, weights, previous))

    return metrics


# ---------------------------------------------------------------------------
# Run naming
# ---------------------------------------------------------------------------


def make_run_name(model_name: str, tag: str | None = None) -> str:
    """``<model>[-<tag>]-<YYYYmmdd-HHMMSS>``, unique enough to sort by time."""
    parts = [model_name]
    if tag:
        parts.append(tag)
    parts.append(time.strftime("%Y%m%d-%H%M%S"))
    return "-".join(parts)


def unique_path(path: Path) -> Path:
    """``path``, or ``path`` with ``-1``, ``-2``, ... appended until it is free.

    Two runs started within the same second would otherwise share a TensorBoard
    directory and interleave their scalars.
    """
    if not path.exists():
        return path
    for suffix in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free name near {path}")


def resolve_run_dir(log_dir: Path, run_name: str) -> Path:
    """The TensorBoard directory for this run, created and guaranteed unused."""
    directory = unique_path(Path(log_dir) / run_name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_checkpoint_path(
    checkpoint: Path | None, model_name: str, run_name: str
) -> Path:
    """``checkpoint`` if given, else ``checkpoints/<model>/<run_name>.pt``."""
    path = (
        Path(checkpoint)
        if checkpoint is not None
        else DEFAULT_CHECKPOINT_DIR / model_name / f"{run_name}.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


_RUN_TIMESTAMP = re.compile(r"\d{8}-\d{6}")


def checkpoint_stamp(path: Path) -> str:
    """The ``YYYYmmdd-HHMMSS`` a checkpoint was created at.

    :func:`make_run_name` bakes this into the run name, and
    :func:`resolve_checkpoint_path` bakes the run name into the checkpoint's
    filename, so it can be read back out of the stem. Falls back to the file's
    mtime for a checkpoint named by hand, so the figures this stamps stay
    sorted with the checkpoint they came from either way.
    """
    match = _RUN_TIMESTAMP.search(Path(path).stem)
    if match:
        return match.group(0)
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(Path(path).stat().st_mtime))


def resolve_figure_path(out: Path | None, *parts: str, stamp: str | None = None) -> Path:
    """``out`` if given, else an auto-named path under ``figures/``.

    A bare filename in ``out`` (no directory component) still lands under
    ``figures/``, so ``--out plot.png`` and omitting ``--out`` both keep
    figures in one place; a path with a directory in it is used as-is.
    ``parts`` -- e.g. ``(model, f"P{power:g}", plane)`` -- are joined with
    ``_`` and suffixed with ``stamp`` (see :func:`checkpoint_stamp`; defaults
    to the current time) when ``out`` is omitted entirely.
    """
    if out is None:
        stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        name = "_".join(parts) + f"_{stamp}.png"
        path = DEFAULT_FIGURES_DIR / name
    elif out.parent == Path("."):
        path = DEFAULT_FIGURES_DIR / out
    else:
        path = out
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically.

    A long run interrupted mid-``torch.save`` would otherwise leave a truncated
    file where the best checkpoint used to be, so the new one is written beside
    it and moved into place only once it is complete.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, map_location: torch.device | str = "cpu"
) -> dict[str, Any]:
    """Read a checkpoint written by :class:`BestCheckpoint`."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    for key in ("model", "architecture"):
        if key not in checkpoint:
            raise KeyError(f"{path}: checkpoint is missing the `{key}` key")
    return checkpoint


class BestCheckpoint:
    """Keeps the single checkpoint with the best value of a monitored metric.

    ``mode='min'`` for losses and errors, ``mode='max'`` for scores. The payload
    is supplied by the caller so each model can store whatever ``agent.py`` needs
    to rebuild itself -- but ``model`` (the state dict) and ``architecture`` (the
    keyword arguments of its constructor) are mandatory, since the loaders
    depend on them.
    """

    def __init__(self, path: Path, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.path = Path(path)
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.step: int | None = None

    def is_better(self, value: float) -> bool:
        return value < self.best if self.mode == "min" else value > self.best

    def update(self, value: float, payload: dict[str, Any], step: int | None = None) -> bool:
        """Save ``payload`` if ``value`` improves on the best seen. Returns whether it did."""
        if not self.is_better(value):
            return False

        self.best = value
        self.step = step
        save_checkpoint(self.path, {**payload, "metric": value, "step": step})
        return True
