"""Train PiDoN on laser-scan temperature fields with PINN losses.

Invoked through the top-level dispatcher::

    python train.py pidon --iterations 20000

Every quantity below is SI (metres, seconds, Kelvin); the raw ``.npy`` files
store millimetres and are converted by :mod:`dataset` on load. ``P`` is the
branch input and is constant within a file, so the branch network only has
something to learn once several files at different powers are present; every
file under the data directory is globbed and concatenated automatically.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import DEFAULT_DATA_DIR, Domain, SimulationDataset
from utils import (
    DEFAULT_LOG_DIR,
    BestCheckpoint,
    add_optimizer_args,
    alignment_group,
    build_optimizer,
    count_parameters,
    gradient_alignment,
    make_run_name,
    resolve_checkpoint_path,
    resolve_device,
    resolve_lr,
    resolve_run_dir,
    seed_everything,
)

from .dataset import PiDoNDataset
from .loss import (
    PROPERTIES,
    LossWeights,
    PINNLoss,
    ResidualScales,
    peak_laser_flux,
)
from .model import PiDoN

MODEL_NAME = "pidon"

# Sub-ambient rows (~5%, minimum 289.8 K) are a solver artefact: with heating
# only, T can never fall below T_amb. Left untouched by default rather than
# silently editing the source data.
CLIP_SUBAMBIENT = False


@torch.no_grad()
def evaluate(
    model: PiDoN, split: SimulationDataset, chunk: int = 65536
) -> tuple[float, float]:
    """Return ``(rmse, max_abs_error)`` in Kelvin over every point of ``split``."""
    squared_error = 0.0
    worst = 0.0
    for start in range(0, len(split), chunk):
        stop = start + chunk
        error = (
            model(split.power[start:stop], split.coords[start:stop])
            - split.temperature[start:stop]
        )
        squared_error += float(error.pow(2).sum())
        worst = max(worst, float(error.abs().max()))
    return math.sqrt(squared_error / len(split)), worst


def build_model(
    domain: Domain,
    temperature_rise: float,
    max_power: float,
    gaussian_exponent_scale: float = 1.0,
    hidden_layers: tuple[int, ...] = (128, 128, 128, 128),
    latent_dim: int = 128,
) -> tuple[PiDoN, dict]:
    """Instantiate the network with normalisation baked in from the data statistics.

    The keyword dict is returned alongside so it can be stored in the checkpoint;
    ``agent.py`` rebuilds the network from it without re-reading the dataset.

    ``hidden_layers`` sizes the branch and trunk subnetworks alike, so
    ``--hidden 64 64 64`` puts this model at the same depth and width as
    ``pimlp``'s ``64x3``. ``latent_dim`` -- the width of the inner product the
    two subnetworks meet in -- has no counterpart in a dense stack and is left
    alone.
    """
    architecture = dict(
        branch_input_dim=1,
        hidden_layers=tuple(hidden_layers),
        latent_dim=latent_dim,
        coord_mean=domain.center.tolist(),
        coord_scale=domain.half_width.tolist(),
        branch_mean=[0.0],
        branch_scale=[max_power],
        temperature_offset=PROPERTIES.ambient_temperature,
        temperature_scale=temperature_rise,
        gaussian_exponent_scale=gaussian_exponent_scale,
    )
    return PiDoN(**architecture), architecture


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f"train.py {MODEL_NAME}", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--exclude",
        type=int,
        default=0,
        metavar="N",
        help="drop the first N time steps of the corpus before training. Applied "
        "before the split, so they are absent from the validation half too; the "
        "domain contracts with them, which moves the initial-condition term onto "
        "the earliest step left -- where it is fitted to the field the solver had "
        "there rather than to ambient",
    )
    parser.add_argument("--iterations", type=int, default=20000)
    add_optimizer_args(parser)
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[128, 128, 128, 128],
        help="hidden layer widths of the branch and trunk subnetworks alike",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="width of the inner product the branch and trunk meet in",
    )
    parser.add_argument("--batch-data", type=int, default=4096)
    parser.add_argument("--batch-physics", type=int, default=2048)
    parser.add_argument("--batch-boundary", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=250, help="validation and console cadence")
    parser.add_argument("--scalar-every", type=int, default=25, help="TensorBoard loss cadence")
    parser.add_argument(
        "--align-every",
        type=int,
        default=0,
        metavar="N",
        help="every N iterations, differentiate each loss term separately and log "
        "the pairwise cosines and gradient norms under `align/`; 0 disables it. "
        "Costs one extra backward pass per term when it fires, so it is off by "
        "default. See utils.gradient_alignment",
    )
    parser.add_argument(
        "--align-curvature",
        action="store_true",
        help="add the second-order family to --align-every: how each term's "
        "gradient CHANGED since the last measurement, against the step the "
        "parameters took. These are the (s, y) pairs L-BFGS builds its curvature "
        "from, so `curv_<term>` (= s'y) shows which term supplies the positive "
        "curvature the update needs and which one opposes it. Only meaningful at "
        "a small --align-every",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--double", action="store_true", help="run in float64")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="defaults to checkpoints/pidon/<run-name>.pt",
    )
    parser.add_argument("--logdir", type=Path, default=DEFAULT_LOG_DIR, help="TensorBoard output")
    parser.add_argument("--run-name", type=str, default=None, help="subdirectory under --logdir")
    parser.add_argument("--tag", type=str, default=None, help="label folded into the run name")
    parser.add_argument("--no-progress", action="store_true", help="disable the tqdm bar")
    parser.add_argument("--w-data", type=float, default=1.0)
    parser.add_argument("--w-pde", type=float, default=1e-4)
    parser.add_argument("--w-bc", type=float, default=1e-4)
    parser.add_argument("--w-ic", type=float, default=1e-4)
    parser.add_argument(
        "--gaussian-exponent-scale",
        type=float,
        default=1.0,
        help="multiplies the exponent of the top-surface laser gaussian; "
        ">1 narrows the assumed source, <1 broadens it",
    )
    args = parser.parse_args(argv)
    if args.align_curvature and args.align_every <= 0:
        parser.error("--align-curvature requires --align-every")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dtype = torch.float64 if args.double else torch.float32
    torch.set_default_dtype(dtype)
    device = resolve_device(args.device)
    seed_everything(args.seed)

    corpus = SimulationDataset.from_dir(
        args.data_dir,
        dtype=dtype,
        device=device,
        clip_below=PROPERTIES.ambient_temperature if CLIP_SUBAMBIENT else None,
        exclude_steps=args.exclude,
    )
    domain = corpus.domain
    max_power = corpus.max_power
    temperature_rise = corpus.temperature_rise(PROPERTIES.ambient_temperature)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    train_split, val_split = corpus.split(args.val_fraction, generator)
    sampler = PiDoNDataset(
        train_split,
        generator,
        domain=domain,
        gaussian_exponent_scale=args.gaussian_exponent_scale,
    )
    # One batch, drawn once and reused for every measurement, so the only thing
    # moving in the `align/` series is the parameters. Its generator is separate
    # from the training one: --align-every has to be an observer, and sharing the
    # stream would make the batches training draws depend on whether it is on.
    probe_batches = None
    if args.align_every > 0:
        probe_sampler = PiDoNDataset(
            train_split,
            torch.Generator(device=device).manual_seed(args.seed + 1),
            domain=domain,
            gaussian_exponent_scale=args.gaussian_exponent_scale,
        )
        probe_batches = probe_sampler.batches(
            args.batch_data, args.batch_physics, args.batch_boundary
        )
    # Carried across measurements so each one can be compared with the last; see
    # utils._secant_metrics. None keeps the probe first-order only.
    probe_state = {} if (probe_batches is not None and args.align_curvature) else None

    model, architecture = build_model(
        domain,
        temperature_rise,
        max_power,
        args.gaussian_exponent_scale,
        hidden_layers=tuple(args.hidden),
        latent_dim=args.latent_dim,
    )
    model = model.to(device=device, dtype=dtype)
    scales = ResidualScales.characteristic(
        properties=PROPERTIES,
        temperature_rise=temperature_rise,
        time_scale=float(domain.upper[3]),
        peak_flux=float(peak_laser_flux(max_power)),
    )
    weights = LossWeights(data=args.w_data, pde=args.w_pde, bc=args.w_bc, ic=args.w_ic)
    criterion = PINNLoss(PROPERTIES, weights=weights, scales=scales)

    learning_rate = resolve_lr(args)
    optimizer = build_optimizer(model.parameters(), args, learning_rate)
    use_lbfgs = args.optimizer == "lbfgs"
    # L-BFGS *requires* a fixed objective; adam merely allows one.
    freeze_batch = use_lbfgs or args.freeze_batch
    # L-BFGS estimates curvature from the gradients of previous steps, which is
    # only meaningful if they all come from the same function. So the physics
    # points -- normally resampled every iteration -- are drawn once and frozen,
    # and the LR schedule is dropped: the strong-Wolfe line search sets the step.
    scheduler = (
        None
        if use_lbfgs
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations)
    )
    fixed_batches = (
        sampler.batches(args.batch_data, args.batch_physics, args.batch_boundary)
        if freeze_batch
        else None
    )

    run_name = args.run_name or make_run_name(MODEL_NAME, args.tag)
    run_dir = resolve_run_dir(args.logdir, run_name)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, MODEL_NAME, run_name)
    best = BestCheckpoint(checkpoint_path, mode="min")
    writer = SummaryWriter(log_dir=str(run_dir))

    print(f"[setup] tensorboard run -> {run_dir}")
    print(f"[setup] checkpoint      -> {checkpoint_path}")
    print(f"[setup] device={device} dtype={dtype} params={count_parameters(model)}")
    print(f"[setup] lower={domain.lower.tolist()} upper={domain.upper.tolist()}")
    print(f"[setup] powers={corpus.powers.tolist()} W  T_rise={temperature_rise:.1f} K")
    print(f"[setup] gaussian_exponent_scale={args.gaussian_exponent_scale:g}")
    print(
        f"[setup] scales temperature={scales.temperature:.4g} K "
        f"pde={scales.pde:.4g} W/m^3 flux={scales.flux:.4g} W/m^2"
    )
    print(f"[setup] train={len(train_split)} val={len(val_split)}")

    progress = tqdm(
        range(1, args.iterations + 1),
        desc="train",
        unit="it",
        disable=args.no_progress,
        dynamic_ncols=True,
    )
    for iteration in progress:
        model.train()

        # Measured before the step, so the geometry reported is the one at the
        # parameters this iteration is about to move away from.
        if probe_batches is not None and (
            iteration % args.align_every == 0 or iteration == 1
        ):
            _, probe_components = criterion(model, **probe_batches)
            alignment = gradient_alignment(
                probe_components, model.parameters(), weights, previous=probe_state
            )
            for name, value in alignment.items():
                writer.add_scalar(f"{alignment_group(name)}/{name}", value, iteration)
            cosines = " ".join(
                f"{key.removeprefix('cos_')}={alignment[key]:+.3f}"
                for key in ("cos_data_pde", "cos_data_bc", "cos_data_ic", "cos_data_physics")
                if key in alignment
            )
            norms = " ".join(
                f"{term}={alignment[f'gnorm_{term}']:.2e}"
                for term in ("data", "pde", "bc", "ic")
                if f"gnorm_{term}" in alignment
            )
            ratio = alignment.get("norm_ratio_physics_data")
            progress.write(
                f"[align {iteration:6d}] cos {cosines} | |g| {norms}"
                + ("" if ratio is None else f" | w|g| phys/data={ratio:.3e}")
            )
            # Absent on the first measurement, which has nothing to difference
            # against, and whenever the step moved nothing.
            curvature = " ".join(
                f"{term}={alignment[f'curv_{term}']:+.2e}"
                for term in ("data", "pde", "bc", "ic")
                if f"curv_{term}" in alignment
            )
            if curvature:
                pair = alignment.get("cos_y_data_pde")
                total = alignment.get("curv_total")
                progress.write(
                    f"[curv  {iteration:6d}] sTy {curvature}"
                    + ("" if total is None else f" | total={total:+.2e}")
                    + ("" if pair is None else f" | cos_y(data,pde)={pair:+.3f}")
                    + f" | |s|={alignment.get('snorm', 0.0):.2e}"
                )

        batches = fixed_batches or sampler.batches(
            args.batch_data, args.batch_physics, args.batch_boundary
        )
        parts: dict = {}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            total, components = criterion(model, **batches)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            parts.clear()
            parts.update(components)
            return total

        if use_lbfgs:
            # L-BFGS re-evaluates the closure several times per step to run its
            # line search, so it -- not the caller -- drives the loop.
            total = optimizer.step(closure)
        else:
            total = closure()
            optimizer.step()
            scheduler.step()
        components = parts

        # .item() synchronises with the GPU, so only pull the scalars we log.
        if iteration % args.scalar_every == 0 or iteration == 1:
            total_value = total.detach().item()
            writer.add_scalar("loss/total", total_value, iteration)
            for name, value in components.items():
                writer.add_scalar(f"loss/{name}", value.detach().item(), iteration)
            current_lr = learning_rate if scheduler is None else scheduler.get_last_lr()[0]
            writer.add_scalar("lr", current_lr, iteration)
            progress.set_postfix(loss=f"{total_value:.3e}", best=f"{best.best:.1f}K")

        if iteration % args.log_every == 0 or iteration == 1:
            model.eval()
            rmse, worst = evaluate(model, val_split)
            writer.add_scalar("val/rmse", rmse, iteration)
            writer.add_scalar("val/max_error", worst, iteration)

            parts = " ".join(
                f"{name}={value.detach().item():.3e}" for name, value in components.items()
            )
            progress.write(
                f"[{iteration:6d}] total={total.detach().item():.4e} {parts} "
                f"| val_rmse={rmse:7.3f}K val_max={worst:8.3f}K lr={learning_rate if scheduler is None else scheduler.get_last_lr()[0]:.2e}"
            )
            improved = best.update(
                rmse,
                {
                    "model": model.state_dict(),
                    "architecture": architecture,
                    "bounds": domain.bounds.detach().cpu(),
                    "val_rmse": rmse,
                    "properties": PROPERTIES,
                    "scales": scales,
                    "weights": weights,
                },
                step=iteration,
            )
            if improved:
                progress.set_postfix(
                    loss=f"{total.detach().item():.3e}", best=f"{best.best:.1f}K"
                )

    progress.close()
    writer.add_hparams(
        {
            "optimizer": args.optimizer,
            "lr": learning_rate,
            "iterations": args.iterations,
            "hidden": str(tuple(args.hidden)),
            "latent_dim": args.latent_dim,
            "batch_data": args.batch_data,
            "batch_physics": args.batch_physics,
            "batch_boundary": args.batch_boundary,
            "w_data": weights.data,
            "w_pde": weights.pde,
            "w_bc": weights.bc,
            "w_ic": weights.ic,
            "gaussian_exponent_scale": args.gaussian_exponent_scale,
        },
        {"hparam/val_rmse": best.best},
    )
    writer.close()
    print(f"[done] best val RMSE {best.best:.3f} K -> {checkpoint_path}")
    print(f"[done] tensorboard --logdir {args.logdir}")
