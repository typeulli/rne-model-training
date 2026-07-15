"""Train GDoN on laser-scan temperature fields with a supervised loss alone.

Invoked through the top-level dispatcher::

    python train.py gdon --iterations 20000

Identical to ``models/gpidon/train.py`` except that the objective is a scaled MSE
against the labelled temperatures -- no PDE, boundary or initial-condition
residuals, hence no collocation batches and no ``--w-*`` weights. Everything else
is held fixed so the two runs can be compared directly.

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
    build_optimizer,
    count_parameters,
    make_run_name,
    resolve_checkpoint_path,
    resolve_device,
    resolve_lr,
    resolve_run_dir,
)

from .dataset import GDoNDataset
from .loss import ScaledMSELoss
from .model import GDoN

MODEL_NAME = "gdon"

# Matches models/gpidon/loss.py; only used to centre the output, so an exact
# value is not critical.
AMBIENT_TEMPERATURE = 298.0

# Sub-ambient rows (~5%, minimum 289.8 K) are a solver artefact: with heating
# only, T can never fall below T_amb. Left untouched by default rather than
# silently editing the source data.
CLIP_SUBAMBIENT = False


@torch.no_grad()
def evaluate(
    model: GDoN, split: SimulationDataset, chunk: int = 65536
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
    gate_offset: float = 0.5,
    hidden_layers: tuple[int, ...] = (128, 128, 128, 128),
    latent_dim: int = 128,
) -> tuple[GDoN, dict]:
    """Instantiate the network with normalisation baked in from the data statistics.

    The keyword dict is returned alongside so it can be stored in the checkpoint;
    ``agent.py`` rebuilds the network from it without re-reading the dataset.

    ``hidden_layers`` sizes the branch and trunk subnetworks alike, so
    ``--hidden 64 64 64`` puts this model at the same depth and width as
    ``mlp``'s ``64x3``. ``latent_dim`` -- the width of the inner product the two
    subnetworks meet in -- has no counterpart in a dense stack and is left alone.
    """
    architecture = dict(
        branch_input_dim=1,
        hidden_layers=tuple(hidden_layers),
        latent_dim=latent_dim,
        coord_mean=domain.center.tolist(),
        coord_scale=domain.half_width.tolist(),
        branch_mean=[0.0],
        branch_scale=[max_power],
        temperature_offset=AMBIENT_TEMPERATURE,
        temperature_scale=temperature_rise,
        gaussian_exponent_scale=gaussian_exponent_scale,
        gate_offset=gate_offset,
    )
    return GDoN(**architecture), architecture


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f"train.py {MODEL_NAME}", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
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
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=250, help="validation and console cadence")
    parser.add_argument("--scalar-every", type=int, default=25, help="TensorBoard loss cadence")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--double", action="store_true", help="run in float64")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="defaults to checkpoints/gdon/<run-name>.pt",
    )
    parser.add_argument("--logdir", type=Path, default=DEFAULT_LOG_DIR, help="TensorBoard output")
    parser.add_argument("--run-name", type=str, default=None, help="subdirectory under --logdir")
    parser.add_argument("--tag", type=str, default=None, help="label folded into the run name")
    parser.add_argument("--no-progress", action="store_true", help="disable the tqdm bar")
    parser.add_argument(
        "--gaussian-exponent-scale",
        type=float,
        default=1.0,
        help="multiplies the exponent of the network's gaussian gate around the "
        "beam; >1 tightens the envelope, <1 widens it",
    )
    parser.add_argument(
        "--gate-offset",
        type=float,
        default=0.5,
        help="initial value of the learnable `p` in the gate G = g + p; it is only "
        "the starting point, training moves it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dtype = torch.float64 if args.double else torch.float32
    torch.set_default_dtype(dtype)
    device = resolve_device(args.device)

    corpus = SimulationDataset.from_dir(
        args.data_dir,
        dtype=dtype,
        device=device,
        clip_below=AMBIENT_TEMPERATURE if CLIP_SUBAMBIENT else None,
    )
    domain = corpus.domain
    max_power = corpus.max_power
    temperature_rise = corpus.temperature_rise(AMBIENT_TEMPERATURE)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    train_split, val_split = corpus.split(args.val_fraction, generator)
    sampler = GDoNDataset(train_split, generator)

    model, architecture = build_model(
        domain,
        temperature_rise,
        max_power,
        args.gaussian_exponent_scale,
        args.gate_offset,
        hidden_layers=tuple(args.hidden),
        latent_dim=args.latent_dim,
    )
    model = model.to(device=device, dtype=dtype)
    criterion = ScaledMSELoss(scale=temperature_rise)

    learning_rate = resolve_lr(args)
    optimizer = build_optimizer(model.parameters(), args, learning_rate)
    use_lbfgs = args.optimizer == "lbfgs"
    # L-BFGS *requires* a fixed objective; adam merely allows one.
    freeze_batch = use_lbfgs or args.freeze_batch
    # L-BFGS descends one fixed function, so its batch is drawn once and reused;
    # the strong-Wolfe line search sets the step, so there is no LR schedule.
    scheduler = (
        None
        if use_lbfgs
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations)
    )
    fixed_batch = sampler.batch(args.lbfgs_batch) if freeze_batch else None

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
    print(f"[setup] gate_offset p0={args.gate_offset:g} (learnable)")
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

        power, coords, temperature = (
            fixed_batch if freeze_batch else sampler.batch(args.batch_data)
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(power, coords), temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            return loss

        if use_lbfgs:
            # L-BFGS re-evaluates the closure several times per step to run its
            # line search, so it -- not the caller -- drives the loop.
            loss = optimizer.step(closure)
        else:
            loss = closure()
            optimizer.step()
            scheduler.step()

        # .item() synchronises with the GPU, so only pull the scalars we log.
        if iteration % args.scalar_every == 0 or iteration == 1:
            value = loss.detach().item()
            writer.add_scalar("loss/data", value, iteration)
            current_lr = learning_rate if scheduler is None else scheduler.get_last_lr()[0]
            writer.add_scalar("lr", current_lr, iteration)
            progress.set_postfix(loss=f"{value:.3e}", best=f"{best.best:.1f}K")

        if iteration % args.log_every == 0 or iteration == 1:
            model.eval()
            rmse, worst = evaluate(model, val_split)
            gate_offset = float(model.gate_offset.detach())
            writer.add_scalar("val/rmse", rmse, iteration)
            writer.add_scalar("val/max_error", worst, iteration)
            writer.add_scalar("gate/offset", gate_offset, iteration)

            progress.write(
                f"[{iteration:6d}] data={loss.detach().item():.4e} "
                f"| val_rmse={rmse:7.3f}K val_max={worst:8.3f}K p={gate_offset:+7.4f} "
                f"lr={learning_rate if scheduler is None else scheduler.get_last_lr()[0]:.2e}"
            )
            improved = best.update(
                rmse,
                {
                    "model": model.state_dict(),
                    "architecture": architecture,
                    "bounds": domain.bounds.detach().cpu(),
                    "val_rmse": rmse,
                },
                step=iteration,
            )
            if improved:
                progress.set_postfix(loss=f"{loss.detach().item():.3e}", best=f"{best.best:.1f}K")

    progress.close()
    writer.add_hparams(
        {
            "optimizer": args.optimizer,
            "lr": learning_rate,
            "iterations": args.iterations,
            "hidden": str(tuple(args.hidden)),
            "latent_dim": args.latent_dim,
            "batch_data": args.batch_data,
            "gaussian_exponent_scale": args.gaussian_exponent_scale,
            "gate_offset_init": args.gate_offset,
        },
        {"hparam/val_rmse": best.best},
    )
    writer.close()
    print(f"[done] learned gate offset p = {float(model.gate_offset.detach()):+.4f}")
    print(f"[done] best val RMSE {best.best:.3f} K -> {checkpoint_path}")
    print(f"[done] tensorboard --logdir {args.logdir}")
