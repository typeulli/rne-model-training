"""Train the CMLP control on ``(t, z, y, x) -> T``.

Invoked through the top-level dispatcher::

    python train.py cmlp --epochs 20 --batch-size 8192

Identical to ``models/mlp/train.py`` -- same loop, same objective, same
hyperparameters, same batches drawn from the same rows -- except that the laser
power is not among the network's inputs. Everything else is held fixed so the two
runs can be compared directly.

The whole corpus is loaded into memory up front, split once into train and
validation, and every step draws a random batch (with replacement) from the
train split. An "epoch" is not a shuffled pass over every row; it is
``--steps-per-epoch`` such batches, after which the model is scored against the
full validation split and the best checkpoint is updated.

The validation RMSE is not expected to reach ``mlp``'s. The split is by row, so
every power in the corpus appears in both halves, and the same ``(t, z, y, x)``
carries seven different temperatures across the seven files; no function of the
four coordinates alone can hit them all. The residual that remains is the spread
across powers, and that is the quantity this run exists to measure.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import DEFAULT_DATA_DIR, SimulationDataset
from utils import (
    DEFAULT_LOG_DIR,
    BestCheckpoint,
    count_parameters,
    make_run_name,
    resolve_checkpoint_path,
    resolve_device,
    resolve_run_dir,
)

from .dataset import CMLPDataset
from .loss import ScaledMSELoss
from .model import ControlMLP

MODEL_NAME = "cmlp"

# Matches models/gpidon/loss.py; only used to centre the output, so an exact
# value is not critical.
AMBIENT_TEMPERATURE = 298.0

# Sub-ambient rows (~5%, minimum 289.8 K) are a solver artefact: with heating
# only, T can never fall below T_amb. Left untouched by default rather than
# silently editing the source data.
CLIP_SUBAMBIENT = False


@torch.no_grad()
def evaluate(
    model: ControlMLP, inputs: Tensor, target: Tensor, chunk: int = 65536
) -> tuple[float, float]:
    """Return ``(rmse, max_abs_error)`` in Kelvin over every point given."""
    squared_error = 0.0
    worst = 0.0
    for start in range(0, inputs.size(0), chunk):
        stop = start + chunk
        error = model(inputs[start:stop]) - target[start:stop]
        squared_error += float(error.pow(2).sum())
        worst = max(worst, float(error.abs().max()))
    return math.sqrt(squared_error / inputs.size(0)), worst


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f"train.py {MODEL_NAME}", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=1000,
        help="batches per epoch; sampling is with replacement, so this is a free choice",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[256, 256, 256, 256], help="hidden layer widths"
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--scalar-every", type=int, default=25, help="TensorBoard loss cadence")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--double", action="store_true", help="run in float64")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="defaults to checkpoints/cmlp/<run-name>.pt"
    )
    parser.add_argument("--logdir", type=Path, default=DEFAULT_LOG_DIR, help="TensorBoard output")
    parser.add_argument("--run-name", type=str, default=None, help="subdirectory under --logdir")
    parser.add_argument("--tag", type=str, default=None, help="label folded into the run name")
    parser.add_argument("--no-progress", action="store_true", help="disable the tqdm bar")
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
    temperature_rise = corpus.temperature_rise(AMBIENT_TEMPERATURE)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    train_split, val_split = corpus.split(args.val_fraction, generator)
    sampler = CMLPDataset(train_split, generator)
    val_inputs, val_target = CMLPDataset(val_split, generator).all()

    input_mean, input_scale = sampler.normalisation()
    architecture = dict(
        hidden_layers=tuple(args.hidden),
        input_mean=input_mean,
        input_scale=input_scale,
        temperature_offset=AMBIENT_TEMPERATURE,
        temperature_scale=temperature_rise,
    )
    model = ControlMLP(**architecture).to(device=device, dtype=dtype)
    criterion = ScaledMSELoss(scale=temperature_rise)

    total_steps = args.epochs * args.steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

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
    print("[setup] P is NOT an input: the model sees (t, z, y, x) and must average over them")
    print(f"[setup] train={len(train_split)} val={len(val_split)}")
    print(f"[setup] {args.epochs} epochs x {args.steps_per_epoch} steps x {args.batch_size} rows")

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(
            range(args.steps_per_epoch),
            desc=f"epoch {epoch}/{args.epochs}",
            unit="it",
            disable=args.no_progress,
            dynamic_ncols=True,
            leave=False,
        )
        running = 0.0
        for _ in progress:
            step += 1
            inputs, target = sampler.batch(args.batch_size)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # .item() synchronises with the GPU, so only pull the scalars we log.
            if step % args.scalar_every == 0 or step == 1:
                value = loss.detach().item()
                running = value
                writer.add_scalar("loss/train", value, step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
                progress.set_postfix(loss=f"{value:.3e}", best=f"{best.best:.1f}K")
        progress.close()

        model.eval()
        rmse, worst = evaluate(model, val_inputs, val_target)
        writer.add_scalar("val/rmse", rmse, step)
        writer.add_scalar("val/max_error", worst, step)
        improved = best.update(
            rmse,
            {
                "model": model.state_dict(),
                "architecture": architecture,
                "bounds": domain.bounds.detach().cpu(),
                "val_rmse": rmse,
                "epoch": epoch,
            },
            step=step,
        )
        print(
            f"[epoch {epoch:3d}/{args.epochs}] train={running:.4e} "
            f"val_rmse={rmse:8.3f}K val_max={worst:9.3f}K "
            f"lr={scheduler.get_last_lr()[0]:.2e}{'  *' if improved else ''}"
        )

    writer.add_hparams(
        {
            "lr": args.lr,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "hidden": str(tuple(args.hidden)),
        },
        {"hparam/val_rmse": best.best},
    )
    writer.close()
    print(f"[done] best val RMSE {best.best:.3f} K -> {checkpoint_path}")
    print(f"[done] tensorboard --logdir {args.logdir}")
