"""Fit ``(P, t) -> spatial Fourier coefficients``, and score the volume it implies.

Invoked through the top-level dispatcher::

    python train.py fmlp --epochs 20000
    python train.py fmlp --energy-target 0.99999 --detrend
    python train.py fmlp --no-derotate --tag ablation

Three things about this run are not like the other nine, and all three follow from
the sample being a *snapshot* rather than a point.

**There is no batch.** The corpus is seven powers of eight snapshots -- 56 rows,
a few megabytes -- so every step is a full pass over all of them and an "epoch" is
one gradient step on the whole training set. There is nothing to sample.

**There is a floor.** A network that predicted every stored coefficient perfectly
would still be wrong, by whatever the truncation box threw away. So the floor is
computed up front and reported next to the model's own error, and the gap between
them is the only part the network is answerable for. Chasing the floor is the goal;
beating it is impossible.

**Validation is the held-out power, not a row split.** The other models score
themselves on a random 10% of rows, which for a model that always reconstructs the
entire volume would mean holding out points it predicts anyway. The honest question
for this one is whether it can interpolate in ``P``, so the metric is the real-space
RMSE in Kelvin on the 160 W run under ``data/valid`` -- a power it never sees. That
also makes the number directly comparable with the table in ``README.md``, which is
the point of quoting it in Kelvin rather than in coefficient space.

It is worth being clear about what that metric does *not* test. The held-out run is
sampled at the same eight instants as the training runs, so nothing here scores the
model between snapshots -- and that is exactly where a model fitted to the raw,
aliased coefficients falls apart while its score barely moves. The evidence is in
``dataset.py``, under de-rotation; the short version is that ``--no-derotate``
costs 0.07 K on this number and roughly 460 K on the peak at a time it was not shown.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import DEFAULT_DATA_DIR, DEFAULT_VALID_DIR, find_data_files
from utils import (
    DEFAULT_LOG_DIR,
    BestCheckpoint,
    count_parameters,
    load_checkpoint,
    make_run_name,
    resolve_checkpoint_path,
    resolve_device,
    resolve_run_dir,
)

from .dataset import AMBIENT_TEMPERATURE, SCAN_SPEED, FourierCorpus, bounds_of, build
from .loss import CoefficientMSELoss
from .model import FourierMLP

MODEL_NAME = "fmlp"


@torch.no_grad()
def evaluate(model: FourierMLP, inputs: Tensor, truth: Tensor) -> tuple[float, float]:
    """``(rmse, max_abs_error)`` in Kelvin, over every node of every snapshot given."""
    error = model.field(inputs) - truth
    return float(error.pow(2).mean().sqrt()), float(error.abs().max())


def score(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """``(rmse, linf, peak_error_percent)`` for one power, in Kelvin.

    The peak is quoted relative to the temperature *rise*, not the absolute
    temperature, because the 298 K pedestal is not something a model can get wrong
    and including it would flatter every number by a factor of three.
    """
    error = predicted - truth
    rise = truth.max() - AMBIENT_TEMPERATURE
    peak = (predicted.max() - truth.max()) / rise * 100.0
    return (
        float(np.sqrt((error**2).mean())),
        float(np.abs(error).max()),
        float(peak),
    )


@torch.no_grad()
def report(model: FourierMLP, corpus: FourierCorpus, device, dtype, held: float) -> None:
    """The table this model is judged by: the box's floor, and the network's gap to it."""
    print(f"\n{'P':>6} {'':>6} {'RMSE':>10} {'Linf':>9} {'peak':>9}")
    print("-" * 45)
    for power in np.unique(corpus.powers):
        rows = corpus.powers == power
        inputs = torch.tensor(corpus.inputs[rows], dtype=dtype, device=device)
        predicted = model.field(inputs).cpu().numpy()

        for name, field in (("floor", corpus.floor[rows]), ("model", predicted)):
            rmse, linf, peak = score(field, corpus.truth[rows])
            tail = "HELD OUT" if power == held and name == "model" else ""
            print(
                f"{power:>5.0f}W {name:>6} {rmse:>9.3f}K {linf:>8.1f}K "
                f"{peak:>8.2f}%   {tail}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f"train.py {MODEL_NAME}", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--valid-dir",
        type=Path,
        default=DEFAULT_VALID_DIR,
        help="the held-out power the checkpoint is selected on",
    )
    parser.add_argument(
        "--energy-target",
        type=float,
        default=0.9999,
        help="the box must retain this fraction of EVERY power's energy (default "
        "0.9999 -> kx=19, ky=5, floor 0.53 K on the shipped sweep)",
    )
    parser.add_argument(
        "--box",
        type=int,
        nargs=2,
        default=None,
        metavar=("KX", "KY"),
        help="set the box by hand, overriding --energy-target",
    )
    parser.add_argument(
        "--detrend",
        action="store_true",
        help="split the x-wrap ramp off before transforming and predict it separately; "
        "collapses the high-|mx| tail the ramp would otherwise pay for",
    )
    parser.add_argument(
        "--no-derotate",
        dest="derotate",
        action="store_false",
        help="learn the raw coefficients rather than the ones in the frame that moves "
        "with the laser. Costs little at the snapshot times (0.421 K against 0.353 K) "
        "and everything between them; see models/fmlp/dataset.py",
    )
    parser.add_argument(
        "--norm",
        choices=("global", "per-coef"),
        default="global",
        help="global keeps Parseval, so the loss is the field's L2 error; per-coef "
        "standardises every coefficient, so the faintest mode counts as much as DC",
    )
    parser.add_argument("--scan-speed", type=float, default=SCAN_SPEED, help="v [m/s]")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument(
        "--epochs",
        type=int,
        default=20000,
        help="full-batch gradient steps; there are only 56 rows, so there is no batch",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--double", action="store_true", help="run in float64")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="defaults to checkpoints/fmlp/<run-name>.pt"
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
    torch.manual_seed(args.seed)

    train_paths = find_data_files(args.data_dir)
    valid_paths = find_data_files(args.valid_dir)

    train, architecture = build(
        train_paths,
        box=tuple(args.box) if args.box else None,
        energy_target=args.energy_target,
        detrend=args.detrend,
        derotate=args.derotate,
        speed=args.scan_speed,
        normalisation=args.norm,
    )
    # The held-out power reuses the training box and the training statistics: one
    # that chose its own would not be measuring the same model.
    valid, _ = build(valid_paths, reference=train, detrend=args.detrend,
                     derotate=args.derotate, speed=args.scan_speed)

    architecture["hidden_layers"] = tuple(args.hidden)
    architecture["input_scale"] = (
        float(train.inputs[:, 0].max()),
        float(train.inputs[:, 1].max()),
    )
    model = FourierMLP(**architecture).to(device=device, dtype=dtype)
    criterion = CoefficientMSELoss()

    inputs = torch.tensor(train.inputs, dtype=dtype, device=device)
    targets = torch.tensor(train.targets, dtype=dtype, device=device)
    val_inputs = torch.tensor(valid.inputs, dtype=dtype, device=device)
    val_truth = torch.tensor(valid.truth, dtype=dtype, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = args.run_name or make_run_name(MODEL_NAME, args.tag)
    run_dir = resolve_run_dir(args.logdir, run_name)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, MODEL_NAME, run_name)
    best = BestCheckpoint(checkpoint_path, mode="min")
    writer = SummaryWriter(log_dir=str(run_dir))

    bounds = torch.tensor(bounds_of(train_paths[0]), dtype=dtype)
    held = float(valid.powers[0])
    floor_rmse, floor_linf, floor_peak = score(valid.floor, valid.truth)
    kx, ky = train.box

    print(f"\n[setup] tensorboard run -> {run_dir}")
    print(f"[setup] checkpoint      -> {checkpoint_path}")
    print(f"[setup] device={device} dtype={dtype} params={count_parameters(model)}")
    print(
        f"[setup] box |mx|<={kx} |my|<={ky}, z kept whole -> {model.n_coefficient} "
        f"coefficients{f' + {model.n_ramp} ramp' if model.detrend else ''}"
    )
    print(f"[setup] derotate={args.derotate} detrend={args.detrend} norm={args.norm}")
    print(f"[setup] train={len(train)} snapshots, valid={len(valid)} at {held:.0f} W")
    print(
        f"[setup] truncation floor at {held:.0f} W: RMSE {floor_rmse:.3f} K, "
        f"Linf {floor_linf:.1f} K, peak {floor_peak:+.2f}%  <- the model cannot beat this"
    )

    progress = tqdm(
        range(1, args.epochs + 1),
        desc="fmlp",
        unit="ep",
        disable=args.no_progress,
        dynamic_ncols=True,
    )
    for epoch in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            rmse, worst = evaluate(model, val_inputs, val_truth)
            value = loss.detach().item()

            writer.add_scalar("loss/train", value, epoch)
            writer.add_scalar("val/rmse", rmse, epoch)
            writer.add_scalar("val/max_error", worst, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
            progress.set_postfix(loss=f"{value:.3e}", val=f"{rmse:.2f}K", best=f"{best.best:.2f}K")

            best.update(
                rmse,
                {
                    "model": model.state_dict(),
                    "architecture": architecture,
                    "bounds": bounds,
                    "val_rmse": rmse,
                    "epoch": epoch,
                },
                step=epoch,
            )
    progress.close()

    writer.add_hparams(
        {
            "lr": args.lr,
            "epochs": args.epochs,
            "hidden": str(tuple(args.hidden)),
            "energy_target": args.energy_target,
            "kx": kx,
            "ky": ky,
            "derotate": args.derotate,
            "detrend": args.detrend,
            "norm": args.norm,
        },
        {"hparam/val_rmse": best.best, "hparam/floor_rmse": floor_rmse},
    )
    writer.close()

    # The table below is the checkpoint's, not the last step's -- the two differ
    # whenever the run overfitted after its best epoch.
    model.eval()
    if not math.isinf(best.best):
        model.load_state_dict(load_checkpoint(checkpoint_path, map_location=device)["model"])

    print("\nAgainst the solver. 'floor' is the best the kept modes can do.")
    report(model, train, device, dtype, held)
    report(model, valid, device, dtype, held)

    print(f"\n[done] best val RMSE {best.best:.3f} K at {held:.0f} W "
          f"(floor {floor_rmse:.3f} K) -> {checkpoint_path}")
    print(f"[done] tensorboard --logdir {args.logdir}")
