"""Train the CPKMLP patch on ``(t, z, dy, dx) -> T`` in the beam frame.

Invoked through the top-level dispatcher::

    python train.py cpkmlp --data-dir ../data/train_175W \
        --hidden 64 64 64 --exclude 1 --tag 175W

``models/cmlp/train.py`` with one thing changed: the corpus is
:func:`models.cpkmlp.dataset.build` instead of the whole plate, so every row is a
node within ``--radius`` of where the beam is at that instant and its ``x`` and
``y`` are measured from the beam. Same stack, same objective, same loop, same
hyperparameters.

The beam path is stored in the checkpoint under ``anchor``, the way ``cpimlp``
stores the power its physics was written against. It is not a convenience: the
patch is *pasted* in the same frame it is fitted in, and a frame that lived only
in a module constant could be recalibrated between the two. Stored, it cannot.
:mod:`models.cpkmlp.dataset` gives the longer argument for anchoring on the beam
rather than on the hottest node.

**Give it a single-power directory.** The power is not among the inputs, as in
every ``c``-prefixed model, so with the default seven-power corpus the same
``(t, z, dy, dx)`` would carry seven different temperatures and the best the
network could do is their average. Working in the beam frame does not change that
-- it sharpens it, since the pool is exactly where the powers differ most.

``--exclude 1`` drops ``t = 0``, where the plate is still uniformly at ambient.
Nothing here requires it -- the beam has a position at ``t = 0`` like any other
time -- but the base models this patch is pasted onto are trained the same way,
and a patch fitted over a wider clock than the field it corrects would be answering
questions its host cannot ask.

The validation RMSE this reports is over the *patch only*, so it is not
comparable with ``cmlp``'s number over the whole plate -- it is the error on the
hardest few millimetres of it, which is a harder question and should be expected
to read worse. What it is comparable to is ``cmlp``'s error restricted to the same
window, which is what ``visualize.py --pkcorrect`` and ``scanline.py --pkcorrect``
put side by side.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MM, DEFAULT_DATA_DIR, find_data_files
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

from .dataset import DEFAULT_RADIUS, CMLPDataset, build
from .laser import BeamPath
from .loss import ScaledMSELoss
from .model import PeakMLP

MODEL_NAME = "cpkmlp"

# Matches models/gpidon_old/loss.py; only used to centre the output, so an exact
# value is not critical.
AMBIENT_TEMPERATURE = 298.0

# Hidden-layer activations selectable from the command line. The chosen class is
# stored in the architecture dict, so the agent rebuilds the network with the same
# one rather than the constructor default.
ACTIVATIONS = {
    "silu": torch.nn.SiLU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
}


@torch.no_grad()
def evaluate(
    model: PeakMLP, inputs: Tensor, target: Tensor, chunk: int = 65536
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
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS / MM,
        metavar="MM",
        help="half-width in mm of the window kept around the beam, in x and in y; "
        "z is kept whole. The nodes it selects set the checkpoint's bounds, so a "
        "patch is never pasted wider than the data it was fitted on",
    )
    parser.add_argument(
        "--exclude",
        type=int,
        default=0,
        metavar="N",
        help="drop the first N time steps of every file. t = 0 is well defined "
        "here; --exclude 1 matches how the models this patch corrects were trained",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=1000,
        help="batches per epoch; sampling is with replacement, so this is a free choice",
    )
    add_optimizer_args(parser)
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[256, 256, 256, 256], help="hidden layer widths"
    )
    parser.add_argument(
        "--activation",
        choices=tuple(ACTIVATIONS),
        default="silu",
        help="hidden-layer activation; stored in the checkpoint so the agent "
        "rebuilds the network with the same one",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--scalar-every", type=int, default=25, help="TensorBoard loss cadence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--double", action="store_true", help="run in float64")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="defaults to checkpoints/cpkmlp/<run-name>.pt"
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="warm-start from an existing checkpoint's weights before training; the "
        "architecture (--hidden) must match. Used to continue an Adam-trained model "
        "with L-BFGS instead of starting L-BFGS from a random init",
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
    seed_everything(args.seed)

    radius = args.radius * MM
    beam = BeamPath()
    corpus = build(
        find_data_files(args.data_dir),
        radius=radius,
        beam=beam,
        exclude=args.exclude,
        dtype=dtype,
        device=device,
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
        activation=ACTIVATIONS[args.activation],
        input_mean=input_mean,
        input_scale=input_scale,
        temperature_offset=AMBIENT_TEMPERATURE,
        temperature_scale=temperature_rise,
    )
    model = PeakMLP(**architecture).to(device=device, dtype=dtype)
    if args.init_from is not None:
        state = load_checkpoint(args.init_from, map_location=device)
        model.load_state_dict(state["model"])
        print(f"[setup] warm-started from {args.init_from} (val RMSE {state['val_rmse']:.3f} K)")
    criterion = ScaledMSELoss(scale=temperature_rise)

    total_steps = args.epochs * args.steps_per_epoch
    learning_rate = resolve_lr(args)
    optimizer = build_optimizer(model.parameters(), args, learning_rate)
    use_lbfgs = args.optimizer == "lbfgs"
    # L-BFGS *requires* a fixed objective; adam merely allows one.
    freeze_batch = use_lbfgs or args.freeze_batch
    # L-BFGS descends one fixed function, so its batch is drawn once and reused;
    # cosine-annealing its step size on top of the line search would only fight
    # the line search, so the schedule is Adam's alone.
    scheduler = (
        None
        if use_lbfgs
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
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
    print(
        f"[setup] the beam neighbourhood only: |dx|,|dy| <= {args.radius:g} mm about "
        f"x_l(t) = {beam.start_x / MM:g} + {beam.speed / MM:g} t mm, "
        f"y = {beam.y / MM:g} mm, z whole"
    )
    print("[setup] P is NOT an input: give this a single-power --data-dir")
    print(f"[setup] train={len(train_split)} val={len(val_split)}")
    print(f"[setup] optimizer={args.optimizer} lr={learning_rate:g} activation={args.activation}")
    if use_lbfgs:
        print(
            f"[setup] L-BFGS on ONE fixed batch of {args.lbfgs_batch} rows "
            f"(history={args.lbfgs_history}, max_iter={args.lbfgs_max_iter}); no LR schedule"
        )
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
            inputs, target = fixed_batch if freeze_batch else sampler.batch(args.batch_size)

            def closure() -> Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                return loss

            if use_lbfgs:
                # L-BFGS re-evaluates the closure several times per step to run
                # its line search, so it -- not the caller -- drives the loop.
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
                scheduler.step()

            # .item() synchronises with the GPU, so only pull the scalars we log.
            if step % args.scalar_every == 0 or step == 1:
                value = loss.detach().item()
                running = value
                current_lr = learning_rate if scheduler is None else scheduler.get_last_lr()[0]
                writer.add_scalar("loss/train", value, step)
                writer.add_scalar("lr", current_lr, step)
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
                "radius": (radius, radius),
                "anchor": beam.as_dict(),
                "val_rmse": rmse,
                "epoch": epoch,
            },
            step=step,
        )
        current_lr = learning_rate if scheduler is None else scheduler.get_last_lr()[0]
        print(
            f"[epoch {epoch:3d}/{args.epochs}] train={running:.4e} "
            f"val_rmse={rmse:8.3f}K val_max={worst:9.3f}K "
            f"lr={current_lr:.2e}{'  *' if improved else ''}"
        )

    writer.add_hparams(
        {
            "optimizer": args.optimizer,
            "lr": learning_rate,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.lbfgs_batch if use_lbfgs else args.batch_size,
            "hidden": str(tuple(args.hidden)),
            "activation": args.activation,
            "radius": args.radius,
        },
        {"hparam/val_rmse": best.best},
    )
    writer.close()
    print(f"[done] best val RMSE {best.best:.3f} K -> {checkpoint_path}")
    print(f"[done] tensorboard --logdir {args.logdir}")
