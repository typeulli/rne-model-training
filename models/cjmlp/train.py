"""Train CJMLP on ``(t, z, y, x) -> T``, gated by a moving-source field.

Invoked through the top-level dispatcher::

    python train.py cjmlp --data-dir ../rne-am-simulation/data/train_175W \
        --hidden 64 64 64 --exclude 1 --tag 64x3-175W-ex1

Identical to ``models/cgmlp/train.py`` -- same loop, same objective, same
hyperparameters, same learnable ``--gate-offset`` -- except for what the gate is.
``cgmlp`` multiplies the network by a Gaussian on the beam; this multiplies it by
Rosenthal's travelling-source solution, imaged for the plate's two faces and
softened at the origin (:mod:`models.cjmlp.laser`). Everything else is held fixed
so the two runs can be compared directly, and the difference between them is what
the prior is worth when it is the shape of the *field* rather than of the source.

The learned gate offset ``p`` is the quantity to watch, exactly as in ``cgmlp``:
it is the floor the gate relaxes to away from the beam, so it says how far the fit
had to back away from the prior it was given. A prior that already contains the
wake should need less of it than one that dies off symmetrically.

``--soft-core`` is the beam radius, 1.70 mm, and is worth understanding before it
is changed. Fitting it to the field's shape instead gives 0.45 mm -- a visibly
better envelope that trains to 2.149 K against this one's 1.204 K. The module
docstring of :mod:`models.cjmlp.laser` has the runs and the reason; the short
version is that the network divides by the gate, so a sharp gate is a worse factor
than a smooth one even when it is the better picture.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MM, DEFAULT_DATA_DIR, SimulationDataset
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

from .dataset import CJMLPDataset
from .laser import DIFFUSIVITY, IMAGES, SOFT_CORE, THICKNESS
from .loss import ScaledMSELoss
from .model import ControlSourceMLP

MODEL_NAME = "cjmlp"

# Matches models/gpidon_old/loss.py; only used to centre the output, so an exact
# value is not critical.
AMBIENT_TEMPERATURE = 298.0

# Sub-ambient rows (~5%, minimum 289.8 K) are a solver artefact: with heating
# only, T can never fall below T_amb. Left untouched by default rather than
# silently editing the source data.
CLIP_SUBAMBIENT = False

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
    model: ControlSourceMLP, inputs: Tensor, target: Tensor, chunk: int = 65536
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
        "--exclude",
        type=int,
        default=0,
        metavar="N",
        help="drop the first N time steps of the corpus before training. Applied "
        "before the split, so they are absent from the validation half too",
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
        "--checkpoint", type=Path, default=None, help="defaults to checkpoints/cjmlp/<run-name>.pt"
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
    parser.add_argument(
        "--diffusivity",
        type=float,
        default=DIFFUSIVITY,
        metavar="ALPHA",
        help="thermal diffusivity in m^2/s; with the scan speed it sets how far "
        "the gate's wake reaches, as 2*alpha/v",
    )
    parser.add_argument(
        "--soft-core",
        type=float,
        default=SOFT_CORE / MM,
        metavar="MM",
        help="sigma of the gate in mm: the length that removes the point source's "
        "singularity. Fitted at 0.45 mm, NOT the 1.70 mm beam radius; see the "
        "module docstring of models/cjmlp/laser.py",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=THICKNESS / MM,
        metavar="MM",
        help="plate thickness d in mm, the depth at which the image series pins "
        "the gate to ambient",
    )
    parser.add_argument(
        "--images",
        type=int,
        default=IMAGES,
        metavar="N",
        help="terms of the image series kept either side of the real source",
    )
    parser.add_argument(
        "--gate-offset",
        type=float,
        default=0.5,
        help="initial value of the learnable `p` in the gate G = j + p; it is only "
        "the starting point, training moves it",
    )
    return parser.parse_args(argv)


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
        clip_below=AMBIENT_TEMPERATURE if CLIP_SUBAMBIENT else None,
        exclude_steps=args.exclude,
    )
    domain = corpus.domain
    temperature_rise = corpus.temperature_rise(AMBIENT_TEMPERATURE)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    train_split, val_split = corpus.split(args.val_fraction, generator)
    sampler = CJMLPDataset(train_split, generator)
    val_inputs, val_target = CJMLPDataset(val_split, generator).all()

    input_mean, input_scale = sampler.normalisation()
    architecture = dict(
        hidden_layers=tuple(args.hidden),
        activation=ACTIVATIONS[args.activation],
        input_mean=input_mean,
        input_scale=input_scale,
        temperature_offset=AMBIENT_TEMPERATURE,
        temperature_scale=temperature_rise,
        diffusivity=args.diffusivity,
        soft_core=args.soft_core * MM,
        thickness=args.thickness * MM,
        images=args.images,
        gate_offset=args.gate_offset,
    )
    model = ControlSourceMLP(**architecture).to(device=device, dtype=dtype)
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
    print("[setup] P is NOT an input: the model sees (t, z, y, x) and must average over them")
    print(
        f"[setup] moving-source gate: alpha={args.diffusivity:g} m^2/s "
        f"sigma={args.soft_core:g} mm d={args.thickness:g} mm images=+/-{args.images}"
    )
    print(f"[setup] gate_offset p0={args.gate_offset:g} (learnable)")
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
        gate_offset = float(model.gate_offset.detach())
        writer.add_scalar("val/rmse", rmse, step)
        writer.add_scalar("val/max_error", worst, step)
        writer.add_scalar("gate/offset", gate_offset, step)
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
        current_lr = learning_rate if scheduler is None else scheduler.get_last_lr()[0]
        print(
            f"[epoch {epoch:3d}/{args.epochs}] train={running:.4e} "
            f"val_rmse={rmse:8.3f}K val_max={worst:9.3f}K p={gate_offset:+7.4f} "
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
            "diffusivity": args.diffusivity,
            "soft_core": args.soft_core,
            "gate_offset_init": args.gate_offset,
        },
        {"hparam/val_rmse": best.best},
    )
    writer.close()
    print(f"[done] learned gate offset p = {float(model.gate_offset.detach()):+.4f}")
    print(f"[done] best val RMSE {best.best:.3f} K -> {checkpoint_path}")
    print(f"[done] tensorboard --logdir {args.logdir}")
