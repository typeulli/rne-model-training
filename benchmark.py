"""Time a trained model's full-field reconstruction: one ``(t, P)`` to every ``(x, y, z)``.

This is the query the downstream tooling actually makes -- "give me the whole
volume at this time and this laser power" -- so it is the one worth timing.
:meth:`~agent.BaseAgent.predict_of` already expresses it: ``[B, 2]`` of ``(t, P)``
in, ``[B, 1, D, H, W]`` of Kelvin out. Nothing here reaches past that contract, so
any model under ``models/`` can be timed without changing a line.

Two numbers come out of it. The wall-clock of one full-field call is what a caller
waits for; the throughput in nodes per second is what survives a change of grid
resolution, and is the fair way to compare a model timed at one ``--shape``
against another timed at a different one.

The first call is excluded by default (``--warmup``). It pays for CUDA context
setup, cuDNN's algorithm search, and building the reconstruction grid that
:attr:`~agent.BaseAgent.grid` then caches for the life of the agent -- none of
which a second call repeats, so folding them into the average would flatter a
one-shot use and mislead a repeated one.

Examples::

    python benchmark.py --model mlp --checkpoint checkpoints/mlp/mlp-small-20260713-150632.pt
    python benchmark.py --model gmlp --checkpoint best.pt --repeats 50 --device cpu
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from agent import BaseAgent, peak_corrected
from dataset import DEFAULT_FIELD_SHAPE
from models import available_models, build_agent
from utils import count_parameters, load_checkpoint, resolve_device


def synchronise(device: torch.device) -> None:
    """Block until the device has finished the work already queued on it.

    CUDA kernels are launched asynchronously, so without this the timer would
    measure how long it takes to *enqueue* the forward pass rather than to run
    it -- reporting a full-field reconstruction as microseconds of Python.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def time_field(
    agent: BaseAgent, time_s: float, power: float, repeats: int, warmup: int
) -> list[float]:
    """Return the wall-clock of ``repeats`` full-field reconstructions, in seconds."""
    query = [[time_s, power]]

    for _ in range(warmup):
        agent.predict_of(query)
    synchronise(agent.device)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        agent.predict_of(query)
        synchronise(agent.device)
        samples.append(time.perf_counter() - start)
    return samples


def report(name: str, samples: list[float], nodes: int) -> None:
    """Print the median call, its spread, and the throughput it implies."""
    median = statistics.median(samples)
    fastest, slowest = min(samples), max(samples)
    spread = statistics.stdev(samples) if len(samples) > 1 else 0.0

    print(
        f"{name:8s} {median * 1e3:8.2f} ms/field  "
        f"(min {fastest * 1e3:7.2f}, max {slowest * 1e3:7.2f}, sd {spread * 1e3:6.2f})  "
        f"{nodes / median / 1e6:7.2f} M nodes/s  {median / nodes * 1e9:6.1f} ns/node"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default="mlp", choices=available_models(), help="which model to load"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--power", type=float, default=200.0, help="P in W")
    parser.add_argument("--time", type=float, default=1.2, help="t in s")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=list(DEFAULT_FIELD_SHAPE),
        metavar=("D", "H", "W"),
        help="reconstruction grid (nz, ny, nx); defaults to the simulation's own",
    )
    parser.add_argument("--repeats", type=int, default=20, help="timed calls")
    parser.add_argument(
        "--warmup", type=int, default=3, help="untimed calls first; see the module docstring"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="query points per forward pass; defaults to the agent's own",
    )
    parser.add_argument(
        "--pkcorrect",
        nargs="?",
        const="",
        default=None,
        metavar="CHECKPOINT",
        help="time the field with a cpkmlp patch pasted over its peak; takes a "
        "cpkmlp checkpoint, or nothing for the most recent under checkpoints/cpkmlp/",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    shape = (args.shape[0], args.shape[1], args.shape[2])
    nodes = shape[0] * shape[1] * shape[2]

    agent = build_agent(args.model, args.checkpoint, shape=shape, device=device)
    if args.chunk is not None:
        agent.chunk = args.chunk

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    print(f"[load] {args.checkpoint.name}: val RMSE {checkpoint['val_rmse']:.3f} K")

    if args.pkcorrect is not None:
        agent = peak_corrected(agent, args.pkcorrect or None, device=device)

    print(
        f"[setup] device={device} dtype={agent.dtype} "
        f"params={count_parameters(agent.model)} chunk={agent.chunk}"
    )
    print(
        f"[setup] field {shape} = {nodes} nodes at (t, P) = ({args.time} s, {args.power} W)"
    )
    print(f"[setup] {args.warmup} warmup + {args.repeats} timed calls\n")

    samples = time_field(agent, args.time, args.power, args.repeats, args.warmup)
    report(args.model + ("+pk" if args.pkcorrect is not None else ""), samples, nodes)


if __name__ == "__main__":
    main()
