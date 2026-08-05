"""Train FDoN: a DeepONet whose answer is a temperature *history*.

Invoked through the top-level dispatcher::

    python train.py fdon --iterations 140000 --hidden 64 64 64 --latent-dim 128

The floor of the sequence family: nothing multiplies the inner product,
so whatever ``gfdon`` and ``jfdon`` gain, they gain from their gate.

**How it differs from ``don``.** The trunk takes ``(x, y, z)`` and no clock;
the output is ``[B, Nt]``, the temperature at that point at every stored time.
Time stops being a coordinate the network is queried at and becomes the axis its
answer lives on -- which is the shape the input already had, since the branch
reads the toolpath as a time series over 64 fixed sensors. A forward pass now
costs one evaluation per point instead of ``Nt`` of them, and a point's history
comes out of a single latent, so it cannot be self-inconsistent across time.

**What it gives up.** The answer exists only on the corpus's grid -- 27 times,
0.4 s apart. ``don`` is continuous in ``t`` because ``t`` goes into its trunk;
this is not. :class:`~models._pathdon.SequenceAgent` interpolates linearly
between stored times so the pointwise ``[B, 5]`` contract still holds, and that
interpolation is the agent's, not the model's.

**The ragged clock.** The four scans take 9.20 s to 10.78 s, so they store 24 to
27 snapshots. The output is 27 wide and a shorter run is a prefix of it, carrying
a mask: the loss never scores a time the solver did not compute for that run.

See :mod:`models.don` for the corpus, the holdout and the units.
"""

from __future__ import annotations

from .. import _pathdon

MODEL_NAME = "fdon"


def main(argv: list[str] | None = None) -> None:
    _pathdon.run(MODEL_NAME, __doc__, argv)
