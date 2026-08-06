"""Train RSJFDoN: :mod:`models.rjfdon` with the wake also relaxed into each corner.

Invoked through the top-level dispatcher::

    python train.py rsjfdon --iterations 140000 --batch-data 4096 \
        --hidden 64 64 64 --latent-dim 16 --cede-radius 2.5e-3

Everything is :mod:`models.jfdon`'s -- same corpus, same split, same network, same
objective, same loop -- except the beam state the gate reads, which differs in
three ways at once. Each is a separate flag and each is recorded in the
checkpoint.

**Held over dark travel.** ``raster`` spends 22% of its run travelling dark and
``nested_l`` 16%, and the plain beam frame follows the head there, to somewhere
with no melt pool in it. :func:`~models._pathdon.retouched_beam_state` holds the
position, heading and speed at their last lit values so the frame tracks the pool
rather than the head.

**Fading while it is held.** Holding ``lit`` at 1 says the pool is there, which it
is, and that it is as hot as it was under the beam, which it is not. The gate's
amplitude column ``q`` now decays as ``(tau / (dt + tau))^(3/2)`` from the instant
the laser cut out -- the time part of the three-dimensional heat kernel, shifted so
it starts at 1 rather than at infinity -- with ``dark_tau`` defaulting to the
pool's own diffusion time ``a^2 / 4 alpha = 0.196 s``. The speed is deliberately
*not* ramped with it: Rosenthal's series loses its ``exp(-v (r + xi) / 2 alpha)``
suppression as ``v`` falls and its normalised peak grows from 1.00 to 3.15, so
fading the amplitude while inflating the shape it multiplies would be two knobs
fighting over one number. ``q`` moves; ``v`` does not.

**Lagging through corners.** :func:`~models._pathdon.slewed_beam_state` starts each
lit segment pointing the way the previous lit segment did and relaxes to the new
heading as ``1 - exp(-dt / slew_tau)``. Without it the gate moves by the whole
normalised peak in one stored time at every turn, which the plate does not do.

The control for the pair of these is :mod:`models.rjfdon`; the control for the
slew alone is :mod:`models.sjfdon` against :mod:`models.jfdon`.

See :mod:`models.don` for the corpus, the holdout and the units.
"""

from __future__ import annotations

from .. import _pathdon

MODEL_NAME = "rsjfdon"


def main(argv: list[str] | None = None) -> None:
    _pathdon.run(MODEL_NAME, __doc__, argv)
