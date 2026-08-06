"""Train SJFDoN: :mod:`models.jfdon` with the wake relaxed into each corner.

Invoked through the top-level dispatcher::

    python train.py sjfdon --iterations 140000 --batch-data 4096 \
        --hidden 64 64 64 --latent-dim 16 --cede-radius 2.5e-3

Everything is :mod:`models.jfdon`'s -- same corpus, same split, same network,
same objective, same loop -- except the direction Rosenthal's wake is drawn down
while the beam is coming out of a turn.

**What changes.** ``G = j + p`` points its tail along the beam's velocity, so at a
node where the path turns the tail turns with it, all at once. Measured on the
beam-centred window, the gate is identical from one stored time to the next down
a straight track and moves by the *entire* normalised peak across a corner: 0.98
at 90 degrees, 1.00 at 180. That happens at 8 of ``serpentine``'s steps, 8 of
``spiral``'s, and 13 of ``nested_l``'s -- and the plate does no such thing, since
the trail behind a beam that has just turned is still lying down the leg it came
in on.

:func:`~models._pathdon.slewed_beam_state` starts each lit segment pointing the
way the previous lit segment did and relaxes to the new heading as
``1 - exp(-dt / slew_tau)``, with ``slew_tau`` defaulting to the pool's own
diffusion time ``a^2 / 4 alpha = 0.196 s``. The speed is untouched: taking it to
zero through the corner would inflate the series' normalised peak from 1.00 to
3.15 and amplify exactly what this is meant to smooth.

Nothing else in the family is slewed, so ``jfdon`` is the control and the pair
differ in this alone.

See :mod:`models.don` for the corpus, the holdout and the units.
"""

from __future__ import annotations

from .. import _pathdon

MODEL_NAME = "sjfdon"


def main(argv: list[str] | None = None) -> None:
    _pathdon.run(MODEL_NAME, __doc__, argv)
