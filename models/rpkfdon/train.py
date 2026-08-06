"""Train RPKFDoN: :mod:`models.pkfdon` with the beam frame held over dark travel.

Invoked through the top-level dispatcher::

    python train.py rpkfdon --iterations 140000 --batch-data 4096         --hidden 64 64 64 --latent-dim 16 --cede-radius 2.5e-3

Everything is :mod:`models.pkfdon`'s -- same corpus, same split, same network,
same objective, same loop -- except where the beam is taken to be while the laser
is off.

**What changes.** ``raster`` spends 22% of its run travelling dark and
``nested_l`` 16%. Over those stretches the head is moving to the next track at
30 mm/s and depositing nothing, and the plain beam frame follows it: the window
rides along to somewhere with no melt pool in it, and the gate multiplies by
``lit = 0`` so the models have no shape at all for the pool that is in fact
sitting where the laser was last on, cooling. That pool is real and it is a
sixth to a fifth of two of the four runs.

:func:`~models._pathdon.retouched_beam_state` holds the position, the heading and
the speed at their last lit values and keeps ``lit`` at 1, so the frame tracks
the pool rather than the head. Nothing is held before the first lit instant or
after the scan ends -- there is no pool yet in the one case and the run is over
in the other.

``serpentine`` and ``spiral`` never lift, so for them this model is
:mod:`models.pkfdon` exactly.

See :mod:`models.don` for the corpus, the holdout and the units.
"""

from __future__ import annotations

from .. import _pathdon

MODEL_NAME = "rpkfdon"


def main(argv: list[str] | None = None) -> None:
    _pathdon.run(MODEL_NAME, __doc__, argv)
