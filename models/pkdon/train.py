"""Train PKDoN, the melt-pool patch, in the frame that rides with the beam.

Invoked through the top-level dispatcher::

    python train.py pkdon --iterations 40000
    python train.py pkdon --radius 3e-3

``models/don/train.py`` with one thing changed: the corpus is
:class:`~models._pathdon.PatchCorpus` instead of the whole plate, so every row is
a node within ``--radius`` of where the beam is at that instant, with its position
rewritten as ``(along, across)`` in the beam's direction of travel. Same network,
same branch, same objective, same loop, same hyperparameters -- exactly the
relation ``models/cpkmlp`` has to ``models/cmlp``.

**Why a patch is needed.** The pool is a few millimetres of a 20 mm square, so it
is a fraction of a percent of the points a mean squared error averages over. The
whole-plate models are worth 1.7-2.2 K RMSE on the held-out power and still miss
the peak by tens of Kelvin; the error is almost entirely inside a beam radius of
the source. Fitting that neighbourhood on its own gives it the whole of the
network's capacity instead of a rounding error's worth.

**Why the beam frame, and why rotated.** A quasi-steady melt pool is stationary in
the frame of the source, which is what makes it a function a network can learn at
all. ``cpkmlp`` gets that frame from three calibrated numbers and a clock, and it
does not rotate because its beam only ever travels ``+x``. These paths turn
corners, so the frame is rotated into the direction of travel -- otherwise the
wake would point a different way in every snapshot and the network would have to
infer the heading from the clock, which is the one thing the frame exists to
remove.

**Where it is pasted.** :class:`~models._pathdon.PathPeakCorrectedAgent`, reached
from ``visualize_don.py --pkcorrect``. The window is the same square in the same
frame, so fitted region and pasted region coincide by construction -- there is no
anchor to estimate and nothing to be a node out by.

See :mod:`models.don` for the corpus, the holdout and the units.
"""

from __future__ import annotations

from .. import _pathdon

MODEL_NAME = "pkdon"


def main(argv: list[str] | None = None) -> None:
    _pathdon.run(MODEL_NAME, __doc__, argv)
