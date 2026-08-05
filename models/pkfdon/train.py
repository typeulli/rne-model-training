"""Train PKFDoN: the melt pool as a history, in the frame that rides with the beam.

Invoked through the top-level dispatcher::

    python train.py pkfdon --iterations 140000 --hidden 64 64 64 --latent-dim 16

``models/pkdon`` is to ``don`` what this is to ``fdon``: the same network as the
family's ungated member, fitted to the beam's neighbourhood alone, in coordinates
that ride and turn with it -- and, here, returning the whole temperature history
at each offset rather than one number at one instant.

**What a row means.** "The temperature 1 mm behind the beam, over the scan." That
is a real quantity and not the same one ``pkdon`` learns: a quasi-steady pool
would make it a flat line, so everything that is *not* flat in it is the
accumulation, the corners, and the cooling while the laser travels dark.

**Where the targets come from.** ``pkdon`` keeps whichever nodes fall in the window
at each snapshot, so its offsets differ slightly from instant to instant. A history
needs the *same* offset at every time and no node is, so the lattice is fixed in
the beam frame and the field is interpolated onto it -- bilinear in ``(x, y)``,
exact in ``z``. On this grid that is worth ~11 K at worst and 0.2 K in RMS
(~5 K inside the pool), measured rather than assumed; see
:class:`~models._pathdon.SequencePatchCorpus`. It is the only interpolated target
in this repository, and it touches training only: the paste is scored against raw
solver nodes.

**Where it is pasted.** :class:`~models._pathdon.GenericPeakCorrectedAgent`, reached
from ``visualize_don.py --pkcorrect``. The window is the same square in the same
frame the patch was fitted in, so there is no anchor to estimate.

See :mod:`models.fdon` for the output shape and :mod:`models.don` for the corpus.
"""

from __future__ import annotations

from .. import _pathdon

MODEL_NAME = "pkfdon"


def main(argv: list[str] | None = None) -> None:
    _pathdon.run(MODEL_NAME, __doc__, argv)
