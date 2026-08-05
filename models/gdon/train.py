"""Train GDoN: :mod:`models.don` with a Gaussian riding on the beam as the gate.

Invoked through the top-level dispatcher::

    python train.py gdon --iterations 40000
    python train.py gdon --gaussian-exponent-scale 2 --gate-offset 0.5

``T = T_amb + dT * (<branch, trunk> * G + c)`` with ``G = g + p``, ``g`` the
unit-peak Gaussian on the beam and ``p`` learnable. Identical to ``don`` in every
other respect -- same corpus, same split, same subnetworks, same objective, same
loop -- so the difference between the two runs is the gate.

Unlike ``models/gmlp`` and ``models/gdon_old``, whose ``laser.py`` can only
evaluate ``LASER_START_X + SCAN_SPEED * t``, the Gaussian here is placed by the
row's own toolpath at the row's own time. Over a segment travelled with the laser
off it is zero, and the fit falls back on ``p``.

See :mod:`models.don` for the corpus, the holdout and the units.
"""

from __future__ import annotations

from .. import _pathdon

MODEL_NAME = "gdon"


def main(argv: list[str] | None = None) -> None:
    _pathdon.run(MODEL_NAME, __doc__, argv)
