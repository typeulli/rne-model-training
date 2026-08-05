"""Train JDoN: :mod:`models.don` gated by Rosenthal's travelling-source field.

Invoked through the top-level dispatcher::

    python train.py jdon --iterations 40000

``G = j + p`` where ``j`` is the moving-source solution -- imaged for the plate's
insulated top and ambient bottom, softened at the origin, normalised to a unit
peak -- exactly as ``models/cjmlp/laser.py`` builds it. Where a Gaussian is
symmetric and has no wake, this has both: ahead of the beam the field dies within
``alpha / v``, behind it a ``1/r`` tail is left.

The one thing generalised off ``cjmlp`` is orientation. Rosenthal's ``xi`` is the
displacement measured *down the direction of travel*, which on a straight pass is
``x - x_l(t)`` and on these paths is not. Here it is the displacement projected
on the beam's own velocity, so the wake turns when the path turns -- which for
``spiral`` and ``nested_l`` is most of what there is to get right.

See :mod:`models.don` for the corpus, the holdout and the units.
"""

from __future__ import annotations

from .. import _pathdon

MODEL_NAME = "jdon"


def main(argv: list[str] | None = None) -> None:
    _pathdon.run(MODEL_NAME, __doc__, argv)
