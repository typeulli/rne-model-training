"""The network CPKMLP fits: CMLP's, with two of its columns re-read as offsets.

:class:`~models.cmlp.model.ControlMLP` maps ``[B, 4]`` of ``(t, z, y, x)`` to a
temperature. Nothing in it inspects what the last two columns *mean* -- they are
normalised by the buffers the run stored and handed to a dense stack -- so
feeding it ``(t, z, dy, dx)``, the same coordinates measured from the peak rather
than from the origin, needs no change to the architecture at all. What changes is
the corpus (:mod:`models.cpkmlp.dataset`) and the normalisation that follows from
it: ``dx`` and ``dy`` span a 5 mm window centred on zero instead of the plate.

This is the one place in ``models/`` that imports another model rather than
copying it. The others copy deliberately, so an experiment on one cannot shift
another underneath it; here the sharing is the point -- ``cpkmlp`` is ``cmlp``
restricted to the peak, and if the two stacks ever stopped being the same stack
the comparison between them would stop meaning anything.
"""

from __future__ import annotations

from models.cmlp.model import INPUT_DIM, INPUT_NAMES, ControlMLP

# The same class under the name this package refers to it by. The alias is not a
# subclass: a checkpoint's `architecture` dict is the constructor's keyword
# arguments, and those are ControlMLP's.
PeakMLP = ControlMLP

__all__ = ["PeakMLP", "ControlMLP", "INPUT_DIM", "INPUT_NAMES"]
