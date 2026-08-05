"""FDoN -- the ungated member of the sequence DeepONet family.

The floor of the sequence family: nothing multiplies the inner product,
so whatever ``gfdon`` and ``jfdon`` gain, they gain from their gate.

The architecture is :class:`~models._pathdon.SequenceDeepONet`, shared by the
three so a difference between their numbers is a difference between their gates.
"""

from __future__ import annotations

from .._pathdon import GATES, SequenceDeepONet

GATE = GATES["fdon"]

FDoN = SequenceDeepONet

__all__ = ["FDoN", "GATE"]
