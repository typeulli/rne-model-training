"""GFDoN -- the Gaussian-gated member of the sequence DeepONet family.

``fdon`` with ``G = g + p``, a unit-peak Gaussian on the beam -- evaluated
at every stored time, because the answer is a history and the beam has
moved between each of its entries.

The architecture is :class:`~models._pathdon.SequenceDeepONet`, shared by the
three so a difference between their numbers is a difference between their gates.
"""

from __future__ import annotations

from .._pathdon import GATES, SequenceDeepONet

GATE = GATES["gfdon"]

GFDoN = SequenceDeepONet

__all__ = ["GFDoN", "GATE"]
