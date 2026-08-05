"""JFDoN -- the moving-source-gated member of the sequence DeepONet family.

``fdon`` with ``G = j + p``, Rosenthal's travelling source, oriented by the
beam's own velocity at each stored time so the wake turns as the path does.

The architecture is :class:`~models._pathdon.SequenceDeepONet`, shared by the
three so a difference between their numbers is a difference between their gates.
"""

from __future__ import annotations

from .._pathdon import GATES, SequenceDeepONet

GATE = GATES["jfdon"]

JFDoN = SequenceDeepONet

__all__ = ["JFDoN", "GATE"]
