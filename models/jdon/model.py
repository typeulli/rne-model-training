"""JDoN -- the moving-source-gated member of the toolpath-conditioned DeepONet family.

``don`` with ``G = j + p``: Rosenthal's travelling source, imaged for the
plate's two faces and oriented by the beam's own velocity, so the wake
turns when the path turns.

The architecture itself is :class:`~models._pathdon.PathDeepONet`, shared by the
three so that a difference between their numbers is a difference between their
gates. This module names it, which is what the rest of the repository imports.
"""

from __future__ import annotations

from .._pathdon import GATES, PathDeepONet

GATE = GATES["jdon"]

JDoN = PathDeepONet

__all__ = ["JDoN", "GATE"]
