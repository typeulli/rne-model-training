"""GDoN -- the Gaussian-gated member of the toolpath-conditioned DeepONet family.

``don`` with ``G = g + p``: a unit-peak Gaussian riding on the beam,
placed by the row's own toolpath rather than by a hardcoded straight pass.

The architecture itself is :class:`~models._pathdon.PathDeepONet`, shared by the
three so that a difference between their numbers is a difference between their
gates. This module names it, which is what the rest of the repository imports.
"""

from __future__ import annotations

from .._pathdon import GATES, PathDeepONet

GATE = GATES["gdon"]

GDoN = PathDeepONet

__all__ = ["GDoN", "GATE"]
