"""DoN -- the ungated member of the toolpath-conditioned DeepONet family.

A DeepONet over ``(toolpath, P)`` with nothing multiplying the inner
product. The floor the other two are measured against.

The architecture itself is :class:`~models._pathdon.PathDeepONet`, shared by the
three so that a difference between their numbers is a difference between their
gates. This module names it, which is what the rest of the repository imports.
"""

from __future__ import annotations

from .._pathdon import GATES, PathDeepONet

GATE = GATES["don"]

DoN = PathDeepONet

__all__ = ["DoN", "GATE"]
