"""PKDoN -- the melt-pool patch of the toolpath-conditioned DeepONet family.

The same :class:`~models._pathdon.PathDeepONet` the other three use, ungated. What
makes it a patch is not the architecture but the corpus it is fitted to and the
coordinates it is fitted in: the beam's neighbourhood, in the beam's travel frame.
See :mod:`models.pkdon.train`.
"""

from __future__ import annotations

from .._pathdon import DEFAULT_RADIUS, GATES, PathDeepONet

GATE = GATES["pkdon"]

PKDoN = PathDeepONet

__all__ = ["PKDoN", "GATE", "DEFAULT_RADIUS"]
