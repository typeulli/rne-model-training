"""PKFDoN -- the melt-pool patch of the sequence DeepONet family.

:class:`~models._pathdon.SequenceDeepONet`, ungated, with its trunk reading
``(along, across, z)`` in the beam's travel frame. What makes it a patch is the
corpus, not the architecture -- exactly how ``pkdon`` relates to ``don``.
"""

from __future__ import annotations

from .._pathdon import DEFAULT_RADIUS, GATES, SequenceDeepONet

GATE = GATES["pkfdon"]

PKFDoN = SequenceDeepONet

__all__ = ["PKFDoN", "GATE", "DEFAULT_RADIUS"]
