"""RPKFDoN -- the melt-pool patch member of the retouched sequence family.

:class:`~models._pathdon.SequenceDeepONet`, as :mod:`models.pkfdon` uses it. What
makes it retouched is the beam frame, not the architecture: see
:func:`~models._pathdon.retouched_beam_state`.
"""

from __future__ import annotations

from .._pathdon import GATES, SequenceDeepONet

GATE = GATES["rpkfdon"]

RPKFDoN = SequenceDeepONet

__all__ = ["RPKFDoN", "GATE"]
