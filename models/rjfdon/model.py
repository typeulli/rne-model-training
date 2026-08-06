"""RJFDoN -- the moving-source-gated member of the retouched sequence family.

:class:`~models._pathdon.SequenceDeepONet`, as :mod:`models.jfdon` uses it. What
makes it retouched is the beam frame, not the architecture: see
:func:`~models._pathdon.retouched_beam_state`.
"""

from __future__ import annotations

from .._pathdon import GATES, SequenceDeepONet

GATE = GATES["rjfdon"]

RJFDoN = SequenceDeepONet

__all__ = ["RJFDoN", "GATE"]
