"""RSJFDoN -- retouched and slewed, on the moving-source gate.

:class:`~models._pathdon.SequenceDeepONet`, as :mod:`models.jfdon` uses it. Both
differences are in the beam state the gate reads, not in the architecture: see
:func:`~models._pathdon.retouched_beam_state` and
:func:`~models._pathdon.slewed_beam_state`.
"""

from __future__ import annotations

from .._pathdon import GATES, SequenceDeepONet

GATE = GATES["rsjfdon"]

RSJFDoN = SequenceDeepONet

__all__ = ["RSJFDoN", "GATE"]
