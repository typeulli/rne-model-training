"""SJFDoN -- the slewed member of the moving-source-gated sequence family.

:class:`~models._pathdon.SequenceDeepONet`, as :mod:`models.jfdon` uses it. What
makes it slewed is the heading the gate is drawn with, not the architecture: see
:func:`~models._pathdon.slewed_beam_state`.
"""

from __future__ import annotations

from .._pathdon import GATES, SequenceDeepONet

GATE = GATES["sjfdon"]

SJFDoN = SequenceDeepONet

__all__ = ["SJFDoN", "GATE"]
