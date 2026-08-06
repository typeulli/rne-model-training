"""SJFDoN -- moving-source-gated, with the wake relaxed into each corner.

:mod:`models.jfdon` in every respect but that. See
:func:`~models._pathdon.slewed_beam_state`.
"""

from .agent import build_agent
from .model import GATE, SJFDoN

__all__ = ["SJFDoN", "GATE", "build_agent"]
