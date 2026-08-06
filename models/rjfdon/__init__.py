"""RJFDoN -- moving-source-gated, with the beam frame held while the laser is off.

:mod:`models.jfdon` in every respect but that. See
:func:`~models._pathdon.retouched_beam_state`.
"""

from .agent import build_agent
from .model import GATE, RJFDoN

__all__ = ["RJFDoN", "GATE", "build_agent"]
