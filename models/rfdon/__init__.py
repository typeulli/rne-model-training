"""RFDoN -- ungated, with the beam frame held while the laser is off.

:mod:`models.fdon` in every respect but that. See
:func:`~models._pathdon.retouched_beam_state`.
"""

from .agent import build_agent
from .model import GATE, RFDoN

__all__ = ["RFDoN", "GATE", "build_agent"]
