"""RGFDoN -- Gaussian-gated, with the beam frame held while the laser is off.

:mod:`models.gfdon` in every respect but that. See
:func:`~models._pathdon.retouched_beam_state`.
"""

from .agent import build_agent
from .model import GATE, RGFDoN

__all__ = ["RGFDoN", "GATE", "build_agent"]
