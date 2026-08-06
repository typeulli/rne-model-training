"""RSJFDoN -- the beam frame held over dark travel, and the wake slewed at corners.

:mod:`models.jfdon` with both retouches at once. See
:func:`~models._pathdon.retouched_beam_state` and
:func:`~models._pathdon.slewed_beam_state`.
"""

from .agent import build_agent
from .model import GATE, RSJFDoN

__all__ = ["RSJFDoN", "GATE", "build_agent"]
