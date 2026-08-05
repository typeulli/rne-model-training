"""PKDoN -- the melt-pool patch for the toolpath-conditioned DeepONets.

Fitted to the beam's neighbourhood alone, in coordinates that ride and turn with
it. Not a surrogate for the plate: it is pasted over one, by
:func:`~models.pkdon.agent.peak_corrected`.
"""

from .agent import build_agent, peak_corrected
from .model import DEFAULT_RADIUS, GATE, PKDoN

__all__ = ["PKDoN", "GATE", "DEFAULT_RADIUS", "build_agent", "peak_corrected"]
