"""PKFDoN -- the melt-pool patch for the sequence DeepONets.

Fitted to the beam's neighbourhood alone, in coordinates that ride and turn with
it, and returning the whole history at each offset. Not a surrogate for the
plate: it is pasted over one, by :func:`~models.pkfdon.agent.peak_corrected`.
"""

from .agent import build_agent, peak_corrected
from .model import DEFAULT_RADIUS, GATE, PKFDoN

__all__ = ["PKFDoN", "GATE", "DEFAULT_RADIUS", "build_agent", "peak_corrected"]
