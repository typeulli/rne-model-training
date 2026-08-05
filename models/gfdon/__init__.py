"""GFDoN -- Gaussian-gated DeepONet returning a temperature history, ``[B, Nt]``.

``fdon`` with ``G = g + p``, a unit-peak Gaussian on the beam -- evaluated
at every stored time, because the answer is a history and the beam has
moved between each of its entries.

Corpus, split, subnetworks, objective and loop are :mod:`models._pathdon`'s and
are identical across ``fdon``, ``gfdon`` and ``jfdon``.
"""

from .agent import build_agent
from .model import GATE, GFDoN

__all__ = ["GFDoN", "GATE", "build_agent"]
