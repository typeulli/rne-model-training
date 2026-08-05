"""GDoN -- Gaussian-gated DeepONet whose branch reads the laser toolpath and power.

``don`` with ``G = g + p``: a unit-peak Gaussian riding on the beam,
placed by the row's own toolpath rather than by a hardcoded straight pass.

Corpus, split, subnetworks, objective and training loop are
:mod:`models._pathdon`'s and are identical across ``don``, ``gdon`` and ``jdon``.
"""

from .agent import build_agent
from .model import GATE, GDoN

__all__ = ["GDoN", "GATE", "build_agent"]
