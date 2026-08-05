"""JDoN -- moving-source-gated DeepONet whose branch reads the laser toolpath and power.

``don`` with ``G = j + p``: Rosenthal's travelling source, imaged for the
plate's two faces and oriented by the beam's own velocity, so the wake
turns when the path turns.

Corpus, split, subnetworks, objective and training loop are
:mod:`models._pathdon`'s and are identical across ``don``, ``gdon`` and ``jdon``.
"""

from .agent import build_agent
from .model import GATE, JDoN

__all__ = ["JDoN", "GATE", "build_agent"]
