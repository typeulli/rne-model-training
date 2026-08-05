"""DoN -- ungated DeepONet whose branch reads the laser toolpath and power.

A DeepONet over ``(toolpath, P)`` with nothing multiplying the inner
product. The floor the other two are measured against.

Corpus, split, subnetworks, objective and training loop are
:mod:`models._pathdon`'s and are identical across ``don``, ``gdon`` and ``jdon``.
"""

from .agent import build_agent
from .model import GATE, DoN

__all__ = ["DoN", "GATE", "build_agent"]
