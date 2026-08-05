"""JFDoN -- moving-source-gated DeepONet returning a temperature history, ``[B, Nt]``.

``fdon`` with ``G = j + p``, Rosenthal's travelling source, oriented by the
beam's own velocity at each stored time so the wake turns as the path does.

Corpus, split, subnetworks, objective and loop are :mod:`models._pathdon`'s and
are identical across ``fdon``, ``gfdon`` and ``jfdon``.
"""

from .agent import build_agent
from .model import GATE, JFDoN

__all__ = ["JFDoN", "GATE", "build_agent"]
