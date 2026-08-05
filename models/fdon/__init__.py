"""FDoN -- ungated DeepONet returning a temperature history, ``[B, Nt]``.

The floor of the sequence family: nothing multiplies the inner product,
so whatever ``gfdon`` and ``jfdon`` gain, they gain from their gate.

Corpus, split, subnetworks, objective and loop are :mod:`models._pathdon`'s and
are identical across ``fdon``, ``gfdon`` and ``jfdon``.
"""

from .agent import build_agent
from .model import GATE, FDoN

__all__ = ["FDoN", "GATE", "build_agent"]
