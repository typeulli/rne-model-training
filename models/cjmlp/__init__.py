"""CJMLP -- CGMLP with the source's shape swapped for the field's.

:mod:`models.cgmlp` gates its output with a Gaussian riding on the beam. A
Gaussian is what the laser *deposits*; it is not what the metal *does* with it.
This gates with the second thing instead:

    ``T_hat = T_amb + dT * (net(t, z, y, x) * G(x, y, z, t) + b)``
    ``G = j(x, y, z, t) + p``

``j`` is Rosenthal's solution for a source travelling at ``v`` through a
conducting medium, summed over image sources so that the plate's insulated top
and ambient-held bottom are satisfied exactly, and softened over the beam's width
so it is finite at the origin. Unlike a Gaussian it is asymmetric -- steep ahead
of the beam, a long ``1/r`` wake behind it -- and it decays into the depth to an
ambient floor rather than to nothing.

Everything else is ``cgmlp``'s, unchanged, so the pair isolates the prior alone.
On the 175 W split at 64x3 it is worth a third of the error:

    cmlp (no gate) 1.689 K    cgmlp (gaussian) 1.532 K    cjmlp 1.204 K
"""

from .agent import CJMLPAgent, build_agent
from .laser import moving_source
from .model import ControlSourceMLP

__all__ = ["ControlSourceMLP", "CJMLPAgent", "build_agent", "moving_source"]
