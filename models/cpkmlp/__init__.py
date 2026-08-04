"""CPKMLP -- the melt pool alone, in coordinates that ride with the beam.

:mod:`models.cmlp` fits the whole plate and spends its capacity accordingly: the
pool is a few millimetres of a 40 x 10 x 6 mm block, so the squared error is
dominated by the cold remainder and the peak is a rare event the loss barely
notices. This model is given nothing but that neighbourhood --

    ``(t, z, dy, dx) -> T``     ``|dx| <= r``, ``|dy| <= r``, ``z`` whole

-- where ``dx`` and ``dy`` are measured from the beam, ``x_l(t) = x_0 + v t``,
not from the origin. Every sample it sees is therefore a point of the pool, and
the beam sits at ``(dx, dy) = (0, 0)`` in every one of them, so a quasi-steady
pool is very nearly a function of ``(dx, dy, z)`` alone.

The laser power is not an input, exactly as in :mod:`models.cmlp`, so a run must
be given a single-power corpus for the labels to be a function of its inputs.

It is not a surrogate on its own -- it knows the pool's shape and nothing about
the rest of the plate. It is the correction :class:`~agent.PeakCorrectedAgent`
pastes over a model that has the rest of the plate and blunts the pool.
"""

from .agent import CPKMLPAgent, build_agent
from .laser import BeamPath
from .model import PeakMLP

__all__ = ["PeakMLP", "BeamPath", "CPKMLPAgent", "build_agent"]
