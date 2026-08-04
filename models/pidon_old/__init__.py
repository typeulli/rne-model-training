"""PiDoNOld -- Physics-informed Deep Operator Network.

The predecessor of :mod:`models.gpidon_old`, kept so the checkpoints trained before
the Gaussian gate was added to the operator output can still be loaded, plotted
and compared. Here the only Gaussian in the problem is the laser source term of
the top boundary condition; the network output itself is unconstrained.
"""

from .agent import PiDoNOldAgent, build_agent
from .model import PiDoNOld

__all__ = ["PiDoNOld", "PiDoNOldAgent", "build_agent"]
