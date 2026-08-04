"""GPiDoNOld -- Gaussian-source Physics-informed Deep Operator Network."""

from .agent import GPiDoNOldAgent, build_agent
from .model import GPiDoNOld

__all__ = ["GPiDoNOld", "GPiDoNOldAgent", "build_agent"]
