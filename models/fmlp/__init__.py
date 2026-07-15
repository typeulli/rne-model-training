"""A spectral surrogate: ``(P, t) -> Fourier coefficients -> the whole volume``."""

from .agent import FMLPAgent, build_agent
from .model import FourierMLP

__all__ = ["FourierMLP", "FMLPAgent", "build_agent"]
