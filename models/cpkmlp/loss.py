"""The supervised objective of CPKMLP: :mod:`models.cmlp.loss`, unchanged.

The residual is divided by a characteristic temperature rise before squaring, so
the reported loss is dimensionless and comparable against ``cmlp``'s -- which is
the comparison this model exists to make, since the two differ only in *which*
points they are asked about.

Shared rather than copied for the reason given in :mod:`models.cpkmlp.model`.
"""

from __future__ import annotations

from models.cmlp.loss import ScaledMSELoss

__all__ = ["ScaledMSELoss"]
