"""The supervised objective of ``fmlp``: mean squared error on the coefficients.

The residual is taken in spectral space rather than on the grid, and under the
default ``--norm global`` that is not a compromise -- it is the same quantity. The
transform :mod:`~models.fmlp.dataset` applies is orthonormal, so Parseval's theorem
says the sum of squared coefficient errors *is* the sum of squared field errors:

    ``|T_hat - T|^2 summed over kappa  =  |T - T|^2 summed over the grid``

up to the modes the box threw away, which no choice of loss can recover. One scale
for every coefficient preserves that identity, so minimising this minimises the
field's L2 error in Kelvin, and the number the optimiser sees is the number the
physics cares about.

``--norm per-coef`` deliberately breaks the identity: standardising each
coefficient by its own spread makes a mode carrying a millionth of the energy count
as much as the DC term. That is the right objective if the *spectrum* is the object
of interest and the wrong one if the field is, which is why it is a flag and not
the default.
"""

from __future__ import annotations

from torch import Tensor, nn


class CoefficientMSELoss(nn.Module):
    """``mean((C_hat - C)^2)`` over the normalised coefficient vector.

    The normalisation lives in the dataset and the model's buffers -- the targets
    arrive standardised and the network emits standardised values -- so there is
    no scale left to divide by here, and this stays a bare MSE. The ``scale`` the
    other models carry is not missing; it has already been applied, once, to both
    sides.
    """

    def forward(self, predicted: Tensor, measured: Tensor) -> Tensor:
        return (predicted - measured).pow(2).mean()
