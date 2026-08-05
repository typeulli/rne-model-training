"""Train DoN: a DeepONet over ``(toolpath, P)``, with no gate at all.

Invoked through the top-level dispatcher::

    python train.py don --iterations 40000
    python train.py don --holdout 175 --sensors 64 --hidden 128 128 128 128

The branch reads the scan path -- ``(x_l, y_l, lit)`` at 64 fixed times -- and the
laser power. The trunk reads ``(x, y, z, t)``. Their inner product plus a bias is
the temperature; nothing multiplies it. That makes DoN the floor the other two are
measured against: whatever ``gdon`` and ``jdon`` gain, they gain from their gate
and from nothing else, because every other part of the three is this file's.

The corpus is ``data/toolpath`` -- four scan patterns on a 20 x 20 x 6 mm plate at
seven powers. One power is held out entirely (``--holdout``, 175 W by default) and
is scored but never trained on, so the number reported is generalisation across
``P`` on paths the model has seen, not interpolation between points it has.

Everything is SI (metres, seconds, Kelvin); the ``.npy`` files store millimetres
and are converted on load.
"""

from __future__ import annotations

from .. import _pathdon

MODEL_NAME = "don"


def main(argv: list[str] | None = None) -> None:
    _pathdon.run(MODEL_NAME, __doc__, argv)
