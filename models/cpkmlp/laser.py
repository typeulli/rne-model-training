"""Where the beam is -- the frame this model's coordinates are measured in.

A copy of ``models/cgmlp/laser.py`` with the gate dropped, since nothing here
needs the Gaussian: only the *path*. Kept as a copy rather than an import for the
reason that module gives for being one itself, so a recalibration of one model
cannot move another underneath it.

``BEAM_RADIUS``, ``LASER_START_X``, ``LASER_Y`` and ``SCAN_SPEED`` are recovered
from the data by ``python calibrate.py``. Every quantity is SI.

:class:`BeamPath` exists so the three numbers that define the frame can be
written into a checkpoint and read back out of it. That is the whole point: the
patch is fitted in coordinates centred on the beam and pasted in coordinates
centred on the beam, and if those two beams were ever different the patch would
land in the wrong place. Travelling with the weights makes them the same beam by
construction rather than by two modules agreeing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MM = 1.0e-3

# ---------------------------------------------------------------------------
# Fitted jointly across all seven powers by `python calibrate.py`.
# ---------------------------------------------------------------------------
BEAM_RADIUS = 1.6971 * MM  # r_b [m]
LASER_START_X = 4.8683 * MM
LASER_Y = 4.9929 * MM
SCAN_SPEED = 10.0000 * MM  # [m s^-1]


@dataclass(frozen=True)
class BeamPath:
    """``x_l(t) = start_x + speed * t`` at a fixed ``y``, in metres and seconds."""

    start_x: float = LASER_START_X
    speed: float = SCAN_SPEED
    y: float = LASER_Y

    def centre(self, time: float) -> tuple[float, float]:
        """``(x, y)`` of the beam at ``time``."""
        return (self.start_x + self.speed * float(time), self.y)

    def as_dict(self) -> dict[str, float]:
        """The three numbers, for storing in a checkpoint."""
        return asdict(self)
