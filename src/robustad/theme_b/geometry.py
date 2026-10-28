"""Shared canvas for the theme B angle series.

The width is the smallest integer margin whose center 512 crop does not sample
outside the padded canvas, for every single angle in the series and for every
sequential round trip of those angles. It is not copied from the theme A 22 px
margin, and it is not chosen from anomaly scores.
"""

from __future__ import annotations

import math

from robustad.theme_a.boundary import fill_is_outside
from robustad.theme_a.warp import REFERENCE_SIZE

SERIES_DEGREES = (0.0, -2.5, 2.5, -5.0, 5.0, -10.0, 10.0)
MAX_SERIES_DEGREES = 10.0
SERIES_MARGIN = 41
CORRECTION_MARGIN = 106
SERIES_CONDITIONS = (
    ("series_0", 0.0),
    ("series_m2_5", -2.5),
    ("series_p2_5", 2.5),
    ("series_m5", -5.0),
    ("series_p5", 5.0),
    ("series_m10", -10.0),
    ("series_p10", 10.0),
)


def geometric_margin(degrees: float, size: int = REFERENCE_SIZE) -> int:
    """Pixels needed so a rotated square of ``size`` stays inside the canvas.

    After a rotation by ``degrees`` about the center, the axis-aligned half
    width of the square is ``(size / 2) * (|cos| + |sin|)``. The margin is the
    amount of that half width beyond ``size / 2``, rounded up.
    """

    theta = math.radians(abs(float(degrees)))
    half = size / 2.0
    extra = half * (abs(math.cos(theta)) + abs(math.sin(theta))) - half
    if extra <= 0:
        return 0
    return int(math.ceil(extra - 1e-9))


def series_angle_lists() -> tuple[tuple[float, ...], ...]:
    """Single angles, then both orders of each nonzero round trip."""

    singles = tuple((degrees,) for degrees in SERIES_DEGREES)
    trips = []
    for degrees in (2.5, 5.0, 10.0):
        trips.append((degrees, -degrees))
        trips.append((-degrees, degrees))
    return singles + tuple(trips)


def margin_covering_series(start: int | None = None) -> int:
    """Smallest margin that keeps torchvision fill out of the center crop.

    ``start`` defaults to the axis-aligned bound at ±10°. One pixel is added
    at a time until the bilinear probe passes. This checks the sampling rule.
    It does not compare padding modes.
    """

    margin = geometric_margin(MAX_SERIES_DEGREES) if start is None else int(start)
    lists = series_angle_lists()
    while margin < 160:
        if all(fill_is_outside(angles, margin) for angles in lists):
            return margin
        margin += 1
    raise RuntimeError("±10° の系列を収める反射延長幅が 160 px 以内にありません。")
