"""In-plane rotation warps for the first Theme A comparison.

Positive degrees follow torchvision and are counter-clockwise. The dataset
label ``5%`` is not converted to degrees here.
"""

from __future__ import annotations

import torch
from torchvision.transforms.functional import InterpolationMode, rotate


def rotate_tensor(
    image: torch.Tensor,
    degrees: float,
    *,
    nearest: bool,
    fill: float = 0.0,
) -> torch.Tensor:
    """Rotate around the center without changing the tensor height or width."""

    if image.ndim not in (2, 3):
        raise ValueError(f"画像は H×W または C×H×W です: {tuple(image.shape)}")
    mode = InterpolationMode.NEAREST if nearest else InterpolationMode.BILINEAR
    output = rotate(image, angle=float(degrees), interpolation=mode, expand=False, fill=fill)
    return output.to(dtype=image.dtype)


def inverse_degrees(degrees: float) -> float:
    return -float(degrees)


def identity_warp(image: torch.Tensor, *, nearest: bool) -> torch.Tensor:
    """Pass the image through the same rotator at 0 degrees."""

    return rotate_tensor(image, 0.0, nearest=nearest)


def round_trip(image: torch.Tensor, degrees: float, *, nearest: bool) -> torch.Tensor:
    """Rotate and invert, leaving the pose unchanged and the resampling applied."""

    forward = rotate_tensor(image, degrees, nearest=nearest)
    return rotate_tensor(forward, inverse_degrees(degrees), nearest=nearest)


def restore(image: torch.Tensor, degrees: float, *, nearest: bool) -> torch.Tensor:
    """Apply the inverse of a declared in-plane angle."""

    return rotate_tensor(image, inverse_degrees(degrees), nearest=nearest)


def valid_after_rotation(height: int, width: int, degrees: float) -> torch.Tensor:
    """Pixels that do not mix with the rotation fill.

    A value of one stays valid. A pixel that interpolates with the empty
    fill is outside the valid region. This is not a workpiece ROI.
    """

    ones = torch.ones((1, height, width), dtype=torch.float32)
    rotated = rotate_tensor(ones, degrees, nearest=False, fill=0.0)
    return rotated[0] >= 1.0 - 1e-5


def rotated_mask(mask: torch.Tensor, degrees: float) -> torch.Tensor:
    """Nearest rotation of a binary mask, then ``> 0``."""

    array = mask.to(dtype=torch.float32)
    if array.ndim == 2:
        array = array[None]
    rotated = rotate_tensor(array, degrees, nearest=True, fill=0.0)
    return rotated > 0
