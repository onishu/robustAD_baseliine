"""Synthetic in-plane rotation in the 512 reference frame.

Positive degrees are counter-clockwise, matching torchvision. The center is
the image center ``(width / 2, height / 2)`` in top-left pixel coordinates.
``expand`` is false and the outside fill is 0. RGB is rotated before ImageNet
normalization. Masks use nearest sampling and then ``> 0``.

The saved matrix is the 2×3 sampling matrix torchvision applies, which maps
an output location back to the source. Its inverse is stored beside it.
This is not an inverse warp of a model score map.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode, _get_inverse_affine_matrix, rotate

REFERENCE_SIZE = 512
FILL = 0.0
EXPAND = False
POSITIVE_DIRECTION = "counter-clockwise"


def rotation_center(height: int, width: int) -> tuple[float, float]:
    """Return ``(x, y)`` of the rotation center. Origin is the top-left pixel."""

    return (width * 0.5, height * 0.5)


def sampling_matrix(degrees: float, height: int, width: int) -> np.ndarray:
    """3×3 matrix mapping an output point to the source point.

    The top two rows are exactly the matrix passed to torchvision. The angle
    sign is negated there because torchvision's affine helper and ``rotate``
    disagree; after that negation, positive ``degrees`` is counter-clockwise.
    """

    center_x, center_y = rotation_center(height, width)
    center_f = [center_x - width * 0.5, center_y - height * 0.5]
    raw = _get_inverse_affine_matrix(center_f, -float(degrees), [0.0, 0.0], 1.0, [0.0, 0.0])
    return np.array(
        [[raw[0], raw[1], raw[2]], [raw[3], raw[4], raw[5]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    inverted = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    return inverted


def _as_channel_first(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if image.ndim == 2:
        return image[None], True
    if image.ndim == 3:
        return image, False
    raise ValueError(f"画像は H×W または C×H×W です: {tuple(image.shape)}")


def rotate_tensor(
    image: torch.Tensor,
    degrees: float,
    *,
    nearest: bool,
    fill: float = FILL,
) -> torch.Tensor:
    """Rotate without changing height or width."""

    channel_first, squeezed = _as_channel_first(image)
    height, width = channel_first.shape[-2], channel_first.shape[-1]
    mode = InterpolationMode.NEAREST if nearest else InterpolationMode.BILINEAR
    center = list(rotation_center(height, width))
    output = rotate(
        channel_first,
        angle=float(degrees),
        interpolation=mode,
        expand=EXPAND,
        center=center,
        fill=fill,
    )
    if squeezed:
        return output[0].to(dtype=image.dtype)
    return output.to(dtype=image.dtype)


def apply_angles(image: torch.Tensor, angles: tuple[float, ...], *, nearest: bool) -> tuple[torch.Tensor, list[dict[str, list[float]]]]:
    """Apply each angle to the previous pre-normalization image.

    An empty angle list leaves the tensor untouched and records an identity
    matrix. That path is the original image, not a 0° resample.
    """

    channel_first, squeezed = _as_channel_first(image)
    height, width = int(channel_first.shape[-2]), int(channel_first.shape[-1])
    output = channel_first
    steps: list[dict[str, list[float]]] = []
    if not angles:
        identity = np.eye(3, dtype=np.float64)
        steps.append(_matrix_record(identity, identity, 0.0, resampled=False))
    for degrees in angles:
        sampling = sampling_matrix(degrees, height, width)
        output = rotate_tensor(output, degrees, nearest=nearest)
        steps.append(_matrix_record(sampling, inverse_matrix(sampling), float(degrees), resampled=True))
    if squeezed:
        output = output[0]
    return output.to(dtype=image.dtype), steps


def _matrix_record(sampling: np.ndarray, forward: np.ndarray, degrees: float, *, resampled: bool) -> dict[str, list[float] | float | bool]:
    return {
        "degrees": degrees,
        "resampled": resampled,
        "output_to_source": sampling.reshape(-1).tolist(),
        "source_to_output": forward.reshape(-1).tolist(),
    }


def inverse_degrees(degrees: float) -> float:
    return -float(degrees)


def identity_warp(image: torch.Tensor, *, nearest: bool) -> torch.Tensor:
    """Resample through the rotator at 0 degrees."""

    output, _steps = apply_angles(image, (0.0,), nearest=nearest)
    return output


def round_trip(image: torch.Tensor, degrees: float, *, nearest: bool) -> torch.Tensor:
    """Rotate by ``degrees`` and then by the opposite angle."""

    output, _steps = apply_angles(image, (float(degrees), inverse_degrees(degrees)), nearest=nearest)
    return output


def restore(image: torch.Tensor, degrees: float, *, nearest: bool) -> torch.Tensor:
    output, _steps = apply_angles(image, (inverse_degrees(degrees),), nearest=nearest)
    return output


def valid_after_angles(height: int, width: int, angles: tuple[float, ...]) -> torch.Tensor:
    """Pixels that never mixed with the fill along the whole angle sequence."""

    if not angles:
        return torch.ones((height, width), dtype=torch.bool)
    ones = torch.ones((1, height, width), dtype=torch.float32)
    rotated, _steps = apply_angles(ones, angles, nearest=False)
    return rotated[0] >= 1.0 - 1e-5


def valid_after_rotation(height: int, width: int, degrees: float) -> torch.Tensor:
    return valid_after_angles(height, width, (float(degrees),))


def rotated_mask(mask: torch.Tensor, angles: tuple[float, ...]) -> torch.Tensor:
    """Nearest rotation of a binary mask, then ``> 0``."""

    array = mask.to(dtype=torch.float32)
    if array.ndim == 2:
        array = array[None]
    rotated, _steps = apply_angles(array, angles, nearest=True)
    return rotated[0] > 0
