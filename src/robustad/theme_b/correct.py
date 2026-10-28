"""Shared 512 input, then one correction warp. The angle is the only difference."""

from __future__ import annotations

import torch

from robustad.theme_a.boundary import extend_and_warp
from robustad.theme_b.geometry import CORRECTION_MARGIN, SERIES_MARGIN


def series_rgb(image: torch.Tensor, degrees: float) -> tuple[torch.Tensor, list[dict]]:
    cropped, steps = extend_and_warp(image, (float(degrees),), nearest=False, margin=SERIES_MARGIN, pad_mode="reflect")
    return cropped, steps


def series_mask(mask: torch.Tensor, degrees: float) -> torch.Tensor:
    cropped, _steps = extend_and_warp(mask, (float(degrees),), nearest=True, margin=SERIES_MARGIN, pad_mode="constant")
    return cropped


def correction_rgb(generated: torch.Tensor, degrees: float) -> tuple[torch.Tensor, list[dict]]:
    """Correct a finished 512 image. This does not continue the 41 px canvas."""

    cropped, steps = extend_and_warp(
        generated,
        (float(degrees),),
        nearest=False,
        margin=CORRECTION_MARGIN,
        pad_mode="reflect",
    )
    return cropped, steps


def correction_mask(generated_mask: torch.Tensor, degrees: float) -> torch.Tensor:
    cropped, _steps = extend_and_warp(
        generated_mask,
        (float(degrees),),
        nearest=True,
        margin=CORRECTION_MARGIN,
        pad_mode="constant",
    )
    return cropped
