"""Move a confirmed label along the theme B image path.

The series warp and the correction warp are separate 512 crops. The known
correction does not continue the series canvas. This module does not score
and does not read unconfirmed board masks.
"""

from __future__ import annotations

import numpy as np
import torch

from robustad.theme_a.action import map_crop_pixel_to_original, warp_label_and_origin
from robustad.theme_a.boundary import FILL_EPS, content_probe, fill_probe
from robustad.theme_a.warp import REFERENCE_SIZE, sampling_matrix
from robustad.theme_b.correct import correction_mask, series_mask
from robustad.theme_b.geometry import CORRECTION_MARGIN, SERIES_MARGIN


def _source_grid(margin: int, degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous source coordinates of every crop pixel, in the pre-pad square."""

    side = REFERENCE_SIZE + 2 * int(margin)
    matrix = sampling_matrix(float(degrees), side, side)
    cols = np.arange(REFERENCE_SIZE, dtype=np.float64) + margin
    rows = np.arange(REFERENCE_SIZE, dtype=np.float64) + margin
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    centered = np.stack((xx + 0.5 - side * 0.5, yy + 0.5 - side * 0.5, np.ones_like(xx)), axis=0)
    source = np.tensordot(matrix, centered, axes=(1, 0))
    src_x = source[0] + side * 0.5 - 0.5 - margin
    src_y = source[1] + side * 0.5 - 0.5 - margin
    inside = (src_x >= 0.0) & (src_x < REFERENCE_SIZE) & (src_y >= 0.0) & (src_y < REFERENCE_SIZE)
    return src_x, src_y, inside


def composed_origin(col: int, row: int, added_degrees: float, correction_degrees: float) -> dict[str, object]:
    """Back-project one final pixel through the correction crop, then the series crop."""

    series_point = map_crop_pixel_to_original(col, row, (float(correction_degrees),), margin=CORRECTION_MARGIN)
    series_inside = bool(series_point["inside_original_square"])
    if not series_inside:
        return {
            "inside_original_square": False,
            "correction_samples_series_image": False,
            "series_col": series_point["source_col"],
            "series_row": series_point["source_row"],
        }
    series_col = int(np.floor(series_point["source_col"]))
    series_row = int(np.floor(series_point["source_row"]))
    original = map_crop_pixel_to_original(series_col, series_row, (float(added_degrees),), margin=SERIES_MARGIN)
    return {
        "inside_original_square": bool(original["inside_original_square"]),
        "correction_samples_series_image": True,
        "series_col": series_point["source_col"],
        "series_row": series_point["source_row"],
        "source_col": original["source_col"],
        "source_row": original["source_row"],
    }


def series_then_correction_label(
    label: torch.Tensor,
    added_degrees: float,
    correction_degrees: float | None,
) -> dict[str, torch.Tensor]:
    """Nearest label with constant-0 padding, and the origin of each final pixel."""

    array = torch.as_tensor(label, dtype=torch.float32)
    if tuple(array.shape) != (REFERENCE_SIZE, REFERENCE_SIZE):
        raise ValueError(f"ラベルは 512×512 です: {tuple(array.shape)}")
    if correction_degrees is None:
        moved = warp_label_and_origin(array, (float(added_degrees),), margin=SERIES_MARGIN)
        moved["correction_applied"] = torch.tensor(False)
        return moved
    series = series_mask(array, float(added_degrees))
    corrected = correction_mask(series, float(correction_degrees))
    src_x, src_y, samples_series = _source_grid(CORRECTION_MARGIN, float(correction_degrees))
    series_content = (content_probe((float(added_degrees),), SERIES_MARGIN) >= (1.0 - FILL_EPS)).numpy()
    series_valid = (fill_probe((float(added_degrees),), SERIES_MARGIN) >= (1.0 - FILL_EPS)).numpy()
    correction_valid = (fill_probe((float(correction_degrees),), CORRECTION_MARGIN) >= (1.0 - FILL_EPS)).numpy()
    content = np.zeros((REFERENCE_SIZE, REFERENCE_SIZE), dtype=bool)
    valid = np.zeros((REFERENCE_SIZE, REFERENCE_SIZE), dtype=bool)
    flat_rows, flat_cols = np.nonzero(samples_series)
    src_cols = np.floor(src_x[flat_rows, flat_cols]).astype(np.int64)
    src_rows = np.floor(src_y[flat_rows, flat_cols]).astype(np.int64)
    content[flat_rows, flat_cols] = series_content[src_rows, src_cols]
    valid[flat_rows, flat_cols] = series_valid[src_rows, src_cols] & correction_valid[flat_rows, flat_cols]
    return {
        "roi": corrected > 0,
        "content_from_original": torch.from_numpy(content),
        "geometrically_valid": torch.from_numpy(valid),
        "correction_applied": torch.tensor(True),
    }
