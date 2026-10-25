"""Image-score reductions that stay distinct from each other.

The reproduction image score is whatever the fitted model returns as
``pred_score``. For PatchCore that value is the reweighted worst-patch
distance. For PaDiM it is the maximum of the smoothed anomaly map. The
diagnostic reductions below are computed from the same patch distances and
are not written back into the reproduction metrics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def finite_or_raise(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size and not np.isfinite(array).all():
        raise ValueError(f"{name} に NaN または Inf があります。")
    return array


def max_reduction(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    if not np.isfinite(array).all():
        return float("nan")
    return float(array.max())


def quantile_reduction(values: np.ndarray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"分位点は [0, 1] です: {quantile}")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    if not np.isfinite(array).all():
        return float("nan")
    return float(np.quantile(array, quantile))


def patchcore_reweight(score: torch.Tensor, support_distances: torch.Tensor) -> torch.Tensor:
    """Match ``PatchcoreModel.compute_anomaly_score`` steps 5 and 6.

    ``support_distances`` has shape ``(batch, neighbors)`` and is ordered like
    ``topk(largest=False)``: column 0 is the nearest support sample. The
    returned weight uses that column only, then multiplies the worst-patch
    distance. This is not the maximum of a smoothed map.
    """

    weights = (1 - F.softmax(support_distances, dim=1))[:, 0]
    return weights * score


def nearest_upsample(patch_scores: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Match PatchCore's ``F.interpolate`` call, which leaves mode at the default."""

    return F.interpolate(patch_scores, size=(image_size[0], image_size[1]))


def bilinear_upsample(distance: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Match PaDiM's upsampling."""

    return F.interpolate(distance, size=image_size, mode="bilinear", align_corners=False)
