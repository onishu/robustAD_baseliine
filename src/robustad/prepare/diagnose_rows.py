"""Per-image rows for the diagnostic refit.

Maps are reduced here. The adoption of a score or a border width is not.
"""

from __future__ import annotations

import numpy as np
import torch

from robustad.prepare.regions import (
    attribution_masks,
    border_band,
    max_pixel_membership,
    region_score_stats,
)
from robustad.prepare.spec import BORDER_MARGINS, SMOOTH_RADIUS

REGION_NAMES = ("defect", "border", "nondefect_unpartitioned")


def storage_estimate() -> dict[str, object]:
    """Bytes for banks, PaDiM statistics, and float16 maps.

    Grids were measured at input 512. The coreset length is ``int(n * 0.1)``
    and is replaced by the saved tensor's shape after fitting. CNN embeddings
    are excluded.
    """

    patchcore_images = 171 + 200 + 195 + 140 + 140 + 140
    padim_images = 210
    coreset = int(180 * 64 * 64 * 0.1)
    bank = coreset * 1536 * 4
    padim_cov = (128 * 128) * 100 * 100 * 4
    padim_mean = 100 * (128 * 128) * 4
    # The pre-blur image map is the deterministic upsample of the patch grid
    # (nearest for PatchCore, bilinear for PaDiM), so only the patch grid and
    # the smoothed map are stored.
    patchcore_image = (64 * 64 + 512 * 512) * 2
    padim_image = (128 * 128 + 512 * 512) * 2
    maps = patchcore_images * patchcore_image + padim_images * padim_image
    return {
        "patchcore_images": patchcore_images,
        "padim_images": padim_images,
        "assumed_coreset_patches": coreset,
        "two_patchcore_banks_bytes": bank * 2,
        "padim_statistics_bytes": padim_cov + padim_mean,
        "maps_float16_bytes": maps,
        "total_bytes": bank * 2 + padim_cov + padim_mean + maps,
        "excluded": "CNN embeddings and the pre-blur full-resolution map. The latter is the upsample of the stored patch grid.",
    }


def blur_sigma8(upsampled: torch.Tensor) -> torch.Tensor:
    from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

    generator = getattr(blur_sigma8, "generator", None)
    if generator is None:
        generator = AnomalyMapGenerator(sigma=8)
        blur_sigma8.generator = generator
    return generator.blur(upsampled)


def map_max_shares(score: np.ndarray, defect: np.ndarray, margin: int) -> dict[str, float]:
    border = border_band(score.shape[0], score.shape[1], margin)
    masks = attribution_masks(defect, border)
    return max_pixel_membership(score, {name: masks[name] for name in REGION_NAMES})


def patch_block_shares(patch_map: np.ndarray, defect: np.ndarray, margin: int) -> dict[str, float]:
    """Attribute the worst patch's image block, which drives the PatchCore score."""

    peak = float(np.max(patch_map))
    if not np.isfinite(peak):
        return {name: float("nan") for name in REGION_NAMES}
    peaks = np.asarray(patch_map) == peak
    stride_y = defect.shape[0] // patch_map.shape[0]
    stride_x = defect.shape[1] // patch_map.shape[1]
    if stride_y < 1 or stride_x < 1:
        raise ValueError("パッチ格子が画像より細かい前提が崩れています。")
    block = np.zeros(defect.shape, dtype=bool)
    for y, x in zip(*np.where(peaks), strict=False):
        block[y * stride_y : (y + 1) * stride_y, x * stride_x : (x + 1) * stride_x] = True
    border = border_band(defect.shape[0], defect.shape[1], margin)
    masks = attribution_masks(defect, border)
    total = int(block.sum())
    return {name: float((block & masks[name]).sum() / total) for name in REGION_NAMES}


def score_row_regions(
    maps: dict[str, np.ndarray],
    defect: np.ndarray,
    patch_map: np.ndarray,
) -> dict[str, float]:
    row: dict[str, float] = {}
    for margin in BORDER_MARGINS:
        patch_shares = patch_block_shares(patch_map, defect, margin)
        for name in REGION_NAMES:
            row[f"patchmax_m{margin}_frac_{name}"] = patch_shares[name]
        for source, array in maps.items():
            shares = map_max_shares(array, defect, margin)
            for name in REGION_NAMES:
                row[f"{source}_m{margin}_maxfrac_{name}"] = shares[name]
    border = border_band(defect.shape[0], defect.shape[1], SMOOTH_RADIUS)
    masks = attribution_masks(defect, border)
    for name in REGION_NAMES:
        stats = region_score_stats(maps["sigma4"], masks[name])
        row[f"sigma4_m16_{name}_n"] = stats["n"]
        row[f"sigma4_m16_{name}_q50"] = stats["q50"]
        row[f"sigma4_m16_{name}_q95"] = stats["q95"]
        row[f"sigma4_m16_{name}_max"] = stats["max"]
    return row
