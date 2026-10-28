"""Counts and output guard for theme B. This module does not score images."""

from __future__ import annotations

from pathlib import Path

SD_TEST_IMAGES = 195
SERIES_CONDITIONS = 7
STAGE1_BANKS = 2
# Known inverse at ±10°, estimated inverse at ±10°, and estimated inverse at 0°.
STAGE2_EXTRA_CONDITIONS = 5
STAGE2_ESTIMATES_PER_IMAGE = 3
CONTROL0_SCORE_SECONDS = 191.4
CONTROL0_FORWARDS = SD_TEST_IMAGES * 4
PATCH_GRID_BYTES = 64 * 64 * 4
UINT8_RGB_BYTES = 512 * 512 * 3

PROTECTED_OUTPUTS = {
    "results_a0",
    "results_a0_boundary",
    "results_a0_feature",
    "results_a0_augment",
    "results_a0_control0",
    "results_a1",
    "results_tv",
    "diag",
}


def assert_new_output(path: Path) -> Path:
    """Refuse to write into a finished theme A or preparation directory."""

    resolved = path.resolve()
    if resolved.name in PROTECTED_OUTPUTS or any(part in PROTECTED_OUTPUTS for part in resolved.parts):
        raise RuntimeError(f"保護された出力先です: {resolved}")
    if "results_p0" in resolved.parts:
        raise RuntimeError(f"results_p0 は変更しません: {resolved}")
    return resolved


def stage_counts() -> dict[str, int]:
    stage1 = SD_TEST_IMAGES * SERIES_CONDITIONS * STAGE1_BANKS
    stage2 = SD_TEST_IMAGES * STAGE2_EXTRA_CONDITIONS
    shared_images = SD_TEST_IMAGES * SERIES_CONDITIONS
    return {
        "stage1_forwards": stage1,
        "stage2_extra_forwards": stage2,
        "shared_series_images": shared_images,
        "stage2_angle_estimates": SD_TEST_IMAGES * STAGE2_ESTIMATES_PER_IMAGE,
        "stage1_patch_grid_bytes": stage1 * PATCH_GRID_BYTES,
        "stage2_patch_grid_bytes": stage2 * PATCH_GRID_BYTES,
        "shared_uint8_rgb_bytes": shared_images * UINT8_RGB_BYTES,
    }


def scaled_score_seconds(forwards: int) -> float:
    """Scale the measured control-0° scoring interval. This is not a new timing."""

    return CONTROL0_SCORE_SECONDS * forwards / CONTROL0_FORWARDS
