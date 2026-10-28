"""Counts for a Metal Parts confirmation that has not been fit.

The saved patch count is fixed from the 144-image research split and the
64×64 fused grid. Matching the coreset ratio 0.1 on the 3-view candidate
set would keep a different number of patches, so that ratio is not the rule.
"""

from __future__ import annotations

from pathlib import Path

from robustad.theme_a.action import candidate_index, projection_components, ratio_matching_coreset
from robustad.theme_b.geometry import CORRECTION_MARGIN, SERIES_MARGIN
from robustad.theme_b.protocol import assert_new_output

FIT_IMAGES = 144
CALIBRATION_IMAGES = 36
FUSED_GRID = 64
FUSED_CHANNELS = 1536
SAVED_PATCHES = int(FIT_IMAGES * FUSED_GRID * FUSED_GRID * 0.1)
CANDIDATES_ONE_VIEW = FIT_IMAGES * FUSED_GRID * FUSED_GRID
CANDIDATES_THREE_VIEW = FIT_IMAGES * 3 * FUSED_GRID * FUSED_GRID
THREE_VIEW_RATIO = ratio_matching_coreset(CANDIDATES_THREE_VIEW, SAVED_PATCHES)
PROJECTION_COMPONENTS = projection_components(CANDIDATES_THREE_VIEW)
PROJECTION_COMPONENTS_ONE_VIEW_JL = projection_components(CANDIDATES_ONE_VIEW)
# Sorted fit-list index, 0° view, fused row, fused col. Not chosen from scores.
INITIAL_IMAGE = 35
INITIAL_ROW = 9
INITIAL_COL = 38
METAL_SD_NORMAL = 120
METAL_SD_ANOMALY = 51
METAL_SD_IMAGES = METAL_SD_NORMAL + METAL_SD_ANOMALY
PRIMARY_TEST_ANGLES = (0.0, -5.0, 5.0, -10.0, 10.0)
DEFERRED_TEST_ANGLES = (-2.5, 2.5)
FIT_VIEW_ANGLES = (0.0, -5.0, 5.0)
OUTPUT = Path("results_m0")
PREP = Path("results_m0_prep")


def assert_shared_bank_counts() -> None:
    if int(CANDIDATES_ONE_VIEW * 0.1) != SAVED_PATCHES:
        raise RuntimeError("1ビューの 0.1 が保存数と一致しません。")
    if int(CANDIDATES_THREE_VIEW * THREE_VIEW_RATIO) != SAVED_PATCHES:
        raise RuntimeError("3ビューの保存数が1ビューと一致しません。")
    if int(CANDIDATES_THREE_VIEW * 0.1) == SAVED_PATCHES:
        raise RuntimeError("3ビューを coreset 比 0.1 のままにすると保存数が揃いません。")
    if PROJECTION_COMPONENTS_ONE_VIEW_JL == PROJECTION_COMPONENTS:
        raise RuntimeError("共有する射影次元が、1ビューのJL次元と偶然一致しています。設定を見直してください。")


def initial_indexes() -> dict[str, int]:
    one = candidate_index(INITIAL_IMAGE, 0, INITIAL_ROW, INITIAL_COL, n_views=1)
    three = candidate_index(INITIAL_IMAGE, 0, INITIAL_ROW, INITIAL_COL, n_views=3)
    if one == three:
        raise RuntimeError("1ビューと3ビューで初期中心の整数が一致しています。別の候補順です。")
    return {"one_view": one, "three_view": three}


def assert_output_allowed(path: Path) -> Path:
    resolved = assert_new_output(path)
    if "results_b0" in resolved.parts or resolved.name == "results_b0":
        raise RuntimeError(f"results_b0 は変更しません: {resolved}")
    return resolved


def primary_forwards() -> dict[str, int]:
    """Feature extracts and PatchCore forwards in the proposed minimum run."""

    uncorrected = METAL_SD_IMAGES * len(PRIMARY_TEST_ANGLES) * 2
    known = METAL_SD_IMAGES * 2
    calibration = CALIBRATION_IMAGES * 2
    deferred = METAL_SD_IMAGES * len(DEFERRED_TEST_ANGLES) * 2
    return {
        "fit_extracts_one_view": FIT_IMAGES,
        "fit_extracts_three_view": FIT_IMAGES * 3,
        "calibration_forwards": calibration,
        "uncorrected_forwards": uncorrected,
        "known_correction_forwards": known,
        "scoring_forwards": calibration + uncorrected + known,
        "deferred_pm2_5_forwards": deferred,
        "series_margin_px": SERIES_MARGIN,
        "correction_margin_px": CORRECTION_MARGIN,
        "saved_patches": SAVED_PATCHES,
        "projection_components_shared": PROJECTION_COMPONENTS,
    }
