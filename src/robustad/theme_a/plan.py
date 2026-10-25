"""Capacity estimate for the synthetic PCB rotation comparison.

The count is six conditions on the 195 PCB SD test images, plus 36
untransformed calibration images. The real Rotation folder is not included.
"""

from __future__ import annotations

import shutil
from pathlib import Path

SD_TEST_IMAGES = 195
CALIBRATION_IMAGES = 36
CONDITIONS = (
    "original",
    "warp_0",
    "synthetic_m5",
    "synthetic_p5",
    "restore_m5",
    "restore_p5",
)


def rotation_protocol() -> dict[str, object]:
    return {
        "dataset": "pcb",
        "model": "patchcore",
        "seed": 0,
        "image_size": 512,
        "fit_images": 144,
        "calibration_images": CALIBRATION_IMAGES,
        "source_images": "pcb_sd_test",
        "source_count": SD_TEST_IMAGES,
        "conditions": CONDITIONS,
        "real_rotation_folder_used": False,
        "angle_degrees": (-5.0, 5.0),
        "angle_meaning": "synthetic experiment setting, not RobustAD ±5%",
        "refit_per_condition": False,
        "recalibrate_per_condition": False,
        "primary_statistic_dtype": "float32",
        "save": ("memory_bank", "patch_grid", "valid_region", "matrices", "reproduction_score", "component_max"),
        "do_not_save": ("cnn_embedding", "float16_diagnostic_cache", "inverse_warped_score_map"),
        "do_not_modify": ("results_a1", "results_p0/diag"),
    }


def storage_estimate() -> dict[str, int]:
    coreset = int(144 * 64 * 64 * 0.1)
    bank = coreset * 1536 * 4
    scored = SD_TEST_IMAGES * len(CONDITIONS)
    patch_grids = scored * 64 * 64 * 4
    # The valid region depends only on the angle list, so it is stored once
    # per condition plus the common intersection, not once per image.
    valid_masks = (len(CONDITIONS) + 1) * 512 * 512
    return {
        "coreset_patches": coreset,
        "bank_bytes": bank,
        "scored_condition_images": scored,
        "calibration_images": CALIBRATION_IMAGES,
        "patch_grids_bytes": patch_grids,
        "valid_masks_bytes": valid_masks,
        "total_bytes": bank + patch_grids + valid_masks,
    }


def capacity_report(output_dir: Path) -> dict[str, object]:
    anchor = output_dir if output_dir.exists() else output_dir.parent
    if not anchor.exists():
        anchor = Path(".")
    usage = shutil.disk_usage(anchor)
    estimate = storage_estimate()
    return {
        "output_dir": str(output_dir),
        "disk_free_bytes": usage.free,
        "estimate": estimate,
        "model_was_fit": False,
        "free_exceeds_estimate": usage.free > int(estimate["total_bytes"]) + 400_000_000,
        "runtime_note": (
            "p0 の PatchCore 180 枚は coreset 約 11 分、その後 200 枚の推論が約 3 分だった。"
            "今回の推論は SD 195 枚×6 条件で 1170 枚。見込は fit 7–12 分、推論 15–25 分、合計 25–40 分。"
            "計測した所要時間ではない。実撮影 Rotation 140 枚は含まない。"
        ),
    }
