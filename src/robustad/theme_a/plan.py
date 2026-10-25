"""Capacity and time estimate for the PCB rotation comparison.

No model is built here.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def rotation_protocol() -> dict[str, object]:
    """One PatchCore fit, then four test-time image conditions."""

    return {
        "dataset": "pcb",
        "model": "patchcore",
        "seed": 0,
        "image_size": 512,
        "fit_images": 144,
        "calibration_images": 36,
        "conditions": (
            "source_original",
            "rotation_as_captured",
            "rotation_restored",
            "rotation_identity_warp",
            "rotation_round_trip",
        ),
        "refit_per_condition": False,
        "primary_statistic_dtype": "float32",
        "save": ("memory_bank", "patch_grid", "reproduction_score", "component_max"),
        "do_not_save": ("cnn_embedding", "float16_diagnostic_cache"),
        "do_not_modify": ("results_a1", "results_p0/diag"),
    }


def storage_estimate() -> dict[str, int]:
    """Bytes for one 144-image PatchCore bank and the patch grids we would keep."""

    coreset = int(144 * 64 * 64 * 0.1)
    bank = coreset * 1536 * 4
    # SD 195 images once. Rotation 140 images in four warps. Calibration 36.
    scored_images = 195 + 140 * 4 + 36
    patch_grids = scored_images * 64 * 64 * 4
    return {
        "coreset_patches": coreset,
        "bank_bytes": bank,
        "scored_images": scored_images,
        "patch_grids_bytes": patch_grids,
        "total_bytes": bank + patch_grids,
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
            "144 枚の coreset はパッチ数比で短く、見込みは fit 7–12 分、推論 15–25 分、合計 30–45 分。"
            "計測した所要時間ではない。"
        ),
    }
