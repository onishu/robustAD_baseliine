"""Print the Metal Parts confirmation estimate. This module does not fit."""

from __future__ import annotations

import json
from pathlib import Path

from robustad.metal_confirm.spec import (
    CANDIDATES_ONE_VIEW,
    CANDIDATES_THREE_VIEW,
    FUSED_CHANNELS,
    FUSED_GRID,
    PREP,
    SAVED_PATCHES,
    THREE_VIEW_RATIO,
    assert_output_allowed,
    assert_shared_bank_counts,
    initial_indexes,
    primary_forwards,
)

CONTROL_SUMMARY = Path("results_a0_control0/summary.json")
THREE_SUMMARY = Path("results_a0_augment/summary.json")
PCB_SCORE_FORWARDS = 195 * 4


def _pcb_basis() -> dict[str, float]:
    control = json.loads(CONTROL_SUMMARY.read_text(encoding="utf-8"))
    three = json.loads(THREE_SUMMARY.read_text(encoding="utf-8"))
    return {
        "control_fit_seconds": float(control["fit"]["fit_seconds"]),
        "three_view_fit_seconds": float(three["fit"]["fit_seconds"]),
        "control_calibration_seconds": float(control["calibration_seconds"]),
        "three_view_calibration_seconds": float(three["calibration_seconds"]),
        "control_score_seconds": float(control["score_seconds"]),
        "control_score_forwards": PCB_SCORE_FORWARDS,
        "control_coreset_gpu_peak_bytes": int(control["fit"]["coreset_gpu_peak_bytes"]),
        "three_view_coreset_gpu_peak_bytes": int(three["fit"]["coreset_gpu_peak_bytes"]),
    }


def estimate() -> dict[str, object]:
    assert_shared_bank_counts()
    basis = _pcb_basis()
    forwards = primary_forwards()
    per_forward = basis["control_score_seconds"] / basis["control_score_forwards"]
    scoring_seconds = forwards["scoring_forwards"] * per_forward
    fit_seconds = basis["control_fit_seconds"] + basis["three_view_fit_seconds"]
    calibration_seconds = basis["control_calibration_seconds"] + basis["three_view_calibration_seconds"]
    grid_bytes = forwards["scoring_forwards"] * FUSED_GRID * FUSED_GRID * 4
    # Five generated angles, not the deferred ±2.5.
    cache_images = 171 * 5
    cache_bytes = cache_images * 512 * 512 * 3
    bank_bytes = 2 * SAVED_PATCHES * FUSED_CHANNELS * 4
    return {
        "fit": False,
        "patchcore_called": False,
        "basis": "PCBで保存された構築・校正・採点時間からの比例。Metal Partsでの計測ではない。",
        "forwards": forwards,
        "candidates_one_view": CANDIDATES_ONE_VIEW,
        "candidates_three_view": CANDIDATES_THREE_VIEW,
        "three_view_ratio_not_0_1": THREE_VIEW_RATIO,
        "initial_indexes": initial_indexes(),
        "scaled_fit_seconds": fit_seconds,
        "scaled_calibration_seconds": calibration_seconds,
        "scaled_scoring_seconds": scoring_seconds,
        "scaled_total_seconds": fit_seconds + calibration_seconds + scoring_seconds,
        "construction_increment_seconds": basis["three_view_fit_seconds"] - basis["control_fit_seconds"],
        "seconds_per_forward_from_pcb_control_score": per_forward,
        "disk_bytes": {
            "two_banks": bank_bytes,
            "patch_grids": grid_bytes,
            "uint8_cache_five_angles": cache_bytes,
        },
        "pcb_gpu_peaks_bytes": {
            "control_coreset": basis["control_coreset_gpu_peak_bytes"],
            "three_view_coreset": basis["three_view_coreset_gpu_peak_bytes"],
        },
        "same_performance_not_assumed": True,
    }


def main() -> None:
    assert_output_allowed(PREP)
    payload = estimate()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
