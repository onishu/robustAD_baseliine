"""Confirmed board-surface masks and the auxiliary patch summary.

Unconfirmed review images and dummy masks produce no scores. The same cell
rule is used later for the saved grids of input alignment, feature alignment,
and augmentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from robustad.theme_a.action import (
    FUSED_GRID,
    board_patch_summary,
    classify_board_cells,
    warp_label_and_origin,
)

# Frames of grids that already exist, or that the augmentation plan will write.
# Angles move the confirmed 512 label. Feature inverse stays in the original
# frame and drops reflection-referenced feature cells.
# Saved 64×64 grids. Confirmed masks are applied later. This table does not score them.
SAVED_GRID_SOURCES = {
    "uncorrected": "results_a0_boundary/maps",
    "input_alignment": "results_a0_boundary/maps",
    "feature_alignment": "results_a0_feature/maps",
    "augmentation": "results_a0_augment/maps",
}
COMPARISON_NOTE = (
    "①の主比較は、回転ビューなしと3ビューである。"
    "②③は旧144枚バンク上の作用点比較である。"
    "全方式の条件をそろえきった最終順位ではない。"
)

BOARD_FRAMES = {
    "sd_original": {"angles": (), "exclude_feature_reflection": False},
    "pad_warp_0": {"angles": (0.0,), "exclude_feature_reflection": False},
    "pad_synthetic_m5": {"angles": (-5.0,), "exclude_feature_reflection": False},
    "pad_synthetic_p5": {"angles": (5.0,), "exclude_feature_reflection": False},
    "pad_restore_m5": {"angles": (-5.0, 5.0), "exclude_feature_reflection": False},
    "pad_restore_p5": {"angles": (5.0, -5.0), "exclude_feature_reflection": False},
    "feature_m5": {"angles": (), "exclude_feature_reflection": True},
    "feature_p5": {"angles": (), "exclude_feature_reflection": True},
}


def load_confirmed_board_masks(roi_dir: Path) -> dict[str, np.ndarray]:
    """Load researcher-painted masks. Any unconfirmed or missing file stops first."""

    manifest_path = roi_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "confirmed" or not manifest.get("masks_created"):
        raise RuntimeError(f"基板面マスクは未確認です: {roi_dir}")
    stems = [str(item["stem"]) for item in manifest["stems"]]
    if any(str(item.get("status")) != "confirmed" for item in manifest["stems"]):
        raise RuntimeError(f"未確認のファイルが残っています: {roi_dir}")
    masks: dict[str, np.ndarray] = {}
    for stem in stems:
        path = roi_dir / "masks" / f"{stem}.png"
        if not path.is_file():
            raise RuntimeError(f"確認済みマスクがありません: {path}")
        array = np.asarray(Image.open(path))
        if array.shape != (512, 512):
            raise RuntimeError(f"マスクは 512×512 です: {path} {array.shape}")
        if array.ndim != 2:
            raise RuntimeError(f"マスクはグレーです: {path}")
        masks[stem] = array > 0
    if not masks:
        raise RuntimeError(f"確認済みマスクがありません: {roi_dir}")
    return masks


def auxiliary_board_metrics(
    patch_map: np.ndarray,
    roi: np.ndarray,
    threshold: float,
    frame: str,
    feature_reflection: np.ndarray | None = None,
) -> dict[str, object]:
    """One image, primary cells only. Partial and reflected cells stay out of the summary."""

    if frame not in BOARD_FRAMES:
        raise KeyError(f"基板面の枠が未定義です: {frame}")
    spec = BOARD_FRAMES[frame]
    grid = np.asarray(patch_map, dtype=np.float32)
    if grid.shape != (FUSED_GRID, FUSED_GRID):
        raise ValueError(f"パッチ格子は 64×64 です: {grid.shape}")
    moved = warp_label_and_origin(np.asarray(roi, dtype=np.float32), spec["angles"])
    cells = classify_board_cells(
        moved["roi"].numpy(),
        moved["content_from_original"].numpy(),
        moved["geometrically_valid"].numpy(),
    )
    excluded = 0
    if spec["exclude_feature_reflection"]:
        if feature_reflection is None:
            raise ValueError("特徴逆回転の枠には反射参照マスクが要ります。")
        reflection = np.asarray(feature_reflection, dtype=bool)
        if reflection.shape != grid.shape:
            raise ValueError(f"反射参照マスクの形状が違います: {reflection.shape}")
        excluded = int(np.sum(cells["board_entire"] & reflection))
        cells["board_entire"] = cells["board_entire"] & ~reflection
    summary = board_patch_summary(grid, cells["board_entire"], threshold)
    summary["n_partial"] = int(np.sum(cells["board_partial"]))
    summary["n_reflection_or_invalid"] = int(np.sum(cells["reflection_or_invalid"]))
    summary["n_feature_reflection_removed_from_primary"] = excluded
    summary["frame"] = frame
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="確認済み基板面マスクがあるときだけ補助指標を計算する。")
    parser.add_argument("--roi-dir", type=Path, default=Path("roi_pending/pcb_sd_normal8"))
    args = parser.parse_args()
    try:
        load_confirmed_board_masks(args.roi_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(
        "確認済みマスクは読めました。数値は出していません。"
        "保存済み格子は補正なし、②、③、①です。新しいfitは要りません。"
        + COMPARISON_NOTE
    )


if __name__ == "__main__":
    main()
