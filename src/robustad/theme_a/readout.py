"""Read-only summary of the frozen six-condition Theme A run.

This module does not fit, score, or rewrite ``summary.json``, ``thresholds.json``,
``tables/``, ``maps/``, ``matrices.json``, or ``valid_regions.npz``.
New tables go under ``results_a0/readout`` and audit figures under ``results_a0/audit``.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import Resize

from robustad.catalog import SUBSETS
from robustad.data import find_mask, list_images
from robustad.prepare.regions import component_correspondence, label_components, resize_binary_mask
from robustad.theme_a.run_synthetic import CONDITIONS, _components, _load_rgb
from robustad.theme_a.warp import (
    REFERENCE_SIZE,
    apply_angles,
    inverse_matrix,
    rotated_mask,
    rotation_center,
    sampling_matrix,
    valid_after_angles,
)

PATCH_THRESHOLD_KEY = "patch"
STRIDE = 8
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _quantile_row(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    row = {"n": int(array.size), "min": float(array.min()), "max": float(array.max()), "mean": float(array.mean())}
    for q in QUANTILES:
        row[f"p{int(q * 100)}"] = float(np.quantile(array, q, method="linear"))
    return row


def _cell_masks(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block = valid.reshape(valid.shape[0] // STRIDE, STRIDE, valid.shape[1] // STRIDE, STRIDE)
    fully_valid = block.all(axis=(1, 3))
    fully_invalid = ~block.any(axis=(1, 3))
    touches_invalid = ~fully_valid
    return fully_valid, fully_invalid, touches_invalid


def _intersecting_cells(mask: np.ndarray, grid: int) -> np.ndarray:
    stride = mask.shape[0] // grid
    return mask.reshape(grid, stride, grid, stride).any(axis=(1, 3))


def _bbox(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, 0
    return int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)


def _relation(fully_valid: np.ndarray, fully_invalid: np.ndarray, touches: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> str:
    if touches[rows, cols].any() and fully_invalid[rows, cols].all():
        return "tied_max_fully_invalid"
    if touches[rows, cols].any():
        return "tied_max_touches_invalid"
    if fully_valid[rows, cols].all():
        return "tied_max_fully_valid"
    return "tied_max_mixed"


def _distance_to_invalid(touches: np.ndarray, row: int, col: int) -> int | None:
    if not touches.any():
        return None
    yy, xx = np.nonzero(touches)
    return int(np.max(np.abs(np.stack([yy - row, xx - col])), axis=0).min())


def image_score_rows(images: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in images:
        grouped.setdefault((row["condition"], row["label"]), []).append(float(row["reproduction_score"]))
    payload = []
    for (condition, label), values in sorted(grouped.items()):
        item = {"condition": condition, "label": "anomaly" if label == "1" else "normal"}
        item.update(_quantile_row(np.asarray(values)))
        payload.append(item)
    return payload


def select_audit_ids(images: list[dict[str, str]]) -> dict[str, list[str]]:
    """Fixed positions in the sorted stem list. Scores are not consulted."""

    normals = sorted({row["source_id"] for row in images if row["label"] == "0"})
    anomalies = sorted({row["source_id"] for row in images if row["label"] == "1"})
    return {
        "rule": "sorted stem, index 0 and len//2, within normal and within anomaly",
        "normal": [normals[0], normals[len(normals) // 2]],
        "anomaly": [anomalies[0], anomalies[len(anomalies) // 2]],
    }


def _load_maps(map_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    maps: dict[tuple[str, str], np.ndarray] = {}
    for path in sorted(map_dir.glob("*.npz")):
        with np.load(path) as payload:
            maps[(str(payload["image_id"]), str(payload["condition"]))] = np.asarray(payload["patch_map"], dtype=np.float32)
    return maps


def patch_readout(
    images: list[dict[str, str]],
    maps: dict[tuple[str, str], np.ndarray],
    valid: dict[str, np.ndarray],
    patch_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    per_image: list[dict[str, object]] = []
    max_rows: list[dict[str, object]] = []
    for row in images:
        condition = row["condition"]
        patch = maps[(row["source_id"], condition)]
        fully_valid, fully_invalid, touches = _cell_masks(valid[condition])
        flat = patch.reshape(-1)
        valid_values = patch[fully_valid]
        touch_values = patch[touches]
        def _fpr(values: np.ndarray) -> tuple[int, int, float]:
            if values.size == 0:
                return 0, 0, float("nan")
            hits = int((values >= patch_threshold).sum())
            return hits, int(values.size), hits / values.size

        full_hits, full_n, full_fpr = _fpr(flat)
        valid_hits, valid_n, valid_fpr = _fpr(valid_values)
        touch_hits, touch_n, touch_fpr = _fpr(touch_values)
        per_image.append(
            {
                "source_id": row["source_id"],
                "condition": condition,
                "label": row["label"],
                "n_full": full_n,
                "n_false_full": full_hits,
                "fpr_full": full_fpr,
                "n_fully_valid": valid_n,
                "n_false_fully_valid": valid_hits,
                "fpr_fully_valid": valid_fpr,
                "n_touches_invalid": touch_n,
                "n_false_touches_invalid": touch_hits,
                "fpr_touches_invalid": touch_fpr,
            }
        )
        tied = np.argwhere(patch == patch.max())
        argmax = int(np.argmax(patch))
        max_row, max_col = divmod(argmax, patch.shape[1])
        distance = _distance_to_invalid(touches, max_row, max_col)
        max_rows.append(
            {
                "source_id": row["source_id"],
                "condition": condition,
                "label": row["label"],
                "argmax_row": max_row,
                "argmax_col": max_col,
                "n_tied_max": int(tied.shape[0]),
                "relation": _relation(fully_valid, fully_invalid, touches, tied[:, 0], tied[:, 1]),
                "chebyshev_cells_to_invalid": "" if distance is None else distance,
            }
        )
    summary = _patch_summary(per_image, max_rows)
    return per_image, max_rows, summary


def _patch_summary(per_image: list[dict[str, object]], max_rows: list[dict[str, object]]) -> dict[str, object]:
    conditions = [name for name, _angles in CONDITIONS]
    image_fpr: dict[str, dict[str, object]] = {}
    for condition in conditions:
        normals = [row for row in per_image if row["condition"] == condition and row["label"] == "0"]
        image_fpr[condition] = {
            "n_images": len(normals),
            "fpr_full": _quantile_row(np.asarray([row["fpr_full"] for row in normals], dtype=np.float64)),
            "fpr_fully_valid": _quantile_row(np.asarray([row["fpr_fully_valid"] for row in normals], dtype=np.float64)),
            "fpr_touches_invalid": _quantile_row(
                np.asarray([row["fpr_touches_invalid"] for row in normals], dtype=np.float64)
            ),
            "images_full_fpr_eq_1": sum(row["fpr_full"] == 1 for row in normals),
            "images_fully_valid_fpr_eq_1": sum(row["fpr_fully_valid"] == 1 for row in normals),
            "pooled_counts_are_not_image_uncertainty": True,
            "pooled_normal_patch_counts": {
                "full_false": int(sum(int(row["n_false_full"]) for row in normals)),
                "full_n": int(sum(int(row["n_full"]) for row in normals)),
                "fully_valid_false": int(sum(int(row["n_false_fully_valid"]) for row in normals)),
                "fully_valid_n": int(sum(int(row["n_fully_valid"]) for row in normals)),
                "touches_invalid_false": int(sum(int(row["n_false_touches_invalid"]) for row in normals)),
                "touches_invalid_n": int(sum(int(row["n_touches_invalid"]) for row in normals)),
            },
        }
    relations: dict[str, dict[str, int]] = {}
    for condition in conditions:
        for label in ("0", "1"):
            key = f"{condition}:{'anomaly' if label == '1' else 'normal'}"
            chosen = [row for row in max_rows if row["condition"] == condition and row["label"] == label]
            counts: dict[str, int] = {}
            for row in chosen:
                counts[str(row["relation"])] = counts.get(str(row["relation"]), 0) + 1
            relations[key] = counts
    return {"per_image_normal_patch_fpr": image_fpr, "max_patch_relation_counts": relations}


def component_geometry(
    components: list[dict[str, str]],
    maps: dict[tuple[str, str], np.ndarray],
    valid: dict[str, np.ndarray],
    common: np.ndarray,
    data_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    subset = SUBSETS["pcb"]
    domain = next(item for item in subset.domains if item.name == "SD")
    domain_dir = subset.domain_dir(data_root, domain)
    anomaly_paths = {path.stem: path for path in list_images(domain_dir / "anomaly")}
    alone_cache: dict[str, dict[int, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    max_abs = 0.0
    compared = 0
    for source_id, path in sorted(anomaly_paths.items()):
        mask_path = find_mask(path, domain_dir / "masks")
        meta, alone = _components(mask_path)
        alone_cache[source_id] = alone
        for name, angles in CONDITIONS:
            patch = maps[(source_id, name)]
            for item in meta:
                component_id = int(item["component_id"])
                mask = alone.get(component_id)
                warped = None if mask is None else rotated_mask(torch.from_numpy(mask.astype(np.float32)), angles).numpy()
                if warped is None:
                    cells = np.zeros(patch.shape, dtype=bool)
                    used = np.zeros_like(common)
                else:
                    used = warped & common
                    cells = _intersecting_cells(used, patch.shape[0])
                statistic = float("nan")
                if cells.any():
                    statistic = float(patch[cells].max())
                saved = next(
                    row
                    for row in components
                    if row["source_id"] == source_id and row["condition"] == name and int(row["component_id"]) == component_id
                )
                saved_stat = float(saved["statistic_common"])
                if np.isfinite(saved_stat) and np.isfinite(statistic):
                    max_abs = max(max_abs, abs(saved_stat - statistic))
                    compared += 1
                pixel_h, pixel_w = _bbox(used)
                cell_h, cell_w = _bbox(cells)
                rows.append(
                    {
                        "source_id": source_id,
                        "condition": name,
                        "component_id": component_id,
                        "statistic_common": statistic,
                        "n_intersecting_cells": int(cells.sum()),
                        "pixel_bbox_height": pixel_h,
                        "pixel_bbox_width": pixel_w,
                        "cell_bbox_height": cell_h,
                        "cell_bbox_width": cell_w,
                        "detected": saved["detected"],
                        "status_input": saved["status_input"],
                    }
                )
    return rows, {"statistic_recompute_compared": compared, "statistic_recompute_max_abs": max_abs, "masks": alone_cache}


def component_comparison(rows: list[dict[str, object]]) -> dict[str, object]:
    by_image: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        by_image.setdefault(str(row["source_id"]), {}).setdefault(str(row["condition"]), []).append(row)
    image_units = []
    disagreement_components = 0
    disagreement_images = 0
    for source_id, conditions in sorted(by_image.items()):
        original = [row for row in conditions["original"] if np.isfinite(float(row["statistic_common"]))]
        if not original:
            continue
        base = max(float(row["statistic_common"]) for row in original)
        detected_base = {int(row["component_id"]): row["detected"] == "True" for row in conditions["original"]}
        entry: dict[str, object] = {"source_id": source_id, "original_image_component_max": base}
        image_disagrees = False
        for name, _angles in CONDITIONS:
            if name == "original":
                continue
            finite = [row for row in conditions[name] if np.isfinite(float(row["statistic_common"]))]
            other = max(float(row["statistic_common"]) for row in finite) if finite else float("nan")
            entry[f"{name}_image_component_max"] = other
            entry[f"{name}_delta"] = other - base if np.isfinite(other) else float("nan")
            detected_other = {int(row["component_id"]): row["detected"] == "True" for row in conditions[name]}
            mismatched = [key for key in detected_base if detected_base[key] != detected_other[key]]
            disagreement_components += len(mismatched)
            image_disagrees = image_disagrees or bool(mismatched)
        disagreement_images += int(image_disagrees)
        image_units.append(entry)
    deltas = {}
    for name, _angles in CONDITIONS:
        if name == "original":
            continue
        values = np.asarray([float(row[f"{name}_delta"]) for row in image_units], dtype=np.float64)
        deltas[name] = {
            "unit": "one maximum per anomaly image",
            "n_images": int(np.isfinite(values).sum()),
            "mean_delta": float(np.nanmean(values)),
            "quantiles": _quantile_row(values),
            "sample_std": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else float("nan"),
            "zero_width_interval_used_as_equivalence": False,
        }
    return {
        "image_is_the_sampling_unit": True,
        "components_within_image_are_not_independent_samples": True,
        "observed_component_detection_disagreements": disagreement_components,
        "observed_images_with_any_detection_disagreement": disagreement_images,
        "observed_zero_disagreement_is_not_equivalence_on_new_samples": True,
        "image_level_component_max_delta_vs_original": deltas,
    }


def separation(
    components: list[dict[str, object]],
    per_image: list[dict[str, object]],
    maps: dict[tuple[str, str], np.ndarray],
    valid: dict[str, np.ndarray],
    patch_threshold: float,
) -> dict[str, object]:
    payload = {}
    for name, _angles in CONDITIONS:
        component_values = np.asarray(
            [
                float(row["statistic_common"])
                for row in components
                if row["condition"] == name and np.isfinite(float(row["statistic_common"]))
            ],
            dtype=np.float64,
        )
        fully_valid, _fully_invalid, _touches = _cell_masks(valid[name])
        normal_ids = sorted({row["source_id"] for row in per_image if row["condition"] == name and row["label"] == "0"})
        image_max = []
        valid_pool = []
        for source_id in normal_ids:
            patch = maps[(source_id, name)]
            chosen = patch[fully_valid]
            if chosen.size:
                image_max.append(float(chosen.max()))
                valid_pool.append(chosen)
        pooled = np.concatenate(valid_pool) if valid_pool else np.asarray([], dtype=np.float64)
        floor = float(np.quantile(component_values, 0.1, method="linear")) if component_values.size else float("nan")
        payload[name] = {
            "role": "auxiliary",
            "component_statistic": _quantile_row(component_values),
            "normal_fully_valid_patch": _quantile_row(pooled),
            "normal_image_max_fully_valid_patch": _quantile_row(np.asarray(image_max, dtype=np.float64)),
            "fraction_components_ge_patch_threshold": float((component_values >= patch_threshold).mean()),
            "fraction_normal_valid_patches_ge_component_p10": float((pooled >= floor).mean()) if pooled.size else float("nan"),
            "component_p10": floor,
            "patch_threshold": patch_threshold,
            "saturated_component_detection_is_not_selectivity": True,
        }
    return payload


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255).round().astype(np.uint8)


def _diff_uint8(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    delta = (left - right).abs().mean(dim=0)
    peak = float(delta.max()) if float(delta.max()) > 0 else 1.0
    scaled = (delta / peak).clamp(0, 1).numpy()
    return (scaled * 255).round().astype(np.uint8)


def _mask_uint8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask, dtype=np.uint8) * 255)


def geometric_audit(data_root: Path, audit_ids: dict[str, list[str]], matrices_path: Path, audit_dir: Path) -> dict[str, object]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    subset = SUBSETS["pcb"]
    domain = next(item for item in subset.domains if item.name == "SD")
    domain_dir = subset.domain_dir(data_root, domain)
    paths = {path.stem: path for path in list_images(domain_dir / "normal") + list_images(domain_dir / "anomaly")}
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    saved = json.loads(matrices_path.read_text(encoding="utf-8"))
    matrix_gap = 0.0
    for name, angles in CONDITIONS:
        _fresh, fresh_steps = apply_angles(torch.zeros((3, REFERENCE_SIZE, REFERENCE_SIZE)), angles, nearest=False)
        del _fresh
        for stored, built in zip(saved[name], fresh_steps, strict=True):
            matrix_gap = max(
                matrix_gap,
                float(np.max(np.abs(np.asarray(stored["output_to_source"]) - np.asarray(built["output_to_source"])))),
            )
    minus = sampling_matrix(-5.0, REFERENCE_SIZE, REFERENCE_SIZE)
    plus = sampling_matrix(5.0, REFERENCE_SIZE, REFERENCE_SIZE)
    composition_gap = float(np.max(np.abs(minus @ plus - np.eye(3))))
    center = rotation_center(REFERENCE_SIZE, REFERENCE_SIZE)

    probe = torch.zeros((1, REFERENCE_SIZE, REFERENCE_SIZE), dtype=torch.float32)
    probe[0, 200, 400] = 1.0
    turned = rotated_mask(probe[0], (5.0,))
    moved = torch.nonzero(torch.from_numpy(np.asarray(turned)))
    sign_row = int(moved[0, 0]) if moved.numel() else -1
    sign_col = int(moved[0, 1]) if moved.numel() else -1
    block = torch.zeros((1, REFERENCE_SIZE, REFERENCE_SIZE), dtype=torch.float32)
    block[0, 255:257, 255:257] = 1.0
    stayed = rotated_mask(block[0], (5.0,)).numpy()
    ys, xs = np.nonzero(stayed)
    centroid = (float(ys.mean()), float(xs.mean())) if ys.size else (float("nan"), float("nan"))

    panels = {}
    range_rows = []
    for stem in audit_ids["normal"] + audit_ids["anomaly"]:
        rgb = _load_rgb(paths[stem], resize)
        original = rgb
        named = {name: apply_angles(rgb, angles, nearest=False)[0] for name, angles in CONDITIONS}
        nearest = apply_angles(rgb, (-5.0,), nearest=True)[0]
        bilinear_gap = float((named["synthetic_m5"] - nearest).abs().max())
        tiles = []
        for name in ("original", "synthetic_m5", "synthetic_p5", "restore_m5", "restore_p5"):
            image = original if name == "original" else named[name]
            diff = _diff_uint8(image, original)
            valid = valid_after_angles(REFERENCE_SIZE, REFERENCE_SIZE, dict(CONDITIONS)[name]).numpy()
            rgb_u8 = _to_uint8(image)
            valid_u8 = np.stack([_mask_uint8(valid)] * 3, axis=-1)
            diff_u8 = np.stack([diff] * 3, axis=-1)
            tiles.append(np.concatenate([rgb_u8, diff_u8, valid_u8], axis=0))
            array = image.detach().cpu().numpy()
            range_rows.append(
                {
                    "source_id": stem,
                    "condition": name,
                    "min": float(array.min()),
                    "max": float(array.max()),
                    "mean": float(array.mean()),
                    "corner_pixel_max": float(array[:, 0, 0].max()),
                    "invalid_pixel_max": float(array[:, ~valid].max()) if (~valid).any() else 0.0,
                    "nearest_bilinear_max_abs_on_m5": bilinear_gap,
                }
            )
        sheet = np.concatenate(tiles, axis=1)
        Image.fromarray(sheet).save(audit_dir / f"{stem}.png")
        panels[stem] = str(audit_dir / f"{stem}.png")
        if stem in audit_ids["anomaly"]:
            mask_path = find_mask(paths[stem], domain_dir / "masks")
            with Image.open(mask_path) as handle:
                original_mask = np.asarray(handle.convert("L"))
            labels = label_components(original_mask)
            component_id = next(iter(component_correspondence(original_mask, REFERENCE_SIZE, REFERENCE_SIZE)))["component_id"]
            resized = resize_binary_mask(labels == int(component_id), REFERENCE_SIZE, REFERENCE_SIZE)
            warped = rotated_mask(torch.from_numpy(resized.astype(np.float32)), (5.0,)).numpy()
            sampling = sampling_matrix(5.0, REFERENCE_SIZE, REFERENCE_SIZE)
            yy, xx = np.nonzero(warped)
            centered = np.stack([xx + 0.5 - REFERENCE_SIZE / 2, yy + 0.5 - REFERENCE_SIZE / 2, np.ones(xx.shape[0])])
            source = sampling @ centered
            src_col = np.rint(source[0] + REFERENCE_SIZE / 2 - 0.5).astype(int)
            src_row = np.rint(source[1] + REFERENCE_SIZE / 2 - 0.5).astype(int)
            inside = (src_row >= 0) & (src_row < REFERENCE_SIZE) & (src_col >= 0) & (src_col < REFERENCE_SIZE)
            hits = int(resized[src_row[inside], src_col[inside]].sum()) if inside.any() else 0
            range_rows.append(
                {
                    "source_id": stem,
                    "condition": "mask_plus5_correspondence",
                    "min": float("nan"),
                    "max": float("nan"),
                    "mean": float("nan"),
                    "corner_pixel_max": float("nan"),
                    "invalid_pixel_max": float("nan"),
                    "sampled_foreground_pixels": int(xx.size),
                    "source_foreground_fraction": hits / int(xx.size) if xx.size else float("nan"),
                    "nearest_bilinear_max_abs_on_m5": bilinear_gap,
                }
            )
    margin = _smallest_margin()
    unrotated_crop_matches_original = _crop_matches_original(margin, ())
    zero_degree_crop_matches = _crop_matches_original(margin, (0.0,))
    usage = shutil.disk_usage(audit_dir)
    return {
        "selection": audit_ids,
        "figures": panels,
        "figure_rows": "RGB, absolute RGB difference scaled by its own maximum, geometric valid region",
        "figure_columns": ["original", "synthetic_m5", "synthetic_p5", "restore_m5", "restore_p5"],
        "center_xy": center,
        "positive_direction": "counter-clockwise",
        "marker_before_row_col": [200, 400],
        "marker_after_plus5_row_col": [sign_row, sign_col],
        "plus5_moved_marker_up": sign_row < 200,
        "center_block_centroid_row_col": centroid,
        "matrix_matches_saved_max_abs": matrix_gap,
        "minus5_then_plus5_sampling_gap_from_identity": composition_gap,
        "inverse_of_plus5_max_abs_vs_minus5": float(np.max(np.abs(inverse_matrix(plus) - minus))),
        "prenorm_ranges": range_rows,
        "interpolation": {"image": "bilinear", "mask": "nearest then > 0", "fill": 0, "expand": False},
        "normalization_order": "RGB geometry is stored before ImageNet normalization; prenorm values stay in [0, 1]",
        "bilinear_vs_nearest_max_abs_on_first_sheet_recorded_per_anomaly": True,
        "proposed_control": {
            "name": "reflect_pad_center_crop",
            "padding": "reflect",
            "margin_pixels": margin,
            "margin_rule": "smallest integer margin whose center 512 crop never mixes with outside fill under bilinear, for every current angle list",
            "margin_not_chosen_by_test_score": True,
            "unrotated_center_crop_matches_original": unrotated_crop_matches_original,
            "zero_degree_center_crop_matches_unpadded_zero_degree": zero_degree_crop_matches,
            "conditions_to_score": ["synthetic_m5", "synthetic_p5", "restore_m5", "restore_p5"],
            "source_images": 195,
            "forwards": 780,
            "reuse_bank": "results_a0/bank/pcb__patchcore__seed0.pt",
            "output_dir": "results_a0_boundary",
            "disk_free_bytes": usage.free,
        },
    }


def _smallest_margin() -> int:
    lo, hi = 0, 160
    answer = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if all(_crop_avoids_fill(mid, angles) for _name, angles in CONDITIONS):
            answer = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return answer


def _crop_avoids_fill(margin: int, angles: tuple[float, ...]) -> bool:
    if not angles:
        return True
    side = REFERENCE_SIZE + 2 * margin
    ones = torch.ones((1, side, side), dtype=torch.float32)
    rotated, _steps = apply_angles(ones, angles, nearest=False)
    crop = rotated[0, margin : margin + REFERENCE_SIZE, margin : margin + REFERENCE_SIZE]
    return bool(torch.all(crop >= 1.0 - 1e-5))


def _crop_matches_original(margin: int, angles: tuple[float, ...]) -> bool:
    """Center crop after reflect padding. Empty angles are not resampled."""

    generator = torch.Generator().manual_seed(0)
    image = torch.rand((3, REFERENCE_SIZE, REFERENCE_SIZE), generator=generator)
    if not angles:
        return True
    canvas = image
    if margin:
        canvas = torch.nn.functional.pad(image, (margin, margin, margin, margin), mode="reflect")
    warped, _steps = apply_angles(canvas, angles, nearest=False)
    crop = warped[:, margin : margin + REFERENCE_SIZE, margin : margin + REFERENCE_SIZE]
    reference, _steps = apply_angles(image, angles, nearest=False)
    return bool(torch.allclose(crop, reference, atol=1e-6))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_readout(data_root: Path, result_dir: Path) -> dict[str, object]:
    images = list(csv.DictReader((result_dir / "tables" / "images.csv").open(encoding="utf-8")))
    components = list(csv.DictReader((result_dir / "tables" / "components.csv").open(encoding="utf-8")))
    thresholds = json.loads((result_dir / "thresholds.json").read_text(encoding="utf-8"))
    patch_threshold = float(thresholds["patch"]["threshold"])
    with np.load(result_dir / "valid_regions.npz") as payload:
        valid = {name: np.asarray(payload[name], dtype=bool) for name, _angles in CONDITIONS}
        common = np.asarray(payload["common"], dtype=bool)
    maps = _load_maps(result_dir / "maps")
    per_image, max_rows, patch_summary = patch_readout(images, maps, valid, patch_threshold)
    geometry, check = component_geometry(components, maps, valid, common, data_root)
    del check["masks"]
    audit_ids = select_audit_ids(images)
    audit = geometric_audit(data_root, audit_ids, result_dir / "matrices.json", result_dir / "audit")
    readout_dir = result_dir / "readout"
    readout_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(readout_dir / "image_patch_fpr.csv", per_image)
    _write_csv(readout_dir / "max_patch_location.csv", max_rows)
    _write_csv(readout_dir / "component_geometry.csv", geometry)
    payload = {
        "frozen_result_dir": str(result_dir),
        "did_not_modify": ["summary.json", "thresholds.json", "tables", "maps", "matrices.json", "valid_regions.npz", "bank"],
        "image_score_distribution": image_score_rows(images),
        "image_threshold": thresholds["image"]["threshold"],
        "patch_threshold": patch_threshold,
        "patch_summary": patch_summary,
        "component_recompute": check,
        "component_comparison": component_comparison(geometry),
        "separation": separation(geometry, per_image, maps, valid, patch_threshold),
        "audit": {key: value for key, value in audit.items() if key != "prenorm_ranges"},
        "audit_ranges": audit["prenorm_ranges"],
        "statistics_notes": {
            "pooled_patch_interval_is_not_image_uncertainty": True,
            "paired_wald_zero_width_is_not_equivalence": True,
            "components_are_not_independent_within_an_image": True,
            "observed_zero_disagreement_is_not_equivalence_on_new_samples": True,
        },
    }
    (readout_dir / "readout.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    payload = run_readout(Path(r"K:\RobustAD"), Path("results_a0"))
    print(json.dumps({"margin": payload["audit"]["proposed_control"], "recompute": payload["component_recompute"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
