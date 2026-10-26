"""Score reflect-padded ±5° PCB images after inverse-rotating fused features.

195 source images and two angles are 390 evaluations. The 144-image bank and
the frozen thresholds are read and not refit. results_a0 and
results_a0_boundary are comparison inputs, not outputs.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import Normalize, Resize

from robustad.benchmark import IMAGE_NET_MEAN, IMAGE_NET_STD, build_model
from robustad.prepare.spec import VERDICT_UNDECIDED
from robustad.theme_a.action import select_board_roi_stems
from robustad.theme_a.boundary import (
    MARGIN,
    _load_bank,
    _paired_decisions,
    _quantile_row,
    _read_csv,
    _test_items,
    _write_csv,
    extend_and_warp,
)
from robustad.theme_a.components import component_record
from robustad.theme_a.feature_warp import reflection_reference_mask, score_embedding, score_fused_feature_inverse
from robustad.theme_a.run_synthetic import _components, _load_rgb
from robustad.theme_a.warp import REFERENCE_SIZE

LOGGER = logging.getLogger("robustad.theme_a.feature")

OUTPUT_NAME = "results_a0_feature"
CONDITIONS = (
    {"name": "feature_m5", "image_degrees": -5.0, "feature_degrees": 5.0, "rotated": "pad_synthetic_m5", "round_trip": "pad_restore_m5"},
    {"name": "feature_p5", "image_degrees": 5.0, "feature_degrees": -5.0, "rotated": "pad_synthetic_p5", "round_trip": "pad_restore_p5"},
)
PREFLIGHT_STEMS = (
    "Test_PCB_resized_None_None_Normal_Set2_181",
    "Test_PCB_resized_None_None_Normal_Set2_241",
    "Test_PCB_None_None_Missing Component_1",
    "Test_PCB_None_None_Scratch_20",
)
IMAGE_THRESHOLD = 36.28824043273926
PATCH_THRESHOLD = 24.215826511383057
BATCH_SIZE = 2
MAP_BYTES = 195 * 2 * 64 * 64 * 4
FORBIDDEN = {"results_a0", "results_a0_boundary", "results_a1", "results_p0", "results", "results_tv", "diag"}


def _refuse(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if resolved.name in FORBIDDEN or (resolved.name == "diag" and "results_p0" in resolved.parts):
        raise RuntimeError(f"保護された出力先です: {resolved}")
    if "results_p0" in resolved.parts and "diag" in resolved.parts:
        raise RuntimeError(f"results_p0/diag は変更しません: {resolved}")
    return resolved


def _cells(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    return array.reshape(64, 8, 64, 8).any(axis=(1, 3))


def _inside_statistic(patch: np.ndarray, component: np.ndarray, reflection: np.ndarray) -> tuple[float, int, bool]:
    cells = _cells(component)
    chosen = patch[cells]
    inside = patch[cells & ~reflection]
    statistic = float(chosen.max()) if chosen.size else float("nan")
    inside_statistic = float(inside.max()) if inside.size else float("nan")
    flat = int(np.argmax(np.where(cells, patch, -np.inf)))
    row, col = divmod(flat, patch.shape[1])
    return inside_statistic, int((cells & reflection).sum()), bool(reflection[row, col]) if cells.any() else False


def write_roi_pack(data_root: Path, pack_dir: Path) -> dict[str, object]:
    """Save the eight review images. Masks stay absent until a person paints them."""

    pack_dir.mkdir(parents=True, exist_ok=True)
    image_dir = pack_dir / "images"
    image_dir.mkdir(exist_ok=True)
    items = [(path, label, _mask) for path, label, _mask in _test_items(data_root) if label == 0]
    selected = select_board_roi_stems([path.stem for path, _label, _mask in items])
    by_stem = {path.stem: path for path, _label, _mask in items}
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    written = []
    for item in selected:
        destination = image_dir / f"{item['stem']}.png"
        if not destination.is_file():
            tensor = _load_rgb(by_stem[str(item["stem"])], resize)
            array = tensor.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
            Image.fromarray(array).save(destination)
        written.append(str(item["stem"]))
    payload = {
        "status": "unconfirmed",
        "ground_truth": False,
        "masks_created": False,
        "rule": "sorted PCB SD test-normal stems, round(i * 119 / 7) for i = 0..7",
        "stems": selected,
        "images": written,
        "instruction": "正規化前の512画像を見て、基板面だけを masks/<stem>.png に塗る。塗るまで confirmed にしない。",
    }
    (pack_dir / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _preflight(torch_model, items: list[tuple[Path, int, str]], normalize: Normalize, resize: Resize) -> dict[str, object]:
    by_stem = {path.stem: path for path, _label, _mask in items}
    missing = [stem for stem in PREFLIGHT_STEMS if stem not in by_stem]
    if missing:
        raise RuntimeError(f"実行前確認の画像がありません: {missing}")
    device = torch_model.memory_bank.device
    rows = []
    worst_score = 0.0
    worst_patch = 0.0
    for stem in PREFLIGHT_STEMS:
        rgb = _load_rgb(by_stem[stem], resize)
        for image_degrees in (-5.0, 5.0):
            rotated = extend_and_warp(rgb, (float(image_degrees),), nearest=False)[0]
            normalized = normalize(rotated).unsqueeze(0).to(device)
            with torch.no_grad():
                reference = torch_model(normalized).pred_score.detach()
                embedding = score_fused_feature_inverse(torch_model, normalized, 0.0)
                direct = score_embedding(torch_model, embedding["unwarped_embedding"])
            score_gap = float((embedding["pred_score"].detach() - reference).abs().max())
            direct_gap = float((direct["pred_score"].detach() - reference).abs().max())
            patch_gap = float((embedding["patch_map"].detach() - direct["patch_map"].detach()).abs().max())
            worst_score = max(worst_score, score_gap, direct_gap)
            worst_patch = max(worst_patch, patch_gap)
            rows.append(
                {
                    "source_id": stem,
                    "image_degrees": image_degrees,
                    "zero_warp_versus_model_forward": score_gap,
                    "unwarped_reimplementation_versus_model_forward": direct_gap,
                    "zero_warp_versus_unwarped_patch": patch_gap,
                }
            )
    if worst_score > 1e-4 or worst_patch > 1e-4:
        raise RuntimeError(f"0°特徴ワープが元経路と一致しません: score {worst_score} patch {worst_patch}")
    return {"stems": list(PREFLIGHT_STEMS), "max_abs_score": worst_score, "max_abs_patch": worst_patch, "rows": rows}


def _compare(image_rows: list[dict[str, object]], boundary_rows: list[dict[str, str]], reference_name: str, condition: str) -> dict[str, object]:
    boundary = {(row["source_id"], row["condition"]): row for row in boundary_rows}
    current = {(str(row["source_id"]), str(row["condition"])): row for row in image_rows}
    ids = [str(row["source_id"]) for row in image_rows if row["condition"] == condition]
    labels = np.asarray([int(current[(item, condition)]["label"]) for item in ids])
    ours = np.asarray([float(current[(item, condition)]["reproduction_score"]) for item in ids], dtype=np.float64)
    reference = np.asarray([float(boundary[(item, reference_name)]["reproduction_score"]) for item in ids], dtype=np.float64)
    ours_hit = np.asarray([bool(current[(item, condition)]["detected"]) for item in ids])
    reference_hit = np.asarray([boundary[(item, reference_name)]["detected"] == "True" for item in ids])
    return {
        "reference_condition": reference_name,
        "score_delta_feature_minus_reference": _quantile_row(ours - reference),
        "normal_score_delta": _quantile_row(ours[labels == 0] - reference[labels == 0]),
        "anomaly_score_delta": _quantile_row(ours[labels == 1] - reference[labels == 1]),
        "decisions": _paired_decisions(reference_hit, ours_hit),
        "normal_decisions": _paired_decisions(reference_hit[labels == 0], ours_hit[labels == 0]),
        "anomaly_decisions": _paired_decisions(reference_hit[labels == 1], ours_hit[labels == 1]),
        "observed_zero_disagreement_is_not_equivalence": True,
    }


def _component_image_delta(component_rows: list[dict[str, object]], boundary_components: list[dict[str, str]], condition: str, reference: str) -> dict[str, object]:
    def maxima(rows: list[dict[str, object]], key_condition: str) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            if str(row["condition"]) != key_condition:
                continue
            value = float(row["statistic_common"])
            if np.isfinite(value):
                grouped.setdefault(str(row["source_id"]), []).append(value)
        return {source_id: max(values) for source_id, values in grouped.items()}

    ours = maxima(component_rows, condition)
    reference_max = maxima(boundary_components, reference)
    shared = sorted(set(ours) & set(reference_max))
    delta = np.asarray([ours[item] - reference_max[item] for item in shared], dtype=np.float64)
    disagreements = 0
    ours_detected: dict[str, dict[int, bool]] = {}
    reference_detected: dict[str, dict[int, bool]] = {}
    for row in component_rows:
        if str(row["condition"]) == condition:
            ours_detected.setdefault(str(row["source_id"]), {})[int(row["component_id"])] = bool(row["detected"])
    for row in boundary_components:
        if row["condition"] == reference:
            reference_detected.setdefault(row["source_id"], {})[int(row["component_id"])] = row["detected"] == "True"
    for source_id, flags in ours_detected.items():
        other = reference_detected.get(source_id, {})
        for component_id, detected in flags.items():
            if component_id in other and detected != other[component_id]:
                disagreements += 1
    return {
        "unit": "one maximum per anomaly image",
        "n_images": len(shared),
        "delta": _quantile_row(delta),
        "component_detection_disagreements": disagreements,
        "disagreement_is_not_selectivity_or_equivalence": True,
        "mask_frame": "feature inverse uses the unrotated 512 mask; the reference uses that condition's warped mask",
    }


def run_feature_inverse(data_root: Path, output_dir: Path, source_dir: Path, boundary_dir: Path, roi_dir: Path) -> dict[str, object]:
    output_dir = _refuse(output_dir)
    if (output_dir / "summary.json").is_file():
        raise RuntimeError(f"既存の特徴逆回転結果は上書きしません: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads((source_dir / "thresholds.json").read_text(encoding="utf-8"))
    image_cut = float(thresholds["image"]["threshold"])
    patch_cut = float(thresholds["patch"]["threshold"])
    if image_cut != IMAGE_THRESHOLD or patch_cut != PATCH_THRESHOLD:
        raise RuntimeError(f"しきい値が凍結値と違います: {image_cut}, {patch_cut}")
    free = shutil.disk_usage(output_dir).free
    if free < MAP_BYTES + 200_000_000:
        raise RuntimeError(f"空き容量が不足します: {free}")
    roi_manifest = write_roi_pack(data_root, roi_dir)
    reflections = {item["name"]: reflection_reference_mask(item["feature_degrees"]) for item in CONDITIONS}
    np.savez(output_dir / "feature_reference.npz", **{name: mask.numpy() for name, mask in reflections.items()})
    started = time.perf_counter()
    manifest = {
        "insertion": {
            "function": "robustad.theme_a.feature_warp.score_fused_feature_inverse",
            "tensor": "embedding",
            "producer": "PatchcoreModel.generate_embedding",
            "consumer_after_warp": "PatchcoreModel.reshape_embedding",
            "shape": [None, 1536, 64, 64],
            "score": "PatchcoreModel.compute_anomaly_score on the warped rows",
            "stored_scores_reused": False,
        },
        "grid_sample": {
            "mode": "bilinear",
            "padding_mode": "reflection",
            "align_corners": False,
            "matches": "torchvision rotate / _gen_affine_grid",
        },
        "conditions": [dict(item) for item in CONDITIONS],
        "forwards": 390,
        "batch_size": BATCH_SIZE,
        "thresholds_recomputed": False,
        "image_threshold": image_cut,
        "patch_threshold": patch_cut,
        "bank": str(source_dir / "bank" / "pcb__patchcore__seed0.pt"),
        "refit": False,
        "map_bytes": MAP_BYTES,
        "disk_free_bytes_before_score": free,
        "roi_pack": str(roi_dir),
        "roi_status": roi_manifest["status"],
        "roi_metrics": False,
        "reflection_cells": {name: int(mask.sum()) for name, mask in reflections.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "thresholds_used.json").write_text(
        json.dumps({"source": str(source_dir / "thresholds.json"), "recomputed": False, "image_threshold": image_cut, "patch_threshold": patch_cut}, indent=2),
        encoding="utf-8",
    )
    lightning_model = build_model("patchcore", REFERENCE_SIZE, "timm")
    bank = _load_bank(lightning_model, source_dir / "bank" / "pcb__patchcore__seed0.pt")
    torch_model = lightning_model.model
    normalize = Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    items = _test_items(data_root)
    preflight = _preflight(torch_model, items, normalize, resize)
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    LOGGER.info("preflight max abs score %s patch %s", preflight["max_abs_score"], preflight["max_abs_patch"])
    score_started = time.perf_counter()
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    fpr_rows: list[dict[str, object]] = []
    location_rows: list[dict[str, object]] = []
    ones = np.ones((REFERENCE_SIZE, REFERENCE_SIZE), dtype=bool)
    device = bank.device
    loaded = []
    for index, (path, label, mask_path) in enumerate(items):
        components, alone = _components(mask_path) if label == 1 else ([], {})
        loaded.append((index, path, label, components, alone, _load_rgb(path, resize)))
    for spec in CONDITIONS:
        reflection = reflections[spec["name"]].numpy()
        batch_start = 0
        while batch_start < len(loaded):
            chunk = loaded[batch_start : batch_start + BATCH_SIZE]
            normalized = torch.stack(
                [normalize(extend_and_warp(item[5], (spec["image_degrees"],), nearest=False)[0]) for item in chunk]
            ).to(device)
            with torch.no_grad():
                scored = score_fused_feature_inverse(torch_model, normalized, spec["feature_degrees"])
            if not torch.equal(scored["reflection"], reflections[spec["name"]]):
                raise RuntimeError("反射参照マスクが角度ごとの固定マスクと違います。")
            for offset, (index, path, label, components, alone, _rgb) in enumerate(chunk):
                score = float(scored["pred_score"][offset].detach().cpu())
                grid = scored["patch_map"][offset].detach().cpu().numpy().astype(np.float32, copy=False)
                np.savez(
                    map_dir / f"{index:05d}__{spec['name']}.npz",
                    image_id=np.array(path.stem),
                    condition=np.array(spec["name"]),
                    label=np.int32(label),
                    reproduction_score=np.float32(score),
                    patch_map=grid,
                    reflection=reflection,
                    image_degrees=np.float32(spec["image_degrees"]),
                    feature_degrees=np.float32(spec["feature_degrees"]),
                )
                image_rows.append(
                    {
                        "source_id": path.stem,
                        "condition": spec["name"],
                        "label": label,
                        "reproduction_score": score,
                        "detected": bool(score >= image_cut),
                        "image_threshold": image_cut,
                        "patch_threshold": patch_cut,
                    }
                )
                flat = int(np.argmax(grid))
                row_i, col_i = divmod(flat, grid.shape[1])
                location_rows.append(
                    {
                        "source_id": path.stem,
                        "condition": spec["name"],
                        "label": label,
                        "argmax_row": row_i,
                        "argmax_col": col_i,
                        "argmax_is_reflection_reference": bool(reflection[row_i, col_i]),
                        "n_reflection_cells": int(reflection.sum()),
                    }
                )
                if label == 0:
                    full_hits = int((grid >= patch_cut).sum())
                    inside = grid[~reflection]
                    inside_hits = int((inside >= patch_cut).sum())
                    fpr_rows.append(
                        {
                            "source_id": path.stem,
                            "condition": spec["name"],
                            "n_full": int(grid.size),
                            "n_false_full": full_hits,
                            "fpr_full": full_hits / grid.size,
                            "n_inside_grid": int(inside.size),
                            "n_false_inside_grid": inside_hits,
                            "fpr_inside_grid": inside_hits / inside.size,
                        }
                    )
                for row in components:
                    component_id = int(row["component_id"])
                    mask = alone.get(component_id)
                    record = component_record(
                        component_id=component_id,
                        status_input=str(row["status"]),
                        area_original=int(row["area_original"]),
                        area_512=int(row["area_resized_alone"]),
                        warped=None if mask is None else mask,
                        condition_valid=ones,
                        common_valid=ones,
                        patch_map=grid,
                        threshold=patch_cut,
                    )
                    if mask is None:
                        record.update({"statistic_inside_grid": float("nan"), "n_reflection_cells": 0, "max_cell_is_reflection": False})
                    else:
                        inside_statistic, n_reflection, max_is_reflection = _inside_statistic(grid, mask, reflection)
                        record.update(
                            {
                                "statistic_inside_grid": inside_statistic,
                                "n_reflection_cells": n_reflection,
                                "max_cell_is_reflection": max_is_reflection,
                            }
                        )
                    record.update({"source_id": path.stem, "condition": spec["name"]})
                    component_rows.append(record)
            batch_start += BATCH_SIZE
            LOGGER.info("feature %s %d / 195", spec["name"], min(batch_start, 195))
    score_seconds = time.perf_counter() - score_started
    _write_csv(output_dir / "tables" / "images.csv", image_rows)
    _write_csv(output_dir / "tables" / "components.csv", component_rows)
    _write_csv(output_dir / "tables" / "image_patch_fpr.csv", fpr_rows)
    _write_csv(output_dir / "tables" / "max_patch_location.csv", location_rows)
    boundary_images = _read_csv(boundary_dir / "tables" / "images.csv")
    boundary_components = _read_csv(boundary_dir / "tables" / "components.csv")
    achieved = {}
    comparisons = {}
    for spec in CONDITIONS:
        name = spec["name"]
        rows = [row for row in image_rows if row["condition"] == name]
        normals = [row for row in rows if row["label"] == 0]
        anomalies = [row for row in rows if row["label"] == 1]
        fprs = [row for row in fpr_rows if row["condition"] == name]
        components = [row for row in component_rows if row["condition"] == name]
        locations = [row for row in location_rows if row["condition"] == name and row["label"] == 0]
        achieved[name] = {
            "anomaly_detected": int(sum(bool(row["detected"]) for row in anomalies)),
            "anomaly_n": len(anomalies),
            "normal_detected": int(sum(bool(row["detected"]) for row in normals)),
            "normal_n": len(normals),
            "image_score": {
                "normal": _quantile_row(np.asarray([float(row["reproduction_score"]) for row in normals])),
                "anomaly": _quantile_row(np.asarray([float(row["reproduction_score"]) for row in anomalies])),
            },
            "per_image_normal_patch_fpr_full": _quantile_row(np.asarray([float(row["fpr_full"]) for row in fprs])),
            "per_image_normal_patch_fpr_inside_grid": _quantile_row(np.asarray([float(row["fpr_inside_grid"]) for row in fprs])),
            "image_decision_includes_reflection": True,
            "pooled_patch_interval_omitted": True,
            "normal_argmax_on_reflection": int(sum(bool(row["argmax_is_reflection_reference"]) for row in locations)),
            "component_detected": int(sum(bool(row["detected"]) for row in components)),
            "component_n": len(components),
            "component_saturation_is_not_selectivity": True,
            "input_unevaluable": int(sum(row["status_input"] == "入力変換による評価不能" for row in components)),
        }
        comparisons[name] = {
            "against_rotated": _compare(image_rows, boundary_images, spec["rotated"], name),
            "against_image_round_trip": _compare(image_rows, boundary_images, spec["round_trip"], name),
            "component_against_rotated": _component_image_delta(component_rows, boundary_components, name, spec["rotated"]),
            "component_against_image_round_trip": _component_image_delta(component_rows, boundary_components, name, spec["round_trip"]),
        }
    summary = {
        "verdict": VERDICT_UNDECIDED,
        "called_board_surface_improvement": False,
        "roi_metrics": False,
        "output_dir": str(output_dir),
        "image_threshold": image_cut,
        "patch_threshold": patch_cut,
        "thresholds_recomputed": False,
        "score_seconds": score_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "conditions": achieved,
        "comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["score_seconds"] = score_seconds
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_feature_inverse(
        Path(r"K:\RobustAD"),
        Path("results_a0_feature"),
        Path("results_a0"),
        Path("results_a0_boundary"),
        Path("roi_pending") / "pcb_sd_normal8",
    )
