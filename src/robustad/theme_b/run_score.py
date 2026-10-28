"""Score the shared 41 px series, then the 106 px corrections.

Existing theme A directories are not written. A finished stage is not scored again.
Calibration scores are not recomputed.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import Normalize, Resize

from robustad.benchmark import BATCH_SIZE, IMAGE_NET_MEAN, IMAGE_NET_STD, build_model
from robustad.prepare.capture import MapCapture, attach_patchcore_capture
from robustad.theme_a.boundary import _test_items, fill_probe
from robustad.theme_a.components import component_record
from robustad.theme_a.run_synthetic import _components, _load_rgb, _score_batch
from robustad.theme_a.warp import REFERENCE_SIZE
from robustad.theme_b.correct import correction_mask, correction_rgb, series_mask, series_rgb
from robustad.theme_b.estimate import ReferenceFeatures, diagnostic_added_angle
from robustad.theme_b.geometry import CORRECTION_MARGIN, SERIES_CONDITIONS, SERIES_MARGIN
from robustad.theme_b.protocol import assert_new_output
from robustad.theme_b.reference import resolve_reference_path

LOGGER = logging.getLogger("robustad.theme_b")
DATA_ROOT = Path(r"K:\RobustAD")
OUTPUT = Path("results_b0")
SMOKE_DIR = Path("results_b0_smoke")
BANK_SHAPE = (58982, 1536)
MINIMUM_FREE_BYTES = 3_000_000_000
CONTROL_BANK = Path("results_a0_control0")
THREE_BANK = Path("results_a0_augment")
STAGE2 = (
    ("known_m10", "series_m10", -10.0, "known"),
    ("known_p10", "series_p10", 10.0, "known"),
    ("estimated_0", "series_0", 0.0, "estimated"),
    ("estimated_m10", "series_m10", -10.0, "estimated"),
    ("estimated_p10", "series_p10", 10.0, "estimated"),
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _status(path: Path) -> dict:
    if not path.is_file():
        return {"state": "absent"}
    return json.loads(path.read_text(encoding="utf-8"))


def _uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def _float_image(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.astype(np.float32) / 255.0))


def _require_space(path: Path) -> int:
    anchor = path if path.exists() else path.parent
    free = shutil.disk_usage(anchor).free
    if free < MINIMUM_FREE_BYTES:
        raise RuntimeError(f"空き容量が {MINIMUM_FREE_BYTES} バイト未満です: {free}")
    return free


def _load_torch_model(bank_dir: Path):
    bank_path = bank_dir / "bank" / "pcb__patchcore__seed0.pt"
    saved = torch.load(bank_path, map_location="cpu", weights_only=False)
    if tuple(saved["memory_bank"].shape) != BANK_SHAPE:
        raise RuntimeError(f"バンク形状が 58982×1536 ではありません: {tuple(saved['memory_bank'].shape)}")
    if bool(saved.get("spatial_constraint", False)):
        raise RuntimeError("空間制約入りのバンクは使いません。")
    lightning_model = build_model("patchcore", REFERENCE_SIZE, "timm")
    lightning_model.model.memory_bank = saved["memory_bank"]
    lightning_model.model.num_neighbors = int(saved["num_neighbors"])
    lightning_model.model.eval()
    if not torch.cuda.is_available():
        raise RuntimeError("採点には CUDA が必要です。")
    lightning_model.cuda()
    torch_model = lightning_model.model
    capture = MapCapture()
    attach_patchcore_capture(torch_model, capture)
    return torch_model, capture, Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)


def _thresholds(bank_dir: Path) -> tuple[float, float]:
    payload = json.loads((bank_dir / "thresholds.json").read_text(encoding="utf-8"))
    return float(payload["image"]["threshold"]), float(payload["patch"]["threshold"])


def _fill_mask(degrees: float, margin: int) -> np.ndarray:
    probe = fill_probe((float(degrees),), margin)
    return (probe >= 1.0 - 1e-5).numpy()


def _component_masks(alone: dict[int, np.ndarray], added: float, correction: float | None) -> dict[int, np.ndarray]:
    warped = {}
    for component_id, mask in alone.items():
        tensor = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
        series = series_mask(tensor, added)
        if correction is None:
            warped[component_id] = (series > 0).numpy()
        else:
            warped[component_id] = (correction_mask(series, correction) > 0).numpy()
    return warped


def build_series_cache(output_dir: Path, items: list[tuple[Path, int, str]]) -> np.ndarray:
    status_path = output_dir / "series_status.json"
    cache_path = output_dir / "series_uint8.npy"
    shape = (len(items), len(SERIES_CONDITIONS), 3, REFERENCE_SIZE, REFERENCE_SIZE)
    if _status(status_path).get("state") == "complete" and cache_path.is_file():
        cached = np.load(cache_path, mmap_mode="r")
        if tuple(cached.shape) == shape:
            return cached
    _require_space(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(status_path, {"state": "running", "n_images": len(items)})
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8, shape=shape)
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    matrices = {}
    started = time.perf_counter()
    for index, (path, _label, _mask) in enumerate(items):
        rgb = _load_rgb(path, resize)
        for column, (name, degrees) in enumerate(SERIES_CONDITIONS):
            image, steps = series_rgb(rgb, degrees)
            cache[index, column] = _uint8(image)
            if index == 0:
                matrices[name] = {
                    "added_degrees": degrees,
                    "margin_pixels": SERIES_MARGIN,
                    "pad": "reflect",
                    "interpolation": "bilinear",
                    "crop": "center 512 once",
                    "steps": steps,
                }
        if index % 25 == 0:
            cache.flush()
            LOGGER.info("series cache %d / %d", index + 1, len(items))
    cache.flush()
    del cache
    valid = {name: _fill_mask(degrees, SERIES_MARGIN) for name, degrees in SERIES_CONDITIONS}
    np.savez(output_dir / "series_valid.npz", **{f"fill_{name}": mask for name, mask in valid.items()})
    _write_json(output_dir / "series_matrices.json", matrices)
    _write_json(
        output_dir / "series_index.json",
        {"source_id": [path.stem for path, _label, _mask in items], "conditions": [name for name, _degrees in SERIES_CONDITIONS]},
    )
    _write_json(status_path, {"state": "complete", "seconds": time.perf_counter() - started, "margin_pixels": SERIES_MARGIN})
    return np.load(cache_path, mmap_mode="r")


def _score_images(torch_model, capture, normalize, images: list[torch.Tensor]) -> tuple[list[float], list[np.ndarray], float]:
    scores: list[float] = []
    grids: list[np.ndarray] = []
    _sync()
    started = time.perf_counter()
    batch = BATCH_SIZE["patchcore"]
    for start in range(0, len(images), batch):
        predicted, patch = _score_batch(torch_model, images[start : start + batch], capture, normalize)
        _sync()
        scores.extend(float(value) for value in predicted)
        grids.extend(patch[offset].astype(np.float32, copy=False) for offset in range(len(predicted)))
    return scores, grids, time.perf_counter() - started


def _write_stage_tables(stage_dir: Path, rows: list[dict]) -> None:
    table_dir = stage_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    image_rows = [row for item in rows for row in item["images"]]
    component_rows = [row for item in rows for row in item["components"]]
    fpr_rows = [row for item in rows for row in item["fpr"]]
    estimate_rows = [row for item in rows for row in item["estimates"]]
    for name, content in (
        ("images.csv", image_rows),
        ("components.csv", component_rows),
        ("image_patch_fpr.csv", fpr_rows),
        ("estimates.csv", estimate_rows),
    ):
        if not content:
            continue
        with (table_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(content[0].keys()))
            writer.writeheader()
            writer.writerows(content)


def _quantile_median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.5, method="linear"))


def _summarize_stage(stage_dir: Path, image_cut: float, patch_cut: float) -> dict:
    image_rows = []
    component_rows = []
    fpr_rows = []
    estimate_rows = []
    score_seconds = 0.0
    estimate_seconds = 0.0
    warp_seconds = 0.0
    for path in sorted((stage_dir / "rows").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        image_rows.extend(payload["images"])
        component_rows.extend(payload["components"])
        fpr_rows.extend(payload["fpr"])
        estimate_rows.extend(payload["estimates"])
        score_seconds += float(payload.get("score_seconds") or 0.0)
        estimate_seconds += float(payload.get("estimate_seconds") or 0.0)
        warp_seconds += float(payload.get("warp_seconds") or 0.0)
    conditions = sorted({row["condition"] for row in image_rows})
    report = {}
    for condition in conditions:
        images = [row for row in image_rows if row["condition"] == condition]
        normal = [row for row in images if int(row["label"]) == 0]
        anomaly = [row for row in images if int(row["label"]) == 1]
        components = [row for row in component_rows if row["condition"] == condition]
        rates = [float(row["fpr_full"]) for row in fpr_rows if row["condition"] == condition]
        estimates = [row for row in estimate_rows if row["condition"] == condition]
        failures = [row for row in estimates if row["fallback"]]
        reasons: dict[str, int] = {}
        for row in failures:
            key = str(row.get("reason") or "failed")
            reasons[key] = reasons.get(key, 0) + 1
        report[condition] = {
            "normal_detected": f"{sum(bool(row['detected']) for row in normal)}/{len(normal)}",
            "anomaly_detected": f"{sum(bool(row['detected']) for row in anomaly)}/{len(anomaly)}",
            "components_detected": f"{sum(bool(row['detected']) for row in components)}/{len(components)}",
            "normal_patch_fpr_median": _quantile_median(rates),
            "estimates": len(estimates),
            "fallbacks": len(failures),
            "fallback_reasons": reasons,
        }
    summary = {
        "image_threshold": image_cut,
        "patch_threshold": patch_cut,
        "thresholds_recomputed": False,
        "conditions": report,
        "score_seconds": score_seconds,
        "estimate_seconds": estimate_seconds,
        "warp_seconds": warp_seconds,
        "auxiliary_auroc_not_used_for_selection": True,
        "roi_metrics": False,
        "verdict": "未判定",
    }
    _write_json(stage_dir / "summary.json", summary)
    return summary


def _stage_complete(stage_dir: Path, n_images: int) -> bool:
    status = _status(stage_dir / "status.json")
    rows = list((stage_dir / "rows").glob("*.json")) if (stage_dir / "rows").is_dir() else []
    return status.get("state") == "complete" and len(rows) == n_images


def run_formal_smoke(data_root: Path = DATA_ROOT, output_dir: Path = SMOKE_DIR) -> dict:
    """Two fixed images through the 41 px series and the 106 px correction."""

    assert_new_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    done = output_dir / "formal_smoke.json"
    if done.is_file():
        payload = json.loads(done.read_text(encoding="utf-8"))
        if payload.get("passed") is True:
            return payload
    items = _test_items(data_root)
    chosen = [next(item for item in items if item[1] == 0), next(item for item in items if item[1] == 1)]
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    reference = _load_rgb(resolve_reference_path(data_root), resize)
    features = ReferenceFeatures(reference, reference_path_stem(data_root))
    torch_model, capture, normalize = _load_torch_model(CONTROL_BANK)
    image_cut, patch_cut = _thresholds(CONTROL_BANK)
    cases = []
    for path, label, mask_path in chosen:
        rgb = _load_rgb(path, resize)
        _components_rows, alone = _components(mask_path) if label == 1 else ([], {})
        generated = {}
        estimates = {}
        for name, degrees in (("series_0", 0.0), ("series_m10", -10.0), ("series_p10", 10.0)):
            image, steps = series_rgb(rgb, degrees)
            if tuple(image.shape) != (3, REFERENCE_SIZE, REFERENCE_SIZE):
                raise RuntimeError(f"系列画像の形状が違います: {tuple(image.shape)}")
            generated[name] = image
            estimates[name] = features.estimate(_uint8(image))
            if steps[0]["degrees"] != degrees:
                raise RuntimeError("系列の変換行列の角度が違います。")
        uncorrected_scores, uncorrected_maps, _seconds = _score_images(
            torch_model, capture, normalize, [generated["series_0"], generated["series_p10"]]
        )
        if any(not np.isfinite(score) for score in uncorrected_scores):
            raise RuntimeError("未補正スコアが有限ではありません。")
        if any(tuple(grid.shape) != (64, 64) for grid in uncorrected_maps):
            raise RuntimeError(f"パッチ格子の形状が違います: {[tuple(grid.shape) for grid in uncorrected_maps]}")
        known, known_steps = correction_rgb(generated["series_p10"], -10.0)
        if tuple(known.shape) != (3, REFERENCE_SIZE, REFERENCE_SIZE):
            raise RuntimeError("既知補正の形状が違います。")
        if known_steps[0]["degrees"] != -10.0:
            raise RuntimeError("既知補正の角度が違います。")
        estimate = estimates["series_p10"]
        if estimate.correction_degrees() is None:
            estimated = generated["series_p10"]
            warped = False
        else:
            estimated, _estimated_steps = correction_rgb(generated["series_p10"], float(estimate.correction_degrees()))
            warped = True
        if tuple(estimated.shape) != (3, REFERENCE_SIZE, REFERENCE_SIZE):
            raise RuntimeError("推定補正の形状が違います。")
        scored, maps, _seconds = _score_images(torch_model, capture, normalize, [known, estimated])
        if any(not np.isfinite(score) for score in scored) or any(tuple(grid.shape) != (64, 64) for grid in maps):
            raise RuntimeError("補正後のスコア経路が確認できません。")
        mask_ok = True
        if alone:
            sample = next(iter(alone.values()))
            series = series_mask(torch.from_numpy(sample.astype(np.float32)), 10.0)
            corrected = correction_mask(series, -10.0)
            mask_ok = tuple(corrected.shape[-2:]) == (REFERENCE_SIZE, REFERENCE_SIZE)
        fill = _fill_mask(-10.0, CORRECTION_MARGIN)
        cases.append(
            {
                "source_id": path.stem,
                "label": label,
                "series_shape": [3, REFERENCE_SIZE, REFERENCE_SIZE],
                "patch_shape": [64, 64],
                "uncorrected_0_score": uncorrected_scores[0],
                "uncorrected_p10_score": uncorrected_scores[1],
                "known_p10_score": scored[0],
                "estimated_p10_score": scored[1],
                "estimated_p10_warped": warped,
                "estimated_p10_fallback": estimate.correction_degrees() is None,
                "estimated_p10_reason": estimate.fallback_reason(),
                "estimated_p10_degrees": estimate.angle_degrees,
                "diagnostic_p10": None
                if estimate.angle_degrees is None or estimates["series_0"].angle_degrees is None
                else diagnostic_added_angle(estimate.angle_degrees, estimates["series_0"].angle_degrees),
                "image_threshold": image_cut,
                "patch_threshold": patch_cut,
                "mask_shape_ok": mask_ok,
                "correction_fill_valid_pixels": int(fill.sum()),
                "correction_margin_pixels": CORRECTION_MARGIN,
                "series_margin_pixels": SERIES_MARGIN,
                "parameters_changed": False,
            }
        )
    payload = {
        "passed": True,
        "patchcore_forwards": 8,
        "parameter_update": False,
        "cases": cases,
        "note": "この smoke のスコアで参照、採用数、比、残差閾値は変えていない。",
    }
    _write_json(done, payload)
    del torch_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def reference_path_stem(data_root: Path) -> str:
    return resolve_reference_path(data_root).stem


def _run_stage1(output_dir: Path, items: list, cache: np.ndarray, bank_dir: Path, stage_name: str) -> dict:
    stage_dir = output_dir / stage_name
    if _stage_complete(stage_dir, len(items)):
        LOGGER.info("skip complete %s", stage_name)
        return json.loads((stage_dir / "summary.json").read_text(encoding="utf-8"))
    _require_space(output_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    row_dir = stage_dir / "rows"
    map_dir = stage_dir / "maps"
    row_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    image_cut, patch_cut = _thresholds(bank_dir)
    _write_json(
        stage_dir / "thresholds_used.json",
        {"source": str(bank_dir / "thresholds.json"), "recomputed": False, "image_threshold": image_cut, "patch_threshold": patch_cut},
    )
    _write_json(stage_dir / "status.json", {"state": "running"})
    torch_model, capture, normalize = _load_torch_model(bank_dir)
    torch.cuda.reset_peak_memory_stats()
    wall = time.perf_counter()
    valid = {name: _fill_mask(degrees, SERIES_MARGIN) for name, degrees in SERIES_CONDITIONS}
    try:
        for index, (path, label, mask_path) in enumerate(items):
            row_path = row_dir / f"{index:05d}.json"
            if row_path.is_file():
                continue
            components, alone = _components(mask_path) if label == 1 else ([], {})
            images = [_float_image(cache[index, column]) for column in range(len(SERIES_CONDITIONS))]
            scores, grids, score_seconds = _score_images(torch_model, capture, normalize, images)
            image_rows = []
            fpr_rows = []
            component_rows = []
            estimate_rows = []
            for column, (name, degrees) in enumerate(SERIES_CONDITIONS):
                score = scores[column]
                grid = grids[column]
                np.savez(
                    map_dir / f"{index:05d}__{name}.npz",
                    image_id=np.array(path.stem),
                    condition=np.array(name),
                    label=np.int32(label),
                    reproduction_score=np.float32(score),
                    patch_map=grid,
                )
                image_rows.append(
                    {
                        "source_id": path.stem,
                        "condition": name,
                        "added_degrees": degrees,
                        "label": label,
                        "reproduction_score": score,
                        "detected": bool(score >= image_cut),
                        "image_threshold": image_cut,
                        "patch_threshold": patch_cut,
                    }
                )
                estimate_rows.append(
                    {
                        "source_id": path.stem,
                        "condition": name,
                        "added_degrees": degrees,
                        "estimated_degrees": "",
                        "diagnostic_added_degrees": "",
                        "correction_degrees": "",
                        "n_candidates": "",
                        "n_inliers": "",
                        "residual_rms_px": "",
                        "fallback": False,
                        "reason": "not_estimated",
                        "reference_id": "",
                    }
                )
                if label == 0:
                    hits = int((grid >= patch_cut).sum())
                    fpr_rows.append(
                        {
                            "source_id": path.stem,
                            "condition": name,
                            "n_full": int(grid.size),
                            "n_false_full": hits,
                            "fpr_full": hits / int(grid.size),
                        }
                    )
                masks = _component_masks(alone, degrees, None)
                for row in components:
                    component_id = int(row["component_id"])
                    record = component_record(
                        component_id=component_id,
                        status_input=str(row["status"]),
                        area_original=int(row["area_original"]),
                        area_512=int(row["area_resized_alone"]),
                        warped=masks.get(component_id),
                        condition_valid=valid[name],
                        common_valid=valid[name],
                        patch_map=grid,
                        threshold=patch_cut,
                    )
                    record.update({"source_id": path.stem, "condition": name})
                    component_rows.append(record)
            _write_json(
                row_path,
                {
                    "source_id": path.stem,
                    "score_seconds": score_seconds,
                    "images": image_rows,
                    "components": component_rows,
                    "fpr": fpr_rows,
                    "estimates": estimate_rows,
                },
            )
            if index % 25 == 0:
                LOGGER.info("%s %d / %d", stage_name, index + 1, len(items))
    except Exception:
        _write_json(stage_dir / "status.json", {"state": "failed", "seconds": time.perf_counter() - wall})
        raise
    summary = _summarize_stage(stage_dir, image_cut, patch_cut)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(row_dir.glob("*.json"))]
    _write_stage_tables(stage_dir, rows)
    summary["wall_seconds"] = time.perf_counter() - wall
    summary["gpu_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    summary["bank"] = str(bank_dir)
    _write_json(stage_dir / "summary.json", summary)
    _write_json(stage_dir / "status.json", {"state": "complete", "n_images": len(items), "seconds": summary["wall_seconds"]})
    del torch_model
    torch.cuda.empty_cache()
    return summary


def _run_stage2(output_dir: Path, data_root: Path, items: list, cache: np.ndarray) -> dict:
    stage_dir = output_dir / "stage2_control0"
    if _stage_complete(stage_dir, len(items)):
        LOGGER.info("skip complete stage2")
        return json.loads((stage_dir / "summary.json").read_text(encoding="utf-8"))
    _require_space(output_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    row_dir = stage_dir / "rows"
    map_dir = stage_dir / "maps"
    row_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    image_cut, patch_cut = _thresholds(CONTROL_BANK)
    _write_json(
        stage_dir / "thresholds_used.json",
        {
            "source": str(CONTROL_BANK / "thresholds.json"),
            "recomputed": False,
            "image_threshold": image_cut,
            "patch_threshold": patch_cut,
            "correction_input": "series uint8 512",
            "correction_margin_pixels": CORRECTION_MARGIN,
            "known_uses_series_canvas": False,
        },
    )
    _write_json(stage_dir / "status.json", {"state": "running"})
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    reference_stem = reference_path_stem(data_root)
    features = ReferenceFeatures(_load_rgb(resolve_reference_path(data_root), resize), reference_stem)
    torch_model, capture, normalize = _load_torch_model(CONTROL_BANK)
    torch.cuda.reset_peak_memory_stats()
    wall = time.perf_counter()
    name_to_column = {name: column for column, (name, _degrees) in enumerate(SERIES_CONDITIONS)}
    try:
        for index, (path, label, mask_path) in enumerate(items):
            row_path = row_dir / f"{index:05d}.json"
            if row_path.is_file():
                continue
            components, alone = _components(mask_path) if label == 1 else ([], {})
            needed = ("series_0", "series_m10", "series_p10")
            generated = {name: _float_image(cache[index, name_to_column[name]]) for name in needed}
            estimate_started = time.perf_counter()
            estimates = {}
            for source_name in needed:
                column = name_to_column[source_name]
                estimates[source_name] = features.estimate(np.ascontiguousarray(cache[index, column]))
            estimate_seconds = time.perf_counter() - estimate_started
            prepared = []
            meta = []
            warp_seconds = 0.0
            for condition, source_name, added, kind in STAGE2:
                source = generated[source_name]
                estimate = estimates[source_name]
                diagnostic = None
                if estimate.angle_degrees is not None and estimates["series_0"].angle_degrees is not None:
                    diagnostic = diagnostic_added_angle(estimate.angle_degrees, estimates["series_0"].angle_degrees)
                if kind == "known":
                    correction = -added
                    fallback = False
                    reason = ""
                else:
                    correction = estimate.correction_degrees()
                    fallback = correction is None
                    reason = estimate.fallback_reason()
                warp_started = time.perf_counter()
                if correction is None:
                    image = source
                    steps = []
                    valid = _fill_mask(added, SERIES_MARGIN)
                else:
                    image, steps = correction_rgb(source, correction)
                    valid = _fill_mask(correction, CORRECTION_MARGIN)
                warp_seconds += time.perf_counter() - warp_started
                prepared.append(image)
                meta.append(
                    {
                        "condition": condition,
                        "source_condition": source_name,
                        "added_degrees": added,
                        "kind": kind,
                        "estimate": estimate,
                        "diagnostic": diagnostic,
                        "correction": correction,
                        "fallback": fallback,
                        "reason": reason,
                        "steps": steps,
                        "valid": valid,
                    }
                )
            scores, grids, score_seconds = _score_images(torch_model, capture, normalize, prepared)
            image_rows = []
            fpr_rows = []
            component_rows = []
            estimate_rows = []
            for offset, item in enumerate(meta):
                score = scores[offset]
                grid = grids[offset]
                condition = item["condition"]
                np.savez(
                    map_dir / f"{index:05d}__{condition}.npz",
                    image_id=np.array(path.stem),
                    condition=np.array(condition),
                    label=np.int32(label),
                    reproduction_score=np.float32(score),
                    patch_map=grid,
                )
                image_rows.append(
                    {
                        "source_id": path.stem,
                        "condition": condition,
                        "added_degrees": item["added_degrees"],
                        "label": label,
                        "reproduction_score": score,
                        "detected": bool(score >= image_cut),
                        "image_threshold": image_cut,
                        "patch_threshold": patch_cut,
                    }
                )
                estimate = item["estimate"]
                estimate_rows.append(
                    {
                        "source_id": path.stem,
                        "condition": condition,
                        "added_degrees": item["added_degrees"],
                        "estimated_degrees": "" if estimate.angle_degrees is None else estimate.angle_degrees,
                        "diagnostic_added_degrees": "" if item["diagnostic"] is None else item["diagnostic"],
                        "correction_degrees": "" if item["correction"] is None else item["correction"],
                        "n_candidates": estimate.n_candidates,
                        "n_inliers": estimate.n_inliers,
                        "residual_rms_px": "" if estimate.residual_rms_px is None else estimate.residual_rms_px,
                        "fallback": item["fallback"],
                        "reason": item["reason"],
                        "reference_id": reference_stem,
                        "input_image": "series_uint8_512",
                        "series_margin_pixels": SERIES_MARGIN,
                        "correction_margin_pixels": "" if item["correction"] is None else CORRECTION_MARGIN,
                        "transform_steps": item["steps"],
                        "fill_valid_pixels": int(item["valid"].sum()),
                        "fill_pixels": int(item["valid"].size),
                    }
                )
                if label == 0:
                    hits = int((grid >= patch_cut).sum())
                    fpr_rows.append(
                        {
                            "source_id": path.stem,
                            "condition": condition,
                            "n_full": int(grid.size),
                            "n_false_full": hits,
                            "fpr_full": hits / int(grid.size),
                        }
                    )
                masks = _component_masks(alone, item["added_degrees"], None if item["fallback"] and item["kind"] == "estimated" else item["correction"])
                for row in components:
                    component_id = int(row["component_id"])
                    record = component_record(
                        component_id=component_id,
                        status_input=str(row["status"]),
                        area_original=int(row["area_original"]),
                        area_512=int(row["area_resized_alone"]),
                        warped=masks.get(component_id),
                        condition_valid=item["valid"],
                        common_valid=item["valid"],
                        patch_map=grid,
                        threshold=patch_cut,
                    )
                    record.update({"source_id": path.stem, "condition": condition})
                    component_rows.append(record)
            _write_json(
                row_path,
                {
                    "source_id": path.stem,
                    "estimate_seconds": estimate_seconds,
                    "warp_seconds": warp_seconds,
                    "score_seconds": score_seconds,
                    "images": image_rows,
                    "components": component_rows,
                    "fpr": fpr_rows,
                    "estimates": estimate_rows,
                },
            )
            if index % 25 == 0:
                LOGGER.info("stage2 %d / %d", index + 1, len(items))
    except Exception:
        _write_json(stage_dir / "status.json", {"state": "failed", "seconds": time.perf_counter() - wall})
        raise
    summary = _summarize_stage(stage_dir, image_cut, patch_cut)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(row_dir.glob("*.json"))]
    _write_stage_tables(stage_dir, rows)
    summary["wall_seconds"] = time.perf_counter() - wall
    summary["gpu_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    summary["reference_seconds"] = features.seconds
    summary["bank"] = str(CONTROL_BANK)
    summary["reference_id"] = reference_stem
    _write_json(stage_dir / "summary.json", summary)
    _write_json(stage_dir / "status.json", {"state": "complete", "n_images": len(items), "seconds": summary["wall_seconds"]})
    del torch_model
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    output_dir = assert_new_output(OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    free = _require_space(output_dir)
    items = _test_items(DATA_ROOT)
    _write_json(output_dir / "disk_before.json", {"free_bytes": free})
    smoke = run_formal_smoke()
    if smoke.get("passed") is not True:
        raise RuntimeError("正式経路の smoke が通っていません。")
    cache = build_series_cache(output_dir, items)
    stage1_control = _run_stage1(output_dir, items, cache, CONTROL_BANK, "stage1_control0")
    stage1_three = _run_stage1(output_dir, items, cache, THREE_BANK, "stage1_three")
    stage2 = _run_stage2(output_dir, DATA_ROOT, items, cache)
    _write_json(
        output_dir / "run_status.json",
        {
            "state": "complete",
            "stage1_control0": stage1_control.get("conditions"),
            "stage1_three": stage1_three.get("conditions"),
            "stage2_control0": stage2.get("conditions"),
            "fit": False,
            "recalibrated": False,
            "roi_metrics": False,
        },
    )
    LOGGER.info("theme B scoring complete")


if __name__ == "__main__":
    main()
