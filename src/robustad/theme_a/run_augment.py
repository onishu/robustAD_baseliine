"""Build one 3-view PatchCore bank and score the four approved conditions.

The 1536-d candidate matrix stays on CPU. A finished summary is not overwritten,
and a saved bank is not fit again when scoring fails.
"""

from __future__ import annotations

import ctypes
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import Normalize, Resize

from robustad.benchmark import IMAGE_NET_MEAN, IMAGE_NET_STD, accelerate_patchcore_coreset, build_model
from robustad.catalog import SUBSETS
from robustad.data import list_images
from robustad.prepare.calibration import fpr_report, image_threshold, patch_threshold, pool_patch_scores
from robustad.prepare.capture import MapCapture, attach_patchcore_capture
from robustad.prepare.spec import TARGET_IMAGE_FPR, TARGET_PATCH_FPR, VERDICT_UNDECIDED
from robustad.theme_a.action import (
    AUGMENT_DECISION,
    AUGMENT_OUTPUT_NAME,
    AUGMENT_QUANTILE,
    AUGMENT_TEST_CONDITIONS,
    AUGMENT_ZERO_PATH_REASON,
    BASELINE_CORESET,
    FUSED_CHANNELS,
    FUSED_GRID,
    LAYER2_STRIDE_PX,
    OLD_IMAGE_THRESHOLD,
    OLD_PATCH_THRESHOLD,
    PROVENANCE_FIELDS,
    SPATIAL_CONSTRAINT_IN_AUGMENT,
    assert_augment_sources,
    assert_integer_coreset,
    augment_storage,
    augment_views,
    candidate_index,
    projection_components,
    ratio_matching_coreset,
    reference_cell_center,
    required_free_ram_bytes,
)
from robustad.theme_a.boundary import (
    FILL_EPS,
    _cell_fully_valid,
    _paired_decisions,
    _quantile_row,
    _read_csv,
    _test_items,
    _write_csv,
    content_probe,
    extend_and_warp,
    fill_probe,
)
from robustad.theme_a.components import component_record
from robustad.theme_a.feature_warp import fused_embedding
from robustad.theme_a.run_synthetic import _components, _load_rgb, _score_batch, _stem_from_key
from robustad.theme_a.warp import REFERENCE_SIZE

LOGGER = logging.getLogger("robustad.theme_a.augment")

FORBIDDEN = {
    "results_a0",
    "results_a0_boundary",
    "results_a0_feature",
    "results_a1",
    "results_p0",
    "results",
    "results_tv",
    "diag",
}
BANK_NAME = "pcb__patchcore__seed0.pt"
BATCH_SIZE = 2
REFERENCE_TABLES = {
    "sd_original": ("baseline", "original"),
    "pad_warp_0": ("boundary", "pad_warp_0"),
    "pad_synthetic_m5": ("boundary", "pad_synthetic_m5"),
    "pad_synthetic_p5": ("boundary", "pad_synthetic_p5"),
}


def refuse_augment_run(
    output_dir: Path,
    *,
    confirmed: bool,
    capacity_ok: bool = False,
    manifest_saved: bool = False,
) -> None:
    """Open only the approved output after capacity is checked and the manifest is on disk.

    A finished summary is never replaced. A saved bank is not a reason to fit again.
    """

    resolved = output_dir.resolve()
    if resolved.name in FORBIDDEN or resolved.name != AUGMENT_OUTPUT_NAME:
        raise RuntimeError(f"①の出力先は {AUGMENT_OUTPUT_NAME} だけです: {resolved}")
    if "results_p0" in resolved.parts and "diag" in resolved.parts:
        raise RuntimeError(f"results_p0/diag は変更しません: {resolved}")
    if (resolved / "summary.json").is_file():
        raise RuntimeError(f"既存の①結果は上書きしません: {resolved}")
    if not confirmed:
        raise RuntimeError("3ビューの fit には --confirm-augment-fit が必要です。")
    if not capacity_ok:
        raise RuntimeError("空きメモリ、GPU、ディスクの確認を満たしていません。")
    if not manifest_saved:
        raise RuntimeError("実行前 manifest が保存されていません。")


def physical_memory() -> dict[str, int]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = (
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        )

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("空きメモリを読めません。")
    return {"total_bytes": int(status.ullTotalPhys), "free_bytes": int(status.ullAvailPhys)}


def capacity_snapshot(output_dir: Path) -> dict[str, object]:
    storage = augment_storage()
    ram = physical_memory()
    usage = shutil.disk_usage(output_dir if output_dir.exists() else output_dir.parent)
    gpu = {"available": bool(torch.cuda.is_available())}
    if gpu["available"]:
        free, total = torch.cuda.mem_get_info()
        gpu.update({"free_bytes": int(free), "total_bytes": int(total), "name": torch.cuda.get_device_name(0)})
    required_ram = required_free_ram_bytes()
    required_disk = int(storage["saved_bytes"]) + 2 * 1024**3
    required_gpu = int(storage["projected_bytes"]) * 2 + 1024**3
    ok = (
        ram["free_bytes"] >= required_ram
        and usage.free >= required_disk
        and gpu["available"]
        and int(gpu.get("free_bytes", 0)) >= required_gpu
    )
    return {
        "storage": storage,
        "ram": ram,
        "disk_free_bytes": int(usage.free),
        "gpu": gpu,
        "required_free_ram_bytes": required_ram,
        "required_free_disk_bytes": required_disk,
        "required_free_gpu_bytes": required_gpu,
        "ok": ok,
    }


def write_view_features(
    storage: torch.Tensor,
    image_indexes: list[int],
    view_index: int,
    embedding: torch.Tensor,
    *,
    n_views: int = 3,
) -> None:
    """Copy one view's fused rows into the preallocated CPU matrix.

    The persistent copy is the destination slice. A batch-sized CPU tensor is
    the only temporary, and the 1536-d matrix is never moved to CUDA.
    """

    if storage.device.type != "cpu" or storage.dtype != torch.float32 or not storage.is_contiguous():
        raise RuntimeError("候補特徴の格納先は連続した CPU float32 です。")
    if embedding.dtype != torch.float32:
        raise RuntimeError("候補特徴が float32 ではありません。型変換による複製はしません。")
    if embedding.ndim != 4 or embedding.shape[0] != len(image_indexes) or embedding.shape[1] != FUSED_CHANNELS:
        raise RuntimeError(f"融合特徴の形状が違います: {tuple(embedding.shape)}")
    ordered = embedding.detach().permute(0, 2, 3, 1).reshape(len(image_indexes), -1, FUSED_CHANNELS)
    if ordered.device.type != "cpu":
        ordered = ordered.to("cpu")
    cells = int(ordered.shape[1])
    if cells != FUSED_GRID * FUSED_GRID:
        raise RuntimeError(f"1ビューのセル数が 4096 ではありません: {cells}")
    for offset, image_index in enumerate(image_indexes):
        start = candidate_index(image_index, view_index, 0, 0, n_views=n_views)
        storage[start : start + cells].copy_(ordered[offset])
    del ordered


def _view_flags(angles: tuple[float, ...]) -> dict[str, np.ndarray]:
    if not angles:
        return {
            "from_reflect_pad": np.zeros((FUSED_GRID, FUSED_GRID), dtype=bool),
            "geometrically_valid": np.ones((FUSED_GRID, FUSED_GRID), dtype=bool),
        }
    content = (content_probe(angles) >= (1.0 - FILL_EPS)).numpy()
    valid = (fill_probe(angles) >= (1.0 - FILL_EPS)).numpy()
    content_all = content.reshape(FUSED_GRID, LAYER2_STRIDE_PX, FUSED_GRID, LAYER2_STRIDE_PX).all(axis=(1, 3))
    valid_all = valid.reshape(FUSED_GRID, LAYER2_STRIDE_PX, FUSED_GRID, LAYER2_STRIDE_PX).all(axis=(1, 3))
    return {"from_reflect_pad": ~content_all, "geometrically_valid": valid_all}


def _decode(index: int, n_views: int) -> tuple[int, int, int, int]:
    block = FUSED_GRID * FUSED_GRID
    col = int(index % FUSED_GRID)
    row = int((index // FUSED_GRID) % FUSED_GRID)
    view = int((index // block) % n_views)
    image = int(index // (n_views * block))
    return image, view, row, col


def _fit_paths(data_root: Path, manifest_path: Path) -> tuple[list[Path], list[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fit_stems = {_stem_from_key(key) for key in manifest["datasets"]["pcb"]["fit"]}
    calibration_stems = {_stem_from_key(key) for key in manifest["datasets"]["pcb"]["calibration"]}
    subset = SUBSETS["pcb"]
    by_stem = {path.stem: path for path in list_images(subset.train_dir(data_root) / "normal")}
    if set(by_stem) != fit_stems | calibration_stems:
        raise RuntimeError("固定 manifest と train 正常画像のファイル名が一致しません。")
    test_ids = [path.stem for path, _label, _mask in _test_items(data_root)]
    assert_augment_sources(sorted(fit_stems), sorted(calibration_stems), test_ids)
    if len(fit_stems) != 144 or len(calibration_stems) != 36:
        raise RuntimeError(f"fit 144 / calibration 36 ではありません: {len(fit_stems)} / {len(calibration_stems)}")
    return [by_stem[stem] for stem in sorted(fit_stems)], [by_stem[stem] for stem in sorted(calibration_stems)]


def _condition_image(rgb: torch.Tensor, angles: tuple[float, ...]) -> torch.Tensor:
    if not angles:
        return rgb
    return extend_and_warp(rgb, angles, nearest=False)[0]


def _condition_mask(mask: np.ndarray | None, angles: tuple[float, ...]) -> np.ndarray | None:
    if mask is None:
        return None
    tensor = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
    if not angles:
        return tensor.numpy() > 0
    return extend_and_warp(tensor, angles, nearest=True, pad_mode="constant")[0].numpy() > 0


def _build_bank(output_dir: Path, fit_paths: list[Path], storage: dict[str, object]) -> dict[str, object]:
    started_flag = output_dir / "fit_started.json"
    bank_path = output_dir / "bank" / BANK_NAME
    if bank_path.is_file():
        LOGGER.info("saved bank exists; fit is not repeated")
        return {"fit_repeated": False, "fit_seconds": 0.0, "bank_path": str(bank_path)}
    if started_flag.is_file():
        raise RuntimeError("fit は開始済みでバンクがありません。自動では繰り返しません。")
    n_candidates = int(storage["n_candidates"])
    ratio = float(storage["coreset_ratio"])
    assert_integer_coreset(n_candidates, ratio, BASELINE_CORESET)
    if projection_components(n_candidates) != 355:
        raise RuntimeError("射影次元が 355 ではありません。")
    views = augment_views()
    ram_before = physical_memory()
    candidates = torch.full((n_candidates, FUSED_CHANNELS), float("nan"), dtype=torch.float32)
    if candidates.device.type != "cpu":
        raise RuntimeError("候補特徴が CPU にありません。")
    LOGGER.info("allocated candidates %s free_ram %s", tuple(candidates.shape), physical_memory()["free_bytes"])
    lightning_model = build_model("patchcore", REFERENCE_SIZE, "timm")
    torch_model = lightning_model.model
    torch_model.eval()
    if torch.cuda.is_available():
        lightning_model.cuda()
    normalize = Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    device = next(torch_model.parameters()).device
    started_flag.write_text(json.dumps({"started": True, "views": [dict(view) for view in views]}, ensure_ascii=False), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    fit_started = time.perf_counter()
    loaded = [_load_rgb(path, resize) for path in fit_paths]
    for view_index, view in enumerate(views):
        angles = () if not view["reflect_pad"] else (float(view["angle_degrees"]),)
        for batch_start in range(0, len(loaded), BATCH_SIZE):
            chunk = loaded[batch_start : batch_start + BATCH_SIZE]
            images = torch.stack([normalize(_condition_image(image, angles)) for image in chunk]).to(device)
            with torch.no_grad():
                embedding = fused_embedding(torch_model, images)
            write_view_features(candidates, list(range(batch_start, batch_start + len(chunk))), view_index, embedding)
            del embedding, images
        LOGGER.info("extracted view %s", view["angle_degrees"])
    del loaded
    if torch_model.embedding_store:
        raise RuntimeError("embedding_store に候補が残っています。")
    if not torch.isfinite(candidates).all():
        raise RuntimeError("候補特徴に未書き込みまたは非有限値があります。")
    extract_gpu_peak = int(torch.cuda.max_memory_allocated())
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated() > 3 * 1024**3:
        raise RuntimeError("抽出後も GPU に候補特徴が残っています。")
    from lightning.pytorch import seed_everything

    seed_everything(0, workers=True)
    accelerate_patchcore_coreset()
    from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy

    torch.cuda.reset_peak_memory_stats()
    sampler = KCenterGreedy(embedding=candidates, sampling_ratio=ratio)
    selected = [int(index) for index in sampler.select_coreset_idxs()]
    coreset_gpu_peak = int(torch.cuda.max_memory_allocated())
    if sampler.model.n_components != 355:
        raise RuntimeError(f"射影次元が 355 ではありません: {sampler.model.n_components}")
    if candidates.device.type != "cpu":
        raise RuntimeError("coreset 中に候補特徴が CPU から外れました。")
    if len(selected) != BASELINE_CORESET or len(set(selected)) != BASELINE_CORESET:
        raise RuntimeError(f"選択パッチ数が 58982 ちょうどではありません: {len(selected)}")
    assert_integer_coreset(n_candidates, ratio, BASELINE_CORESET)
    bank = candidates.index_select(0, torch.tensor(selected, dtype=torch.long)).contiguous()
    if tuple(bank.shape) != (BASELINE_CORESET, FUSED_CHANNELS):
        raise RuntimeError(f"バンク形状が違います: {tuple(bank.shape)}")
    del candidates
    del sampler.features
    del sampler
    torch.cuda.empty_cache()
    provenance, view_counts = _provenance(selected, [path.stem for path in fit_paths], views)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": bank,
            "num_neighbors": 9,
            "n_fit": 144,
            "n_views": 3,
            "seed": 0,
            "coreset_ratio": ratio,
            "spatial_constraint": False,
        },
        bank_path,
    )
    np.savez(output_dir / "bank" / "provenance.npz", **provenance)
    (output_dir / "bank" / "view_counts.json").write_text(json.dumps(view_counts, indent=2, ensure_ascii=False), encoding="utf-8")
    fit_seconds = time.perf_counter() - fit_started
    return {
        "fit_repeated": False,
        "fit_seconds": fit_seconds,
        "bank_path": str(bank_path),
        "initial_index": selected[0],
        "projection_components": 355,
        "extract_gpu_peak_bytes": extract_gpu_peak,
        "coreset_gpu_peak_bytes": coreset_gpu_peak,
        "ram_before_allocation": ram_before,
        "ram_after_fit": physical_memory(),
        "view_counts": view_counts,
        "n_selected": len(selected),
    }


def _provenance(selected: list[int], image_ids: list[str], views: tuple[dict[str, object], ...]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    flags = [_view_flags(() if not view["reflect_pad"] else (float(view["angle_degrees"]),)) for view in views]
    rows = len(selected)
    image_index = np.empty(rows, dtype=np.int16)
    view_id = np.empty(rows, dtype=np.int8)
    angle = np.empty(rows, dtype=np.float32)
    grid_row = np.empty(rows, dtype=np.int16)
    grid_col = np.empty(rows, dtype=np.int16)
    reference_x = np.empty(rows, dtype=np.float32)
    reference_y = np.empty(rows, dtype=np.float32)
    from_reflect = np.empty(rows, dtype=bool)
    valid = np.empty(rows, dtype=bool)
    inside = np.empty(rows, dtype=bool)
    names = np.empty(rows, dtype="U80")
    for bank_row, index in enumerate(selected):
        image, view, row, col = _decode(index, len(views))
        mapped = reference_cell_center(row, col, () if not views[view]["reflect_pad"] else (float(views[view]["angle_degrees"]),))
        image_index[bank_row] = image
        view_id[bank_row] = view
        angle[bank_row] = float(views[view]["angle_degrees"])
        grid_row[bank_row] = row
        grid_col[bank_row] = col
        reference_x[bank_row] = float(mapped["reference_x"])
        reference_y[bank_row] = float(mapped["reference_y"])
        inside[bank_row] = bool(mapped["inside_original_square"])
        from_reflect[bank_row] = bool(flags[view]["from_reflect_pad"][row, col])
        valid[bank_row] = bool(flags[view]["geometrically_valid"][row, col])
        names[bank_row] = image_ids[image]
    payload = {
        "coreset_index": np.asarray(selected, dtype=np.int64),
        "bank_row": np.arange(rows, dtype=np.int32),
        "fit_image_index": image_index,
        "fit_image_id": names,
        "view_id": view_id,
        "angle_degrees": angle,
        "grid_row": grid_row,
        "grid_col": grid_col,
        "reference_x": reference_x,
        "reference_y": reference_y,
        "from_reflect_pad": from_reflect,
        "geometrically_valid": valid,
        "inside_original_square": inside,
    }
    missing = [name for name in PROVENANCE_FIELDS if name not in payload and name not in {"fit_image_id"}]
    if missing:
        raise RuntimeError(f"由来の項目が欠けています: {missing}")
    counts = []
    for view in views:
        chosen = view_id == int(view["view_id"])
        counts.append(
            {
                "view_id": int(view["view_id"]),
                "angle_degrees": float(view["angle_degrees"]),
                "n_patches": int(chosen.sum()),
                "n_images": int(len(set(image_index[chosen].tolist()))),
                "quota_enforced": False,
            }
        )
    return payload, {"views": counts, "n_selected": rows, "spatial_constraint": False}


def _score(
    output_dir: Path,
    data_root: Path,
    calibration_paths: list[Path],
    baseline_dir: Path,
    boundary_dir: Path,
) -> dict[str, object]:
    bank_path = output_dir / "bank" / BANK_NAME
    lightning_model = build_model("patchcore", REFERENCE_SIZE, "timm")
    saved = torch.load(bank_path, map_location="cpu", weights_only=False)
    if tuple(saved["memory_bank"].shape) != (BASELINE_CORESET, FUSED_CHANNELS):
        raise RuntimeError(f"保存バンクが 58982×1536 ではありません: {tuple(saved['memory_bank'].shape)}")
    if bool(saved.get("spatial_constraint", False)):
        raise RuntimeError("空間制約がバンクに入っています。")
    lightning_model.model.memory_bank = saved["memory_bank"]
    lightning_model.model.num_neighbors = int(saved["num_neighbors"])
    lightning_model.model.eval()
    lightning_model.cuda()
    torch_model = lightning_model.model
    capture = MapCapture()
    attach_patchcore_capture(torch_model, capture)
    normalize = Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    calibration_started = time.perf_counter()
    image_scores = []
    patch_maps = []
    for path in calibration_paths:
        score, patch = _score_batch(torch_model, [_load_rgb(path, resize)], capture, normalize)
        image_scores.append(float(score[0]))
        patch_maps.append(patch[0])
    image_cut = image_threshold(np.asarray(image_scores, dtype=np.float32), TARGET_IMAGE_FPR)
    patch_scores = pool_patch_scores(patch_maps)
    patch_cut = patch_threshold(patch_scores, TARGET_PATCH_FPR)
    if image_cut == OLD_IMAGE_THRESHOLD and patch_cut == OLD_PATCH_THRESHOLD:
        LOGGER.info("recomputed thresholds numerically match the old bank; they were not copied")
    thresholds = {
        "recomputed": True,
        "reused_old_thresholds": False,
        "old_image_threshold_not_used": OLD_IMAGE_THRESHOLD,
        "old_patch_threshold_not_used": OLD_PATCH_THRESHOLD,
        "quantile": AUGMENT_QUANTILE,
        "decision": AUGMENT_DECISION,
        "ties": "counted as anomalous",
        "alpha": 0.05,
        "calibration": "untransformed_36",
        "fit_images_included": False,
        "recalibrated_per_condition": False,
        "image": fpr_report(np.asarray(image_scores, dtype=np.float64), image_cut, TARGET_IMAGE_FPR),
        "patch": fpr_report(patch_scores, patch_cut, TARGET_PATCH_FPR),
    }
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")
    calibration_seconds = time.perf_counter() - calibration_started
    ones = np.ones((REFERENCE_SIZE, REFERENCE_SIZE), dtype=bool)
    valid = {}
    content = {}
    for name, angles in AUGMENT_TEST_CONDITIONS:
        if not angles:
            valid[name] = ones
            content[name] = ones
        else:
            valid[name] = (fill_probe(angles) >= (1.0 - FILL_EPS)).numpy()
            content[name] = (content_probe(angles) >= (1.0 - FILL_EPS)).numpy()
    common = np.ones_like(ones)
    for mask in valid.values():
        common &= mask
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    fpr_rows: list[dict[str, object]] = []
    score_started = time.perf_counter()
    items = _test_items(data_root)
    for index, (path, label, mask_path) in enumerate(items):
        rgb = _load_rgb(path, resize)
        components, alone = _components(mask_path) if label == 1 else ([], {})
        prepared = [_condition_image(rgb, angles) for _name, angles in AUGMENT_TEST_CONDITIONS]
        for batch_start in range(0, len(prepared), BATCH_SIZE):
            names = AUGMENT_TEST_CONDITIONS[batch_start : batch_start + BATCH_SIZE]
            predicted, patch = _score_batch(torch_model, prepared[batch_start : batch_start + BATCH_SIZE], capture, normalize)
            for offset, (name, angles) in enumerate(names):
                score = float(predicted[offset])
                grid = patch[offset].astype(np.float32, copy=False)
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
                        "label": label,
                        "reproduction_score": score,
                        "detected": bool(score >= image_cut),
                        "image_threshold": image_cut,
                        "patch_threshold": patch_cut,
                    }
                )
                if label == 0:
                    hits = int((grid >= patch_cut).sum())
                    content_cells = _cell_fully_valid(content[name])
                    content_values = grid[content_cells]
                    content_hits = int((content_values >= patch_cut).sum()) if content_values.size else 0
                    fpr_rows.append(
                        {
                            "source_id": path.stem,
                            "condition": name,
                            "n_full": int(grid.size),
                            "n_false_full": hits,
                            "fpr_full": hits / int(grid.size),
                            "n_content": int(content_values.size),
                            "n_false_content": content_hits,
                            "fpr_content": content_hits / content_values.size if content_values.size else float("nan"),
                        }
                    )
                for row in components:
                    component_id = int(row["component_id"])
                    record = component_record(
                        component_id=component_id,
                        status_input=str(row["status"]),
                        area_original=int(row["area_original"]),
                        area_512=int(row["area_resized_alone"]),
                        warped=_condition_mask(alone.get(component_id), angles),
                        condition_valid=valid[name],
                        common_valid=common,
                        patch_map=grid,
                        threshold=patch_cut,
                    )
                    record.update({"source_id": path.stem, "condition": name})
                    component_rows.append(record)
        if index % 25 == 0:
            LOGGER.info("scored %d / 195", index + 1)
    score_seconds = time.perf_counter() - score_started
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(table_dir / "images.csv", image_rows)
    _write_csv(table_dir / "components.csv", component_rows)
    _write_csv(table_dir / "image_patch_fpr.csv", fpr_rows)
    summary = _summarize(
        image_rows,
        component_rows,
        fpr_rows,
        image_cut,
        patch_cut,
        baseline_dir,
        boundary_dir,
        calibration_seconds,
        score_seconds,
    )
    return summary


def _finite_component_map(rows: list[dict[str, object]], condition: str) -> dict[tuple[str, int], float]:
    mapped = {}
    for row in rows:
        if str(row["condition"]) != condition:
            continue
        value = float(row["statistic_common"])
        if np.isfinite(value):
            mapped[(str(row["source_id"]), int(row["component_id"]))] = value
    return mapped


def _summarize(
    image_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    fpr_rows: list[dict[str, object]],
    image_cut: float,
    patch_cut: float,
    baseline_dir: Path,
    boundary_dir: Path,
    calibration_seconds: float,
    score_seconds: float,
) -> dict[str, object]:
    baseline_images = _read_csv(baseline_dir / "tables" / "images.csv")
    boundary_images = _read_csv(boundary_dir / "tables" / "images.csv")
    baseline_components = _read_csv(baseline_dir / "tables" / "components.csv")
    boundary_components = _read_csv(boundary_dir / "tables" / "components.csv")
    image_lookup = {(row["source_id"], row["condition"]): row for row in baseline_images + boundary_images}
    directories = {"baseline": baseline_dir.name, "boundary": boundary_dir.name}
    achieved = {}
    comparisons = {}
    for name, _angles in AUGMENT_TEST_CONDITIONS:
        rows = [row for row in image_rows if row["condition"] == name]
        normals = [row for row in rows if int(row["label"]) == 0]
        anomalies = [row for row in rows if int(row["label"]) == 1]
        components = [row for row in component_rows if row["condition"] == name]
        fprs = [row for row in fpr_rows if row["condition"] == name]
        achieved[name] = {
            "anomaly_detected": int(sum(bool(row["detected"]) for row in anomalies)),
            "anomaly_n": len(anomalies),
            "normal_detected": int(sum(bool(row["detected"]) for row in normals)),
            "normal_n": len(normals),
            "image_score": {
                "normal": _quantile_row(np.asarray([float(row["reproduction_score"]) for row in normals])),
                "anomaly": _quantile_row(np.asarray([float(row["reproduction_score"]) for row in anomalies])),
            },
            "per_image_normal_patch_fpr": _quantile_row(np.asarray([float(row["fpr_full"]) for row in fprs])),
            "component_detected": int(sum(bool(row["detected"]) for row in components)),
            "component_n": len(components),
            "component_saturation_is_not_selectivity": True,
            "input_unevaluable": int(sum(row["status_input"] == "入力変換による評価不能" for row in components)),
        }
        source_name, reference_condition = REFERENCE_TABLES[name]
        ids = [str(row["source_id"]) for row in rows]
        labels = np.asarray([int(row["label"]) for row in rows])
        ours = np.asarray([float(row["reproduction_score"]) for row in rows], dtype=np.float64)
        reference_rows = [image_lookup[(item, reference_condition)] for item in ids]
        reference = np.asarray([float(row["reproduction_score"]) for row in reference_rows], dtype=np.float64)
        ours_hit = np.asarray([bool(row["detected"]) for row in rows])
        reference_hit = np.asarray([row["detected"] == "True" for row in reference_rows])
        ours_components = _finite_component_map(component_rows, name)
        reference_pool = baseline_components if source_name == "baseline" else boundary_components
        reference_components = _finite_component_map(reference_pool, reference_condition)
        shared = sorted(set(ours_components) & set(reference_components))
        component_delta = np.asarray(
            [ours_components[key] - reference_components[key] for key in shared],
            dtype=np.float64,
        )
        comparisons[name] = {
            "reference_result": directories[source_name],
            "reference_condition": reference_condition,
            "thresholds_differ": True,
            "score_delta_new_minus_reference": _quantile_row(ours - reference),
            "normal_score_delta": _quantile_row(ours[labels == 0] - reference[labels == 0]),
            "anomaly_score_delta": _quantile_row(ours[labels == 1] - reference[labels == 1]),
            "decisions_use_each_models_own_threshold": _paired_decisions(reference_hit, ours_hit),
            "component_score_delta": _quantile_row(component_delta),
            "component_pairs": len(shared),
            "paired_by": "source image id and original component id",
            "disagreement_is_not_equivalence": True,
        }
    return {
        "verdict": VERDICT_UNDECIDED,
        "called_board_surface_improvement": False,
        "roi_metrics": False,
        "image_threshold": image_cut,
        "patch_threshold": patch_cut,
        "thresholds_recomputed": True,
        "recalibrated_per_condition": False,
        "spatial_constraint": SPATIAL_CONSTRAINT_IN_AUGMENT,
        "calibration_seconds": calibration_seconds,
        "score_seconds": score_seconds,
        "conditions": achieved,
        "comparisons": comparisons,
    }


def run_augment(data_root: Path, output_dir: Path, manifest_path: Path, *, confirmed: bool) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if (output_dir / "summary.json").is_file():
        raise RuntimeError(f"既存の①結果は上書きしません: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = capacity_snapshot(output_dir)
    fit_paths, calibration_paths = _fit_paths(data_root, manifest_path)
    storage = augment_storage()
    views = augment_views()
    pre_manifest = {
        "approved": True,
        "output_dir": str(output_dir),
        "zero_path_reason": AUGMENT_ZERO_PATH_REASON,
        "views": [dict(view) for view in views],
        "candidate_order": "sorted fit stems, views (0, -5, +5), row-major cells matching reshape_embedding",
        "fit_image_ids": [path.stem for path in fit_paths],
        "calibration_image_ids": [path.stem for path in calibration_paths],
        "test_conditions": [{"name": name, "angles": list(angles)} for name, angles in AUGMENT_TEST_CONDITIONS],
        "coreset": {
            "method": "anomalib KCenterGreedy with the existing GPU distance loop",
            "initial_center_included": True,
            "projection": "anomalib SparseRandomProjection",
            "projection_eps": 0.9,
            "projection_components": 355,
            "matrix_storage": "dense float32 on the candidate device",
            "candidate_device": "cpu",
            "projected_device": "cuda",
            "seed_immediately_before_coreset": 0,
            "ratio": storage["coreset_ratio"],
            "n_candidates": storage["n_candidates"],
            "n_selected_assert": BASELINE_CORESET,
            "quota_enforced": False,
            "spatial_constraint": False,
        },
        "differences_from_existing_bank": [
            "ビューは未変換1枚から、未変換0°と反射延長±5°の3ビューになった。",
            "候補数は 589,824 から 1,769,472 になった。",
            "coreset比は 0.1 から 58982/1769472 になった。int(n*ratio) はどちらも 58982 である。",
            "同じ eps 0.9 の Johnson-Lindenstrauss 次元は候補数に依存し、328 から 355 になった。",
            "1536次元の候補は GPU に積まず CPU に置いた。既存の144枚は学習ステップ中に GPU 上へ蓄積していた。",
            "候補順はソートした fit 画像、ビュー、行優先である。既存バンクの連結順はシャッフルされた DataLoader で、インデックスは保存されていない。",
            "coreset の直前に seed 0 を入れ直す。既存バンクの初期点は lightning の fit 後の乱数状態で、保存されていない。",
            "同じものは、wide_resnet50_2、layer2 と layer3、同じプーリングと連結、num_neighbors 9、初期点を含む GPU 版 k-center、float32、ImageNet 正規化、入力512、抽出器の eval である。",
        ],
        "quantile": AUGMENT_QUANTILE,
        "decision": AUGMENT_DECISION,
        "reuse_old_thresholds": False,
        "capacity": snapshot,
        "do_not_modify": ["results_a0", "results_a0_boundary", "results_a0_feature", "results_a1", "results_p0/diag"],
    }
    manifest_path_out = output_dir / "pre_manifest.json"
    manifest_path_out.write_text(json.dumps(pre_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    refuse_augment_run(output_dir, confirmed=confirmed, capacity_ok=bool(snapshot["ok"]), manifest_saved=manifest_path_out.is_file())
    if not snapshot["ok"]:
        raise RuntimeError(f"容量が不足します: {snapshot}")
    fit_report = _build_bank(output_dir, fit_paths, storage)
    try:
        summary = _score(output_dir, data_root, calibration_paths, Path("results_a0"), Path("results_a0_boundary"))
    except Exception as exc:
        (output_dir / "score_error.json").write_text(
            json.dumps({"error": str(exc), "fit_repeated": False, "bank_saved": (output_dir / "bank" / BANK_NAME).is_file()}, ensure_ascii=False),
            encoding="utf-8",
        )
        raise
    summary["fit"] = fit_report
    summary["ram_after_score"] = physical_memory()
    summary["disk_free_bytes_after"] = int(shutil.disk_usage(output_dir).free)
    summary["gpu_peak_bytes_after_score"] = int(torch.cuda.max_memory_allocated())
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"K:\RobustAD"))
    parser.add_argument("--output-dir", type=Path, default=Path(AUGMENT_OUTPUT_NAME))
    parser.add_argument("--manifest", type=Path, default=Path("results_p0/research_split_manifest.json"))
    parser.add_argument("--confirm-augment-fit", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_augment(args.data_root, args.output_dir, args.manifest, confirmed=args.confirm_augment_fit)


if __name__ == "__main__":
    main()
