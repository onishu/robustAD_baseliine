"""Build one 0°-only control bank with the same builder as the 3-view bank.

Projection dim is fixed at 355. The initial center is the resolved 0° cell
143974, not the 3-view integer 430694. A saved bank is not fit again if scoring fails.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import Normalize, Resize

from robustad.benchmark import IMAGE_NET_MEAN, IMAGE_NET_STD, build_model
from robustad.prepare.calibration import fpr_report, image_threshold, patch_threshold, pool_patch_scores
from robustad.prepare.capture import MapCapture, attach_patchcore_capture
from robustad.prepare.spec import TARGET_IMAGE_FPR, TARGET_PATCH_FPR, VERDICT_UNDECIDED
from robustad.theme_a.action import (
    AUGMENT_DECISION,
    AUGMENT_QUANTILE,
    AUGMENT_TEST_CONDITIONS,
    AUGMENT_ZERO_PATH,
    AUGMENT_ZERO_PATH_REASON,
    BASELINE_CORESET,
    CONTROL0_FORBIDDEN_THREE_VIEW_INDEX_AS_CANDIDATE,
    CONTROL0_INITIAL_COL,
    CONTROL0_INITIAL_IMAGE_ID,
    CONTROL0_INITIAL_IMAGE_INDEX,
    CONTROL0_INITIAL_INDEX,
    CONTROL0_INITIAL_ROW,
    CONTROL0_N_CANDIDATES,
    CONTROL0_N_VIEWS,
    CONTROL0_OUTPUT_NAME,
    CONTROL0_PROJECTION_COMPONENTS,
    CONTROL0_RATIO,
    CONTROL0_THREE_VIEW_INITIAL_INDEX,
    FUSED_CHANNELS,
    FUSED_GRID,
    OLD_IMAGE_THRESHOLD,
    OLD_PATCH_THRESHOLD,
    PROVENANCE_FIELDS,
    assert_integer_coreset,
    candidate_index,
    control0_storage,
    projection_components,
    reference_cell_center,
    resolve_control0_initial_center,
)
from robustad.theme_a.boundary import (
    FILL_EPS,
    _cell_fully_valid,
    _paired_decisions,
    _quantile_row,
    _read_csv,
    _write_csv,
    content_probe,
    fill_probe,
)
from robustad.theme_a.components import component_record
from robustad.theme_a.feature_warp import fused_embedding
from robustad.theme_a.run_augment import (
    BATCH_SIZE,
    BANK_NAME,
    FORBIDDEN,
    _condition_image,
    _condition_mask,
    _decode,
    _finite_component_map,
    _fit_paths,
    _view_flags,
    physical_memory,
    write_view_features,
)
from robustad.theme_a.run_synthetic import _components, _load_rgb, _score_batch
from robustad.theme_a.warp import REFERENCE_SIZE

LOGGER = logging.getLogger("robustad.theme_a.control0")

PROTECTED = FORBIDDEN | {"results_a0_augment", "results_a0_feature"}
THREE_VIEW_AUGMENT = 37.689165115356445
THREE_VIEW_PATCH = 25.81197500228882


def refuse_control0_run(
    output_dir: Path,
    *,
    confirmed: bool,
    capacity_ok: bool = False,
    manifest_saved: bool = False,
) -> None:
    resolved = output_dir.resolve()
    if resolved.name in PROTECTED or resolved.name != CONTROL0_OUTPUT_NAME:
        raise RuntimeError(f"回転ビューなしの出力先は {CONTROL0_OUTPUT_NAME} だけです: {resolved}")
    if (resolved / "summary.json").is_file():
        raise RuntimeError(f"既存の回転ビューなし結果は上書きしません: {resolved}")
    if not confirmed:
        raise RuntimeError("回転ビューなしの fit には --confirm-control-fit が必要です。")
    if not capacity_ok:
        raise RuntimeError("空きメモリ、GPU、ディスクの確認を満たしていません。")
    if not manifest_saved:
        raise RuntimeError("実行前 manifest が保存されていません。")


def capacity_snapshot(output_dir: Path) -> dict[str, object]:
    storage = control0_storage()
    ram = physical_memory()
    usage = shutil.disk_usage(output_dir if output_dir.exists() else output_dir.parent)
    gpu = {"available": bool(torch.cuda.is_available())}
    if gpu["available"]:
        free, total = torch.cuda.mem_get_info()
        gpu.update({"free_bytes": int(free), "total_bytes": int(total), "name": torch.cuda.get_device_name(0)})
    ok = (
        ram["free_bytes"] >= int(storage["required_free_ram_bytes"])
        and usage.free >= int(storage["required_free_disk_bytes"])
        and gpu["available"]
        and int(gpu.get("free_bytes", 0)) >= int(storage["required_free_gpu_bytes"])
    )
    return {
        "storage": storage,
        "ram": ram,
        "disk_free_bytes": int(usage.free),
        "gpu": gpu,
        "required_free_ram_bytes": storage["required_free_ram_bytes"],
        "required_free_disk_bytes": storage["required_free_disk_bytes"],
        "required_free_gpu_bytes": storage["required_free_gpu_bytes"],
        "ok": ok,
    }


def _projection_library_version() -> str:
    try:
        return importlib.metadata.version("anomalib")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _matrix_sha256(matrix: torch.Tensor) -> str:
    array = matrix.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_projection_355(candidates: torch.Tensor) -> dict[str, object]:
    """Regenerate SparseRandomProjection at 355-d. The 3-view matrix was not saved.

    Projection runs after seed_everything(0) and before the fixed initial center
    is applied, so skipping torch.randint does not change the projection draw.
    """

    from anomalib.models.components.dimensionality_reduction.random_projection import SparseRandomProjection

    if candidates.shape != (CONTROL0_N_CANDIDATES, FUSED_CHANNELS):
        raise RuntimeError(f"候補形状が違います: {tuple(candidates.shape)}")
    jl = projection_components(CONTROL0_N_CANDIDATES)
    if jl != 328:
        raise RuntimeError(f"0°だけの JL 次元が 328 ではありません: {jl}")
    projector = SparseRandomProjection(eps=0.9, random_state=None)
    projector.n_components = CONTROL0_PROJECTION_COMPONENTS
    projector.sparse_random_matrix = projector._sparse_random_matrix(n_features=FUSED_CHANNELS)
    if projector.sparse_random_matrix.shape != (CONTROL0_PROJECTION_COMPONENTS, FUSED_CHANNELS):
        raise RuntimeError(f"射影行列の形状が違います: {tuple(projector.sparse_random_matrix.shape)}")
    features = projector.transform(candidates)
    if features.shape != (CONTROL0_N_CANDIDATES, CONTROL0_PROJECTION_COMPONENTS):
        raise RuntimeError(f"射影後の形状が違います: {tuple(features.shape)}")
    return {
        "projector": projector,
        "features": features,
        "record": {
            "source": "regenerated",
            "three_view_matrix_saved": False,
            "reused_saved_matrix": False,
            "verified_identical_to_three_view": False,
            "reason_not_verified": "既存3ビューの射影行列は保存されていない。同じ生成経路で再生成した。直接照合はできない。",
            "method": "anomalib SparseRandomProjection",
            "library": "anomalib",
            "library_version": _projection_library_version(),
            "eps": 0.9,
            "input_dim": FUSED_CHANNELS,
            "output_dim": CONTROL0_PROJECTION_COMPONENTS,
            "jl_dim_for_n_candidates": jl,
            "forced_output_dim": CONTROL0_PROJECTION_COMPONENTS,
            "random_state_arg": None,
            "rng": "torch.distributions.Binomial and sklearn sample_without_replacement on global NumPy RNG after lightning seed_everything(0)",
            "seed_everything_before_projection": 0,
            "initial_center_drawn_with_randint": False,
            "projection_before_initial_center": True,
            "matrix_sha256": _matrix_sha256(projector.sparse_random_matrix),
            "matrix_shape": list(projector.sparse_random_matrix.shape),
            "matrix_dtype": str(projector.sparse_random_matrix.dtype),
        },
    }


def select_coreset_fixed_start(features: torch.Tensor, initial_index: int, coreset_size: int) -> list[int]:
    """Same GPU k-center loop as accelerate_patchcore_coreset, with a fixed start.

    Projection is already finished. This function does not call torch.randint, so
    the projection draw is not affected by replacing the random start.
    """

    from tqdm import tqdm

    if initial_index == CONTROL0_FORBIDDEN_THREE_VIEW_INDEX_AS_CANDIDATE:
        raise RuntimeError("3ビューの 430694 を回転ビューなし側の候補番号にしてはいけません。")
    if not 0 <= initial_index < features.shape[0]:
        raise RuntimeError(f"初期中心が範囲外です: {initial_index}")
    work = features
    if torch.cuda.is_available() and work.device.type != "cuda":
        work = work.cuda(non_blocking=True)
    min_distances = None
    index = torch.tensor(initial_index, device=work.device)
    selected = [index]

    def update(center: torch.Tensor) -> None:
        nonlocal min_distances
        center_vec = work[center].squeeze()
        distances = torch.linalg.norm(work - center_vec, ord=2, dim=1, keepdim=True)
        if min_distances is None:
            min_distances = distances
        else:
            min_distances = torch.minimum(min_distances, distances)

    for _ in tqdm(range(coreset_size - 1), desc="Selecting Coreset Indices.", mininterval=30):
        update(index)
        _, index = torch.max(min_distances.squeeze(1), dim=0)
        min_distances.scatter_(0, index.unsqueeze(0).unsqueeze(1), 0.0)
        selected.append(index)
    return torch.stack(selected).cpu().tolist()


def assert_initial_center(fit_ids: list[str]) -> dict[str, object]:
    resolved = resolve_control0_initial_center()
    if not resolved["view_is_zero"]:
        raise RuntimeError("3ビュー初期点が 0° ビューではありません。共通初期点にできません。")
    if resolved["control_candidate_index"] != CONTROL0_INITIAL_INDEX:
        raise RuntimeError("回転ビューなし側の初期インデックスが 143974 ではありません。")
    if CONTROL0_INITIAL_INDEX == CONTROL0_FORBIDDEN_THREE_VIEW_INDEX_AS_CANDIDATE:
        raise RuntimeError("430694 を回転ビューなし側の候補番号にしてはいけません。")
    expected = candidate_index(
        CONTROL0_INITIAL_IMAGE_INDEX,
        0,
        CONTROL0_INITIAL_ROW,
        CONTROL0_INITIAL_COL,
        n_views=1,
    )
    if expected != CONTROL0_INITIAL_INDEX:
        raise RuntimeError("初期セルの計算が一致しません。")
    if fit_ids[CONTROL0_INITIAL_IMAGE_INDEX] != CONTROL0_INITIAL_IMAGE_ID:
        raise RuntimeError(
            f"初期画像IDが違います: {fit_ids[CONTROL0_INITIAL_IMAGE_INDEX]} != {CONTROL0_INITIAL_IMAGE_ID}"
        )
    if CONTROL0_N_CANDIDATES != 589824:
        raise RuntimeError(f"候補数が 589824 ではありません: {CONTROL0_N_CANDIDATES}")
    assert_integer_coreset(CONTROL0_N_CANDIDATES, CONTROL0_RATIO, BASELINE_CORESET)
    return resolved


def _build_bank(output_dir: Path, fit_paths: list[Path], storage: dict[str, object]) -> dict[str, object]:
    started_flag = output_dir / "fit_started.json"
    bank_path = output_dir / "bank" / BANK_NAME
    if bank_path.is_file():
        LOGGER.info("saved bank exists; fit is not repeated")
        return {"fit_repeated": False, "fit_seconds": 0.0, "bank_path": str(bank_path)}
    if started_flag.is_file():
        raise RuntimeError("fit は開始済みでバンクがありません。自動では繰り返しません。")
    assert_integer_coreset(CONTROL0_N_CANDIDATES, CONTROL0_RATIO, BASELINE_CORESET)
    resolved = assert_initial_center([path.stem for path in fit_paths])
    ram_before = physical_memory()
    candidates = torch.full((CONTROL0_N_CANDIDATES, FUSED_CHANNELS), float("nan"), dtype=torch.float32)
    if candidates.device.type != "cpu":
        raise RuntimeError("候補特徴が CPU にありません。")
    lightning_model = build_model("patchcore", REFERENCE_SIZE, "timm")
    torch_model = lightning_model.model
    torch_model.eval()
    if torch.cuda.is_available():
        lightning_model.cuda()
    normalize = Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    device = next(torch_model.parameters()).device
    started_flag.write_text(
        json.dumps(
            {
                "started": True,
                "n_views": 1,
                "zero_path": AUGMENT_ZERO_PATH,
                "initial_index": CONTROL0_INITIAL_INDEX,
                "three_view_bank_refit": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    torch.cuda.reset_peak_memory_stats()
    fit_started = time.perf_counter()
    loaded = [_load_rgb(path, resize) for path in fit_paths]
    for batch_start in range(0, len(loaded), BATCH_SIZE):
        chunk = loaded[batch_start : batch_start + BATCH_SIZE]
        images = torch.stack([normalize(image) for image in chunk]).to(device)
        with torch.no_grad():
            embedding = fused_embedding(torch_model, images)
        write_view_features(
            candidates,
            list(range(batch_start, batch_start + len(chunk))),
            0,
            embedding,
            n_views=CONTROL0_N_VIEWS,
        )
        del embedding, images
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
    torch.cuda.reset_peak_memory_stats()
    projected = build_projection_355(candidates)
    features = projected["features"]
    if candidates.device.type != "cpu":
        raise RuntimeError("射影後に候補特徴が CPU から外れました。")
    selected = select_coreset_fixed_start(features, CONTROL0_INITIAL_INDEX, BASELINE_CORESET)
    coreset_gpu_peak = int(torch.cuda.max_memory_allocated())
    del features
    if selected[0] != CONTROL0_INITIAL_INDEX:
        raise RuntimeError(f"初期中心が保存されていません: {selected[0]}")
    if CONTROL0_FORBIDDEN_THREE_VIEW_INDEX_AS_CANDIDATE in selected[:1]:
        raise RuntimeError("430694 が初期中心になっています。")
    if len(selected) != BASELINE_CORESET or len(set(selected)) != BASELINE_CORESET:
        raise RuntimeError(f"選択パッチ数が 58982 ちょうどではありません: {len(selected)}")
    bank = candidates.index_select(0, torch.tensor(selected, dtype=torch.long)).contiguous()
    del candidates
    torch.cuda.empty_cache()
    provenance, view_counts = _provenance(selected, [path.stem for path in fit_paths])
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": bank,
            "num_neighbors": 9,
            "n_fit": 144,
            "n_views": 1,
            "seed": 0,
            "coreset_ratio": CONTROL0_RATIO,
            "projection_components": CONTROL0_PROJECTION_COMPONENTS,
            "initial_index": CONTROL0_INITIAL_INDEX,
            "spatial_constraint": False,
            "control_of": "results_a0_augment",
            "three_view_bank_refit": False,
        },
        bank_path,
    )
    np.savez(output_dir / "bank" / "provenance.npz", **provenance)
    (output_dir / "bank" / "view_counts.json").write_text(json.dumps(view_counts, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "bank" / "projection_record.json").write_text(
        json.dumps(projected["record"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fit_seconds = time.perf_counter() - fit_started
    return {
        "fit_repeated": False,
        "fit_seconds": fit_seconds,
        "bank_path": str(bank_path),
        "initial_index": selected[0],
        "initial_center": resolved,
        "projection": projected["record"],
        "projection_components": CONTROL0_PROJECTION_COMPONENTS,
        "extract_gpu_peak_bytes": extract_gpu_peak,
        "coreset_gpu_peak_bytes": coreset_gpu_peak,
        "ram_before_allocation": ram_before,
        "ram_after_fit": physical_memory(),
        "view_counts": view_counts,
        "n_selected": len(selected),
        "n_candidates": CONTROL0_N_CANDIDATES,
        "three_view_bank_refit": False,
    }


def _provenance(selected: list[int], image_ids: list[str]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    flags = _view_flags(())
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
        image, view, row, col = _decode(index, CONTROL0_N_VIEWS)
        if view != 0:
            raise RuntimeError(f"0°以外のビューが選ばれました: {index}")
        mapped = reference_cell_center(row, col, ())
        image_index[bank_row] = image
        view_id[bank_row] = 0
        angle[bank_row] = 0.0
        grid_row[bank_row] = row
        grid_col[bank_row] = col
        reference_x[bank_row] = float(mapped["reference_x"])
        reference_y[bank_row] = float(mapped["reference_y"])
        inside[bank_row] = bool(mapped["inside_original_square"])
        from_reflect[bank_row] = bool(flags["from_reflect_pad"][row, col])
        valid[bank_row] = bool(flags["geometrically_valid"][row, col])
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
    return (
        payload,
        {
            "views": [
                {
                    "view_id": 0,
                    "angle_degrees": 0.0,
                    "n_patches": rows,
                    "n_images": int(len(set(image_index.tolist()))),
                    "quota_enforced": False,
                }
            ],
            "n_selected": rows,
            "spatial_constraint": False,
        },
    )


def _score(
    output_dir: Path,
    data_root: Path,
    calibration_paths: list[Path],
    baseline_dir: Path,
    boundary_dir: Path,
    augment_dir: Path,
) -> dict[str, object]:
    bank_path = output_dir / "bank" / BANK_NAME
    lightning_model = build_model("patchcore", REFERENCE_SIZE, "timm")
    saved = torch.load(bank_path, map_location="cpu", weights_only=False)
    if tuple(saved["memory_bank"].shape) != (BASELINE_CORESET, FUSED_CHANNELS):
        raise RuntimeError(f"保存バンクが 58982×1536 ではありません: {tuple(saved['memory_bank'].shape)}")
    if bool(saved.get("spatial_constraint", False)):
        raise RuntimeError("空間制約がバンクに入っています。")
    if int(saved.get("n_views", -1)) != 1:
        raise RuntimeError("回転ビューなしのバンクのビュー数が 1 ではありません。")
    lightning_model.model.memory_bank = saved["memory_bank"]
    lightning_model.model.num_neighbors = int(saved["num_neighbors"])
    lightning_model.model.eval()
    lightning_model.cuda()
    torch_model = lightning_model.model
    capture = MapCapture()
    attach_patchcore_capture(torch_model, capture)
    normalize = Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)

    torch.cuda.reset_peak_memory_stats()
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
    if abs(image_cut - THREE_VIEW_AUGMENT) < 1e-12 and abs(patch_cut - THREE_VIEW_PATCH) < 1e-12:
        LOGGER.info("recomputed thresholds numerically match the 3-view bank; they were not copied")
    thresholds = {
        "recomputed": True,
        "reused_old_thresholds": False,
        "reused_three_view_thresholds": False,
        "old_image_threshold_not_used": OLD_IMAGE_THRESHOLD,
        "old_patch_threshold_not_used": OLD_PATCH_THRESHOLD,
        "three_view_image_threshold_not_used": THREE_VIEW_AUGMENT,
        "three_view_patch_threshold_not_used": THREE_VIEW_PATCH,
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
    calibration_gpu_peak = int(torch.cuda.max_memory_allocated())

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
    torch.cuda.reset_peak_memory_stats()
    score_started = time.perf_counter()
    from robustad.theme_a.boundary import _test_items

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
    score_gpu_peak = int(torch.cuda.max_memory_allocated())
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
        augment_dir,
        calibration_seconds,
        score_seconds,
        calibration_gpu_peak,
        score_gpu_peak,
    )
    return summary


def _summarize(
    image_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    fpr_rows: list[dict[str, object]],
    image_cut: float,
    patch_cut: float,
    baseline_dir: Path,
    boundary_dir: Path,
    augment_dir: Path,
    calibration_seconds: float,
    score_seconds: float,
    calibration_gpu_peak: int,
    score_gpu_peak: int,
) -> dict[str, object]:
    baseline_images = _read_csv(baseline_dir / "tables" / "images.csv")
    boundary_images = _read_csv(boundary_dir / "tables" / "images.csv")
    augment_images = _read_csv(augment_dir / "tables" / "images.csv")
    baseline_components = _read_csv(baseline_dir / "tables" / "components.csv")
    boundary_components = _read_csv(boundary_dir / "tables" / "components.csv")
    augment_components = _read_csv(augment_dir / "tables" / "components.csv")
    reference_map = {
        "sd_original": ("old144", "original", baseline_images, baseline_components, baseline_dir.name),
        "pad_warp_0": ("old144_boundary", "pad_warp_0", boundary_images, boundary_components, boundary_dir.name),
        "pad_synthetic_m5": ("old144_boundary", "pad_synthetic_m5", boundary_images, boundary_components, boundary_dir.name),
        "pad_synthetic_p5": ("old144_boundary", "pad_synthetic_p5", boundary_images, boundary_components, boundary_dir.name),
    }
    three_map = {
        "sd_original": "sd_original",
        "pad_warp_0": "pad_warp_0",
        "pad_synthetic_m5": "pad_synthetic_m5",
        "pad_synthetic_p5": "pad_synthetic_p5",
    }
    achieved = {}
    vs_old = {}
    vs_three = {}
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
        _tag, ref_condition, ref_images, ref_components, ref_dir = reference_map[name]
        ids = [str(row["source_id"]) for row in rows]
        labels = np.asarray([int(row["label"]) for row in rows])
        ours = np.asarray([float(row["reproduction_score"]) for row in rows], dtype=np.float64)
        ref_lookup = {(row["source_id"], row["condition"]): row for row in ref_images}
        reference_rows = [ref_lookup[(item, ref_condition)] for item in ids]
        reference = np.asarray([float(row["reproduction_score"]) for row in reference_rows], dtype=np.float64)
        ours_hit = np.asarray([bool(row["detected"]) for row in rows])
        reference_hit = np.asarray([row["detected"] == "True" for row in reference_rows])
        ours_components = _finite_component_map(component_rows, name)
        reference_components_map = _finite_component_map(ref_components, ref_condition)
        shared = sorted(set(ours_components) & set(reference_components_map))
        vs_old[name] = {
            "role": "builder_change_check",
            "reference_result": ref_dir,
            "reference_condition": ref_condition,
            "thresholds_differ": True,
            "score_delta_control_minus_old": _quantile_row(ours - reference),
            "normal_score_delta": _quantile_row(ours[labels == 0] - reference[labels == 0]),
            "anomaly_score_delta": _quantile_row(ours[labels == 1] - reference[labels == 1]),
            "decisions_use_each_models_own_threshold": _paired_decisions(reference_hit, ours_hit),
            "component_score_delta": _quantile_row(
                np.asarray([ours_components[key] - reference_components_map[key] for key in shared], dtype=np.float64)
            ),
            "component_pairs": len(shared),
        }
        three_condition = three_map[name]
        three_lookup = {(row["source_id"], row["condition"]): row for row in augment_images}
        three_rows = [three_lookup[(item, three_condition)] for item in ids]
        three = np.asarray([float(row["reproduction_score"]) for row in three_rows], dtype=np.float64)
        three_hit = np.asarray([row["detected"] == "True" for row in three_rows])
        three_components_map = _finite_component_map(augment_components, three_condition)
        shared_three = sorted(set(ours_components) & set(three_components_map))
        vs_three[name] = {
            "role": "primary_comparison",
            "reference_result": augment_dir.name,
            "reference_condition": three_condition,
            "thresholds_differ": True,
            "score_delta_control_minus_three_view": _quantile_row(ours - three),
            "normal_score_delta": _quantile_row(ours[labels == 0] - three[labels == 0]),
            "anomaly_score_delta": _quantile_row(ours[labels == 1] - three[labels == 1]),
            "decisions_use_each_models_own_threshold": _paired_decisions(three_hit, ours_hit),
            "component_score_delta": _quantile_row(
                np.asarray([ours_components[key] - three_components_map[key] for key in shared_three], dtype=np.float64)
            ),
            "component_pairs": len(shared_three),
        }
    return {
        "verdict": VERDICT_UNDECIDED,
        "called_board_surface_improvement": False,
        "roi_metrics": False,
        "primary_comparison": "control0_vs_three_view",
        "secondary_comparison": "old144_vs_control0_builder_check",
        "image_threshold": image_cut,
        "patch_threshold": patch_cut,
        "thresholds_recomputed": True,
        "recalibrated_per_condition": False,
        "spatial_constraint": False,
        "calibration_seconds": calibration_seconds,
        "calibration_gpu_peak_bytes": calibration_gpu_peak,
        "score_seconds": score_seconds,
        "score_gpu_peak_bytes": score_gpu_peak,
        "conditions": achieved,
        "comparisons_vs_old144": vs_old,
        "comparisons_vs_three_view": vs_three,
    }


def run_control0(data_root: Path, output_dir: Path, manifest_path: Path, *, confirmed: bool) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if (output_dir / "summary.json").is_file():
        raise RuntimeError(f"既存の回転ビューなし結果は上書きしません: {output_dir}")
    if not (Path("results_a0_augment") / "summary.json").is_file():
        raise RuntimeError("既存3ビュー結果がありません。再 fit はしません。")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = capacity_snapshot(output_dir)
    fit_paths, calibration_paths = _fit_paths(data_root, manifest_path)
    storage = control0_storage()
    resolved = assert_initial_center([path.stem for path in fit_paths])
    pre_manifest = {
        "approved": True,
        "output_dir": str(output_dir),
        "purpose": "same builder, no added views; not a reproduction of the old 144 bank",
        "zero_path_reason": AUGMENT_ZERO_PATH_REASON,
        "n_views": 1,
        "candidate_order": "sorted fit stems, single untransformed 0° view, row-major cells",
        "fit_image_ids": [path.stem for path in fit_paths],
        "calibration_image_ids": [path.stem for path in calibration_paths],
        "test_conditions": [{"name": name, "angles": list(angles)} for name, angles in AUGMENT_TEST_CONDITIONS],
        "initial_center": resolved,
        "coreset": {
            "method": "anomalib KCenterGreedy GPU distance loop with fixed start",
            "initial_center_included": True,
            "initial_index": CONTROL0_INITIAL_INDEX,
            "forbidden_as_control_index": CONTROL0_THREE_VIEW_INITIAL_INDEX,
            "projection_components": CONTROL0_PROJECTION_COMPONENTS,
            "projection_components_jl": projection_components(CONTROL0_N_CANDIDATES),
            "candidate_device": "cpu",
            "projected_device": "cuda",
            "seed_immediately_before_projection": 0,
            "ratio": CONTROL0_RATIO,
            "n_candidates": CONTROL0_N_CANDIDATES,
            "n_selected_assert": BASELINE_CORESET,
            "quota_enforced": False,
            "spatial_constraint": False,
            "three_view_bank_refit": False,
        },
        "quantile": AUGMENT_QUANTILE,
        "decision": AUGMENT_DECISION,
        "capacity": snapshot,
        "do_not_modify": [
            "results_a0",
            "results_a0_boundary",
            "results_a0_feature",
            "results_a0_augment",
            "results_a1",
            "results_p0/diag",
        ],
    }
    manifest_path_out = output_dir / "pre_manifest.json"
    manifest_path_out.write_text(json.dumps(pre_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    refuse_control0_run(
        output_dir,
        confirmed=confirmed,
        capacity_ok=bool(snapshot["ok"]),
        manifest_saved=manifest_path_out.is_file(),
    )
    if not snapshot["ok"]:
        raise RuntimeError(f"容量が不足します: {snapshot}")
    if not (Path("results_a0_augment") / "summary.json").is_file():
        raise RuntimeError("既存3ビュー結果がありません。再 fit はしません。")
    fit_report = _build_bank(output_dir, fit_paths, storage)
    try:
        summary = _score(
            output_dir,
            data_root,
            calibration_paths,
            Path("results_a0"),
            Path("results_a0_boundary"),
            Path("results_a0_augment"),
        )
    except Exception as exc:
        (output_dir / "score_error.json").write_text(
            json.dumps(
                {
                    "error": str(exc),
                    "fit_repeated": False,
                    "bank_saved": (output_dir / "bank" / BANK_NAME).is_file(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise
    summary["fit"] = fit_report
    summary["ram_after_score"] = physical_memory()
    summary["disk_free_bytes_after"] = int(shutil.disk_usage(output_dir).free)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"K:\RobustAD"))
    parser.add_argument("--output-dir", type=Path, default=Path(CONTROL0_OUTPUT_NAME))
    parser.add_argument("--manifest", type=Path, default=Path("results_p0/research_split_manifest.json"))
    parser.add_argument("--confirm-control-fit", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_control0(args.data_root, args.output_dir, args.manifest, confirmed=args.confirm_control_fit)


if __name__ == "__main__":
    main()
