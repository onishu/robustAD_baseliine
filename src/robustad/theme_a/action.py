"""Spec helpers for the next Theme A comparison. They do not fit or score.

The four intervention points stay the research axis. This module records the
coordinate rules, the capacity match for augmentation, and the unconfirmed
board-surface subset. It does not decide which point is best.
"""

from __future__ import annotations

import numpy as np
import torch

from robustad.theme_a.boundary import FILL_EPS, MARGIN, content_probe, extend_and_warp, fill_probe
from robustad.theme_a.warp import REFERENCE_SIZE, rotate_tensor, sampling_matrix

# Measured on wide_resnet50_2, input 512, one forward, 2026-10-26.
# AvgPool2d(3, stride=1, padding=1) did not change height or width.
# layer3 is bilinearly resized onto layer2 and then concatenated.
LAYER2_CHANNELS = 512
LAYER2_GRID = 64
LAYER2_STRIDE_PX = REFERENCE_SIZE // LAYER2_GRID
LAYER3_CHANNELS = 1024
LAYER3_GRID = 32
LAYER3_STRIDE_PX = REFERENCE_SIZE // LAYER3_GRID
FUSED_CHANNELS = LAYER2_CHANNELS + LAYER3_CHANNELS
FUSED_GRID = LAYER2_GRID
FEATURE_STAGE = "fused_after_concat"
POOL_CHANGES_GRID = False

BASELINE_FIT_IMAGES = 144
BASELINE_CORESET = int(BASELINE_FIT_IMAGES * LAYER2_GRID * LAYER2_GRID * 0.1)
# Main augmentation plan. Not executed. 0° is the untransformed fit image.
AUGMENT_MAIN_ANGLES = (0.0, -5.0, 5.0)
AUGMENT_MAIN_ZERO_IS_RESAMPLED = False
AUGMENT_ZERO_PATH = "untransformed_fit_image"
AUGMENT_ZERO_PATH_REASON = (
    "0°ビューは、既存バンクを作ったときと同じ未変換のfit画像にする。"
    "反射延長の0°は回転格子による再標本であり、0°経路を替えるとaugmentationと標本化が混ざる。"
    "反射延長は−5°と+5°のビューだけに使う。"
)
AUGMENT_OUTPUT_NAME = "results_a0_augment"
AUGMENT_FIT_ENABLED = False
AUGMENT_REUSE_OLD_THRESHOLDS = False
OLD_IMAGE_THRESHOLD = 36.28824043273926
OLD_PATCH_THRESHOLD = 24.215826511383057
AUGMENT_TEST_CONDITIONS = (
    ("sd_original", ()),
    ("pad_warp_0", (0.0,)),
    ("pad_synthetic_m5", (-5.0,)),
    ("pad_synthetic_p5", (5.0,)),
)
AUGMENT_CALIBRATION = "untransformed_36"
AUGMENT_QUANTILE = "np.quantile(scores, 0.95, method='linear')"
AUGMENT_DECISION = "score >= threshold"
CANDIDATE_DEVICE = "cpu"
PROJECTED_DEVICE = "cuda"
GPU_MEMORY_BYTES = 12 * 1024**3
# ±5° without 0° replaces the normal view by rotated views. It is not the main plan.
AUGMENT_REPLACEMENT_ANGLES = (-5.0, 5.0)
AUGMENT_REPLACEMENT_LABEL = "回転ビューへの置換"
AUGMENT_EXECUTED = False
SAME_ANGLE_TEST_IS_WITHIN_RANGE = True
PROVENANCE_FIELDS = (
    "fit_image_id",
    "view_id",
    "angle_degrees",
    "grid_row",
    "grid_col",
    "reference_x",
    "reference_y",
    "from_reflect_pad",
    "geometrically_valid",
    "inside_original_square",
    "coreset_index",
)
SPATIAL_CONSTRAINT_IN_AUGMENT = False
BOARD_ROI_COUNT = 8
BOARD_UPPER_QUANTILE = 0.9
BOARD_ROI_STATUS = "unconfirmed"
SPATIAL_RADIUS_PX = None
SPATIAL_DISTANCE = "euclidean_cell_centers_in_reference_pixels"


def embedding_count(n_images: int, n_views: int, grid: int = LAYER2_GRID) -> int:
    if n_images < 1 or n_views < 1 or grid < 1:
        raise ValueError(f"枚数と格子は正です: {n_images}, {n_views}, {grid}")
    return int(n_images) * int(grid) * int(grid) * int(n_views)


def coreset_count(n_embeddings: int, ratio: float) -> int:
    """Match ``KCenterGreedy``: ``int(n * ratio)``."""

    if n_embeddings < 1 or not 0.0 < ratio <= 1.0:
        raise ValueError(f"coreset の入力が不正です: {n_embeddings}, {ratio}")
    size = int(n_embeddings * ratio)
    if size < 1:
        raise ValueError(f"coreset が空です: {size}")
    return size


def assert_integer_coreset(n_embeddings: int, ratio: float, target: int = BASELINE_CORESET) -> None:
    """The saved patch count is ``int(n * ratio)``, and it must equal ``target``."""

    got = int(n_embeddings * ratio)
    if got != target:
        raise AssertionError(f"保存パッチ数が {target} ちょうどではありません: int({n_embeddings} * {ratio}) = {got}")


def augment_views() -> tuple[dict[str, object], ...]:
    """Three fit views. Reflect padding is not applied to the 0° view."""

    return (
        {
            "view_id": 0,
            "angle_degrees": 0.0,
            "path": AUGMENT_ZERO_PATH,
            "reflect_pad": False,
            "reason": AUGMENT_ZERO_PATH_REASON,
        },
        {
            "view_id": 1,
            "angle_degrees": -5.0,
            "path": "reflect_pad_rotate_center_crop",
            "reflect_pad": True,
            "reason": "回転ビューだけに、端を22画素折り返してから回し、中央512を1回切る。",
        },
        {
            "view_id": 2,
            "angle_degrees": 5.0,
            "path": "reflect_pad_rotate_center_crop",
            "reflect_pad": True,
            "reason": "回転ビューだけに、端を22画素折り返してから回し、中央512を1回切る。",
        },
    )


def candidate_index(image_index: int, view_index: int, row: int, col: int, n_views: int = 3, grid: int = FUSED_GRID) -> int:
    """Row-major index in image, view, row, column order. Views are not quota-balanced."""

    if not 0 <= view_index < n_views or not 0 <= row < grid or not 0 <= col < grid or image_index < 0:
        raise ValueError(f"候補インデックスが範囲外です: {(image_index, view_index, row, col)}")
    return ((image_index * n_views + view_index) * grid + row) * grid + col


def view_selection_breakdown(candidate_indices: list[int], n_views: int = 3, grid: int = FUSED_GRID) -> dict[str, object]:
    """Count an existing global coreset by view. This does not assign quotas."""

    counts = [0 for _ in range(n_views)]
    block = grid * grid
    for index in candidate_indices:
        if index < 0:
            raise ValueError(f"候補インデックスが負です: {index}")
        view = (int(index) // block) % n_views
        counts[view] += 1
    return {"counts": counts, "n_selected": int(sum(counts)), "quota_enforced": False}


def projection_components(n_samples: int, eps: float = 0.9) -> int:
    """Match anomalib ``SparseRandomProjection._johnson_lindenstrauss_min_dim`` at ``eps=0.9``."""

    if n_samples < 2 or not 0.0 < eps < 1.0:
        raise ValueError(f"射影の入力が不正です: {n_samples}, {eps}")
    denominator = (eps**2 / 2.0) - (eps**3 / 3.0)
    return int((4.0 * np.log(n_samples) / denominator).astype(np.int64))


def augment_storage() -> dict[str, object]:
    """Bytes for the 3-view candidate set, the projected coreset work, and the saved bank.

    The full candidate tensor does not fit on a 12 GB GPU beside the CNN.
    Projected features are the tensor the k-center loop reads.
    """

    n_candidates = embedding_count(BASELINE_FIT_IMAGES, len(AUGMENT_MAIN_ANGLES))
    ratio = ratio_matching_coreset(n_candidates, BASELINE_CORESET)
    assert_integer_coreset(n_candidates, ratio, BASELINE_CORESET)
    components = projection_components(n_candidates)
    candidate_bytes = n_candidates * FUSED_CHANNELS * 4
    projected_bytes = n_candidates * components * 4
    bank_bytes = BASELINE_CORESET * FUSED_CHANNELS * 4
    # Selected provenance is a small table. Image ids are stored once, not per patch.
    provenance_bytes = BASELINE_CORESET * (8 + 4 + 2 + 1 + 4 + 2 + 2 + 4 + 4 + 1 + 1 + 1)
    test_grid_bytes = 195 * len(AUGMENT_TEST_CONDITIONS) * FUSED_GRID * FUSED_GRID * 4
    calibration_grid_bytes = 36 * FUSED_GRID * FUSED_GRID * 4
    if candidate_bytes + 2 * 1024**3 >= GPU_MEMORY_BYTES:
        candidate_device = CANDIDATE_DEVICE
    else:
        candidate_device = "cuda"
    return {
        "n_candidates": n_candidates,
        "coreset_ratio": ratio,
        "coreset_patches": int(n_candidates * ratio),
        "projection_components": components,
        "projection_eps": 0.9,
        "candidate_bytes": candidate_bytes,
        "candidate_device": candidate_device,
        "projected_bytes": projected_bytes,
        "projected_device": PROJECTED_DEVICE,
        "gpu_memory_bytes": GPU_MEMORY_BYTES,
        "bank_bytes": bank_bytes,
        "provenance_bytes": provenance_bytes,
        "test_grid_bytes": test_grid_bytes,
        "calibration_grid_bytes": calibration_grid_bytes,
        "optional_candidate_memmap_bytes": candidate_bytes,
        "saved_bytes": bank_bytes + provenance_bytes + test_grid_bytes + calibration_grid_bytes,
        "quota_enforced": False,
        "fit_enabled": AUGMENT_FIT_ENABLED,
    }


CONTROL0_OUTPUT_NAME = "results_a0_control0"
CONTROL0_N_VIEWS = 1
CONTROL0_N_CANDIDATES = embedding_count(BASELINE_FIT_IMAGES, CONTROL0_N_VIEWS)
CONTROL0_RATIO = 0.1
CONTROL0_PROJECTION_COMPONENTS = 355
CONTROL0_THREE_VIEW_INITIAL_INDEX = 430694
CONTROL0_INITIAL_IMAGE_INDEX = 35
CONTROL0_INITIAL_IMAGE_ID = "Train_PCB_resized_None_None_Normal_Set1_142"
CONTROL0_INITIAL_ROW = 9
CONTROL0_INITIAL_COL = 38
CONTROL0_INITIAL_INDEX = candidate_index(
    CONTROL0_INITIAL_IMAGE_INDEX,
    0,
    CONTROL0_INITIAL_ROW,
    CONTROL0_INITIAL_COL,
    n_views=CONTROL0_N_VIEWS,
)
CONTROL0_FORBIDDEN_THREE_VIEW_INDEX_AS_CANDIDATE = CONTROL0_THREE_VIEW_INITIAL_INDEX


def zero_only_control_agreement() -> dict[str, object]:
    """Say what a 0°-only rerun of this builder would share with the saved 144-image bank.

    This does not fit. The old coreset index and the RNG state after the shuffled
    dataloader were not stored, so those two cannot be checked numerically.
    """

    n_candidates = CONTROL0_N_CANDIDATES
    ratio = CONTROL0_RATIO
    selected = coreset_count(n_candidates, ratio)
    components = projection_components(n_candidates)
    candidate_bytes = n_candidates * FUSED_CHANNELS * 4
    projected_bytes = n_candidates * CONTROL0_PROJECTION_COMPONENTS * 4
    return {
        "fit_executed": False,
        "n_candidates": n_candidates,
        "matches": {
            "fit_images": "同じ未変換144枚",
            "saved_patch_count": selected == BASELINE_CORESET,
            "coreset_ratio_0_1": True,
            "projection_method": "SparseRandomProjection eps 0.9",
            "projection_components_jl": components,
            "projection_components_match_baseline_328": components == 328,
            "projection_components_forced": CONTROL0_PROJECTION_COMPONENTS,
            "within_image_cell_order": "reshape_embedding の行優先",
            "kcenter_includes_initial_center": True,
            "layers_pool_concat_neighbors": True,
        },
        "mismatches": {
            "image_order": "この構築器はファイル名順。既存バンクは学習ローダの shuffle=True で、その順は保存されていない。",
            "seed_state": "この構築器は coreset 直前に seed 0 を入れ直す。既存バンクの初期点は fit 中に進んだ乱数で、値は保存されていない。",
            "candidate_device": "この構築器は 1536 次元を CPU に置く。既存の144枚は学習ステップ中に GPU へ蓄積した。",
            "projection_dim_vs_jl": f"JL は {components}。回転ビューなしは3ビューと同じ {CONTROL0_PROJECTION_COMPONENTS} に固定する。",
        },
        "estimate": {
            "candidate_bytes": candidate_bytes,
            "projected_bytes": projected_bytes,
            "bank_bytes": BASELINE_CORESET * FUSED_CHANNELS * 4,
            "time_basis": "3ビューの k-center は 1361 秒だった。候補数が 3分の1なので、距離更新だけを比例させると約 454 秒。計測値ではない。",
            "necessity": "3ビューの結果から、ビューを増やした効果と、候補順・seed・射影次元の差を分けるときに要る。既存バンクの再現にはならない。",
        },
        "initial_center": resolve_control0_initial_center(),
    }


def resolve_control0_initial_center() -> dict[str, object]:
    """Map the saved 3-view start 430694 onto the 0°-only candidate order.

    The 3-view start is a 0° cell, so the control bank uses the same image and
    cell. The integer 430694 is not a control-side candidate index.
    """

    three_view = CONTROL0_THREE_VIEW_INITIAL_INDEX
    block = FUSED_GRID * FUSED_GRID
    col = three_view % FUSED_GRID
    row = (three_view // FUSED_GRID) % FUSED_GRID
    view = (three_view // block) % 3
    image = three_view // (3 * block)
    control_index = candidate_index(image, 0, row, col, n_views=1)
    return {
        "three_view_candidate_index": three_view,
        "view_id": view,
        "view_is_zero": view == 0,
        "fit_image_index": image,
        "fit_image_id": CONTROL0_INITIAL_IMAGE_ID if image == CONTROL0_INITIAL_IMAGE_INDEX else None,
        "grid_row": row,
        "grid_col": col,
        "control_candidate_index": control_index,
        "control_candidate_index_assert": CONTROL0_INITIAL_INDEX,
        "same_integer_is_not_same_feature": three_view != control_index,
        "do_not_use_three_view_integer_as_control_index": True,
    }


def control0_storage() -> dict[str, object]:
    """Bytes for the 0°-only control bank with projection dim fixed at 355."""

    n_candidates = CONTROL0_N_CANDIDATES
    assert_integer_coreset(n_candidates, CONTROL0_RATIO, BASELINE_CORESET)
    jl_components = projection_components(n_candidates)
    candidate_bytes = n_candidates * FUSED_CHANNELS * 4
    projected_bytes = n_candidates * CONTROL0_PROJECTION_COMPONENTS * 4
    bank_bytes = BASELINE_CORESET * FUSED_CHANNELS * 4
    provenance_bytes = BASELINE_CORESET * (8 + 4 + 2 + 1 + 4 + 2 + 2 + 4 + 4 + 1 + 1 + 1)
    test_grid_bytes = 195 * len(AUGMENT_TEST_CONDITIONS) * FUSED_GRID * FUSED_GRID * 4
    calibration_grid_bytes = 36 * FUSED_GRID * FUSED_GRID * 4
    return {
        "n_candidates": n_candidates,
        "coreset_ratio": CONTROL0_RATIO,
        "coreset_patches": BASELINE_CORESET,
        "projection_components_jl": jl_components,
        "projection_components": CONTROL0_PROJECTION_COMPONENTS,
        "projection_eps": 0.9,
        "candidate_bytes": candidate_bytes,
        "candidate_device": CANDIDATE_DEVICE,
        "projected_bytes": projected_bytes,
        "projected_device": PROJECTED_DEVICE,
        "gpu_memory_bytes": GPU_MEMORY_BYTES,
        "bank_bytes": bank_bytes,
        "provenance_bytes": provenance_bytes,
        "test_grid_bytes": test_grid_bytes,
        "calibration_grid_bytes": calibration_grid_bytes,
        "saved_bytes": bank_bytes + provenance_bytes + test_grid_bytes + calibration_grid_bytes,
        "required_free_ram_bytes": candidate_bytes + projected_bytes + 4 * 1024**3,
        "required_free_gpu_bytes": projected_bytes * 2 + 1024**3,
        "required_free_disk_bytes": bank_bytes + provenance_bytes + test_grid_bytes + calibration_grid_bytes + 2 * 1024**3,
        "quota_enforced": False,
        "spatial_constraint": False,
        "initial_index": CONTROL0_INITIAL_INDEX,
    }


def required_free_ram_bytes() -> int:
    """Candidate tensor plus projected features plus 4 GB of headroom."""

    storage = augment_storage()
    return int(storage["candidate_bytes"]) + int(storage["projected_bytes"]) + 4 * 1024**3


def map_crop_point_to_original(
    col: float,
    row: float,
    angles: tuple[float, ...],
    margin: int = MARGIN,
) -> dict[str, object]:
    """Map a continuous crop-pixel index back to the original square.

    ``col`` and ``row`` use the same index convention as ``map_crop_pixel_to_original``:
    the sampling center of index ``i`` is ``i + 0.5``.
    """

    if angles and margin < 0:
        raise ValueError(f"余白が負です: {margin}")
    use_margin = margin if angles else 0
    canvas = REFERENCE_SIZE + 2 * use_margin
    x = float(col + use_margin)
    y = float(row + use_margin)
    for degrees in reversed(angles):
        matrix = sampling_matrix(float(degrees), canvas, canvas)
        x, y = _apply_output_to_source(x, y, matrix, canvas, canvas)
    src_col = x - use_margin
    src_row = y - use_margin
    inside = bool(0.0 <= src_col < REFERENCE_SIZE and 0.0 <= src_row < REFERENCE_SIZE)
    return {
        "source_col": src_col,
        "source_row": src_row,
        "inside_original_square": inside,
        "reference_x": src_col + 0.5,
        "reference_y": src_row + 0.5,
    }


def reference_cell_center(row: int, col: int, angles: tuple[float, ...]) -> dict[str, object]:
    """Map a fused-cell center to original-image pixels. The center is ``8 * index + 4``."""

    col_index = col * LAYER2_STRIDE_PX + (LAYER2_STRIDE_PX / 2.0 - 0.5)
    row_index = row * LAYER2_STRIDE_PX + (LAYER2_STRIDE_PX / 2.0 - 0.5)
    mapped = map_crop_point_to_original(col_index, row_index, angles, margin=MARGIN if angles else 0)
    mapped["grid_row"] = int(row)
    mapped["grid_col"] = int(col)
    return mapped


def cell_origin_flags(row: int, col: int, angles: tuple[float, ...]) -> dict[str, bool]:
    """Reflect-pad origin of one fused cell. The 0° untransformed view has no reflected pixels."""

    if not angles:
        return {"from_reflect_pad": False, "geometrically_valid": True, "content_from_original": True}
    content = content_probe(angles) >= (1.0 - FILL_EPS)
    valid = fill_probe(angles) >= (1.0 - FILL_EPS)
    r0, r1, c0, c1 = cell_block(row, col, LAYER2_STRIDE_PX)
    content_all = bool(torch.all(content[r0:r1, c0:c1]).item())
    valid_all = bool(torch.all(valid[r0:r1, c0:c1]).item())
    return {
        "from_reflect_pad": not content_all,
        "geometrically_valid": valid_all,
        "content_from_original": content_all,
    }


def ratio_matching_coreset(n_embeddings: int, target: int = BASELINE_CORESET) -> float:
    """Ratio whose ``int(n * ratio)`` equals ``target``.

    The same ratio as the baseline is not a capacity match once the view count
    changes. Callers still have to record the view count separately.
    """

    if not 0 < target <= n_embeddings:
        raise ValueError(f"目標パッチ数が範囲外です: {target} / {n_embeddings}")
    ratio = float(target) / float(n_embeddings)
    if int(n_embeddings * ratio) != target:
        ratio = float(np.nextafter(ratio, 1.0))
    if int(n_embeddings * ratio) != target:
        raise RuntimeError(f"目標パッチ数にちょうど合う比がありません: {target} / {n_embeddings}")
    return ratio


def memory_bank_provenance(payload: dict) -> dict[str, object]:
    """Say whether a saved bank can support a spatial constraint.

    Image ids and grid coordinates are not inferred from feature values.
    """

    if "memory_bank" not in payload:
        raise ValueError("memory_bank がありません。")
    bank = payload["memory_bank"]
    shape = tuple(int(v) for v in bank.shape)
    location_keys = ("image_id", "source_id", "row", "col", "grid_position", "coreset_index")
    present = sorted(str(key) for key in location_keys if key in payload)
    recoverable = bool(present)
    return {
        "keys": sorted(str(key) for key in payload),
        "bank_shape": shape,
        "image_id_stored": any(key in payload for key in ("image_id", "source_id")),
        "grid_position_stored": any(key in payload for key in ("row", "col", "grid_position")),
        "positions_recoverable": recoverable,
        "location_keys_present": present,
        "reason": (
            "選択インデックスと格子位置が残っている。"
            if recoverable
            else "保存物は特徴ベクトルだけで、画像IDと格子位置は復元しない。"
        ),
    }


def cell_block(row: int, col: int, stride: int, image_size: int = REFERENCE_SIZE) -> tuple[int, int, int, int]:
    """Half-open pixel block of one feature cell. This is not a receptive field."""

    if stride < 1 or row < 0 or col < 0:
        raise ValueError(f"セル指定が不正です: {(row, col, stride)}")
    r0, c0 = row * stride, col * stride
    r1, c1 = r0 + stride, c0 + stride
    if r1 > image_size or c1 > image_size:
        raise ValueError(f"セルが画像の外です: {(row, col, stride, image_size)}")
    return r0, r1, c0, c1


def layer3_sample_on_fused(fused_row: int, fused_col: int) -> tuple[float, float]:
    """Continuous layer3 coordinate sampled by bilinear ``align_corners=False``.

    Output index ``i`` reads the input at ``(i + 0.5) * in_size / out_size - 0.5``.
    A fused cell does not belong to exactly one layer3 cell.
    """

    if not 0 <= fused_row < FUSED_GRID or not 0 <= fused_col < FUSED_GRID:
        raise ValueError(f"融合格子の外です: {(fused_row, fused_col)}")
    scale = LAYER3_GRID / FUSED_GRID
    return (fused_row + 0.5) * scale - 0.5, (fused_col + 0.5) * scale - 0.5


def nearest_image_cell_mask(row: int, col: int, degrees: float) -> torch.Tensor:
    """Rotate the cell's 8×8 block in the 512 image, then re-bin."""

    image = torch.zeros((1, REFERENCE_SIZE, REFERENCE_SIZE), dtype=torch.float32)
    r0, r1, c0, c1 = cell_block(row, col, LAYER2_STRIDE_PX)
    image[:, r0:r1, c0:c1] = 1.0
    rotated = rotate_tensor(image, degrees, nearest=True)
    return rotated[0].reshape(FUSED_GRID, LAYER2_STRIDE_PX, FUSED_GRID, LAYER2_STRIDE_PX).any(dim=(1, 3))


def nearest_feature_cell_mask(row: int, col: int, degrees: float) -> torch.Tensor:
    """Rotate one fused cell on the 64×64 grid. The CNN is not rerun."""

    grid = torch.zeros((1, FUSED_GRID, FUSED_GRID), dtype=torch.float32)
    grid[0, row, col] = 1.0
    rotated = rotate_tensor(grid, degrees, nearest=True)
    return rotated[0] > 0


def _apply_output_to_source(col: float, row: float, matrix: np.ndarray, height: int, width: int) -> tuple[float, float]:
    centered = np.array([col + 0.5 - width * 0.5, row + 0.5 - height * 0.5, 1.0], dtype=np.float64)
    source = np.asarray(matrix, dtype=np.float64) @ centered
    return float(source[0] + width * 0.5 - 0.5), float(source[1] + height * 0.5 - 0.5)


def map_crop_pixel_to_original(
    col: int,
    row: int,
    angles: tuple[float, ...],
    margin: int = MARGIN,
) -> dict[str, object]:
    """Map a crop pixel back through the known angles to the original square.

    Coordinates use the same centered convention as the saved sampling matrix.
    A source that lands in the reflected margin is outside the original square.
    """

    if not 0 <= col < REFERENCE_SIZE or not 0 <= row < REFERENCE_SIZE:
        raise ValueError(f"切り出し座標が範囲外です: {(col, row)}")
    side = REFERENCE_SIZE + 2 * margin
    x = float(col + margin)
    y = float(row + margin)
    for degrees in reversed(angles):
        matrix = sampling_matrix(float(degrees), side, side)
        x, y = _apply_output_to_source(x, y, matrix, side, side)
    src_col = x - margin
    src_row = y - margin
    inside = bool(0.0 <= src_col < REFERENCE_SIZE and 0.0 <= src_row < REFERENCE_SIZE)
    return {"source_col": src_col, "source_row": src_row, "inside_original_square": inside}


def cell_center(row: int, col: int, stride: int = LAYER2_STRIDE_PX) -> tuple[float, float]:
    """Return ``(x, y)`` of the block center in reference-image pixels."""

    return (col + 0.5) * stride, (row + 0.5) * stride


def within_radius(
    query: tuple[int, int],
    bank_cell: tuple[int, int],
    radius_px: float,
    stride: int = LAYER2_STRIDE_PX,
) -> bool:
    """Euclidean test on cell centers. The radius is in reference-image pixels."""

    if radius_px < 0:
        raise ValueError(f"半径は 0 以上です: {radius_px}")
    qx, qy = cell_center(query[0], query[1], stride)
    bx, by = cell_center(bank_cell[0], bank_cell[1], stride)
    return (qx - bx) ** 2 + (qy - by) ** 2 <= float(radius_px) ** 2


def no_candidate_record(n_candidates: int) -> dict[str, object]:
    """Zero candidates stay missing. They do not fall back to a global neighbor."""

    if n_candidates < 0:
        raise ValueError(f"候補数は 0 以上です: {n_candidates}")
    if n_candidates == 0:
        return {"n_candidates": 0, "score": None, "used_unconstrained_neighbor": False}
    return {
        "n_candidates": int(n_candidates),
        "score": "not_scored_in_this_check",
        "used_unconstrained_neighbor": False,
    }


def select_board_roi_stems(stems: list[str], count: int = BOARD_ROI_COUNT) -> list[dict[str, object]]:
    """Evenly spaced file-name order. Scores are not an input."""

    ordered = sorted(stems)
    if len(ordered) != len(set(ordered)):
        raise ValueError("ファイルIDが重複しています。")
    if count < 2 or len(ordered) < count:
        raise ValueError(f"選択には2枚以上の異なるIDが要ります: {count} / {len(ordered)}")
    indexes = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    if len(set(indexes)) != count:
        raise ValueError(f"選択位置が重なりました: {indexes}")
    return [
        {"rule_index": int(index), "stem": ordered[index], "status": BOARD_ROI_STATUS}
        for index in indexes
    ]


def assert_augment_sources(fit_ids: list[str], calibration_ids: list[str], test_ids: list[str]) -> None:
    fit = set(fit_ids)
    calibration = set(calibration_ids)
    test = set(test_ids)
    if fit & calibration:
        raise RuntimeError("fit と calibration が重なっています。")
    if fit & test:
        raise RuntimeError("fit と test が重なっています。")
    if calibration & test:
        raise RuntimeError("calibration と test が重なっています。")


def _block_flags(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(mask, dtype=bool)
    if array.shape != (REFERENCE_SIZE, REFERENCE_SIZE):
        raise ValueError(f"512 画像ではありません: {array.shape}")
    blocks = array.reshape(FUSED_GRID, LAYER2_STRIDE_PX, FUSED_GRID, LAYER2_STRIDE_PX)
    return blocks.all(axis=(1, 3)), blocks.any(axis=(1, 3))


def classify_board_cells(roi: np.ndarray, content_from_original: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Split fused cells before any board statistic.

    ``board_entire`` is the only primary set: every pixel of the 8×8 block is
    inside the confirmed ROI, inside the original square, and geometrically
    valid. Partial cells and reflected cells are kept as separate masks.
    """

    roi_all, roi_any = _block_flags(roi)
    content_all, _content_any = _block_flags(content_from_original)
    valid_all, _valid_any = _block_flags(valid)
    clean = content_all & valid_all
    return {
        "board_entire": roi_all & clean,
        "board_partial": roi_any & ~roi_all & clean,
        "reflection_or_invalid": ~clean,
    }


def warp_label_and_origin(label: torch.Tensor, angles: tuple[float, ...]) -> dict[str, torch.Tensor]:
    """Move a confirmed 512 label with the shared reflect-pad path.

    The label is padded with 0, so it is not mirrored. Content and fill probes
    use the same angles and one final crop. An empty angle list is the
    unrotated label, not a 0° resample.
    """

    array = torch.as_tensor(label)
    if tuple(array.shape) != (REFERENCE_SIZE, REFERENCE_SIZE):
        raise ValueError(f"ラベルは 512×512 です: {tuple(array.shape)}")
    if not angles:
        ones = torch.ones((REFERENCE_SIZE, REFERENCE_SIZE), dtype=torch.bool)
        return {"roi": array > 0, "content_from_original": ones, "geometrically_valid": ones}
    warped, _steps = extend_and_warp(array.to(dtype=torch.float32), angles, nearest=True, pad_mode="constant")
    content = content_probe(angles) >= (1.0 - FILL_EPS)
    valid = fill_probe(angles) >= (1.0 - FILL_EPS)
    roi = warped[0] > 0 if warped.ndim == 3 else warped > 0
    return {"roi": roi, "content_from_original": content, "geometrically_valid": valid}


def board_patch_summary(scores: np.ndarray, entire: np.ndarray, threshold: float) -> dict[str, float]:
    """Per-image auxiliary numbers. Patches are not a sample of images."""

    grid = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(entire, dtype=bool)
    if grid.shape != mask.shape:
        raise ValueError(f"スコアとセルマスクの形状が違います: {grid.shape} / {mask.shape}")
    values = grid[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n_cells": 0,
            "median": float("nan"),
            "upper_quantile": float("nan"),
            "fraction_at_or_above_threshold": float("nan"),
        }
    return {
        "n_cells": int(values.size),
        "median": float(np.median(values)),
        "upper_quantile": float(np.quantile(values, BOARD_UPPER_QUANTILE, method="linear")),
        "fraction_at_or_above_threshold": float(np.mean(values >= threshold)),
    }


def paired_source_delta(reference: dict[str, float], other: dict[str, float]) -> dict[str, float]:
    """One delta per source id. Missing ids are not filled in."""

    keys = sorted(set(reference) & set(other))
    return {key: float(other[key]) - float(reference[key]) for key in keys}
