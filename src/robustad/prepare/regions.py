"""Masks, connected components, and score attribution.

Workpiece and background are not inferred from the defect mask. Pixels that
are not defects and not in a declared border band stay in one unresolved
class, ``nondefect_unpartitioned``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


CONNECTIVITY_8 = np.ones((3, 3), dtype=np.int8)


def resize_binary_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize followed by ``> 0``, matching ``_resize_mask``."""

    array = np.asarray(mask)
    tensor = torch.as_tensor(array, dtype=torch.float32)
    resized = F.interpolate(tensor[None, None], size=(height, width), mode="nearest")
    return (resized[0, 0] > 0).numpy()


def label_components(mask: np.ndarray) -> np.ndarray:
    """Label a binary foreground with 8-connectivity.

    Holes stay unlabeled. An empty mask returns an all-zero array and does not
    invent a component. Identifiers are the positive integers from
    ``ndimage.label`` and are not renumbered.
    """

    foreground = np.asarray(mask) > 0
    if foreground.ndim != 2:
        raise ValueError(f"マスクは2次元です: {foreground.shape}")
    labeled, _ = ndimage.label(foreground, structure=CONNECTIVITY_8)
    return labeled.astype(np.int32)


UNEVALUABLE_INPUT_TRANSFORM = "入力変換による評価不能"
EVALUABLE_COMPONENT = "evaluable"


def component_correspondence(mask: np.ndarray, height: int, width: int) -> list[dict[str, object]]:
    """Map each original component onto the resized joint mask.

    A component whose own nearest-neighbor resize is empty stays in the list
    with status ``入力変換による評価不能``. It is not dropped from the count.
    """

    labels = label_components(mask)
    joint_labels = label_components(resize_binary_mask(mask, height, width))
    rows: list[dict[str, object]] = []
    for component_id, area in component_areas(labels).items():
        alone = resize_binary_mask(labels == component_id, height, width)
        area_alone = int(alone.sum())
        if area_alone == 0:
            rows.append(
                {
                    "component_id": component_id,
                    "area_original": area,
                    "area_resized_alone": 0,
                    "joint_component_ids": "",
                    "status": UNEVALUABLE_INPUT_TRANSFORM,
                }
            )
            continue
        overlapped = sorted(int(value) for value in np.unique(joint_labels[alone]) if int(value) > 0)
        rows.append(
            {
                "component_id": component_id,
                "area_original": area,
                "area_resized_alone": area_alone,
                "joint_component_ids": ",".join(str(value) for value in overlapped),
                "status": EVALUABLE_COMPONENT,
            }
        )
    return rows


def component_areas(labels: np.ndarray) -> dict[int, int]:
    labeled = np.asarray(labels)
    ids = [int(value) for value in np.unique(labeled) if value > 0]
    return {component_id: int((labeled == component_id).sum()) for component_id in ids}


@dataclass(frozen=True)
class ComponentRecord:
    component_id: int
    area: int
    area_resized_alone: int
    vanished_on_resize: bool
    statistic_max: float
    statistic_quantile: float


def iter_original_components(mask: np.ndarray, height: int, width: int) -> list[ComponentRecord]:
    """Resize each original component by itself so a disappearance is visible.

    Joint resizing of the whole mask can merge neighbors. That joint mask is a
    separate object and is what the pixel metric uses. This function does not
    replace the joint mask.
    """

    labels = label_components(mask)
    records = []
    for component_id, area in component_areas(labels).items():
        alone = resize_binary_mask(labels == component_id, height, width)
        resized_area = int(alone.sum())
        records.append(
            ComponentRecord(
                component_id=component_id,
                area=area,
                area_resized_alone=resized_area,
                vanished_on_resize=resized_area == 0,
                statistic_max=float("nan"),
                statistic_quantile=float("nan"),
            )
        )
    return records


def restrict_labels(labels: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
    """Keep original component ids inside a valid region.

    Fragments of one component keep that id. They are not relabeled, so a
    split does not increase the defect count. Pixels outside ``valid`` are
    cleared.
    """

    labeled = np.asarray(labels).astype(np.int32, copy=True)
    keep = np.asarray(valid, dtype=bool)
    if labeled.shape != keep.shape:
        raise ValueError(f"ラベルと有効領域の形状が違います: {labeled.shape} / {keep.shape}")
    stats = {}
    for component_id, area in component_areas(labeled).items():
        inside = int(((labeled == component_id) & keep).sum())
        stats[component_id] = {
            "area": area,
            "area_inside": inside,
            "remaining_ratio": float(inside / area) if area else float("nan"),
            "outside_all": inside == 0,
            "outside_some": 0 < inside < area,
        }
    labeled[~keep] = 0
    return labeled, stats


def full_valid_region(height: int, width: int) -> np.ndarray:
    """The primary valid region is the whole reference image.

    The border band is not removed. A missing workpiece ROI is not filled in.
    """

    if height < 1 or width < 1:
        raise ValueError(f"有効領域の大きさは正です: {height}x{width}")
    return np.ones((height, width), dtype=bool)


def pixel_to_patch(row: int, col: int, stride: int) -> tuple[int, int]:
    """Map a reference-image pixel to its native patch cell.

    The origin is the top-left pixel. ``stride`` is 8 for PatchCore and 4 for
    PaDiM at input 512.
    """

    if stride < 1:
        raise ValueError(f"パッチ刻みは 1 以上です: {stride}")
    if row < 0 or col < 0:
        raise ValueError(f"画素座標は 0 以上です: {(row, col)}")
    return row // stride, col // stride


def component_patch_maximum(patch_map: np.ndarray, component: np.ndarray) -> float:
    """Maximum float32 patch score among cells that meet the component.

    An empty component has no score. The caller keeps that component in the
    detection denominator instead of dropping it.
    """

    patch = np.asarray(patch_map, dtype=np.float32)
    mask = np.asarray(component, dtype=bool)
    if patch.ndim != 2 or mask.ndim != 2:
        raise ValueError("パッチ格子と成分マスクは2次元です。")
    if mask.shape[0] % patch.shape[0] or mask.shape[1] % patch.shape[1]:
        raise ValueError(f"刻みが割り切れません: {mask.shape} / {patch.shape}")
    stride_y = mask.shape[0] // patch.shape[0]
    stride_x = mask.shape[1] // patch.shape[1]
    if stride_y < 1 or stride_x < 1:
        raise ValueError("パッチ格子が画像より細かい前提が崩れています。")
    if not mask.any():
        return float("nan")
    cells = mask.reshape(patch.shape[0], stride_y, patch.shape[1], stride_x).any(axis=(1, 3))
    if not cells.any():
        return float("nan")
    selected = patch[cells]
    if not np.isfinite(selected).all():
        raise ValueError("成分に対応するパッチスコアが有限ではありません。")
    return float(selected.max())


def border_band(height: int, width: int, margin: int) -> np.ndarray:
    """True on the outer ``margin`` pixels. Margin 0 is an empty band.

    This band is the image border. It is not a receptive-field guarantee and
    it is not a warp-invalid region.
    """

    if margin < 0:
        raise ValueError(f"境界幅は 0 以上です: {margin}")
    band = np.zeros((height, width), dtype=bool)
    if margin == 0:
        return band
    if margin * 2 >= height or margin * 2 >= width:
        band[:] = True
        return band
    band[:margin, :] = True
    band[-margin:, :] = True
    band[:, :margin] = True
    band[:, -margin:] = True
    return band


def attribution_masks(defect: np.ndarray, border: np.ndarray) -> dict[str, np.ndarray]:
    """Keep the source masks and also build a display partition.

    Display priority, used only for a single-label view, is defect, then
    border, then the unresolved non-defect class. Intersections stay available
    as their own masks.
    """

    defect_mask = np.asarray(defect, dtype=bool)
    border_mask = np.asarray(border, dtype=bool)
    if defect_mask.shape != border_mask.shape:
        raise ValueError("欠陥マスクと境界の形状が違います。")
    nondefect = ~defect_mask
    return {
        "defect": defect_mask,
        "border": border_mask,
        "nondefect_unpartitioned": nondefect,
        "defect_and_border": defect_mask & border_mask,
        "defect_only": defect_mask & ~border_mask,
        "border_only": border_mask & nondefect,
        "interior_nondefect": nondefect & ~border_mask,
    }


def display_labels(defect: np.ndarray, border: np.ndarray) -> np.ndarray:
    """Exclusive labels: 1 defect, 2 border, 3 unresolved non-defect."""

    masks = attribution_masks(defect, border)
    labels = np.full(defect.shape, 3, dtype=np.int8)
    labels[masks["border"]] = 2
    labels[masks["defect"]] = 1
    return labels


def region_score_stats(score: np.ndarray, region: np.ndarray, high_quantile: float = 0.99) -> dict[str, float]:
    values = np.asarray(score, dtype=np.float64)[np.asarray(region, dtype=bool)]
    if values.size == 0:
        return {
            "n": 0,
            "q50": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
            "high_fraction": float("nan"),
        }
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "n": int(values.size),
            "q50": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
            "high_fraction": float("nan"),
        }
    threshold = float(np.quantile(finite, high_quantile))
    return {
        "n": int(values.size),
        "q50": float(np.quantile(finite, 0.50)),
        "q95": float(np.quantile(finite, 0.95)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(finite.max()),
        "high_fraction": float((finite >= threshold).mean()),
    }


def max_pixel_membership(score: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Share of the maximum-valued pixels that fall in each mask.

    Plateaus are counted. A single exclusive region is not forced.
    """

    array = np.asarray(score, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).any():
        return {name: float("nan") for name in masks}
    peak = np.nanmax(array)
    peaks = np.isfinite(array) & (array == peak)
    total = int(peaks.sum())
    shares = {}
    for name, mask in masks.items():
        shares[name] = float((peaks & np.asarray(mask, dtype=bool)).sum() / total)
    return shares
