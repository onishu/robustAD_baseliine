"""Reflect-pad then center-crop control for the frozen 144-image PatchCore.

Rotation happens on the padded canvas. The center 512 crop is taken once,
after every angle in the condition. Thresholds and the memory bank are read
from the frozen Theme A run and are not recomputed.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import Normalize, Resize

from robustad.benchmark import IMAGE_NET_MEAN, IMAGE_NET_STD, build_model
from robustad.catalog import SUBSETS
from robustad.data import find_mask, list_images
from robustad.prepare.capture import MapCapture, attach_patchcore_capture
from robustad.prepare.spec import VERDICT_UNDECIDED
from robustad.theme_a.components import component_record
from robustad.theme_a.run_synthetic import CONDITIONS as SOURCE_CONDITIONS
from robustad.theme_a.run_synthetic import _components, _guard, _load_rgb, _score_batch
from robustad.theme_a.warp import REFERENCE_SIZE, apply_angles, rotation_center

LOGGER = logging.getLogger("robustad.theme_a.boundary")

MARGIN = 22
FILL_EPS = 1e-5
BANK_SHAPE = (int(144 * 64 * 64 * 0.1), 1536)
CONDITIONS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("pad_warp_0", (0.0,)),
    ("pad_synthetic_m5", (-5.0,)),
    ("pad_synthetic_p5", (5.0,)),
    ("pad_restore_m5", (-5.0, 5.0)),
    ("pad_restore_p5", (5.0, -5.0)),
)
SOURCE_CONDITION = {
    "pad_warp_0": "warp_0",
    "pad_synthetic_m5": "synthetic_m5",
    "pad_synthetic_p5": "synthetic_p5",
    "pad_restore_m5": "restore_m5",
    "pad_restore_p5": "restore_p5",
}
LABEL = "反射延長でfill混入を抑えた対照"


def reflect_pad(image: torch.Tensor, margin: int = MARGIN) -> torch.Tensor:
    return F.pad(image, (margin, margin, margin, margin), mode="reflect")


def constant_pad(image: torch.Tensor, margin: int = MARGIN) -> torch.Tensor:
    return F.pad(image, (margin, margin, margin, margin), mode="constant", value=0.0)


def center_crop(image: torch.Tensor, margin: int = MARGIN) -> torch.Tensor:
    return image[..., margin : margin + REFERENCE_SIZE, margin : margin + REFERENCE_SIZE]


def pad_then_crop(image: torch.Tensor, margin: int = MARGIN) -> torch.Tensor:
    """Reflect pad and crop. This path does not call the rotator."""

    return center_crop(reflect_pad(image, margin), margin)


def extend_and_warp(
    image: torch.Tensor,
    angles: tuple[float, ...],
    *,
    nearest: bool,
    margin: int = MARGIN,
    pad_mode: str = "reflect",
) -> tuple[torch.Tensor, list[dict]]:
    """Pad once, apply the whole angle list, then crop once.

    An intermediate crop or a second pad is not part of this path.
    """

    if not angles:
        raise ValueError("この経路は角度を1つ以上持つ。ワープなしの切出しは pad_then_crop を使う。")
    squeezed = image.ndim == 2
    canvas_image = image[None] if squeezed else image
    if pad_mode == "reflect":
        canvas = reflect_pad(canvas_image, margin)
    elif pad_mode == "constant":
        canvas = constant_pad(canvas_image, margin)
    else:
        raise ValueError(f"pad は reflect か constant です: {pad_mode}")
    warped, steps = apply_angles(canvas, angles, nearest=nearest)
    cropped = center_crop(warped, margin)
    if squeezed:
        cropped = cropped[0]
    return cropped, steps


def fill_probe(angles: tuple[float, ...], margin: int = MARGIN) -> torch.Tensor:
    """Sequential bilinear of an all-ones padded canvas, then one center crop."""

    side = REFERENCE_SIZE + 2 * margin
    ones = torch.ones((1, side, side), dtype=torch.float32)
    warped, _steps = apply_angles(ones, angles, nearest=False)
    return center_crop(warped, margin)[0]


def content_probe(angles: tuple[float, ...], margin: int = MARGIN) -> torch.Tensor:
    """Ones on the original square and zeros on the pad, after the same angles."""

    side = REFERENCE_SIZE + 2 * margin
    canvas = torch.zeros((1, side, side), dtype=torch.float32)
    canvas[:, margin : margin + REFERENCE_SIZE, margin : margin + REFERENCE_SIZE] = 1.0
    warped, _steps = apply_angles(canvas, angles, nearest=False)
    return center_crop(warped, margin)[0]


def fill_is_outside(angles: tuple[float, ...], margin: int = MARGIN) -> bool:
    if not angles:
        return True
    return bool(torch.all(fill_probe(angles, margin) >= 1.0 - FILL_EPS))


def asymmetric_marker() -> torch.Tensor:
    """A pattern with no mirror symmetry through the image center."""

    marker = torch.zeros((3, REFERENCE_SIZE, REFERENCE_SIZE), dtype=torch.float32)
    marker[:, 80:90, 400:430] = 1.0
    marker[:, 300:340, 40:55] = 0.4
    marker[0, 200, 350] = 0.73
    marker[1, 210, 120] = 0.8
    marker[2, 400, 256] = 0.3
    return marker


def _peak_matches(reference: torch.Tensor, other: torch.Tensor, level: float) -> bool:
    source = torch.nonzero(reference > level)
    destination = torch.nonzero(other > level)
    if source.shape != destination.shape:
        return False
    return bool(torch.equal(source, destination))


def geometry_preflight(margin: int = MARGIN) -> dict[str, object]:
    """Check the three 0° paths and the single-crop round trip. No model and no scores."""

    generator = torch.Generator().manual_seed(0)
    image = torch.rand((3, REFERENCE_SIZE, REFERENCE_SIZE), generator=generator)
    untouched = pad_then_crop(image, margin)
    pad_crop_max = float((untouched - image).abs().max())
    zero, _steps = extend_and_warp(image, (0.0,), nearest=False, margin=margin)
    zero_abs = (zero - image).abs()
    marker = asymmetric_marker()
    nearest, _steps = extend_and_warp(marker, (0.0,), nearest=True, margin=margin)
    bilinear, _steps = extend_and_warp(marker, (0.0,), nearest=False, margin=margin)
    plus, _steps = extend_and_warp(marker, (5.0,), nearest=True, margin=margin, pad_mode="constant")
    source_pixel = torch.nonzero(torch.isclose(marker[0], torch.tensor(0.73)))
    moved = torch.nonzero(torch.isclose(plus[0], torch.tensor(0.73)))
    center_before = marker[:, 255:257, 255:257].clone()
    marker[:, 255:257, 255:257] = 1.0
    stayed, _steps = extend_and_warp(marker, (5.0,), nearest=True, margin=margin, pad_mode="constant")
    marker[:, 255:257, 255:257] = center_before
    ys, xs = np.nonzero(stayed[0, 255 - 8 : 255 + 10, 255 - 8 : 255 + 10].numpy() > 0)
    centroid = [float(ys.mean() + 247), float(xs.mean() + 247)] if ys.size else [float("nan"), float("nan")]
    canvas = reflect_pad(image, margin)
    sequential, _steps = apply_angles(canvas, (-5.0, 5.0), nearest=False)
    sequential_crop = center_crop(sequential, margin)
    produced, _steps = extend_and_warp(image, (-5.0, 5.0), nearest=False, margin=margin)
    mid, _steps = extend_and_warp(image, (-5.0,), nearest=False, margin=margin)
    forbidden, _steps = extend_and_warp(mid, (5.0,), nearest=False, margin=margin)
    fill_report = {}
    content_report = {}
    for name, angles in CONDITIONS:
        probe = fill_probe(angles, margin)
        valid = probe >= 1.0 - FILL_EPS
        origin = content_probe(angles, margin)
        fill_report[name] = {
            "min": float(probe.min()),
            "invalid_pixels": int((~valid).sum()),
            "angles": list(angles),
            "intermediate_crop": False,
        }
        content_report[name] = {
            "pixels_from_original_square": int((origin >= 1.0 - FILL_EPS).sum()),
            "pixels": int(origin.numel()),
        }
    wider = None
    if any(item["invalid_pixels"] for item in fill_report.values()):
        wider = margin
        while wider < 160 and any(not fill_is_outside(angles, wider) for _name, angles in CONDITIONS):
            wider += 1
    center = rotation_center(REFERENCE_SIZE + 2 * margin, REFERENCE_SIZE + 2 * margin)
    report = {
        "margin_pixels": margin,
        "padding": "reflect",
        "rgb_pad": "reflect",
        "mask_pad": "constant 0",
        "crop": "center 512 once after the full angle list",
        "label": LABEL,
        "pad_crop_matches_original_max_abs": pad_crop_max,
        "pad_zero_warp_max_abs": float(zero_abs.max()),
        "pad_zero_warp_mean_abs": float(zero_abs.mean()),
        "marker_nearest_max_abs": float((nearest - marker).abs().max()),
        "marker_bilinear_max_abs": float((bilinear - marker).abs().max()),
        "marker_bilinear_mean_abs": float((bilinear - marker).abs().mean()),
        "marker_nearest_peaks_match": _peak_matches(marker[0], nearest[0], 0.9),
        "marker_bilinear_peaks_match": _peak_matches(marker[0], bilinear[0], 0.9),
        "plus5_source_row_col": [int(source_pixel[0, 0]), int(source_pixel[0, 1])],
        "plus5_destination_row_col": [int(moved[0, 0]), int(moved[0, 1])] if moved.numel() else None,
        "plus5_marker_pixels": int(moved.shape[0]),
        "plus5_moved_up": bool(moved.shape[0] == 1 and int(moved[0, 0]) < int(source_pixel[0, 0])),
        "center_block_centroid_row_col": centroid,
        "canvas_center_xy": center,
        "content_center_on_canvas_xy": (margin + REFERENCE_SIZE / 2, margin + REFERENCE_SIZE / 2),
        "round_trip_matches_single_canvas": bool(torch.allclose(produced, sequential_crop, atol=0.0)),
        "intermediate_crop_differs": not bool(torch.allclose(produced, forbidden, atol=1e-5)),
        "fill_probe": fill_report,
        "content_probe": content_report,
        "width_increased_because_fill_entered_crop": wider is not None and wider != margin,
        "revised_margin_pixels": wider,
    }
    failures = []
    if pad_crop_max != 0.0:
        failures.append("反射padのあとワープせず切り出すと元画像と一致しません。")
    if not report["marker_nearest_peaks_match"] or not report["marker_bilinear_peaks_match"]:
        failures.append("0°ワープで非対称マーカーの画素位置が動きました。")
    if report["marker_nearest_max_abs"] != 0.0:
        failures.append("0°の nearest がマーカー値を変えました。")
    if int(source_pixel.shape[0]) != 1 or int(moved.shape[0]) != 1 or not report["plus5_moved_up"]:
        failures.append("パディング後の +5° で非対称マーカーが反時計回りに一意対応しません。")
    if abs(centroid[0] - 255.5) > 1.5 or abs(centroid[1] - 255.5) > 1.5:
        failures.append("パディング後の回転中心がずれました。")
    if not report["round_trip_matches_single_canvas"] or not report["intermediate_crop_differs"]:
        failures.append("往復がキャンバス上の一括変換になっていません。")
    if center != (margin + REFERENCE_SIZE / 2, margin + REFERENCE_SIZE / 2):
        failures.append("キャンバス中心と元画像の中心が一致しません。")
    if report["pad_zero_warp_max_abs"] > 1.0 / 255.0:
        failures.append("0°ワープの差が1/255を超え、目視できる差として扱います。")
    report["passed"] = not failures
    report["failures"] = failures
    return report


def real_image_zero_diff(data_root: Path, margin: int = MARGIN) -> dict[str, float | str]:
    subset = SUBSETS["pcb"]
    domain = next(item for item in subset.domains if item.name == "SD")
    path = list_images(subset.domain_dir(data_root, domain) / "normal")[0]
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    image = _load_rgb(path, resize)
    untouched = pad_then_crop(image, margin)
    warped, _steps = extend_and_warp(image, (0.0,), nearest=False, margin=margin)
    delta = (warped - image).abs()
    return {
        "source_id": path.stem,
        "pad_crop_max_abs": float((untouched - image).abs().max()),
        "pad_zero_warp_max_abs": float(delta.max()),
        "pad_zero_warp_mean_abs": float(delta.mean()),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _quantile_row(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.1, method="linear")),
        "p50": float(np.quantile(array, 0.5, method="linear")),
        "p90": float(np.quantile(array, 0.9, method="linear")),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _cell_fully_valid(valid: np.ndarray) -> np.ndarray:
    stride = REFERENCE_SIZE // valid.shape[0] if valid.shape[0] != REFERENCE_SIZE else 8
    if valid.shape == (64, 64):
        return valid
    block = valid.reshape(64, stride, 64, stride)
    return block.all(axis=(1, 3))


def _paired_decisions(reference: np.ndarray, other: np.ndarray) -> dict[str, object]:
    both = int((reference & other).sum())
    reference_only = int((reference & ~other).sum())
    other_only = int((~reference & other).sum())
    neither = int((~reference & ~other).sum())
    return {
        "n": int(reference.size),
        "both_detected": both,
        "reference_only": reference_only,
        "other_only": other_only,
        "neither": neither,
        "observed_disagreement": reference_only + other_only,
        "observed_zero_disagreement_is_not_equivalence": True,
    }


def _load_bank(lightning_model, bank_path: Path) -> torch.Tensor:
    saved = torch.load(bank_path, map_location="cpu", weights_only=False)
    lightning_model.model.memory_bank = saved["memory_bank"]
    lightning_model.model.eval()
    if torch.cuda.is_available():
        lightning_model.cuda()
    bank = lightning_model.model.memory_bank.detach()
    if tuple(bank.shape) != BANK_SHAPE:
        raise RuntimeError(f"メモリバンクの形状が保存済み144枚モデルと違います: {tuple(bank.shape)}")
    if bank.device.type != "cuda":
        raise RuntimeError(f"採点デバイスが CUDA ではありません: {bank.device}")
    return bank


def _test_items(data_root: Path) -> list[tuple[Path, int, str]]:
    subset = SUBSETS["pcb"]
    domain = next(item for item in subset.domains if item.name == "SD")
    domain_dir = subset.domain_dir(data_root, domain)
    items = [(path, 0, "") for path in list_images(domain_dir / "normal")]
    items += [(path, 1, find_mask(path, domain_dir / "masks")) for path in list_images(domain_dir / "anomaly")]
    if len(items) != 195:
        raise RuntimeError(f"PCB SD は 195 枚です: {len(items)}")
    return items


def run_boundary(data_root: Path, output_dir: Path, source_dir: Path, margin: int = MARGIN) -> dict[str, object]:
    output_dir = _guard(output_dir)
    if output_dir.resolve() == source_dir.resolve():
        raise RuntimeError("境界対照は凍結済みの results_a0 には書きません。")
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = geometry_preflight(margin)
    preflight["real_image"] = real_image_zero_diff(data_root, margin)
    if preflight["real_image"]["pad_crop_max_abs"] != 0.0:
        preflight["passed"] = False
        preflight["failures"].append("実画像の pad→切出しが元画像と一致しません。")
    if float(preflight["real_image"]["pad_zero_warp_max_abs"]) > 1.0 / 255.0:
        preflight["passed"] = False
        preflight["failures"].append("実画像の 0°ワープ差が 1/255 を超えました。")
    if preflight["width_increased_because_fill_entered_crop"]:
        raise RuntimeError(f"22px では中央切出しに fill が混ざります。再見積りが必要です: {preflight['revised_margin_pixels']}")
    if not preflight["passed"]:
        raise RuntimeError(f"境界対照の実行前確認に失敗しました: {preflight['failures']}")
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")

    thresholds = json.loads((source_dir / "thresholds.json").read_text(encoding="utf-8"))
    image_cut = float(thresholds["image"]["threshold"])
    patch_cut = float(thresholds["patch"]["threshold"])
    (output_dir / "thresholds_used.json").write_text(
        json.dumps(
            {
                "source": str(source_dir / "thresholds.json"),
                "recomputed": False,
                "image_threshold": image_cut,
                "patch_threshold": patch_cut,
                "decision": thresholds["decision"],
                "ties": thresholds["ties"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fill_valid = {}
    content_valid = {}
    matrices = {}
    dummy = torch.zeros((3, REFERENCE_SIZE, REFERENCE_SIZE), dtype=torch.float32)
    for name, angles in CONDITIONS:
        probe = fill_probe(angles, margin)
        fill_valid[name] = (probe >= 1.0 - FILL_EPS).numpy()
        content_valid[name] = (content_probe(angles, margin) >= 1.0 - FILL_EPS).numpy()
        _cropped, steps = extend_and_warp(dummy, angles, nearest=False, margin=margin)
        matrices[name] = {"canvas_side": REFERENCE_SIZE + 2 * margin, "crop_once_after_angles": True, "steps": steps}
    np.savez(output_dir / "valid_regions.npz", **{f"fill_{name}": mask for name, mask in fill_valid.items()}, **{f"content_{name}": mask for name, mask in content_valid.items()})
    usage = shutil.disk_usage(output_dir)
    manifest = {
        "label": LABEL,
        "not_called": ["真の画像外情報の復元", "境界影響の完全除去", "実用的改善", "同等", "既知変換による性能上限"],
        "margin_pixels": margin,
        "margin_rule": "approved geometric width; increased only if the sequential fill probe entered the center crop",
        "conditions": [{"name": name, "angles": list(angles), "source_condition": SOURCE_CONDITION[name]} for name, angles in CONDITIONS],
        "source_images": 195,
        "forwards": 195 * len(CONDITIONS),
        "bank": str(source_dir / "bank" / "pcb__patchcore__seed0.pt"),
        "refit": False,
        "recalibrate": False,
        "disk_free_bytes_before_score": usage.free,
        "source_conditions_present": [name for name, _angles in SOURCE_CONDITIONS],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "matrices.json").write_text(json.dumps(matrices), encoding="utf-8")

    bank_path = source_dir / "bank" / "pcb__patchcore__seed0.pt"
    lightning_model = build_model("patchcore", REFERENCE_SIZE, "timm")
    bank = _load_bank(lightning_model, bank_path)
    LOGGER.info("boundary score device %s bank %s", bank.device, tuple(bank.shape))
    torch_model = lightning_model.model
    capture = MapCapture()
    attach_patchcore_capture(torch_model, capture)
    normalize = Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    fpr_rows: list[dict[str, object]] = []
    location_rows: list[dict[str, object]] = []
    common_fill = np.ones((REFERENCE_SIZE, REFERENCE_SIZE), dtype=bool)
    for mask in fill_valid.values():
        common_fill &= mask
    for index, (path, label, mask_path) in enumerate(_test_items(data_root)):
        rgb = _load_rgb(path, resize)
        components, alone = _components(mask_path) if label == 1 else ([], {})
        warped = {}
        for name, angles in CONDITIONS:
            warped[name] = extend_and_warp(rgb, angles, nearest=False, margin=margin)[0]
        names = [name for name, _angles in CONDITIONS]
        predicted, patch = _score_batch(torch_model, [warped[name] for name in names], capture, normalize)
        for offset, (name, angles) in enumerate(CONDITIONS):
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
                full_hits = int((grid >= patch_cut).sum())
                content_cells = _cell_fully_valid(content_valid[name])
                content_values = grid[content_cells]
                content_hits = int((content_values >= patch_cut).sum()) if content_values.size else 0
                fpr_rows.append(
                    {
                        "source_id": path.stem,
                        "condition": name,
                        "n_full": int(grid.size),
                        "n_false_full": full_hits,
                        "fpr_full": full_hits / grid.size,
                        "n_content": int(content_values.size),
                        "n_false_content": content_hits,
                        "fpr_content": content_hits / content_values.size if content_values.size else float("nan"),
                    }
                )
            tied = np.argwhere(grid == grid.max())
            flat = int(np.argmax(grid))
            row_i, col_i = divmod(flat, grid.shape[1])
            content_cells = _cell_fully_valid(content_valid[name])
            location_rows.append(
                {
                    "source_id": path.stem,
                    "condition": name,
                    "label": label,
                    "argmax_row": row_i,
                    "argmax_col": col_i,
                    "n_tied_max": int(tied.shape[0]),
                    "tied_max_touches_reflected_pad": bool((~content_cells)[tied[:, 0], tied[:, 1]].any()),
                    "tied_max_inside_original_square": bool(content_cells[tied[:, 0], tied[:, 1]].all()),
                }
            )
            for row in components:
                component_id = int(row["component_id"])
                mask = alone.get(component_id)
                warped_mask = None if mask is None else extend_and_warp(torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32))), angles, nearest=True, margin=margin, pad_mode="constant")[0].numpy() > 0
                record = component_record(
                    component_id=component_id,
                    status_input=str(row["status"]),
                    area_original=int(row["area_original"]),
                    area_512=int(row["area_resized_alone"]),
                    warped=warped_mask,
                    condition_valid=fill_valid[name],
                    common_valid=common_fill,
                    patch_map=grid,
                    threshold=patch_cut,
                )
                record.update({"source_id": path.stem, "condition": name})
                component_rows.append(record)
        if index % 25 == 0:
            LOGGER.info("boundary scored %d / 195", index + 1)
    _write_csv(output_dir / "tables" / "images.csv", image_rows)
    _write_csv(output_dir / "tables" / "components.csv", component_rows)
    _write_csv(output_dir / "tables" / "image_patch_fpr.csv", fpr_rows)
    _write_csv(output_dir / "tables" / "max_patch_location.csv", location_rows)
    summary = _summarize(output_dir, source_dir, image_rows, component_rows, fpr_rows, location_rows, image_cut, patch_cut)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _summarize(output_dir, source_dir, image_rows, component_rows, fpr_rows, location_rows, image_cut, patch_cut) -> dict:
    source_images = _read_csv(source_dir / "tables" / "images.csv")
    source_components = _read_csv(source_dir / "tables" / "components.csv")
    by_source: dict[tuple[str, str], dict[str, str]] = {(row["source_id"], row["condition"]): row for row in source_images}
    by_new: dict[tuple[str, str], dict[str, object]] = {(str(row["source_id"]), str(row["condition"])): row for row in image_rows}
    comparisons_source = {}
    comparisons_pad0 = {}
    for name, _angles in CONDITIONS:
        source_name = SOURCE_CONDITION[name]
        ids = [str(row["source_id"]) for row in image_rows if row["condition"] == name]
        reference_detected = np.asarray([by_source[(item, source_name)]["detected"] == "True" for item in ids], dtype=bool)
        other_detected = np.asarray([bool(by_new[(item, name)]["detected"]) for item in ids], dtype=bool)
        reference_score = np.asarray([float(by_source[(item, source_name)]["reproduction_score"]) for item in ids], dtype=np.float64)
        other_score = np.asarray([float(by_new[(item, name)]["reproduction_score"]) for item in ids], dtype=np.float64)
        labels = np.asarray([int(by_new[(item, name)]["label"]) for item in ids])
        comparisons_source[name] = {
            "reference_condition": source_name,
            "score_delta": _quantile_row(other_score - reference_score),
            "normal_score_delta": _quantile_row(other_score[labels == 0] - reference_score[labels == 0]),
            "anomaly_score_delta": _quantile_row(other_score[labels == 1] - reference_score[labels == 1]),
            "decisions": _paired_decisions(reference_detected, other_detected),
            "normal_decisions": _paired_decisions(reference_detected[labels == 0], other_detected[labels == 0]),
            "anomaly_decisions": _paired_decisions(reference_detected[labels == 1], other_detected[labels == 1]),
        }
        if name == "pad_warp_0":
            continue
        base_detected = np.asarray([bool(by_new[(item, "pad_warp_0")]["detected"]) for item in ids], dtype=bool)
        base_score = np.asarray([float(by_new[(item, "pad_warp_0")]["reproduction_score"]) for item in ids], dtype=np.float64)
        comparisons_pad0[name] = {
            "reference_condition": "pad_warp_0",
            "score_delta": _quantile_row(other_score - base_score),
            "normal_score_delta": _quantile_row(other_score[labels == 0] - base_score[labels == 0]),
            "anomaly_score_delta": _quantile_row(other_score[labels == 1] - base_score[labels == 1]),
            "decisions": _paired_decisions(base_detected, other_detected),
            "normal_decisions": _paired_decisions(base_detected[labels == 0], other_detected[labels == 0]),
            "anomaly_decisions": _paired_decisions(base_detected[labels == 1], other_detected[labels == 1]),
        }
    achieved = {}
    for name, _angles in CONDITIONS:
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
            "per_image_normal_patch_fpr_content": _quantile_row(np.asarray([float(row["fpr_content"]) for row in fprs])),
            "pooled_patch_interval_omitted": True,
            "normal_max_patch_inside_original_square": int(sum(bool(row["tied_max_inside_original_square"]) for row in locations)),
            "normal_max_patch_touches_reflection": int(sum(bool(row["tied_max_touches_reflected_pad"]) for row in locations)),
            "component_detected": int(sum(bool(row["detected"]) for row in components)),
            "component_n": len(components),
            "input_unevaluable": int(sum(row["status_input"] == "入力変換による評価不能" for row in components)),
        }
    component_compare = _component_deltas(component_rows, source_components)
    return {
        "label": LABEL,
        "verdict": VERDICT_UNDECIDED,
        "output_dir": str(output_dir),
        "image_threshold": image_cut,
        "patch_threshold": patch_cut,
        "thresholds_recomputed": False,
        "conditions": achieved,
        "against_unpadded_source": comparisons_source,
        "against_padded_zero": comparisons_pad0,
        "components": component_compare,
    }


def _component_deltas(component_rows: list[dict[str, object]], source_components: list[dict[str, str]]) -> dict[str, object]:
    source = {(row["source_id"], row["condition"], int(row["component_id"])): row for row in source_components}
    by_image: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in component_rows:
        by_image.setdefault(str(row["source_id"]), {}).setdefault(str(row["condition"]), []).append(row)
    disagreements_source = {name: 0 for name, _angles in CONDITIONS}
    disagreements_pad0 = {name: 0 for name, _angles in CONDITIONS if name != "pad_warp_0"}
    image_delta_source: dict[str, list[float]] = {name: [] for name, _angles in CONDITIONS}
    image_delta_pad0: dict[str, list[float]] = {name: [] for name, _angles in CONDITIONS if name != "pad_warp_0"}
    for source_id, grouped in by_image.items():
        finite = {}
        detected = {}
        for name, _angles in CONDITIONS:
            rows = grouped[name]
            values = [float(row["statistic_common"]) for row in rows if np.isfinite(float(row["statistic_common"]))]
            finite[name] = max(values) if values else float("nan")
            detected[name] = {int(row["component_id"]): bool(row["detected"]) for row in rows}
            source_name = SOURCE_CONDITION[name]
            for row in rows:
                key = (source_id, source_name, int(row["component_id"]))
                if key not in source:
                    continue
                if bool(row["detected"]) != (source[key]["detected"] == "True"):
                    disagreements_source[name] += 1
        for name, _angles in CONDITIONS:
            source_name = SOURCE_CONDITION[name]
            source_values = [
                float(source[(source_id, source_name, int(row["component_id"]))]["statistic_common"])
                for row in grouped[name]
                if (source_id, source_name, int(row["component_id"])) in source
                and np.isfinite(float(source[(source_id, source_name, int(row["component_id"]))]["statistic_common"]))
            ]
            if source_values and np.isfinite(finite[name]):
                image_delta_source[name].append(finite[name] - max(source_values))
            if name == "pad_warp_0" or not np.isfinite(finite[name]) or not np.isfinite(finite["pad_warp_0"]):
                continue
            image_delta_pad0[name].append(finite[name] - finite["pad_warp_0"])
            mismatched = [key for key, flag in detected["pad_warp_0"].items() if detected[name][key] != flag]
            disagreements_pad0[name] += len(mismatched)
    return {
        "unit": "one component-maximum per anomaly image",
        "components_within_image_are_not_independent_samples": True,
        "observed_detection_disagreements_against_unpadded": disagreements_source,
        "observed_detection_disagreements_against_padded_zero": disagreements_pad0,
        "image_component_max_delta_against_unpadded": {name: _quantile_row(np.asarray(values)) for name, values in image_delta_source.items()},
        "image_component_max_delta_against_padded_zero": {name: _quantile_row(np.asarray(values)) for name, values in image_delta_pad0.items()},
        "observed_zero_disagreement_is_not_equivalence": True,
    }
