"""Fit one PCB PatchCore on 144 images and score synthetic SD rotations.

The real Rotation folder is not read. Thresholds come from the 36
untransformed calibration images. Conditions do not refit or recalibrate.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import Normalize, Resize

from robustad.benchmark import BATCH_SIZE, IMAGE_NET_MEAN, IMAGE_NET_STD, build_model, make_engine
from robustad.catalog import SUBSETS
from robustad.data import find_mask, list_images
from robustad.prepare.calibration import (
    fpr_report,
    image_threshold,
    patch_threshold,
    pool_patch_scores,
    rate_comparison,
)
from robustad.prepare.capture import MapCapture, attach_patchcore_capture
from robustad.prepare.regions import (
    UNEVALUABLE_INPUT_TRANSFORM,
    component_correspondence,
    label_components,
    resize_binary_mask,
)
from robustad.prepare.spec import TARGET_IMAGE_FPR, TARGET_PATCH_FPR, VERDICT_UNDECIDED
from robustad.theme_a.components import component_record
from robustad.theme_a.plan import capacity_report
from robustad.theme_a.warp import REFERENCE_SIZE, apply_angles, rotated_mask, valid_after_angles

LOGGER = logging.getLogger("robustad.theme_a")

CONDITIONS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("original", ()),
    ("warp_0", (0.0,)),
    ("synthetic_m5", (-5.0,)),
    ("synthetic_p5", (5.0,)),
    ("restore_m5", (-5.0, 5.0)),
    ("restore_p5", (5.0, -5.0)),
)
EXPECTED_CORESET = int(144 * 64 * 64 * 0.1)


def _guard(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if resolved.name in {"results", "results_a1", "results_tv", "diag"}:
        raise RuntimeError(f"保護された出力先です: {resolved}")
    if "results_p0" in resolved.parts and "diag" in resolved.parts:
        raise RuntimeError(f"results_p0/diag は変更しません: {resolved}")
    return resolved


def _stem_from_key(key: str) -> str:
    body, index = key.rsplit("#", 1)
    return f"{body}_{index}"


def _load_rgb(path: Path, resize: Resize) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    return resize(tensor)


def _components(mask_path: str) -> tuple[list[dict[str, object]], dict[int, np.ndarray]]:
    with Image.open(mask_path) as image:
        original = np.asarray(image.convert("L"))
    rows = component_correspondence(original, REFERENCE_SIZE, REFERENCE_SIZE)
    labels = label_components(original)
    alone: dict[int, np.ndarray] = {}
    for row in rows:
        if row["status"] == UNEVALUABLE_INPUT_TRANSFORM:
            continue
        alone[int(row["component_id"])] = resize_binary_mask(labels == int(row["component_id"]), REFERENCE_SIZE, REFERENCE_SIZE)
    return rows, alone


def _score_batch(torch_model, prenorm: list[torch.Tensor], capture: MapCapture, normalize: Normalize) -> tuple[np.ndarray, np.ndarray]:
    capture.clear()
    batch = torch.stack([normalize(image) for image in prenorm])
    device = next(torch_model.parameters()).device
    with torch.no_grad():
        prediction = torch_model(batch.to(device))
    if len(capture.records) != 1:
        raise RuntimeError(f"スコア捕捉が1件ではありません: {len(capture.records)}")
    predicted = prediction.pred_score.detach().cpu().float().view(-1)
    captured = capture.records[0]["reproduction_image"].float().view(-1)
    if predicted.shape != captured.shape or not torch.allclose(predicted, captured, atol=1e-4):
        raise RuntimeError("再現スコアと pred_score が一致しません。")
    patch = capture.records[0]["patch_map"].float()
    if patch.ndim == 4:
        patch = patch[:, 0]
    return predicted.numpy(), patch.numpy()


def _paired(reference: np.ndarray, other: np.ndarray) -> dict[str, object]:
    delta = other.astype(np.float64) - reference.astype(np.float64)
    difference = float(delta.mean()) if delta.size else float("nan")
    if delta.size < 2:
        low = high = float("nan")
    else:
        width = float(1.96 * delta.std(ddof=1) / np.sqrt(delta.size))
        low, high = difference - width, difference + width
    return {
        "difference": difference,
        "difference_interval_low": low,
        "difference_interval_high": high,
        "practical_improvement": VERDICT_UNDECIDED,
        "equivalence": VERDICT_UNDECIDED,
        "known_transform_ceiling": VERDICT_UNDECIDED,
    }


def run_synthetic(data_root: Path, output_dir: Path, manifest_path: Path, *, reuse_bank: bool = False) -> dict:
    from lightning.pytorch import seed_everything

    output_dir = _guard(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = capacity_report(output_dir)
    if not report["free_exceeds_estimate"]:
        raise RuntimeError(f"空き容量が不足します: {report}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fit_stems = {_stem_from_key(key) for key in manifest["datasets"]["pcb"]["fit"]}
    calibration_stems = {_stem_from_key(key) for key in manifest["datasets"]["pcb"]["calibration"]}
    if fit_stems & calibration_stems:
        raise RuntimeError("fit と calibration が重なっています。")
    subset = SUBSETS["pcb"]
    train_paths = list_images(subset.train_dir(data_root) / "normal")
    by_stem = {path.stem: path for path in train_paths}
    if set(by_stem) != fit_stems | calibration_stems:
        raise RuntimeError("固定 manifest と train 正常画像のファイル名が一致しません。")
    domain = next(item for item in subset.domains if item.name == "SD")
    domain_dir = subset.domain_dir(data_root, domain)
    test_items = [(path, 0, "") for path in list_images(domain_dir / "normal")]
    test_items += [(path, 1, find_mask(path, domain_dir / "masks")) for path in list_images(domain_dir / "anomaly")]
    if len(test_items) != 195:
        raise RuntimeError(f"PCB SD は 195 枚です: {len(test_items)}")
    pre_manifest = {
        "protocol": "synthetic_sd_rotation",
        "real_rotation_folder_used": False,
        "angle_meaning": "synthetic experiment setting, not RobustAD ±5%",
        "conditions": [{"name": name, "angles": list(angles)} for name, angles in CONDITIONS],
        "quantile": "np.quantile(scores, 1 - 0.05, method='linear')",
        "decision": "score >= threshold; ties count as anomalous",
        "fit_images": 144,
        "calibration_images": 36,
        "capacity": report,
        "do_not_modify": ["results_a1", "results_p0/diag"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(pre_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    probe = _load_rgb(test_items[0][0], resize)
    if tuple(probe.shape) != (3, REFERENCE_SIZE, REFERENCE_SIZE):
        raise RuntimeError(f"512 基準画像ではありません: {tuple(probe.shape)}")
    for _name, angles in CONDITIONS:
        warped, _steps = apply_angles(probe, angles, nearest=False)
        if tuple(warped.shape) != tuple(probe.shape):
            raise RuntimeError(f"ワープ後の形状が変わりました: {angles}")
    seed_everything(0, workers=True)
    from robustad.data import build_train_module

    train_module, counts = build_train_module(
        subset, data_root, seed=0, val_ratio=0.2, batch_size=BATCH_SIZE["patchcore"], num_workers=4
    )
    if counts["n_train"] != 144 or counts["n_val"] != 36:
        raise RuntimeError(f"研究用分割ではありません: {counts}")
    LOGGER.info("fit pcb / patchcore / 144")
    lightning_model = build_model("patchcore", REFERENCE_SIZE, "timm")
    bank_path = output_dir / "bank" / "pcb__patchcore__seed0.pt"
    if reuse_bank:
        if not bank_path.is_file():
            raise RuntimeError(f"再利用するメモリバンクがありません: {bank_path}")
        saved = torch.load(bank_path, map_location="cpu", weights_only=False)
        lightning_model.model.memory_bank = saved["memory_bank"]
        LOGGER.info("reuse bank %s", tuple(saved["memory_bank"].shape))
    else:
        engine = make_engine(output_dir)
        engine.fit(model=lightning_model, datamodule=train_module)
    torch_model = lightning_model.model
    torch_model.eval()
    if torch.cuda.is_available():
        lightning_model.cuda()
    bank = torch_model.memory_bank.detach()
    if tuple(bank.shape) != (EXPECTED_CORESET, 1536):
        raise RuntimeError(f"メモリバンクが 144 枚の想定と違います: {tuple(bank.shape)}")
    LOGGER.info("score device %s bank %s", bank.device, tuple(bank.shape))
    if not reuse_bank:
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"memory_bank": bank.cpu(), "num_neighbors": int(torch_model.num_neighbors), "n_fit": 144, "seed": 0},
            bank_path,
        )
    capture = MapCapture()
    attach_patchcore_capture(torch_model, capture)
    normalize = Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    calibration_scores = []
    calibration_patches = []
    for stem in sorted(calibration_stems):
        score, patch = _score_batch(torch_model, [_load_rgb(by_stem[stem], resize)], capture, normalize)
        calibration_scores.append(float(score[0]))
        calibration_patches.append(patch[0])
    image_scores = np.asarray(calibration_scores, dtype=np.float32)
    patch_scores = pool_patch_scores(calibration_patches)
    image_cut = image_threshold(image_scores, TARGET_IMAGE_FPR)
    patch_cut = patch_threshold(patch_scores, TARGET_PATCH_FPR)
    thresholds = {
        "image": fpr_report(image_scores, image_cut, TARGET_IMAGE_FPR),
        "patch": fpr_report(patch_scores, patch_cut, TARGET_PATCH_FPR),
        "decision": "score >= threshold",
        "ties": "counted as anomalous",
        "quantile": "np.quantile(scores, 1 - alpha, method='linear')",
        "alpha": 0.05,
        "recalibrated_per_condition": False,
    }
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    valid = {name: valid_after_angles(REFERENCE_SIZE, REFERENCE_SIZE, angles).numpy() for name, angles in CONDITIONS}
    common = np.ones((REFERENCE_SIZE, REFERENCE_SIZE), dtype=bool)
    for mask in valid.values():
        common &= mask
    np.savez(output_dir / "valid_regions.npz", common=common, **{name: mask for name, mask in valid.items()})
    matrix_payload = {}
    dummy = torch.zeros((3, REFERENCE_SIZE, REFERENCE_SIZE), dtype=torch.float32)
    for name, angles in CONDITIONS:
        _unused, steps = apply_angles(dummy, angles, nearest=False)
        matrix_payload[name] = steps
    (output_dir / "matrices.json").write_text(json.dumps(matrix_payload), encoding="utf-8")
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    patch_false = {name: 0 for name, _angles in CONDITIONS}
    patch_count = {name: 0 for name, _angles in CONDITIONS}
    for index, (path, label, mask_path) in enumerate(test_items):
        rgb = _load_rgb(path, resize)
        components, alone = _components(mask_path) if label == 1 else ([], {})
        warped = {name: apply_angles(rgb, angles, nearest=False)[0] for name, angles in CONDITIONS}
        names = [name for name, _angles in CONDITIONS]
        predicted, patch = _score_batch(torch_model, [warped[name] for name in names], capture, normalize)
        scores = {name: float(predicted[offset]) for offset, name in enumerate(names)}
        patches = {name: patch[offset].astype(np.float32, copy=False) for offset, name in enumerate(names)}
        for name, _angles in CONDITIONS:
            np.savez(
                map_dir / f"{index:05d}__{name}.npz",
                image_id=np.array(path.stem),
                condition=np.array(name),
                label=np.int32(label),
                reproduction_score=np.float32(scores[name]),
                patch_map=patches[name],
            )
            image_rows.append(
                {
                    "source_id": path.stem,
                    "condition": name,
                    "label": label,
                    "reproduction_score": scores[name],
                    "detected": scores[name] >= image_cut,
                    "image_threshold": image_cut,
                    "patch_threshold": patch_cut,
                }
            )
            if label == 0:
                values = patches[name].reshape(-1)
                patch_false[name] += int((values >= patch_cut).sum())
                patch_count[name] += int(values.size)
            for row in components:
                component_id = int(row["component_id"])
                mask = alone.get(component_id)
                angles = dict(CONDITIONS)[name]
                warped_mask = None if mask is None else rotated_mask(torch.from_numpy(mask.astype(np.float32)), angles).numpy()
                record = component_record(
                    component_id=component_id,
                    status_input=str(row["status"]),
                    area_original=int(row["area_original"]),
                    area_512=int(row["area_resized_alone"]),
                    warped=warped_mask,
                    condition_valid=valid[name],
                    common_valid=common,
                    patch_map=patches[name],
                    threshold=patch_cut,
                )
                record.update({"source_id": path.stem, "condition": name})
                component_rows.append(record)
        if index % 25 == 0:
            LOGGER.info("scored %d / %d", index + 1, len(test_items))
    _write_csv(output_dir / "tables" / "images.csv", image_rows)
    _write_csv(output_dir / "tables" / "components.csv", component_rows)
    summary = _summarize(image_rows, component_rows, patch_false, patch_count, thresholds)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rate(flags: list[bool]) -> tuple[int, int]:
    return int(sum(bool(flag) for flag in flags)), len(flags)


def _summarize(image_rows, component_rows, patch_false, patch_count, thresholds) -> dict:
    conditions = [name for name, _angles in CONDITIONS]
    original_images = [row for row in image_rows if row["condition"] == "original" and row["label"] == 1]
    original_by_id = {row["source_id"]: row["detected"] for row in original_images}
    original_components = [row for row in component_rows if row["condition"] == "original"]
    original_component_by_id = {(row["source_id"], row["component_id"]): row["detected"] for row in original_components}
    comparisons = {}
    achieved = {}
    for name in conditions:
        anomalies = [row for row in image_rows if row["condition"] == name and row["label"] == 1]
        normals = [row for row in image_rows if row["condition"] == name and row["label"] == 0]
        components = [row for row in component_rows if row["condition"] == name]
        image_hits, image_n = _rate([row["detected"] for row in anomalies])
        normal_hits, normal_n = _rate([row["detected"] for row in normals])
        component_hits, component_n = _rate([row["detected"] for row in components])
        achieved[name] = {
            "anomaly_image_detection": {"detected": image_hits, "n": image_n},
            "normal_image_fpr": fpr_report(
                np.asarray([row["reproduction_score"] for row in normals], dtype=np.float64),
                thresholds["image"]["threshold"],
                TARGET_IMAGE_FPR,
            ),
            "normal_patch_fpr": {
                "n": patch_count[name],
                "n_false": patch_false[name],
                "achieved_fpr": patch_false[name] / patch_count[name],
                "target_fpr": TARGET_PATCH_FPR,
                "target_match_claimed": False,
            },
            "component_detection": {"detected": component_hits, "n": component_n},
            "input_unevaluable": sum(row["status_input"] == UNEVALUABLE_INPUT_TRANSFORM for row in components),
            "warp_unevaluable": sum(row["status_warp"] == "ワープによる評価不能" for row in components),
            "normal_image_hits_check": {"detected": normal_hits, "n": normal_n},
        }
        if name == "original":
            continue
        other_by_id = {row["source_id"]: row["detected"] for row in anomalies}
        reference = np.asarray([original_by_id[key] for key in original_by_id], dtype=bool)
        other = np.asarray([other_by_id[key] for key in original_by_id], dtype=bool)
        component_reference = np.asarray([original_component_by_id[key] for key in original_component_by_id], dtype=bool)
        other_components = {(row["source_id"], row["component_id"]): row["detected"] for row in components}
        component_other = np.asarray([other_components[key] for key in original_component_by_id], dtype=bool)
        base_hits, base_n = _rate([original_by_id[key] for key in original_by_id])
        image_unpaired = rate_comparison(base_hits, base_n, image_hits, image_n)
        image_unpaired["known_transform_ceiling"] = VERDICT_UNDECIDED
        comparisons[name] = {
            "anomaly_image_unpaired": image_unpaired,
            "anomaly_image_paired": _paired(reference, other),
            "component_paired": _paired(component_reference, component_other),
        }
        base_component_hits, base_component_n = _rate(list(original_component_by_id.values()))
        comparisons[name]["component_unpaired"] = rate_comparison(
            base_component_hits, base_component_n, component_hits, component_n
        )
        comparisons[name]["component_unpaired"]["known_transform_ceiling"] = VERDICT_UNDECIDED
    return {
        "verdict": VERDICT_UNDECIDED,
        "called_practical_improvement": False,
        "called_equivalence": False,
        "called_known_transform_ceiling": False,
        "calibration": thresholds,
        "conditions": achieved,
        "comparisons_against_original": comparisons,
    }
