"""Fit the approved diagnostic runs and store maps, banks, and rows.

Seed 0, input 512, and all 180 normal images match the reproduction.
The 144/36 split is not used. results_a1 is only read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from robustad.benchmark import BATCH_SIZE, build_model, make_engine, predict
from robustad.catalog import SUBSETS
from robustad.data import build_domain_module, build_train_module, list_images
from robustad.metrics import best_detection, safe_auroc
from robustad.prepare.capture import (
    MapCapture,
    attach_padim_capture,
    attach_patchcore_capture,
)
from robustad.prepare.diagnose_rows import blur_sigma8, score_row_regions, storage_estimate
from robustad.prepare.manifest import build_manifest
from robustad.prepare.names import parse_anomaly_stem
from robustad.prepare.regions import UNEVALUABLE_INPUT_TRANSFORM, component_correspondence

LOGGER = logging.getLogger("robustad.diag")

DIAG_RUNS = (
    {"dataset": "metal_parts", "model": "patchcore", "domains": ("SD", "Scale")},
    {"dataset": "pcb", "model": "patchcore", "domains": ("SD", "Rotation", "Position", "Shadows")},
    {"dataset": "pcb", "model": "padim", "domains": ("Lighting",)},
)
RUN_ID = "p0-diag-seed0"
ABORT_IMAGE_AUROC_GAP = 0.05


def _fingerprint(paths: list[Path]) -> dict[str, object]:
    lines = [f"{path.name}\t{path.stat().st_size}" for path in paths]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {"n": len(paths), "name_size_sha256": digest}


def _data_manifest(data_root: Path) -> dict:
    wanted = {
        "metal_parts": ("SD", "Scale"),
        "pcb": ("SD", "Rotation", "Position", "Shadows", "Lighting"),
    }
    payload = {}
    for dataset, domains in wanted.items():
        subset = SUBSETS[dataset]
        train = list_images(subset.train_dir(data_root) / "normal")
        entry = {"train_normal": _fingerprint(train), "domains": {}}
        for domain in subset.domains:
            if domain.name not in domains:
                continue
            folder = subset.domain_dir(data_root, domain)
            entry["domains"][domain.name] = {
                "normal": _fingerprint(list_images(folder / "normal")),
                "anomaly": _fingerprint(list_images(folder / "anomaly")),
            }
        payload[dataset] = entry
    return payload


def write_pre_manifest(repo_root: Path, data_root: Path, output_dir: Path) -> dict:
    usage = shutil.disk_usage(output_dir if output_dir.exists() else output_dir.parent)
    estimate = storage_estimate()
    payload = {
        "run_id": RUN_ID,
        "status": "awaiting_researcher_decision",
        "purpose": "準備Pの診断用再推論。独立calibrationによる主評価ではない。",
        "fit": {"seed": 0, "image_size": 512, "val_ratio": 0.0, "n_normal": 180, "num_workers": 4},
        "split_144_36_applied": False,
        "roi_added": False,
        "undecided": ["目標FPR", "改善幅", "同等性幅", "連結領域の主統計量", "144/36の採否", "ROIの要否"],
        "runs": [{"dataset": item["dataset"], "model": item["model"], "domains": list(item["domains"])} for item in DIAG_RUNS],
        "storage_estimate": estimate,
        "disk_free_bytes": usage.free,
        "data": _data_manifest(data_root),
        "environment": build_manifest(repo_root),
        "reference_results": "results_a1",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "infer_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if usage.free < int(estimate["total_bytes"]) + 400_000_000:
        raise RuntimeError(f"空き容量が推定保存量より 400 MB 以上ありません: free={usage.free} need={estimate['total_bytes']}")
    return payload


def _squeeze_map(value: torch.Tensor) -> torch.Tensor:
    array = value.detach().cpu().float()
    if array.ndim == 4:
        array = array[:, 0]
    return array


def _load_defect(mask_path: str, height: int, width: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    if not mask_path:
        return np.zeros((height, width), dtype=bool), []
    with Image.open(mask_path) as image:
        original = np.array(image.convert("L"))
    from robustad.benchmark import _resize_mask

    resized = _resize_mask(mask_path, height, width).numpy() > 0
    return resized, component_correspondence(original, height, width)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_bank(model_name: str, lightning_model, destination: Path, train_counts: dict) -> dict:
    torch_model = lightning_model.model
    destination.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "patchcore":
        bank = torch_model.memory_bank.detach().cpu()
        torch.save(
            {"run_id": RUN_ID, "kind": "patchcore_memory_bank", "memory_bank": bank, "num_neighbors": int(torch_model.num_neighbors), "train": train_counts},
            destination,
        )
        return {"path": str(destination), "shape": list(bank.shape), "bytes": int(bank.numel() * bank.element_size())}
    mean = torch_model.gaussian.mean.detach().cpu()
    covariance = torch_model.gaussian.inv_covariance.detach().cpu()
    index = torch_model.idx.detach().cpu()
    torch.save(
        {"run_id": RUN_ID, "kind": "padim_gaussian", "mean": mean, "inv_covariance": covariance, "idx": index, "train": train_counts},
        destination,
    )
    nbytes = int(mean.numel() * mean.element_size() + covariance.numel() * covariance.element_size())
    return {"path": str(destination), "mean_shape": list(mean.shape), "inv_covariance_shape": list(covariance.shape), "bytes": nbytes}


def _reference(reference_dir: Path, dataset: str, model_name: str, domain: str) -> dict:
    path = reference_dir / "runs" / f"{dataset}__{model_name}__seed0.json"
    return json.loads(path.read_text(encoding="utf-8"))["domains"][domain]


def _rows_for_batch(dataset, model_name, domain, batch, record, map_dir: Path, start: int):
    reproduction = record["reproduction_image"].float().reshape(-1)
    pred_score = batch.pred_score.detach().cpu().float().reshape(-1)
    if reproduction.shape != pred_score.shape or not torch.allclose(reproduction, pred_score, atol=1e-4):
        raise RuntimeError(f"{dataset}/{model_name}/{domain} の再現スコアが pred_score と一致しません。")
    patch = record["patch_map"].float()
    if patch.ndim == 4:
        patch = patch[:, 0]
    upsampled = _squeeze_map(record["upsampled"])
    smoothed = _squeeze_map(record["smoothed"])
    sigma8 = _squeeze_map(blur_sigma8(upsampled[:, None].contiguous()))
    paths = [str(item) for item in list(getattr(batch, "image_path", []) or [])]
    mask_paths = list(getattr(batch, "mask_path", []) or [])
    labels = batch.gt_label.detach().cpu().reshape(-1).to(torch.int32)
    image_rows = []
    component_rows = []
    pixel_maps = []
    pixel_masks = []
    for index in range(int(pred_score.shape[0])):
        image_id = Path(paths[index]).stem
        label = int(labels[index])
        mask_path = str(mask_paths[index]) if index < len(mask_paths) else ""
        if mask_path in ("", "None"):
            mask_path = ""
        defect, components = _load_defect(mask_path if label == 1 else "", int(smoothed.shape[-2]), int(smoothed.shape[-1]))
        if label == 0 and components:
            raise RuntimeError(f"正常画像に欠陥成分があります: {image_id}")
        parsed = parse_anomaly_stem(image_id) if label == 1 else {"defect": "", "condition": ""}
        patch_map = patch[index].numpy()
        maps = {"sigma0": upsampled[index].numpy(), "sigma4": smoothed[index].numpy(), "sigma8": sigma8[index].numpy()}
        max_patch = float(patch_map.max())
        score = float(pred_score[index])
        file_index = start + index
        np.savez(
            map_dir / f"{file_index:05d}.npz",
            run_id=np.array(RUN_ID),
            image_id=np.array(image_id),
            label=np.int32(label),
            patch_map=patch_map.astype(np.float16),
            smoothed=maps["sigma4"].astype(np.float16),
            reproduction_score=np.float32(score),
            max_patch=np.float32(max_patch),
        )
        row = {
            "run_id": RUN_ID,
            "model": model_name,
            "seed": 0,
            "category": dataset,
            "domain": domain,
            "image_id": image_id,
            "split": "test",
            "label": label,
            "defect": parsed["defect"],
            "condition": parsed["condition"],
            "score_definition": "reproduction_image",
            "smoothing": "sigma0_and_sigma4_saved_sigma8_from_upsampled",
            "region_definition": "defect_nondefect_unpartitioned_border",
            "roi": "none",
            "reproduction_score": score,
            "diag_max_patch": max_patch,
            "diag_max_upsampled": float(maps["sigma0"].max()),
            "diag_max_smoothed": float(maps["sigma4"].max()),
            "diag_max_sigma8": float(maps["sigma8"].max()),
            "reweight": float(score / max_patch) if max_patch != 0 else float("nan"),
            "map_file": f"{file_index:05d}.npz",
            "n_components": len(components),
            "n_unevaluable_input_transform": sum(item["status"] == UNEVALUABLE_INPUT_TRANSFORM for item in components),
        }
        row.update(score_row_regions(maps, defect, patch_map))
        image_rows.append(row)
        for item in components:
            component_rows.append(
                {
                    "run_id": RUN_ID,
                    "model": model_name,
                    "seed": 0,
                    "category": dataset,
                    "domain": domain,
                    "image_id": image_id,
                    "split": "test",
                    "component_id": item["component_id"],
                    "area_original": item["area_original"],
                    "area_resized_alone": item["area_resized_alone"],
                    "joint_component_ids": item["joint_component_ids"],
                    "status": item["status"],
                    "score_definition": "not_a_threshold",
                    "smoothing": "not_applied_to_component_status",
                    "region_definition": "original_component_to_nearest_512",
                }
            )
        pixel_maps.append(maps["sigma4"].reshape(-1))
        pixel_masks.append(defect.reshape(-1).astype(np.uint8))
    return image_rows, component_rows, pixel_maps, pixel_masks


def _score_metrics(labels, scores, pixel_labels, pixel_scores) -> dict:
    image_detection = best_detection(labels, scores)
    payload = {
        "image_auroc": safe_auroc(labels, scores),
        "image_f1_test_max": image_detection["f1"],
        "n_images": int(labels.size),
        "pixel_auroc": safe_auroc(pixel_labels, pixel_scores) if pixel_labels.size else float("nan"),
        "pixel_f1_test_max": best_detection(pixel_labels, pixel_scores)["f1"] if pixel_labels.size else float("nan"),
    }
    return payload


def run_diagnostic(data_root: Path, output_dir: Path, reference_dir: Path, repo_root: Path) -> dict:
    from lightning.pytorch import seed_everything

    diag_dir = output_dir / "diag"
    write_pre_manifest(repo_root, data_root, diag_dir)
    summaries = []
    checked = False
    for spec in DIAG_RUNS:
        dataset = spec["dataset"]
        model_name = spec["model"]
        seed_everything(0, workers=True)
        subset = SUBSETS[dataset]
        train_module, train_counts = build_train_module(
            subset, data_root, seed=0, val_ratio=0.0, batch_size=BATCH_SIZE[model_name], num_workers=4
        )
        if train_counts["n_train"] != 180 or train_counts["n_val"] != 0:
            raise RuntimeError(f"fit 枚数が再現条件ではありません: {train_counts}")
        LOGGER.info("fit %s / %s", dataset, model_name)
        lightning_model = build_model(model_name, 512, "timm")
        engine = make_engine(diag_dir)
        engine.fit(model=lightning_model, datamodule=train_module)
        bank_info = _save_bank(model_name, lightning_model, diag_dir / "banks" / f"{dataset}__{model_name}__seed0.pt", train_counts)
        capture = MapCapture()
        if model_name == "patchcore":
            attach_patchcore_capture(lightning_model.model, capture)
        else:
            attach_padim_capture(lightning_model.model, capture)
        domains = {domain.name: domain for domain in subset.domains}
        for domain_name in spec["domains"]:
            capture.clear()
            module, counts = build_domain_module(
                subset, data_root, domains[domain_name], batch_size=BATCH_SIZE[model_name], num_workers=4
            )
            outputs = predict(engine, lightning_model, module)
            if len(outputs) != len(capture.records):
                raise RuntimeError(f"バッチ数 {len(outputs)} と捕捉数 {len(capture.records)} が一致しません。")
            map_dir = diag_dir / "maps" / f"{dataset}__{model_name}__seed0" / domain_name
            map_dir.mkdir(parents=True, exist_ok=True)
            image_rows = []
            component_rows = []
            pixel_maps = []
            pixel_masks = []
            cursor = 0
            for batch, record in zip(outputs, capture.records, strict=True):
                images, components, maps, masks = _rows_for_batch(
                    dataset, model_name, domain_name, batch, record, map_dir, cursor
                )
                cursor += len(images)
                image_rows.extend(images)
                component_rows.extend(components)
                pixel_maps.extend(maps)
                pixel_masks.extend(masks)
            labels = np.asarray([row["label"] for row in image_rows], dtype=np.int32)
            scores = np.asarray([row["reproduction_score"] for row in image_rows], dtype=np.float64)
            pixel_labels = np.concatenate(pixel_masks)
            pixel_scores = np.concatenate(pixel_maps)
            measured = _score_metrics(labels, scores, pixel_labels, pixel_scores)
            for column, key in (
                ("diag_max_patch", "max_patch"),
                ("diag_max_upsampled", "max_upsampled"),
                ("diag_max_smoothed", "max_smoothed"),
                ("diag_max_sigma8", "max_sigma8"),
            ):
                alt = np.asarray([row[column] for row in image_rows], dtype=np.float64)
                measured[f"image_auroc_{key}"] = safe_auroc(labels, alt)
            reference = _reference(reference_dir, dataset, model_name, domain_name)
            gap = float(measured["image_auroc"] - reference["image_auroc"])
            unevaluable = [row for row in component_rows if row["status"] == UNEVALUABLE_INPUT_TRANSFORM]
            payload = {
                "run_id": RUN_ID,
                "dataset": dataset,
                "model": model_name,
                "domain": domain_name,
                "counts": counts,
                "n_rows": len(image_rows),
                "n_components": len(component_rows),
                "n_unevaluable_input_transform": len(unevaluable),
                "unevaluable_kept_in_component_count": True,
                "unevaluable_components": [
                    {"image_id": row["image_id"], "component_id": row["component_id"], "area_original": row["area_original"]}
                    for row in unevaluable
                ],
                "measured": measured,
                "reference_image_auroc": reference["image_auroc"],
                "reference_pixel_auroc": reference.get("pixel_auroc"),
                "image_auroc_gap": gap,
                "pixel_auroc_gap": float(measured["pixel_auroc"] - reference["pixel_auroc"]) if "pixel_auroc" in reference else None,
                "bank": bank_info,
                "auroc_not_used_for_specification": True,
            }
            if not checked and abs(gap) > ABORT_IMAGE_AUROC_GAP:
                raise RuntimeError(f"最初のドメインの Image AUROC 差が {gap:+.4f} で、再現から外れています。")
            checked = True
            table_dir = diag_dir / "tables" / f"{dataset}__{model_name}__seed0"
            _write_csv(table_dir / f"{domain_name}_images.csv", image_rows)
            if component_rows:
                _write_csv(table_dir / f"{domain_name}_components.csv", component_rows)
            (table_dir / f"{domain_name}_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            summaries.append(payload)
            LOGGER.info(
                "%s %s %s image AUROC %.4f reference %.4f gap %+.4f components %d unevaluable %d",
                dataset, model_name, domain_name, measured["image_auroc"], reference["image_auroc"], gap, len(component_rows), len(unevaluable),
            )
            del outputs, pixel_maps, pixel_masks
        del lightning_model, engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (diag_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"domains": summaries}
