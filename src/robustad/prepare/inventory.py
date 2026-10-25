"""Read-only inventory of the released RobustAD folders.

Full-file hashes of every image are not computed here. Metadata files are
hashed. Image identity for the metal train/test name collision is checked
only for the overlapping names.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from robustad.catalog import SUBSETS
from robustad.data import list_images
from robustad.prepare.names import parse_anomaly_stem
from robustad.prepare.regions import iter_original_components, label_components, resize_binary_mask
from robustad.prepare.spec import DIAGNOSTIC_DOMAINS, RUN_ID

EVAL_SIZE = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_split(domain_name: str, label_name: str) -> str:
    if domain_name == "train" and label_name == "normal":
        return "fit_pool"
    if domain_name == "train":
        return "unused_train_anomaly"
    return "test"


def _diagnostic_keys() -> set[tuple[str, str]]:
    return {(item["dataset"], item["domain"]) for item in DIAGNOSTIC_DOMAINS}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mask_values(path: Path) -> tuple[np.ndarray, list[int]]:
    with Image.open(path) as image:
        array = np.array(image.convert("L"))
    values = [int(value) for value in np.unique(array)]
    return array, values


def build_inventory(data_root: Path, output_dir: Path) -> dict:
    image_rows: list[dict] = []
    component_rows: list[dict] = []
    mask_rows: list[dict] = []
    summary_domains = []
    unparsed = []
    nonbinary = []

    for dataset_key, subset in SUBSETS.items():
        folders = [subset.train_dir(data_root)] + [
            subset.domain_dir(data_root, domain) for domain in subset.domains
        ]
        domain_names = ["train"] + [domain.name for domain in subset.domains]
        for folder, domain_name in zip(folders, domain_names, strict=True):
            metadata = folder / "metadata.jsonl"
            normal_paths = list_images(folder / "normal")
            anomaly_paths = list_images(folder / "anomaly") if (folder / "anomaly").is_dir() else []
            mask_dir = folder / "masks"
            has_masks = mask_dir.is_dir()
            sizes: dict[str, int] = {}
            for label_name, paths in (("normal", normal_paths), ("anomaly", anomaly_paths)):
                for path in paths:
                    with Image.open(path) as image:
                        width, height = image.size
                        mode = image.mode
                    sizes[f"{width}x{height}"] = sizes.get(f"{width}x{height}", 0) + 1
                    if dataset_key == "piled_bags" or label_name == "normal":
                        parsed = {"defect": "", "condition": "", "index": -1, "parsed": 0}
                    else:
                        parsed = parse_anomaly_stem(path.stem)
                        if not parsed["parsed"]:
                            unparsed.append(str(path))
                    image_rows.append(
                        {
                            "run_id": RUN_ID,
                            "model": "none",
                            "seed": "",
                            "category": dataset_key,
                            "domain": domain_name,
                            "image_id": path.stem,
                            "split": _image_split(domain_name, label_name),
                            "label": 0 if label_name == "normal" else 1,
                            "defect": parsed["defect"],
                            "condition": parsed["condition"],
                            "score_definition": "not_computed",
                            "smoothing": "not_computed",
                            "region_definition": "not_computed",
                            "width": width,
                            "height": height,
                            "mode": mode,
                            "bytes": path.stat().st_size,
                            "image_path": str(path),
                        }
                    )
            mask_note = "none"
            if has_masks and subset.localization:
                mask_note = "per_anomaly_file"
                diagnose = (dataset_key, domain_name) in _diagnostic_keys()
                for anomaly_path in anomaly_paths:
                    mask_path = _find_existing_mask(mask_dir, anomaly_path.stem)
                    if mask_path is None:
                        mask_rows.append(
                            {
                                "run_id": RUN_ID,
                                "category": dataset_key,
                                "domain": domain_name,
                                "image_id": anomaly_path.stem,
                                "mask_found": 0,
                            }
                        )
                        continue
                    array, values = _mask_values(mask_path)
                    binary = values == [0, 255] or values == [0] or values == [255]
                    if not binary:
                        nonbinary.append(str(mask_path))
                    if diagnose and domain_name != "train":
                        _append_components(
                            component_rows,
                            dataset_key,
                            domain_name,
                            anomaly_path.stem,
                            array,
                        )
                    mask_rows.append(
                        {
                            "run_id": RUN_ID,
                            "category": dataset_key,
                            "domain": domain_name,
                            "image_id": anomaly_path.stem,
                            "mask_found": 1,
                            "mask_height": int(array.shape[0]),
                            "mask_width": int(array.shape[1]),
                            "n_values": len(values),
                            "binary_0_255": int(binary),
                            "positive": int((array > 0).sum()),
                        }
                    )
            summary_domains.append(
                {
                    "dataset": dataset_key,
                    "domain": domain_name,
                    "n_normal": len(normal_paths),
                    "n_anomaly": len(anomaly_paths),
                    "localization": subset.localization,
                    "mask_dir": mask_note,
                    "sizes": sizes,
                    "metadata_sha256": sha256_file(metadata) if metadata.is_file() else "",
                    "diagnostic": (dataset_key, domain_name) in _diagnostic_keys(),
                }
            )

    name_collision = _metal_name_collision(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "image_index.csv", image_rows)
    _write_csv(output_dir / "components_original.csv", _component_dicts(component_rows))
    _write_csv(output_dir / "mask_audit.csv", mask_rows)
    payload = {
        "run_id": RUN_ID,
        "data_root": str(data_root),
        "hash_policy": "metadata.jsonl の sha256。全画像の内容ハッシュは取っていない。名前が衝突した Metal Parts の正常画像だけ両者をハッシュした。",
        "domains": summary_domains,
        "n_unparsed_anomaly_names": len(unparsed),
        "unparsed_anomaly_names": unparsed[:20],
        "n_nonbinary_masks": len(nonbinary),
        "nonbinary_masks": nonbinary[:20],
        "metal_train_test_normal_names": name_collision,
        "workpiece_roi": "metadata.jsonl の列は file_name, label, mask のみ。正常画像にマスクはなく、ワークと背景を分ける ROI は同梱されていない。",
    }
    (output_dir / "inventory.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _find_existing_mask(mask_dir: Path, stem: str) -> Path | None:
    for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = mask_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _append_components(rows: list[dict], dataset: str, domain: str, image_id: str, mask: np.ndarray) -> None:
    records = iter_original_components(mask, EVAL_SIZE, EVAL_SIZE)
    joint = resize_binary_mask(mask, EVAL_SIZE, EVAL_SIZE)
    joint_labels = label_components(joint)
    n_joint = int(joint_labels.max()) if joint_labels.size else 0
    for record in records:
        rows.append(
            {
                "run_id": RUN_ID,
                "model": "none",
                "seed": "",
                "category": dataset,
                "domain": domain,
                "image_id": image_id,
                "split": "test",
                "component_id": record.component_id,
                "area_original": record.area,
                "area_resized_alone": record.area_resized_alone,
                "vanished_on_resize": int(record.vanished_on_resize),
                "n_joint_components_on_512": n_joint,
                "score_definition": "not_computed",
                "smoothing": "not_computed",
                "region_definition": "per_component_nearest_512",
            }
        )


def _component_dicts(rows: list[dict]) -> list[dict]:
    return rows


def _metal_name_collision(data_root: Path) -> dict:
    train = data_root / "MetalParts" / "metal_parts_data_dir_train" / "normal"
    test = data_root / "MetalParts" / "metal_parts_data_dir_test0" / "normal"
    if not train.is_dir() or not test.is_dir():
        return {"status": "missing"}
    train_map = {path.name: path for path in train.iterdir() if path.is_file()}
    test_map = {path.name: path for path in test.iterdir() if path.is_file()}
    shared = sorted(set(train_map) & set(test_map))
    same_hash = 0
    for name in shared:
        if sha256_file(train_map[name]) == sha256_file(test_map[name]):
            same_hash += 1
    return {
        "shared_names": len(shared),
        "identical_sha256": same_hash,
        "train_normals": len(train_map),
        "test_normals": len(test_map),
    }
