"""Tables for the frozen 3-view bank. Saved scores only. No fit and no new forward."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from robustad.metrics import safe_auroc
from robustad.theme_a.action import zero_only_control_agreement
from robustad.theme_a.boundary import _quantile_row

PAIRS = (
    ("original", "results_a0", "original", "results_a0_augment", "sd_original", ()),
    ("minus_5", "results_a0_boundary", "pad_synthetic_m5", "results_a0_augment", "pad_synthetic_m5", (-5.0,)),
    ("plus_5", "results_a0_boundary", "pad_synthetic_p5", "results_a0_augment", "pad_synthetic_p5", (5.0,)),
)
SIZE_BINS = ((0, 0, "0"), (1, 1, "1"), (2, 8, "2-8"), (9, 32, "9-32"), (33, 128, "33-128"), (129, 10**9, "129以上"))
CELL_BINS = ((0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 10**9, "5以上"))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def defect_type(stem: str) -> str:
    body = stem.removeprefix("Test_PCB_None_None_")
    head, separator, tail = body.rpartition("_")
    if separator and tail.isdigit():
        return head
    return body


def _counts(reference: np.ndarray, other: np.ndarray) -> dict[str, int]:
    reference = np.asarray(reference, dtype=bool)
    other = np.asarray(other, dtype=bool)
    return {
        "n": int(reference.size),
        "both": int(np.sum(reference & other)),
        "reference_only": int(np.sum(reference & ~other)),
        "augment_only": int(np.sum(~reference & other)),
        "neither": int(np.sum(~reference & ~other)),
    }


def _bin_name(value: int, bins: tuple[tuple[int, int, str], ...]) -> str:
    for low, high, name in bins:
        if low <= value <= high:
            return name
    raise ValueError(f"区分に入りません: {value}")


def _image_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["source_id"], row["condition"]): row for row in rows}


def _patch_image_summaries(map_dir: Path, condition: str) -> dict[str, dict[str, float]]:
    """One median and one upper decile per image. Cells are not a sample of images."""

    summaries = {}
    for path in map_dir.glob(f"*__{condition}.npz"):
        with np.load(path) as payload:
            if int(payload["label"]) != 0:
                continue
            grid = np.asarray(payload["patch_map"], dtype=np.float64)
            summaries[str(payload["image_id"])] = {
                "median": float(np.median(grid)),
                "p90": float(np.quantile(grid, 0.9, method="linear")),
            }
    return summaries


def _component_cells(data_root: Path) -> dict[tuple[str, int, str], int]:
    from robustad.theme_a.boundary import extend_and_warp
    from robustad.theme_a.run_synthetic import _components
    from robustad.theme_a.boundary import _test_items
    import torch

    counted: dict[tuple[str, int, str], int] = {}
    for path, label, mask_path in _test_items(data_root):
        if label != 1:
            continue
        rows, alone = _components(mask_path)
        for row in rows:
            component_id = int(row["component_id"])
            mask = alone.get(component_id)
            for _key, _ref_dir, _ref_name, _new_dir, new_name, angles in PAIRS:
                if mask is None:
                    cells = 0
                elif not angles:
                    array = np.asarray(mask, dtype=bool)
                    cells = int(array.reshape(64, 8, 64, 8).any(axis=(1, 3)).sum())
                else:
                    tensor = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
                    warped = extend_and_warp(tensor, angles, nearest=True, pad_mode="constant")[0].numpy() > 0
                    cells = int(warped.reshape(64, 8, 64, 8).any(axis=(1, 3)).sum())
                counted[(path.stem, component_id, new_name)] = cells
                if new_name == "sd_original":
                    counted[(path.stem, component_id, "original")] = cells
    return counted


def build_lock(root: Path, data_root: Path | None = None) -> dict[str, object]:
    augment_dir = root / "results_a0_augment"
    baseline_images = _image_index(_read(root / "results_a0" / "tables" / "images.csv"))
    boundary_images = _image_index(_read(root / "results_a0_boundary" / "tables" / "images.csv"))
    augment_images = _image_index(_read(augment_dir / "tables" / "images.csv"))
    lookups = {"results_a0": baseline_images, "results_a0_boundary": boundary_images, "results_a0_augment": augment_images}
    old_threshold = float(json.loads((root / "results_a0" / "thresholds.json").read_text(encoding="utf-8"))["image"]["threshold"])
    new_threshold = float(json.loads((augment_dir / "thresholds.json").read_text(encoding="utf-8"))["image"]["threshold"])
    summary = json.loads((augment_dir / "summary.json").read_text(encoding="utf-8"))
    decisions = []
    crosses = []
    aurocs = []
    for key, ref_dir, ref_name, new_dir, new_name, _angles in PAIRS:
        ids = sorted(row["source_id"] for row in _read(augment_dir / "tables" / "images.csv") if row["condition"] == new_name)
        labels = {item: augment_images[(item, new_name)]["label"] for item in ids}
        ref_hit = {item: lookups[ref_dir][(item, ref_name)]["detected"] == "True" for item in ids}
        new_hit = {item: augment_images[(item, new_name)]["detected"] == "True" for item in ids}
        for label, title in (("0", "normal"), ("1", "anomaly")):
            chosen = [item for item in ids if labels[item] == label]
            row = _counts(
                np.asarray([ref_hit[item] for item in chosen]),
                np.asarray([new_hit[item] for item in chosen]),
            )
            row.update({"condition": key, "group": title, "official": True})
            decisions.append(row)
        types = sorted({defect_type(item) for item in ids if labels[item] == "1"})
        for name in types:
            chosen = [item for item in ids if labels[item] == "1" and defect_type(item) == name]
            row = _counts(
                np.asarray([ref_hit[item] for item in chosen]),
                np.asarray([new_hit[item] for item in chosen]),
            )
            row.update({"condition": key, "group": f"anomaly:{name}", "official": True})
            decisions.append(row)
        for source_name, directory, condition in (("reference", ref_dir, ref_name), ("augment", new_dir, new_name)):
            scores = {item: float(lookups[directory][(item, condition)]["reproduction_score"]) for item in ids}
            for threshold_name, threshold in (("old", old_threshold), ("new", new_threshold)):
                hit = {item: scores[item] >= threshold for item in ids}
                official = (source_name == "reference" and threshold_name == "old") or (
                    source_name == "augment" and threshold_name == "new"
                )
                for label, title in (("0", "normal"), ("1", "anomaly")):
                    chosen = [item for item in ids if labels[item] == label]
                    crosses.append(
                        {
                            "condition": key,
                            "score_source": source_name,
                            "threshold": threshold_name,
                            "threshold_value": threshold,
                            "group": title,
                            "detected": int(sum(hit[item] for item in chosen)),
                            "n": len(chosen),
                            "official": official,
                            "used_to_choose_threshold": False,
                        }
                    )
            y = np.asarray([int(labels[item]) for item in ids])
            score = np.asarray([scores[item] for item in ids])
            aurocs.append(
                {
                    "condition": key,
                    "score_source": source_name,
                    "image_auroc": safe_auroc(y, score),
                    "n_images": len(ids),
                    "unit": "image",
                }
            )
    component_tables = _components(root)
    cells = _component_cells(data_root) if data_root is not None else {}
    patches = _patches(root)
    return {
        "locked_observation": "同一保存パッチ数の3ビューバンクで、想定角度±5°による正常側の誤検出増加が抑えられた。一方、異常画像検出率も低下した。",
        "not_called": ["実用上の最良", "同等", "欠陥保持の証明", "基板面の過反応改善"],
        "roi": "unconfirmed",
        "official_rule": "各モデル固有の校正しきい値。交差適用は診断であり、しきい値の選び直しには使わない。",
        "old_image_threshold": old_threshold,
        "new_image_threshold": new_threshold,
        "timing": {
            "fit_seconds": summary["fit"]["fit_seconds"],
            "calibration_seconds": summary["calibration_seconds"],
            "score_seconds": summary["score_seconds"],
            "extract_gpu_peak_bytes": summary["fit"]["extract_gpu_peak_bytes"],
            "coreset_gpu_peak_bytes": summary["fit"]["coreset_gpu_peak_bytes"],
            "kcenter_loop": "22分41秒",
        },
        "decisions": decisions,
        "cross_thresholds": crosses,
        "image_auroc": aurocs,
        "components": component_tables,
        "separation": patches,
        "cell_counts_computed": bool(cells),
        "size_bins": _size_bins(root, cells),
        "zero_only_control": zero_only_control_agreement(),
    }


def _components(root: Path) -> list[dict[str, object]]:
    augment = _read(root / "results_a0_augment" / "tables" / "components.csv")
    baseline = _read(root / "results_a0" / "tables" / "components.csv")
    boundary = _read(root / "results_a0_boundary" / "tables" / "components.csv")
    pools = {"results_a0": baseline, "results_a0_boundary": boundary, "results_a0_augment": augment}
    rows = []
    for key, ref_dir, ref_name, _new_dir, new_name, _angles in PAIRS:
        paired: dict[str, list[float]] = {}
        images: dict[str, set[str]] = {}
        for ours in augment:
            if ours["condition"] != new_name:
                continue
            match = next(
                (
                    item
                    for item in pools[ref_dir]
                    if item["condition"] == ref_name
                    and item["source_id"] == ours["source_id"]
                    and item["component_id"] == ours["component_id"]
                ),
                None,
            )
            if match is None:
                continue
            ours_value = float(ours["statistic_common"])
            ref_value = float(match["statistic_common"])
            if not np.isfinite(ours_value) or not np.isfinite(ref_value):
                continue
            kind = defect_type(ours["source_id"])
            paired.setdefault(kind, []).append(ours_value - ref_value)
            images.setdefault(kind, set()).add(ours["source_id"])
        for kind, deltas in sorted(paired.items()):
            summary = _quantile_row(np.asarray(deltas, dtype=np.float64))
            summary.update(
                {
                    "condition": key,
                    "defect_type": kind,
                    "n_images": len(images[kind]),
                    "unit": "component; components in one image are not independent images",
                }
            )
            rows.append(summary)
    return rows


def _patches(root: Path) -> list[dict[str, object]]:
    specs = (
        ("original", root / "results_a0" / "maps", "original", root / "results_a0_augment" / "maps", "sd_original"),
        ("minus_5", root / "results_a0_boundary" / "maps", "pad_synthetic_m5", root / "results_a0_augment" / "maps", "pad_synthetic_m5"),
        ("plus_5", root / "results_a0_boundary" / "maps", "pad_synthetic_p5", root / "results_a0_augment" / "maps", "pad_synthetic_p5"),
    )
    rows = []
    for key, ref_maps, ref_name, new_maps, new_name in specs:
        for source, map_dir, condition in (("reference", ref_maps, ref_name), ("augment", new_maps, new_name)):
            summaries = _patch_image_summaries(map_dir, condition)
            maxima: dict[str, list[float]] = {}
            component_rows = _read(
                (root / "results_a0_augment" / "tables" / "components.csv")
                if source == "augment"
                else (root / "results_a0" / "tables" / "components.csv")
                if key == "original"
                else (root / "results_a0_boundary" / "tables" / "components.csv")
            )
            condition_name = new_name if source == "augment" else ref_name
            for row in component_rows:
                if row["condition"] != condition_name:
                    continue
                value = float(row["statistic_common"])
                if np.isfinite(value):
                    maxima.setdefault(row["source_id"], []).append(value)
            image_max = np.asarray([max(values) for values in maxima.values()], dtype=np.float64)
            normal_median = np.asarray([item["median"] for item in summaries.values()], dtype=np.float64)
            normal_p90 = np.asarray([item["p90"] for item in summaries.values()], dtype=np.float64)
            normal_center = float(np.median(normal_median)) if normal_median.size else float("nan")
            rows.append(
                {
                    "condition": key,
                    "score_source": source,
                    "n_normal_images": int(normal_median.size),
                    "normal_image_patch_median": _quantile_row(normal_median),
                    "normal_image_patch_p90": _quantile_row(normal_p90),
                    "n_anomaly_images_with_component": int(image_max.size),
                    "anomaly_image_component_max": _quantile_row(image_max),
                    "anomaly_images_component_max_below_normal_patch_median": int(np.sum(image_max < normal_center)),
                    "unit": "one summary per image; patch cells are not independent images",
                }
            )
    return rows


def _size_bins(root: Path, cells: dict[tuple[str, int, str], int]) -> list[dict[str, object]]:
    augment = _read(root / "results_a0_augment" / "tables" / "components.csv")
    rows = []
    for _key, _ref_dir, _ref_name, _new_dir, new_name, _angles in PAIRS:
        chosen = [row for row in augment if row["condition"] == new_name]
        for label, bins, value_of in (
            ("area_512", SIZE_BINS, lambda row: int(float(row["area_512"]))),
            ("intersection_cells", CELL_BINS, lambda row: cells.get((row["source_id"], int(row["component_id"]), new_name), -1)),
        ):
            if label == "intersection_cells" and not cells:
                continue
            for low, high, name in bins:
                group = [row for row in chosen if low <= value_of(row) <= high]
                finite = [row for row in group if np.isfinite(float(row["statistic_common"]))]
                rows.append(
                    {
                        "condition": new_name,
                        "axis": label,
                        "bin": name,
                        "n_components": len(group),
                        "n_images": len({row["source_id"] for row in group}),
                        "n_detected": int(sum(row["detected"] == "True" for row in group)),
                        "score_median": float(np.median([float(row["statistic_common"]) for row in finite])) if finite else None,
                        "unit": "component; not an independent image",
                    }
                )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    data_root = Path(r"K:\RobustAD")
    payload = build_lock(root, data_root if data_root.exists() else None)
    target = root / "results_a0_augment" / "tables" / "locked_comparison.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
