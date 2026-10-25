"""Freeze the 144/36 research split without fitting a model.

The lists follow ``split_train_val`` on filenames sorted by ``list_images``.
``results_a1`` and ``results_p0/diag`` are not inputs to this writer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from robustad.catalog import SUBSETS
from robustad.data import list_images, split_train_val
from robustad.prepare.calibration import fit_calibration_overlap
from robustad.prepare.spec import RESEARCH_SPLIT, RUN_ID

_STEM_INDEX = re.compile(r"^(?P<body>.+)_(?P<index>\d+)$")


def filename_identity(path: Path) -> dict[str, object]:
    """Identity that can be read from a filename.

    The trailing integer is an image index. A shared text prefix is a
    condition label, not evidence that two files are the same physical object.
    """

    matched = _STEM_INDEX.match(path.stem)
    if matched is None:
        return {"body": path.stem, "index": None, "key": path.stem, "recoverable": False}
    index = int(matched.group("index"))
    body = matched.group("body")
    return {"body": body, "index": index, "key": f"{body}#{index}", "recoverable": True}


def summarize_split(paths: list[Path], seed: int = 0, val_ratio: float = 0.2) -> dict[str, object]:
    train_index, calibration_index = split_train_val(len(paths), val_ratio, seed)
    fit_paths = [paths[index] for index in train_index]
    calibration_paths = [paths[index] for index in calibration_index]
    fit_ids = [filename_identity(path) for path in fit_paths]
    calibration_ids = [filename_identity(path) for path in calibration_paths]
    fit_keys = [str(item["key"]) for item in fit_ids]
    calibration_keys = [str(item["key"]) for item in calibration_ids]
    bodies = sorted({str(item["body"]) for item in fit_ids + calibration_ids})
    indexes = [item["index"] for item in fit_ids + calibration_ids]
    duplicate_indexes = sorted({index for index in indexes if index is not None and indexes.count(index) > 1})
    return {
        "seed": seed,
        "val_ratio": val_ratio,
        "n_fit": len(fit_paths),
        "n_calibration": len(calibration_paths),
        "n_overlap_files": len(fit_calibration_overlap(fit_keys, calibration_keys)),
        "overlap_keys": fit_calibration_overlap(fit_keys, calibration_keys),
        "bodies": bodies,
        "duplicate_indexes": duplicate_indexes,
        "object_dependence": (
            "ファイル名から読めるのは条件ラベルと連番だけである。"
            "連番は fit と calibration で重ならない。"
            "同一個体の別撮影かはファイル名からは判定できない。"
        ),
        "fit": fit_keys,
        "calibration": calibration_keys,
    }


def write_research_split(data_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.name == "diag":
        raise RuntimeError("results_p0/diag は変更しません。")
    datasets = {}
    for key in ("pcb", "metal_parts"):
        subset = SUBSETS[key]
        paths = list_images(subset.train_dir(data_root) / "normal")
        relative = [path.relative_to(data_root) for path in paths]
        summary = summarize_split(relative, RESEARCH_SPLIT["seed"], RESEARCH_SPLIT["val_ratio"])
        if summary["n_fit"] != RESEARCH_SPLIT["n_fit"] or summary["n_calibration"] != RESEARCH_SPLIT["n_calibration"]:
            raise RuntimeError(f"144/36 になりません: {key} {summary['n_fit']}/{summary['n_calibration']}")
        if summary["n_overlap_files"] != 0 or summary["duplicate_indexes"]:
            raise RuntimeError(f"分割が重なっています: {key}")
        datasets[key] = summary
    payload = {
        "run_id": RUN_ID,
        "status": "frozen",
        "adopted_for_research_fit": True,
        "applied_to_results_a1": False,
        "applied_to_results_p0_diag": False,
        "rule": "sorted filename, split_train_val, seed 0, val_ratio 0.2",
        "datasets": datasets,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "research_split_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
