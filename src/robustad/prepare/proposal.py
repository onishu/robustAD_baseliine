"""Propose a fit/calibration split without fitting a model.

The 144/36 counts are the appendix split already implemented by
``split_train_val``. Writing the lists does not adopt them.
"""

from __future__ import annotations

import json
from pathlib import Path

from robustad.catalog import SUBSETS
from robustad.data import list_images, split_train_val
from robustad.prepare.calibration import fit_calibration_overlap, identifiable_fprs, proportion_halfwidth
from robustad.prepare.spec import RUN_ID


def propose_normal_split(paths: list[Path], seed: int = 0, val_ratio: float = 0.2) -> dict:
    train_index, calibration_index = split_train_val(len(paths), val_ratio, seed)
    fit = [str(paths[index]) for index in train_index]
    calibration = [str(paths[index]) for index in calibration_index]
    overlap = fit_calibration_overlap(fit, calibration)
    return {
        "seed": seed,
        "val_ratio": val_ratio,
        "n_fit": len(fit),
        "n_calibration": len(calibration),
        "overlap": overlap,
        "fit": fit,
        "calibration": calibration,
        "status": "awaiting_researcher_decision",
    }


def write_proposal(data_root: Path, output_dir: Path) -> dict:
    datasets = {}
    for key in ("pcb", "metal_parts"):
        subset = SUBSETS[key]
        paths = list_images(subset.train_dir(data_root) / "normal")
        datasets[key] = propose_normal_split(paths)
    payload = {
        "run_id": RUN_ID,
        "status": "awaiting_researcher_decision",
        "rule": "sorted filename, split_train_val, seed 0, val_ratio 0.2。既存の fit180 結果は変更しない。",
        "datasets": datasets,
        "identifiable_fpr_if_calibration_36": identifiable_fprs(36),
        "halfwidth_examples": {
            "n60_rate0.5": proportion_halfwidth(60, 0.5),
            "n20_rate0.5": proportion_halfwidth(20, 0.5),
            "n10_rate0.5": proportion_halfwidth(10, 0.5),
            "n60_rate0.8": proportion_halfwidth(60, 0.8),
        },
        "planning_note": "halfwidth は比率 0.5 または 0.8 を仮定した正規近似。観測された検出率ではない。",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_proposal.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
