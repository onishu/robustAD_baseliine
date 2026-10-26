"""Paired continuous scores for the frozen feature-inverse run.

Reads the saved image and component tables. It does not load a model and does
not read patch grids.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from robustad.theme_a.boundary import _quantile_row

PAIRS = (
    ("feature_m5", "pad_synthetic_m5", "rotated"),
    ("feature_m5", "pad_restore_m5", "image_round_trip"),
    ("feature_p5", "pad_synthetic_p5", "rotated"),
    ("feature_p5", "pad_restore_p5", "image_round_trip"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sign_counts(image_delta: np.ndarray, component_delta: np.ndarray) -> dict[str, int]:
    image = np.asarray(image_delta, dtype=np.float64)
    component = np.asarray(component_delta, dtype=np.float64)
    if image.shape != component.shape:
        raise ValueError(f"画像差と成分差の長さが違います: {image.shape} / {component.shape}")
    return {
        "n": int(image.size),
        "both_lower": int(np.sum((image < 0.0) & (component < 0.0))),
        "both_higher": int(np.sum((image > 0.0) & (component > 0.0))),
        "image_lower_component_higher": int(np.sum((image < 0.0) & (component > 0.0))),
        "image_higher_component_lower": int(np.sum((image > 0.0) & (component < 0.0))),
        "either_unchanged": int(np.sum((image == 0.0) | (component == 0.0))),
    }


def aggregate_paired_continuous(feature_dir: Path, boundary_dir: Path) -> dict[str, object]:
    """Feature minus reference, paired by source id and by component id."""

    feature_images = _read_csv(feature_dir / "tables" / "images.csv")
    boundary_images = _read_csv(boundary_dir / "tables" / "images.csv")
    feature_components = _read_csv(feature_dir / "tables" / "components.csv")
    boundary_components = _read_csv(boundary_dir / "tables" / "components.csv")
    image_index = {(row["source_id"], row["condition"]): row for row in feature_images + boundary_images}
    component_index = {
        (row["source_id"], row["component_id"], row["condition"]): row
        for row in feature_components + boundary_components
    }
    comparisons = []
    image_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for condition, reference, kind in PAIRS:
        anomaly_ids = sorted(
            row["source_id"]
            for row in feature_images
            if row["condition"] == condition and row["label"] == "1"
        )
        image_delta = []
        component_max_delta = []
        missing_component = 0
        for source_id in anomaly_ids:
            ours = image_index[(source_id, condition)]
            other = image_index[(source_id, reference)]
            delta = float(ours["reproduction_score"]) - float(other["reproduction_score"])
            ours_max = _component_max(feature_components, source_id, condition)
            other_max = _component_max(boundary_components, source_id, reference)
            if ours_max is None or other_max is None:
                missing_component += 1
                continue
            component_delta = ours_max - other_max
            image_delta.append(delta)
            component_max_delta.append(component_delta)
            image_rows.append(
                {
                    "source_id": source_id,
                    "condition": condition,
                    "reference_condition": reference,
                    "comparison": kind,
                    "image_score_delta": delta,
                    "component_max_delta": component_delta,
                    "image_minus_component_max": delta - component_delta,
                }
            )
        per_component = []
        keys = {
            (row["source_id"], row["component_id"])
            for row in feature_components
            if row["condition"] == condition
        }
        for source_id, component_id in sorted(keys):
            ours_row = component_index.get((source_id, component_id, condition))
            other_row = component_index.get((source_id, component_id, reference))
            if ours_row is None or other_row is None:
                continue
            ours_value = float(ours_row["statistic_common"])
            other_value = float(other_row["statistic_common"])
            if not np.isfinite(ours_value) or not np.isfinite(other_value):
                continue
            delta = ours_value - other_value
            per_component.append(delta)
            component_rows.append(
                {
                    "source_id": source_id,
                    "component_id": int(component_id),
                    "condition": condition,
                    "reference_condition": reference,
                    "comparison": kind,
                    "component_score_delta": delta,
                }
            )
        image_array = np.asarray(image_delta, dtype=np.float64)
        component_array = np.asarray(component_max_delta, dtype=np.float64)
        comparisons.append(
            {
                "condition": condition,
                "reference_condition": reference,
                "comparison": kind,
                "unit_image": "one anomaly source image",
                "unit_component_max": "maximum finite component score of that image",
                "unit_component": "one original component id",
                "mask_frame": "feature inverse uses the unrotated 512 mask; the reference uses that condition's warped mask",
                "image_score_delta": _quantile_row(image_array),
                "component_max_delta": _quantile_row(component_array),
                "image_minus_component_max": _quantile_row(image_array - component_array),
                "sign_counts": _sign_counts(image_array, component_array),
                "anomaly_images_without_finite_component": missing_component,
                "per_component_score_delta": _quantile_row(np.asarray(per_component, dtype=np.float64)),
            }
        )
    return {
        "inference": False,
        "verdict": "未判定",
        "board_surface_improvement": False,
        "component_detection_is_not_retention": True,
        "delta": "feature minus reference",
        "comparisons": comparisons,
        "image_component_rows": image_rows,
        "component_rows": component_rows,
    }


def _component_max(rows: list[dict[str, str]], source_id: str, condition: str) -> float | None:
    values = []
    for row in rows:
        if row["source_id"] != source_id or row["condition"] != condition:
            continue
        value = float(row["statistic_common"])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(max(values))


def write_paired_continuous(feature_dir: Path, boundary_dir: Path) -> Path:
    """Write derived tables beside the frozen feature result. Do not replace summary.json."""

    target = feature_dir / "tables" / "paired_continuous.json"
    if target.is_file():
        raise RuntimeError(f"対応差の集計は上書きしません: {target}")
    payload = aggregate_paired_continuous(feature_dir, boundary_dir)
    image_rows = payload.pop("image_component_rows")
    component_rows = payload.pop("component_rows")
    _write_rows(feature_dir / "tables" / "paired_image_component.csv", image_rows)
    _write_rows(feature_dir / "tables" / "paired_components.csv", component_rows)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if path.is_file():
        raise RuntimeError(f"対応差の集計は上書きしません: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
