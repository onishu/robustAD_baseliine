"""Estimator smoke on synthetic patterns and two fixed real images.

PatchCore is not loaded. The two real images are the first SD normal file
and the first SD anomaly file in filename order.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from robustad.catalog import SUBSETS
from robustad.data import list_images
from robustad.theme_a.run_synthetic import _load_rgb
from robustad.theme_a.warp import REFERENCE_SIZE, rotate_tensor
from robustad.theme_b.estimate import MAX_CORRECTION_DEGREES, ReferenceFeatures, diagnostic_added_angle
from robustad.theme_b.geometry import geometric_margin, margin_covering_series
from robustad.theme_b.protocol import assert_new_output, scaled_score_seconds, stage_counts
from robustad.theme_b.reference import reference_record, resolve_reference_path
from torchvision.transforms.v2 import Resize

SMOKE_DIR = Path("results_b0_smoke")
DATA_ROOT = Path(r"K:\RobustAD")


def textured_image() -> torch.Tensor:
    """Deterministic pattern with one corner mark and no dependence on labels."""

    yy, xx = np.mgrid[0:REFERENCE_SIZE, 0:REFERENCE_SIZE]
    image = np.zeros((3, REFERENCE_SIZE, REFERENCE_SIZE), dtype=np.float32)
    image[0] = (np.sin(xx / 7.0) + 1.0) * 0.5
    image[1] = (np.sin(yy / 5.0) + 1.0) * 0.5
    image[2] = ((xx * 3 + yy) % 17) / 16.0
    image[:, 30:90, 400:470] = np.array([0.95, 0.15, 0.1], dtype=np.float32)[:, None, None]
    return torch.from_numpy(image)


def _synthetic_case(reference: ReferenceFeatures, degrees: float) -> dict[str, object]:
    query = rotate_tensor(textured_image(), degrees, nearest=False)
    estimate = reference.estimate(query)
    return {
        "added_degrees": degrees,
        "estimated_degrees": estimate.angle_degrees,
        "diagnostic_added_degrees": None
        if estimate.angle_degrees is None
        else diagnostic_added_angle(estimate.angle_degrees, 0.0),
        "correction_degrees": estimate.correction_degrees(),
        "n_candidates": estimate.n_candidates,
        "n_inliers": estimate.n_inliers,
        "residual_rms_px": estimate.residual_rms_px,
        "failed": estimate.failed,
        "reason": estimate.reason,
        "seconds": estimate.seconds,
    }


def _real_cases(data_root: Path) -> dict[str, object]:
    if not data_root.is_dir():
        return {"status": "skipped", "reason": f"データがありません: {data_root}"}
    subset = SUBSETS["pcb"]
    domain = next(item for item in subset.domains if item.name == "SD")
    domain_dir = subset.domain_dir(data_root, domain)
    normal = list_images(domain_dir / "normal")[0]
    anomaly = list_images(domain_dir / "anomaly")[0]
    reference_path = resolve_reference_path(data_root)
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    reference_image = _load_rgb(reference_path, resize)
    features = ReferenceFeatures(reference_image, reference_path.stem)
    rows = []
    for path, label in ((normal, "normal"), (anomaly, "anomaly")):
        original = _load_rgb(path, resize)
        unrotated = features.estimate(original)
        for degrees in (0.0, -10.0, 10.0):
            query = original if degrees == 0.0 else rotate_tensor(original, degrees, nearest=False)
            estimate = unrotated if degrees == 0.0 else features.estimate(query)
            diagnostic = None
            if estimate.angle_degrees is not None and unrotated.angle_degrees is not None:
                diagnostic = diagnostic_added_angle(estimate.angle_degrees, unrotated.angle_degrees)
            rows.append(
                {
                    "source_id": path.stem,
                    "label": label,
                    "added_degrees": degrees,
                    "estimated_degrees": estimate.angle_degrees,
                    "unrotated_estimate_degrees": unrotated.angle_degrees,
                    "diagnostic_added_degrees": diagnostic,
                    "correction_degrees": estimate.correction_degrees(),
                    "n_candidates": estimate.n_candidates,
                    "n_inliers": estimate.n_inliers,
                    "residual_rms_px": estimate.residual_rms_px,
                    "failed": estimate.correction_degrees() is None,
                    "reason": estimate.fallback_reason(),
                    "seconds": estimate.seconds,
                    "diagnostic_unrotated_used_for_correction": False,
                }
            )
    return {
        "status": "ran",
        "reference_id": reference_path.stem,
        "reference_seconds": features.seconds,
        "selection": "SD 正常のファイル名順で最初、SD 異常のファイル名順で最初",
        "rows": rows,
    }


def write_smoke(output_dir: Path = SMOKE_DIR, data_root: Path = DATA_ROOT) -> dict[str, object]:
    assert_new_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = textured_image()
    reference = ReferenceFeatures(pattern, "synthetic_texture")
    warp_started = time.perf_counter()
    rotate_tensor(pattern, 10.0, nearest=False)
    warp_seconds = time.perf_counter() - warp_started
    counts = stage_counts()
    payload = {
        "run_id": "b0_smoke",
        "patchcore_called": False,
        "fit_called": False,
        "full_inference": False,
        "reference_rule": reference_record(),
        "margin_geometric_px": geometric_margin(10.0),
        "margin_series_px": margin_covering_series(),
        "max_correction_degrees": MAX_CORRECTION_DEGREES,
        "one_warp_seconds": warp_seconds,
        "stage_counts": counts,
        "scaled_stage1_score_seconds": scaled_score_seconds(counts["stage1_forwards"]),
        "scaled_stage2_score_seconds": scaled_score_seconds(counts["stage2_extra_forwards"]),
        "synthetic": [_synthetic_case(reference, degrees) for degrees in (0.0, 10.0, -10.0)],
        "real": _real_cases(data_root),
    }
    (output_dir / "smoke.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "reference_manifest.json").write_text(
        json.dumps(payload["reference_rule"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    payload = write_smoke()
    print(json.dumps({"margin_series_px": payload["margin_series_px"], "real": payload["real"]["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
