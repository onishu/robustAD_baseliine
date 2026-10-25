"""Synthetic forward pass that checks capture against pred_score.

This does not read RobustAD and does not fit the 180-image memory bank.
The numeric output is a plumbing check.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from robustad.prepare.capture import MapCapture, assert_capture_matches_prediction, attach_patchcore_capture


def run_patchcore_smoke(output_dir: Path, image_size: int = 64, seed: int = 0) -> dict:
    if image_size > 128:
        raise ValueError("smoke の入力は 128 以下に制限しています。")
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = PatchcoreModel(
        layers=["layer2", "layer3"],
        backbone="wide_resnet50_2",
        pre_trained=True,
        num_neighbors=9,
    ).eval()
    model.to(device)
    embedding_dim = int(sum(model.feature_extractor.out_dims))
    torch.manual_seed(seed)
    model.memory_bank = torch.randn(24, embedding_dim).to(device)
    capture = MapCapture()
    attach_patchcore_capture(model, capture)
    images = torch.randn(2, 3, image_size, image_size, device=device)
    with torch.no_grad():
        prediction = model(images)
    assert_capture_matches_prediction(capture, prediction.pred_score)
    record = capture.records[-1]
    # Recompute the reweight from the support distances of the worst patch by
    # calling the unbound anomalib method's inputs already stored: the stored
    # reproduction score is the reference. Also check it is not defined as the
    # smoothed-map maximum.
    gap = (record["reproduction_image"].reshape(-1) - record["max_smoothed"].reshape(-1)).abs().max().item()
    payload = {
        "kind": "smoke",
        "research_result": False,
        "image_size": image_size,
        "seed": seed,
        "batch": int(images.shape[0]),
        "embedding_dim": embedding_dim,
        "patch_shape": list(record["patch_scores"].shape),
        "patch_map_shape": list(record["patch_map"].shape),
        "smoothed_shape": list(record["smoothed"].shape),
        "max_abs_gap_reweight_vs_smoothed_max": gap,
        "reproduction_matches_pred_score": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_patchcore.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
