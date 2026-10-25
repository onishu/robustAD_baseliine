"""Capture intermediate maps from one forward pass.

The wrappers observe anomalib's own map generator and score function. They do
not reimplement nearest-neighbor search. A captured file is a diagnostic cache,
not a reproduction result, until its image score is checked against ``pred_score``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class MapCapture:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.records.clear()


def attach_patchcore_capture(model, capture: MapCapture):
    original_map = model.anomaly_map_generator.forward
    original_score = model.compute_anomaly_score

    def wrapped_score(patch_scores, locations, embedding):
        score = original_score(patch_scores, locations, embedding)
        capture.records.append(
            {
                "max_patch": patch_scores.detach().amax(dim=1).cpu(),
                "reproduction_image": score.detach().cpu(),
                "patch_scores": patch_scores.detach().cpu(),
            }
        )
        return score

    def wrapped_map(patch_scores, image_size=None):
        if image_size is None:
            upsampled = patch_scores.detach()
        else:
            upsampled = F.interpolate(patch_scores, size=(image_size[0], image_size[1])).detach()
        smoothed = original_map(patch_scores, image_size)
        if capture.records:
            capture.records[-1]["patch_map"] = patch_scores.detach().cpu()
            capture.records[-1]["upsampled"] = upsampled.cpu()
            capture.records[-1]["smoothed"] = smoothed.detach().cpu()
            capture.records[-1]["max_upsampled"] = upsampled.amax(dim=(-2, -1)).cpu()
            capture.records[-1]["max_smoothed"] = smoothed.detach().amax(dim=(-2, -1)).cpu()
        return smoothed

    model.compute_anomaly_score = wrapped_score
    model.anomaly_map_generator.forward = wrapped_map
    return original_score, original_map


def attach_padim_capture(model, capture: MapCapture):
    generator = model.anomaly_map_generator
    original = generator.compute_anomaly_map

    def wrapped(embedding, mean, inv_covariance, image_size=None):
        distance = generator.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), inv_covariance.to(embedding.device)],
        )
        upsampled = distance if not image_size else generator.up_sample(distance, image_size)
        smoothed = generator.smooth_anomaly_map(upsampled)
        capture.records.append(
            {
                "max_patch": distance.detach().amax(dim=(-2, -1)).cpu(),
                "max_upsampled": upsampled.detach().amax(dim=(-2, -1)).cpu(),
                "max_smoothed": smoothed.detach().amax(dim=(-2, -1)).cpu(),
                "reproduction_image": smoothed.detach().amax(dim=(-2, -1)).cpu(),
                "patch_scores": distance.detach().cpu(),
                "patch_map": distance.detach().cpu(),
                "upsampled": upsampled.detach().cpu(),
                "smoothed": smoothed.detach().cpu(),
            }
        )
        return smoothed

    generator.compute_anomaly_map = wrapped
    return original


def assert_capture_matches_prediction(capture: MapCapture, pred_score: torch.Tensor, atol: float = 1e-5) -> None:
    if not capture.records:
        raise RuntimeError("中間スコアが取れていません。")
    stored = capture.records[-1]["reproduction_image"].float().reshape(-1)
    returned = pred_score.detach().cpu().float().reshape(-1)
    if stored.shape != returned.shape or not torch.allclose(stored, returned, atol=atol, equal_nan=True):
        raise RuntimeError("捕捉した再現互換スコアが pred_score と一致しません。")
