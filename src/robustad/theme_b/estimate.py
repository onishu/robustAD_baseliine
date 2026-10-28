"""Rotation about a known center, estimated from correspondences.

The free affine transform is not estimated. A positive angle is the same
counter-clockwise image rotation as torchvision. The ground-truth angle is
not an argument of the estimator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from robustad.theme_a.warp import REFERENCE_SIZE, rotation_center

LOWE_RATIO = 0.8
MIN_RADIUS_PX = 16.0
INLIER_PX = 3.0
MIN_INLIERS = 8
RANSAC_ITERATIONS = 128
RANSAC_SEED = 0
MAX_CORRECTION_DEGREES = 45.0
SIFT_SETTINGS = {
    "nfeatures": 0,
    "nOctaveLayers": 3,
    "contrastThreshold": 0.04,
    "edgeThreshold": 10,
    "sigma": 1.6,
}


def wrap_degrees(angle: float) -> float:
    """Wrap to (-180, 180]. A full turn lands on -180, not on both signs."""

    value = float(angle)
    if not np.isfinite(value):
        return value
    wrapped = (value + 180.0) % 360.0 - 180.0
    if wrapped == 0.0:
        return 0.0
    return float(wrapped)


def diagnostic_added_angle(theta_condition: float, theta_unrotated: float) -> float:
    """Difference used only to score the estimator. It is not a correction."""

    return wrap_degrees(float(theta_condition) - float(theta_unrotated))


def pair_angle_degrees(query_xy: np.ndarray, reference_xy: np.ndarray, center_xy: np.ndarray) -> float:
    """Angle that carries the reference point onto the query point.

    Image y points down. The returned sign matches a torchvision
    counter-clockwise rotation of the reference content.
    """

    reference = np.asarray(reference_xy, dtype=np.float64) - np.asarray(center_xy, dtype=np.float64)
    query = np.asarray(query_xy, dtype=np.float64) - np.asarray(center_xy, dtype=np.float64)
    dot = float(reference[0] * query[0] + reference[1] * query[1])
    cross = float(reference[1] * query[0] - reference[0] * query[1])
    if dot == 0.0 and cross == 0.0:
        return float("nan")
    return wrap_degrees(float(np.degrees(np.arctan2(cross, dot))))


def rotate_points(points_xy: np.ndarray, degrees: float, center_xy: np.ndarray) -> np.ndarray:
    """Move points as image content under a counter-clockwise rotation."""

    theta = math_radians(degrees)
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    matrix = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    centered = np.asarray(points_xy, dtype=np.float64) - np.asarray(center_xy, dtype=np.float64)
    return centered @ matrix.T + np.asarray(center_xy, dtype=np.float64)


def math_radians(degrees: float) -> float:
    return float(np.deg2rad(degrees))


def reprojection_px(query_xy: np.ndarray, reference_xy: np.ndarray, degrees: float, center_xy: np.ndarray) -> np.ndarray:
    """Distance after rotating the reference point by the candidate angle."""

    predicted = rotate_points(reference_xy, degrees, center_xy)
    delta = predicted - np.asarray(query_xy, dtype=np.float64)
    return np.linalg.norm(delta, axis=1)


def _circular_mean(degrees: np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    return wrap_degrees(float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))))


@dataclass(frozen=True)
class RotationEstimate:
    angle_degrees: float | None
    n_candidates: int
    n_inliers: int
    residual_rms_px: float | None
    failed: bool
    reason: str
    seconds: dict[str, float]

    def correction_degrees(self) -> float | None:
        """Inverse angle to apply, or None when the image stays uncorrected."""

        if self.failed or self.angle_degrees is None or not np.isfinite(self.angle_degrees):
            return None
        if abs(self.angle_degrees) > MAX_CORRECTION_DEGREES:
            return None
        return wrap_degrees(-self.angle_degrees)

    def fallback_reason(self) -> str:
        if self.correction_degrees() is not None:
            return ""
        if self.failed:
            return self.reason
        if self.angle_degrees is not None and abs(self.angle_degrees) > MAX_CORRECTION_DEGREES:
            return "angle_outside_prepared_range"
        return self.reason or "failed"


def estimate_from_correspondences(
    query_xy: np.ndarray,
    reference_xy: np.ndarray,
    center_xy: np.ndarray,
    *,
    n_candidates: int | None = None,
) -> RotationEstimate:
    """Robust one-point rotation. Translation, scale, and shear stay at zero."""

    started = time.perf_counter()
    query = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    reference = np.asarray(reference_xy, dtype=np.float64).reshape(-1, 2)
    if query.shape != reference.shape:
        raise ValueError("対応点の数が合いません。")
    finite = np.isfinite(query).all(axis=1) & np.isfinite(reference).all(axis=1)
    query = query[finite]
    reference = reference[finite]
    center = np.asarray(center_xy, dtype=np.float64)
    candidates = int(query.shape[0] if n_candidates is None else n_candidates)
    radius = np.linalg.norm(query - center, axis=1)
    other = np.linalg.norm(reference - center, axis=1)
    usable = (radius >= MIN_RADIUS_PX) & (other >= MIN_RADIUS_PX)
    hypothesis_query = query[usable]
    hypothesis_reference = reference[usable]
    if hypothesis_query.shape[0] < 1:
        return _failed(candidates, 0, "too_few_correspondences", started)
    generator = np.random.default_rng(RANSAC_SEED)
    best_mask: np.ndarray | None = None
    best_count = -1
    for _ in range(RANSAC_ITERATIONS):
        index = int(generator.integers(0, hypothesis_query.shape[0]))
        theta = pair_angle_degrees(hypothesis_query[index], hypothesis_reference[index], center)
        if not np.isfinite(theta):
            continue
        residual = reprojection_px(hypothesis_query, hypothesis_reference, theta, center)
        mask = residual <= INLIER_PX
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count < 1:
        return _failed(candidates, max(best_count, 0), "too_few_inliers", started)
    theta = _refit(hypothesis_query, hypothesis_reference, best_mask, center)
    if not np.isfinite(theta):
        return _failed(candidates, best_count, "non_finite_angle", started)
    # One circular mean, then residuals on every finite correspondence.
    residual = reprojection_px(query, reference, theta, center)
    mask = residual <= INLIER_PX
    inliers = int(mask.sum())
    rms = float(np.sqrt(np.mean(np.square(residual[mask])))) if inliers else None
    if inliers < MIN_INLIERS:
        return _failed(candidates, inliers, "too_few_inliers", started, angle=float(theta), rms=rms)
    if abs(float(theta)) > MAX_CORRECTION_DEGREES:
        return _failed(candidates, inliers, "angle_outside_prepared_range", started, angle=float(theta), rms=rms)
    return RotationEstimate(
        angle_degrees=float(theta),
        n_candidates=candidates,
        n_inliers=inliers,
        residual_rms_px=rms,
        failed=False,
        reason="",
        seconds={"estimate": time.perf_counter() - started},
    )


def _refit(query: np.ndarray, reference: np.ndarray, mask: np.ndarray, center: np.ndarray) -> float:
    angles = [
        pair_angle_degrees(query[index], reference[index], center)
        for index in np.flatnonzero(mask)
    ]
    finite = [angle for angle in angles if np.isfinite(angle)]
    if not finite:
        return float("nan")
    return _circular_mean(np.asarray(finite, dtype=np.float64))


def _failed(
    candidates: int,
    inliers: int,
    reason: str,
    started: float,
    angle: float | None = None,
    rms: float | None = None,
) -> RotationEstimate:
    return RotationEstimate(
        angle_degrees=angle,
        n_candidates=candidates,
        n_inliers=inliers,
        residual_rms_px=rms,
        failed=True,
        reason=reason,
        seconds={"estimate": time.perf_counter() - started},
    )


def _rgb_array(image: np.ndarray | object) -> np.ndarray:
    if hasattr(image, "detach"):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    return np.ascontiguousarray(array)


def gray_uint8(image: np.ndarray | object) -> np.ndarray:
    import cv2

    array = _rgb_array(image)
    if array.dtype != np.uint8:
        array = np.clip(np.rint(np.asarray(array, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return array
    if array.shape[-1] == 1:
        return array[..., 0]
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)


class ReferenceFeatures:
    """SIFT features of the one fixed reference. Built once per run."""

    def __init__(self, image: np.ndarray | object, stem: str) -> None:
        import cv2

        started = time.perf_counter()
        gray = gray_uint8(image)
        if gray.shape != (REFERENCE_SIZE, REFERENCE_SIZE):
            raise ValueError(f"参照は {REFERENCE_SIZE} 正方です: {gray.shape}")
        self.stem = stem
        self.center_xy = rotation_center(REFERENCE_SIZE, REFERENCE_SIZE)
        self.sift = cv2.SIFT_create(**SIFT_SETTINGS)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        self.keypoints = keypoints or []
        self.descriptors = descriptors
        self.seconds = time.perf_counter() - started

    def estimate(self, image: np.ndarray | object) -> RotationEstimate:
        import cv2

        started = time.perf_counter()
        gray = gray_uint8(image)
        if not np.isfinite(gray.astype(np.float64)).all():
            failed = _failed(0, 0, "non_finite_image", started)
            return _with_reference_time(failed, self.seconds)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if (
            not self.keypoints
            or self.descriptors is None
            or len(self.keypoints) < 2
            or not keypoints
            or descriptors is None
            or len(keypoints) < 2
        ):
            failed = _failed(0, 0, "no_keypoints", started)
            return _with_reference_time(failed, self.seconds)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(self.descriptors, descriptors, k=2)
        query_xy = []
        reference_xy = []
        for neighbors in pairs:
            if len(neighbors) < 2:
                continue
            best, second = neighbors
            if best.distance < LOWE_RATIO * second.distance:
                reference_xy.append(self.keypoints[best.queryIdx].pt)
                query_xy.append(keypoints[best.trainIdx].pt)
        if not query_xy:
            failed = _failed(0, 0, "empty_matches", started)
            return _with_reference_time(failed, self.seconds)
        # descriptors were matched reference -> query, so queryIdx is the reference.
        estimate = estimate_from_correspondences(
            np.asarray(query_xy, dtype=np.float64),
            np.asarray(reference_xy, dtype=np.float64),
            np.asarray(self.center_xy, dtype=np.float64),
            n_candidates=len(query_xy),
        )
        seconds = dict(estimate.seconds)
        seconds["match"] = time.perf_counter() - started
        seconds["reference"] = self.seconds
        return RotationEstimate(
            angle_degrees=estimate.angle_degrees,
            n_candidates=estimate.n_candidates,
            n_inliers=estimate.n_inliers,
            residual_rms_px=estimate.residual_rms_px,
            failed=estimate.failed,
            reason=estimate.reason,
            seconds=seconds,
        )


def _with_reference_time(estimate: RotationEstimate, reference_seconds: float) -> RotationEstimate:
    seconds = dict(estimate.seconds)
    seconds["reference"] = reference_seconds
    return RotationEstimate(
        angle_degrees=estimate.angle_degrees,
        n_candidates=estimate.n_candidates,
        n_inliers=estimate.n_inliers,
        residual_rms_px=estimate.residual_rms_px,
        failed=estimate.failed,
        reason=estimate.reason,
        seconds=seconds,
    )


def evaluation_row(
    *,
    source_id: str,
    component_ids: tuple[int, ...] | list[int],
    condition: str,
    added_degrees: float | None,
    estimate: RotationEstimate,
) -> dict[str, object]:
    """Keep the source image and component ids. Warped copies are not new images."""

    return {
        "source_id": source_id,
        "component_ids": [int(item) for item in component_ids],
        "condition": condition,
        "added_degrees": None if added_degrees is None else float(added_degrees),
        "estimated_degrees": estimate.angle_degrees,
        "correction_degrees": estimate.correction_degrees(),
        "fallback": estimate.correction_degrees() is None,
        "fallback_reason": estimate.fallback_reason(),
        "n_candidates": estimate.n_candidates,
        "n_inliers": estimate.n_inliers,
        "residual_rms_px": estimate.residual_rms_px,
        "diagnostic_only": False,
    }
