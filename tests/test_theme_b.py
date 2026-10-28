"""Theme B geometry and rotation estimator. No PatchCore fit or full scoring."""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from robustad.theme_a.boundary import fill_is_outside
from robustad.theme_a.warp import rotate_tensor
from robustad.theme_b.estimate import (
    MAX_CORRECTION_DEGREES,
    ReferenceFeatures,
    diagnostic_added_angle,
    estimate_from_correspondences,
    evaluation_row,
    pair_angle_degrees,
    reprojection_px,
    rotate_points,
    wrap_degrees,
)
from robustad.theme_b.geometry import geometric_margin, margin_covering_series
from robustad.theme_b.protocol import assert_new_output
from robustad.theme_b.reference import reference_stem
from robustad.theme_b.smoke import textured_image


def _moved(point: np.ndarray, degrees: float, center: np.ndarray) -> np.ndarray:
    return rotate_points(np.asarray(point, dtype=np.float64).reshape(1, 2), degrees, center)[0]


class GeometryTests(unittest.TestCase):
    def test_margin_comes_from_the_ten_degree_bound_not_from_22(self) -> None:
        bound = geometric_margin(10.0)
        self.assertGreater(bound, 22)
        self.assertFalse(fill_is_outside((10.0,), 22))
        self.assertFalse(fill_is_outside((10.0, -10.0), 22))
        margin = margin_covering_series()
        self.assertGreaterEqual(margin, bound)
        self.assertTrue(fill_is_outside((10.0,), margin))
        self.assertTrue(fill_is_outside((-10.0, 10.0), margin))
        self.assertTrue(fill_is_outside((0.0,), margin))


class CorrespondenceTests(unittest.TestCase):
    def test_known_center_pairs_recover_the_sign_and_the_wrap(self) -> None:
        center = np.array([256.0, 256.0])
        source = np.array([[320.0, 220.0], [180.0, 300.0], [400.0, 400.0], [140.0, 180.0]])
        for degrees in (10.0, -10.0, 90.0, -90.0, 170.0, -170.0):
            query = np.vstack([_moved(point, degrees, center) for point in source])
            for index, point in enumerate(source):
                recovered = pair_angle_degrees(query[index], point, center)
                self.assertAlmostEqual(recovered, wrap_degrees(degrees), places=6)
            self.assertTrue(np.allclose(rotate_points(source, degrees, center), query, atol=1e-6))
        marker = torch.zeros((1, 512, 512))
        marker[0, 220, 320] = 1
        found = torch.nonzero(rotate_tensor(marker, 10.0, nearest=True)[0] > 0)[0]
        predicted = rotate_points(np.array([[320.0, 220.0]]), 10.0, center)[0]
        self.assertLess(abs(predicted[0] - float(found[1])), 0.6)
        self.assertLess(abs(predicted[1] - float(found[0])), 0.6)
        self.assertAlmostEqual(wrap_degrees(190.0), -170.0, places=6)
        self.assertAlmostEqual(wrap_degrees(-190.0), 170.0, places=6)
        self.assertAlmostEqual(wrap_degrees(180.0), wrap_degrees(-180.0), places=6)
        self.assertAlmostEqual(diagnostic_added_angle(8.0, -2.0), 10.0, places=6)

    def test_outliers_do_not_set_the_angle_and_short_sets_fail(self) -> None:
        center = np.array([256.0, 256.0])
        source = np.array([[300.0, 240.0], [220.0, 180.0], [360.0, 300.0], [180.0, 340.0],
                           [280.0, 400.0], [140.0, 260.0], [400.0, 200.0], [210.0, 150.0]])
        query = rotate_points(source, 12.0, center)
        outliers_query = np.array([[100.0, 100.0], [420.0, 80.0], [90.0, 400.0], [450.0, 450.0]])
        outliers_reference = np.array([[400.0, 400.0], [80.0, 420.0], [300.0, 90.0], [120.0, 120.0]])
        estimate = estimate_from_correspondences(
            np.vstack([query, outliers_query]),
            np.vstack([source, outliers_reference]),
            center,
        )
        self.assertFalse(estimate.failed)
        self.assertAlmostEqual(estimate.angle_degrees or 0.0, 12.0, delta=0.05)
        self.assertGreaterEqual(estimate.n_inliers, 8)
        self.assertIsNotNone(estimate.residual_rms_px)
        self.assertLess(float(estimate.residual_rms_px), 0.05)
        near = np.array([[260.0, 256.0]])
        near_query = rotate_points(near, 12.0, center)
        with_near = estimate_from_correspondences(
            np.vstack([query, near_query]),
            np.vstack([source, near]),
            center,
        )
        self.assertFalse(with_near.failed)
        self.assertGreater(with_near.n_inliers, estimate.n_inliers)
        empty = estimate_from_correspondences(np.zeros((0, 2)), np.zeros((0, 2)), center)
        self.assertTrue(empty.failed)
        self.assertEqual(empty.reason, "too_few_correspondences")
        nan_query = query.copy()
        nan_query[0, 0] = np.nan
        kept = estimate_from_correspondences(nan_query, source, center)
        self.assertTrue(kept.failed)
        self.assertIsNone(kept.correction_degrees())
        residual = reprojection_px(query, source, 12.0, center)
        self.assertTrue(np.all(residual < 1e-6))

    def test_ground_truth_is_not_an_estimator_argument_and_ids_stay(self) -> None:
        signature = inspect.signature(estimate_from_correspondences)
        self.assertNotIn("degrees", signature.parameters)
        self.assertNotIn("added_degrees", signature.parameters)
        center = np.array([256.0, 256.0])
        source = rotate_points(np.array([[300.0, 200.0]]), 0.0, center)
        row = evaluation_row(
            source_id="same-image",
            component_ids=(3, 9),
            condition="series_p10",
            added_degrees=10.0,
            estimate=estimate_from_correspondences(source, source, center),
        )
        self.assertEqual(row["source_id"], "same-image")
        self.assertEqual(row["component_ids"], [3, 9])
        wide = estimate_from_correspondences(
            rotate_points(np.array([[400.0, 256.0], [256.0, 400.0], [180.0, 180.0], [420.0, 180.0],
                                    [180.0, 420.0], [300.0, 140.0], [140.0, 300.0], [360.0, 360.0]]), 50.0, center),
            np.array([[400.0, 256.0], [256.0, 400.0], [180.0, 180.0], [420.0, 180.0],
                      [180.0, 420.0], [300.0, 140.0], [140.0, 300.0], [360.0, 360.0]]),
            center,
        )
        self.assertIsNotNone(wide.angle_degrees)
        self.assertGreater(abs(wide.angle_degrees or 0.0), MAX_CORRECTION_DEGREES)
        self.assertIsNone(wide.correction_degrees())
        self.assertEqual(wide.fallback_reason(), "angle_outside_prepared_range")

    def test_protected_results_and_reference_rule(self) -> None:
        for name in ("results_a0", "results_a0_augment", "results_a0_control0", "results_a1"):
            with self.assertRaises(RuntimeError):
                assert_new_output(Path(name))
        with self.assertRaises(RuntimeError):
            assert_new_output(Path("results_p0") / "diag")
        self.assertEqual(reference_stem(), "Train_PCB_resized_None_None_Normal_Set1_1")


class SiftTests(unittest.TestCase):
    def test_textured_rotation_sign_and_zero(self) -> None:
        image = textured_image()
        features = ReferenceFeatures(image, "synthetic_texture")
        zero = features.estimate(image)
        self.assertFalse(zero.failed)
        self.assertIsNotNone(zero.angle_degrees)
        self.assertAlmostEqual(float(zero.angle_degrees), 0.0, delta=1.0)
        for degrees in (10.0, -10.0):
            estimate = features.estimate(rotate_tensor(image, degrees, nearest=False))
            self.assertFalse(estimate.failed, estimate.reason)
            self.assertIsNotNone(estimate.angle_degrees)
            self.assertAlmostEqual(float(estimate.angle_degrees), degrees, delta=1.5)
            self.assertIsNotNone(estimate.correction_degrees())
            self.assertAlmostEqual(float(estimate.correction_degrees()), -degrees, delta=1.5)


class SmokeArtifactTests(unittest.TestCase):
    def test_smoke_file_does_not_call_patchcore(self) -> None:
        path = Path("results_b0_smoke") / "smoke.json"
        if not path.is_file():
            self.skipTest("smoke はまだ書いていない")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(payload["patchcore_called"])
        self.assertFalse(payload["fit_called"])
        self.assertFalse(payload["full_inference"])


if __name__ == "__main__":
    unittest.main()
