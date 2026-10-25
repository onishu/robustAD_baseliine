"""Geometry and calibration checks. They do not fit a model or read Rotation."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from robustad.prepare.calibration import empirical_threshold, realized_fpr
from robustad.prepare.regions import UNEVALUABLE_INPUT_TRANSFORM
from robustad.theme_a.components import WARP_UNEVALUABLE, component_record
from robustad.theme_a.plan import rotation_protocol, storage_estimate
from robustad.theme_a.warp import (
    apply_angles,
    identity_warp,
    inverse_matrix,
    rotated_mask,
    sampling_matrix,
    valid_after_angles,
    valid_after_rotation,
)


class WarpTests(unittest.TestCase):
    def test_positive_angle_is_counterclockwise_about_the_center(self) -> None:
        image = torch.zeros((1, 32, 32), dtype=torch.float32)
        image[0, 16, 24] = 1
        turned, steps = apply_angles(image, (90.0,), nearest=True)
        row, col = [int(value) for value in torch.nonzero(turned[0] > 0)[0]]
        self.assertLess(row, 16)
        self.assertLess(abs(col - 16), abs(row - 16) + 8)
        sampling = np.array(steps[0]["output_to_source"], dtype=np.float64).reshape(3, 3)
        self.assertTrue(np.allclose(sampling @ np.array(steps[0]["source_to_output"]).reshape(3, 3), np.eye(3), atol=1e-6))
        self.assertTrue(np.allclose(sampling, sampling_matrix(90.0, 32, 32)))

    def test_zero_warp_and_inverse_round_trip(self) -> None:
        image = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
        self.assertTrue(torch.equal(identity_warp(image, nearest=True), image))
        untouched, steps = apply_angles(image, (), nearest=True)
        self.assertTrue(torch.equal(untouched, image))
        self.assertFalse(steps[0]["resampled"])
        there_and_back, pair = apply_angles(image, (-5.0, 5.0), nearest=False)
        self.assertEqual(len(pair), 2)
        self.assertEqual(pair[0]["degrees"], -5.0)
        self.assertEqual(pair[1]["degrees"], 5.0)
        self.assertEqual(there_and_back.shape, image.shape)
        product = sampling_matrix(5.0, 8, 8) @ sampling_matrix(-5.0, 8, 8)
        self.assertTrue(np.allclose(product, np.eye(3), atol=1e-6))
        self.assertTrue(np.allclose(inverse_matrix(sampling_matrix(5.0, 8, 8)) @ sampling_matrix(5.0, 8, 8), np.eye(3)))

    def test_mask_uses_nearest_and_valid_region_keeps_the_center(self) -> None:
        mask = torch.zeros((16, 16))
        mask[7:9, 7:9] = 1
        warped = rotated_mask(mask, (-5.0,))
        self.assertEqual(warped.dtype, torch.bool)
        self.assertTrue(bool(warped[8, 8]))
        original = valid_after_angles(16, 16, ())
        zero = valid_after_rotation(16, 16, 0.0)
        turned = valid_after_rotation(16, 16, 5.0)
        self.assertTrue(torch.all(original))
        self.assertTrue(torch.all(zero))
        self.assertFalse(torch.all(turned))
        self.assertTrue(bool(turned[8, 8]))

    def test_resize_loss_and_warp_loss_both_remain(self) -> None:
        full = np.ones((8, 8), dtype=bool)
        patch = np.zeros((2, 2), dtype=np.float32)
        patch[0, 0] = 4
        vanished = component_record(
            component_id=1,
            status_input=UNEVALUABLE_INPUT_TRANSFORM,
            area_original=1,
            area_512=0,
            warped=None,
            condition_valid=full,
            common_valid=full,
            patch_map=patch,
            threshold=1,
        )
        warped_away = component_record(
            component_id=2,
            status_input="evaluable",
            area_512=4,
            area_original=4,
            warped=np.zeros((8, 8), dtype=bool),
            condition_valid=full,
            common_valid=full,
            patch_map=patch,
            threshold=1,
        )
        kept = component_record(
            component_id=3,
            status_input="evaluable",
            area_512=16,
            area_original=16,
            warped=np.ones((8, 8), dtype=bool),
            condition_valid=full,
            common_valid=full,
            patch_map=np.full((2, 2), 2, dtype=np.float32),
            threshold=1,
        )
        self.assertTrue(vanished["in_denominator"] and warped_away["in_denominator"] and kept["in_denominator"])
        self.assertFalse(vanished["detected"])
        self.assertEqual(warped_away["status_warp"], WARP_UNEVALUABLE)
        self.assertFalse(warped_away["detected"])
        self.assertTrue(kept["detected"])
        self.assertEqual(kept["statistic_common"], 2)

    def test_threshold_is_the_upper_quantile(self) -> None:
        scores = np.arange(10, dtype=np.float64)
        threshold = empirical_threshold(scores, 0.05)
        self.assertEqual(threshold, float(np.quantile(scores, 0.95, method="linear")))
        self.assertEqual(realized_fpr(scores, threshold)["n_false"], 1)
        self.assertEqual(realized_fpr(np.array([1.0, 1.0, 0.0]), 1.0)["n_false"], 2)

    def test_protocol_uses_sd_six_ways_and_not_the_old_count(self) -> None:
        protocol = rotation_protocol()
        self.assertEqual(protocol["source_count"], 195)
        self.assertEqual(len(protocol["conditions"]), 6)
        self.assertFalse(protocol["real_rotation_folder_used"])
        self.assertFalse(protocol["refit_per_condition"])
        estimate = storage_estimate()
        self.assertEqual(estimate["scored_condition_images"], 1170)
        self.assertEqual(estimate["total_bytes"], 383_389_696)
        self.assertNotEqual(estimate["scored_condition_images"], 791)
