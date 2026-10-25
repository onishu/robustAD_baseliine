"""Lightweight checks for the reflect-pad control. They do not score a model."""

from __future__ import annotations

import unittest

import torch

from robustad.theme_a.boundary import MARGIN, extend_and_warp, geometry_preflight, pad_then_crop


class BoundaryGeometryTests(unittest.TestCase):
    def test_three_zero_paths_and_single_crop_round_trip(self) -> None:
        report = geometry_preflight(MARGIN)
        self.assertEqual(report["failures"], [])
        self.assertTrue(report["passed"])
        self.assertEqual(report["pad_crop_matches_original_max_abs"], 0.0)
        self.assertGreater(report["pad_zero_warp_max_abs"], 0.0)
        self.assertLess(report["pad_zero_warp_max_abs"], 1.0 / 255.0)
        self.assertTrue(report["marker_nearest_peaks_match"])
        self.assertTrue(report["marker_bilinear_peaks_match"])
        self.assertEqual(report["marker_nearest_max_abs"], 0.0)
        self.assertTrue(report["plus5_moved_up"])
        self.assertEqual(report["plus5_marker_pixels"], 1)
        self.assertTrue(report["round_trip_matches_single_canvas"])
        self.assertTrue(report["intermediate_crop_differs"])
        self.assertFalse(report["width_increased_because_fill_entered_crop"])
        for item in report["fill_probe"].values():
            self.assertEqual(item["invalid_pixels"], 0)
            self.assertFalse(item["intermediate_crop"])

    def test_pad_crop_is_not_the_zero_warp(self) -> None:
        generator = torch.Generator().manual_seed(1)
        image = torch.rand((3, 512, 512), generator=generator)
        self.assertTrue(torch.equal(pad_then_crop(image), image))
        warped, steps = extend_and_warp(image, (0.0,), nearest=False)
        self.assertTrue(steps[0]["resampled"])
        self.assertFalse(torch.equal(warped, image))
        self.assertLess(float((warped - image).abs().max()), 1.0 / 255.0)
