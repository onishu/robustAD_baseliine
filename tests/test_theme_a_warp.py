"""Warp checks that do not read RobustAD or fit a model."""

from __future__ import annotations

import unittest

import torch

from robustad.theme_a.plan import rotation_protocol, storage_estimate
from robustad.theme_a.warp import identity_warp, inverse_degrees, restore, round_trip, valid_after_rotation


class WarpTests(unittest.TestCase):
    def test_identity_and_declared_inverse(self) -> None:
        image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        self.assertTrue(torch.equal(identity_warp(image, nearest=True), image))
        self.assertEqual(inverse_degrees(5.0), -5.0)
        restored = restore(image, 0.0, nearest=True)
        self.assertTrue(torch.equal(restored, image))

    def test_round_trip_uses_the_same_angle_twice(self) -> None:
        image = torch.zeros((1, 8, 8), dtype=torch.float32)
        image[:, 3:5, 3:5] = 1
        output = round_trip(image, 0.0, nearest=True)
        self.assertTrue(torch.equal(output, image))

    def test_valid_region_shrinks_only_when_the_angle_moves_corners(self) -> None:
        untouched = valid_after_rotation(16, 16, 0.0)
        turned = valid_after_rotation(16, 16, 5.0)
        self.assertTrue(torch.all(untouched))
        self.assertFalse(torch.all(turned))
        self.assertTrue(bool(turned[8, 8]))

    def test_protocol_fits_once(self) -> None:
        protocol = rotation_protocol()
        self.assertEqual(protocol["fit_images"], 144)
        self.assertFalse(protocol["refit_per_condition"])
        self.assertIn("results_a1", protocol["do_not_modify"])
        self.assertIn("results_p0/diag", protocol["do_not_modify"])
        estimate = storage_estimate()
        self.assertEqual(estimate["coreset_patches"], int(144 * 64 * 64 * 0.1))
        self.assertLess(estimate["total_bytes"], 800_000_000)


if __name__ == "__main__":
    unittest.main()
