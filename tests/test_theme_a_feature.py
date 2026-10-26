"""Coordinate checks for the fused-feature inverse. They do not score a model."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torchvision.transforms._functional_tensor import _gen_affine_grid

from robustad.theme_a.feature_warp import (
    FEATURE_ALIGN_CORNERS,
    FEATURE_MODE,
    FEATURE_PADDING,
    feature_sample_grid,
    reflection_reference_mask,
    source_pixels,
    warp_fused_features,
)
from robustad.theme_a.warp import sampling_matrix

SIZE = 64


class FeatureWarpTests(unittest.TestCase):
    def test_settings_match_the_image_grid(self) -> None:
        self.assertEqual(FEATURE_MODE, "bilinear")
        self.assertEqual(FEATURE_PADDING, "reflection")
        self.assertFalse(FEATURE_ALIGN_CORNERS)
        matrix = sampling_matrix(5.0, SIZE, SIZE)
        theta = torch.tensor(matrix[:2, :], dtype=torch.float32).reshape(1, 2, 3)
        reference = _gen_affine_grid(theta, w=SIZE, h=SIZE, ow=SIZE, oh=SIZE)
        ours = feature_sample_grid(5.0, SIZE)
        self.assertTrue(torch.allclose(reference, ours, atol=0.0, rtol=0.0))
        self.assertTrue(np.allclose(sampling_matrix(5.0, SIZE, SIZE), sampling_matrix(5.0, 512, 512)))

    def test_zero_warp_matches_the_unwarped_grid(self) -> None:
        generator = torch.Generator().manual_seed(0)
        embedding = torch.randn((2, 1536, SIZE, SIZE), generator=generator)
        warped, reflection = warp_fused_features(embedding, 0.0)
        self.assertEqual(float((warped - embedding).abs().max()), 0.0)
        self.assertFalse(bool(reflection.any()))
        px, py = source_pixels(0.0)
        columns = torch.arange(SIZE, dtype=torch.float32)
        self.assertTrue(torch.equal(px[0], columns))
        self.assertTrue(torch.equal(py[:, 0], columns))

    def test_center_sign_and_inverse(self) -> None:
        px, py = source_pixels(5.0)
        self.assertLess(abs(float(px[31, 31]) - 31.0) + abs(float(py[31, 31]) - 31.0), 0.6)
        self.assertLess(abs(float(px[32, 32]) - 32.0) + abs(float(py[32, 32]) - 32.0), 0.6)
        # Content to the right of center moves up under counter-clockwise +5°.
        right = source_pixels(-5.0)
        self.assertLess(float(right[1][32, 48]), 32.0)
        back_x, back_y = source_pixels(-5.0)
        forward_x, forward_y = source_pixels(5.0)
        # The -5° source of an interior cell is where +5° placed that cell.
        moved_y = float(back_y[32, 48])
        moved_x = float(back_x[32, 48])
        row = int(round(moved_y))
        col = int(round(moved_x))
        self.assertLess(abs(float(forward_y[row, col]) - 32.0), 0.75)
        self.assertLess(abs(float(forward_x[row, col]) - 48.0), 0.75)
        image_centered = np.array([8 * 16.5, 8 * 0.5, 1.0])
        feature_centered = np.array([16.5, 0.5, 1.0])
        matrix = sampling_matrix(5.0, 512, 512)
        image_source = matrix @ image_centered
        feature_source = matrix @ feature_centered
        self.assertTrue(np.allclose(image_source[:2], 8.0 * feature_source[:2], atol=1e-6))

    def test_reflection_cells_are_outside_and_mask_cells_return(self) -> None:
        reflected = reflection_reference_mask(5.0).numpy()
        px, py = source_pixels(5.0)
        outside = (px < -1e-4) | (px > SIZE - 1 + 1e-4) | (py < -1e-4) | (py > SIZE - 1 + 1e-4)
        self.assertTrue(np.array_equal(reflected, outside.numpy()))
        self.assertTrue(reflected.any())
        self.assertFalse(reflected[32, 32])
        original = np.zeros((512, 512), dtype=bool)
        original[160:192, 320:352] = True
        from robustad.theme_a.boundary import extend_and_warp

        rotated = extend_and_warp(torch.from_numpy(original.astype(np.float32)), (5.0,), nearest=True, pad_mode="constant")[0].numpy() > 0
        from robustad.theme_a.action import map_crop_pixel_to_original

        rows, cols = np.nonzero(rotated)
        checked = 0
        for row, col in zip(rows[::40], cols[::40]):
            mapped = map_crop_pixel_to_original(int(col), int(row), (5.0,))
            if not mapped["inside_original_square"]:
                continue
            source_row = int(round(float(mapped["source_row"])))
            source_col = int(round(float(mapped["source_col"])))
            self.assertTrue(original[np.clip(source_row, 0, 511), np.clip(source_col, 0, 511)])
            feature_row = int(source_row // 8)
            feature_col = int(source_col // 8)
            inv_x, inv_y = source_pixels(-5.0)
            self.assertLess(abs(float(inv_y[feature_row, feature_col]) - (row / 8)), 1.5)
            self.assertLess(abs(float(inv_x[feature_row, feature_col]) - (col / 8)), 1.5)
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
