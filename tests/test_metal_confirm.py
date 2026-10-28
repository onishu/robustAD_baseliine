"""Lightweight checks for the Metal Parts plan and the saved-result helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from robustad.metal_confirm.spec import (
    CANDIDATES_THREE_VIEW,
    PROJECTION_COMPONENTS,
    PROJECTION_COMPONENTS_ONE_VIEW_JL,
    SAVED_PATCHES,
    THREE_VIEW_RATIO,
    assert_output_allowed,
    assert_shared_bank_counts,
    initial_indexes,
    primary_forwards,
)
from robustad.theme_a.boundary import extend_and_warp, fill_is_outside
from robustad.theme_a.board_roi import load_confirmed_board_masks
from robustad.theme_b.geometry import CORRECTION_MARGIN, SERIES_MARGIN
from robustad.theme_b.label_path import composed_origin, series_then_correction_label
from robustad.theme_b.report_saved import build_report, decision_counts
from robustad.theme_a.warp import REFERENCE_SIZE
from torchvision.transforms.v2 import Resize


class MetalCountTests(unittest.TestCase):
    def test_saved_patch_count_matches_and_ratio_0_1_does_not(self) -> None:
        assert_shared_bank_counts()
        self.assertEqual(SAVED_PATCHES, 58982)
        self.assertEqual(int(CANDIDATES_THREE_VIEW * THREE_VIEW_RATIO), SAVED_PATCHES)
        self.assertNotEqual(int(CANDIDATES_THREE_VIEW * 0.1), SAVED_PATCHES)
        self.assertGreater(PROJECTION_COMPONENTS, PROJECTION_COMPONENTS_ONE_VIEW_JL)
        indexes = initial_indexes()
        self.assertNotEqual(indexes["one_view"], indexes["three_view"])
        self.assertEqual(indexes["one_view"], 143974)
        self.assertEqual(indexes["three_view"], 430694)
        forwards = primary_forwards()
        self.assertEqual(forwards["scoring_forwards"], 72 + 1710 + 342)
        self.assertEqual(forwards["series_margin_px"], 41)
        self.assertEqual(forwards["correction_margin_px"], 106)

    def test_outputs_stay_away_from_finished_runs(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_output_allowed(Path("results_b0"))
        with self.assertRaises(RuntimeError):
            assert_output_allowed(Path("results_a0_control0"))
        with self.assertRaises(RuntimeError):
            build_report(Path("results_b0"), Path("results_b0") / "inside")


class GeometryReuseTests(unittest.TestCase):
    def test_512_margin_still_holds_and_two_step_is_not_one_canvas(self) -> None:
        self.assertTrue(fill_is_outside((10.0,), SERIES_MARGIN))
        self.assertTrue(fill_is_outside((45.0,), CORRECTION_MARGIN))
        self.assertFalse(fill_is_outside((10.0, -10.0), 22))
        label = torch.zeros((REFERENCE_SIZE, REFERENCE_SIZE), dtype=torch.float32)
        label[8:40, 8:40] = 1
        one_canvas = extend_and_warp(label, (10.0, -10.0), nearest=True, margin=SERIES_MARGIN, pad_mode="constant")[0] > 0
        two_step = series_then_correction_label(label, 10.0, -10.0)["roi"]
        self.assertFalse(torch.equal(one_canvas, two_step))
        center = composed_origin(256, 256, 10.0, -10.0)
        self.assertTrue(center["correction_samples_series_image"])
        wide = torch.zeros((4, 8), dtype=torch.float32)
        resized = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)(wide[None])
        self.assertEqual(tuple(resized.shape), (1, REFERENCE_SIZE, REFERENCE_SIZE))


class DecisionTests(unittest.TestCase):
    def test_paired_counts_keep_disagreement(self) -> None:
        counts = decision_counts([True, True, False, False], [True, False, True, False])
        self.assertEqual(counts["both_detected"], 1)
        self.assertEqual(counts["reference_only"], 1)
        self.assertEqual(counts["other_only"], 1)
        self.assertEqual(counts["both_missed"], 1)
        self.assertEqual(counts["same_decision"], 2)

    def test_unconfirmed_board_masks_produce_no_scores(self) -> None:
        with self.assertRaises(RuntimeError):
            load_confirmed_board_masks(Path("roi_pending/pcb_sd_normal8"))


if __name__ == "__main__":
    unittest.main()
