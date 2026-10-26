"""Rules for the next Theme A comparison. No model is fit or scored."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np
import torch

from robustad.theme_a.action import (
    AUGMENT_EXECUTED,
    AUGMENT_MAIN_ANGLES,
    AUGMENT_MAIN_ZERO_IS_RESAMPLED,
    AUGMENT_REPLACEMENT_ANGLES,
    AUGMENT_REPLACEMENT_LABEL,
    BASELINE_CORESET,
    BOARD_ROI_STATUS,
    FEATURE_STAGE,
    FUSED_CHANNELS,
    FUSED_GRID,
    LAYER2_CHANNELS,
    LAYER2_STRIDE_PX,
    LAYER3_CHANNELS,
    LAYER3_GRID,
    assert_augment_sources,
    board_patch_summary,
    cell_block,
    classify_board_cells,
    coreset_count,
    embedding_count,
    layer3_sample_on_fused,
    map_crop_pixel_to_original,
    memory_bank_provenance,
    nearest_feature_cell_mask,
    nearest_image_cell_mask,
    no_candidate_record,
    paired_source_delta,
    ratio_matching_coreset,
    select_board_roi_stems,
    warp_label_and_origin,
    within_radius,
)

ROOT = Path(__file__).resolve().parents[1]


class ActionSpecTests(unittest.TestCase):
    def test_layer_grids_match_the_measured_forward(self) -> None:
        self.assertEqual(LAYER2_STRIDE_PX, 8)
        self.assertEqual(512 // LAYER3_GRID, 16)
        self.assertEqual(LAYER2_CHANNELS + LAYER3_CHANNELS, FUSED_CHANNELS)
        self.assertEqual(FUSED_CHANNELS, 1536)
        self.assertEqual(FUSED_GRID, 64)
        self.assertEqual(FEATURE_STAGE, "fused_after_concat")
        self.assertEqual(layer3_sample_on_fused(0, 0), (-0.25, -0.25))
        self.assertEqual(layer3_sample_on_fused(1, 1), (0.25, 0.25))
        block = cell_block(3, 4, LAYER2_STRIDE_PX)
        self.assertEqual(block, (24, 32, 32, 40))
        self.assertNotEqual(512 // LAYER3_GRID, LAYER2_STRIDE_PX)

    def test_feature_grid_rotation_is_not_image_rebin(self) -> None:
        image_cells = nearest_image_cell_mask(10, 40, 5.0)
        feature_cells = nearest_feature_cell_mask(10, 40, 5.0)
        self.assertFalse(torch.equal(image_cells, feature_cells))
        self.assertEqual(int(feature_cells.sum()), 1)

    def test_known_plus5_pixel_maps_back(self) -> None:
        # The saved audit pair (200, 350) -> (192, 345) is (row, col).
        mapped = map_crop_pixel_to_original(345, 192, (5.0,), margin=0)
        self.assertTrue(mapped["inside_original_square"])
        self.assertLess(abs(float(mapped["source_col"]) - 350.0), 1.0)
        self.assertLess(abs(float(mapped["source_row"]) - 200.0), 1.0)

    def test_padded_rotation_keeps_center_and_drops_a_corner(self) -> None:
        center = map_crop_pixel_to_original(255, 255, (5.0,))
        self.assertTrue(center["inside_original_square"])
        corners = [map_crop_pixel_to_original(col, row, (5.0,)) for col, row in ((0, 0), (511, 0), (0, 511), (511, 511))]
        self.assertTrue(any(not item["inside_original_square"] for item in corners))

    def test_coreset_ratio_is_not_a_capacity_match(self) -> None:
        one_view = embedding_count(144, 1)
        replacement = embedding_count(144, len(AUGMENT_REPLACEMENT_ANGLES))
        main = embedding_count(144, len(AUGMENT_MAIN_ANGLES))
        self.assertEqual(coreset_count(one_view, 0.1), BASELINE_CORESET)
        self.assertEqual(BASELINE_CORESET, 58982)
        self.assertNotEqual(coreset_count(replacement, 0.1), BASELINE_CORESET)
        self.assertNotEqual(coreset_count(main, 0.1), BASELINE_CORESET)
        for count in (replacement, main):
            matched = ratio_matching_coreset(count, BASELINE_CORESET)
            self.assertNotEqual(matched, 0.1)
            self.assertEqual(coreset_count(count, matched), BASELINE_CORESET)
        self.assertEqual(AUGMENT_MAIN_ANGLES, (0.0, -5.0, 5.0))
        self.assertFalse(AUGMENT_MAIN_ZERO_IS_RESAMPLED)
        self.assertEqual(AUGMENT_REPLACEMENT_ANGLES, (-5.0, 5.0))
        self.assertEqual(AUGMENT_REPLACEMENT_LABEL, "回転ビューへの置換")
        self.assertFalse(AUGMENT_EXECUTED)

    def test_saved_bank_shape_has_no_location(self) -> None:
        payload = {
            "memory_bank": torch.zeros((58982, 1536)),
            "num_neighbors": 9,
            "n_fit": 144,
            "seed": 0,
        }
        report = memory_bank_provenance(payload)
        self.assertEqual(report["bank_shape"], (58982, 1536))
        self.assertFalse(report["image_id_stored"])
        self.assertFalse(report["grid_position_stored"])
        self.assertFalse(report["positions_recoverable"])
        path = ROOT / "results_a0" / "bank" / "pcb__patchcore__seed0.pt"
        if not path.is_file():
            return
        saved = torch.load(path, map_location="cpu", weights_only=False)
        disk = memory_bank_provenance(saved)
        self.assertEqual(disk["keys"], ["memory_bank", "n_fit", "num_neighbors", "seed"])
        self.assertFalse(disk["positions_recoverable"])
        self.assertEqual(disk["bank_shape"], (58982, 1536))

    def test_radius_and_empty_candidates(self) -> None:
        self.assertTrue(within_radius((0, 0), (0, 1), radius_px=8))
        self.assertFalse(within_radius((0, 0), (0, 1), radius_px=7))
        missing = no_candidate_record(0)
        self.assertIsNone(missing["score"])
        self.assertFalse(missing["used_unconstrained_neighbor"])
        self.assertEqual(no_candidate_record(3)["n_candidates"], 3)

    def test_augment_sources_stay_inside_the_fit_split(self) -> None:
        assert_augment_sources(["a"], ["b"], ["c"])
        with self.assertRaises(RuntimeError):
            assert_augment_sources(["a"], ["a"], ["c"])
        with self.assertRaises(RuntimeError):
            assert_augment_sources(["a"], ["b"], ["a"])

    def test_board_subset_is_filename_order(self) -> None:
        path = ROOT / "results_a0" / "tables" / "images.csv"
        stems = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["condition"] == "original" and row["label"] == "0":
                    stems.append(row["source_id"])
        selected = select_board_roi_stems(list(reversed(stems)))
        self.assertEqual([item["rule_index"] for item in selected], [0, 17, 34, 51, 68, 85, 102, 119])
        self.assertEqual(selected[0]["stem"], "Test_PCB_resized_None_None_Normal_Set2_181")
        self.assertEqual(selected[-1]["stem"], "Test_PCB_resized_None_None_Normal_Set2_300")
        self.assertTrue(all(item["status"] == BOARD_ROI_STATUS for item in selected))
        self.assertEqual(len({item["stem"] for item in selected}), 8)

    def test_partial_and_reflected_cells_are_not_primary_board(self) -> None:
        roi = np.zeros((512, 512), dtype=bool)
        roi[8:16, 8:12] = True
        roi[16:32, 16:32] = True
        content = np.ones((512, 512), dtype=bool)
        content[0:8, :] = False
        valid = np.ones((512, 512), dtype=bool)
        cells = classify_board_cells(roi, content, valid)
        self.assertTrue(cells["board_entire"][2:4, 2:4].all())
        self.assertTrue(bool(cells["board_partial"][1, 1]))
        self.assertFalse(bool(cells["board_entire"][1, 1]))
        self.assertTrue(cells["reflection_or_invalid"][0].all())
        self.assertFalse(cells["board_entire"][0].any())
        summary = board_patch_summary(np.arange(64 * 64, dtype=np.float32).reshape(64, 64), cells["board_entire"], 10.0)
        self.assertGreater(summary["n_cells"], 0)
        empty = board_patch_summary(np.ones((64, 64), dtype=np.float32), np.zeros((64, 64), dtype=bool), 1.0)
        self.assertEqual(empty["n_cells"], 0)
        self.assertTrue(np.isnan(empty["median"]))

    def test_label_warp_does_not_mirror_and_drops_reflected_cells(self) -> None:
        label = torch.zeros((512, 512), dtype=torch.float32)
        label[200:300, 200:300] = 1.0
        moved = warp_label_and_origin(label, (5.0,))
        cells = classify_board_cells(
            moved["roi"].numpy(),
            moved["content_from_original"].numpy(),
            moved["geometrically_valid"].numpy(),
        )
        self.assertTrue((~moved["content_from_original"]).any())
        self.assertFalse(np.logical_and(cells["board_entire"], cells["reflection_or_invalid"]).any())
        self.assertTrue(cells["board_entire"].any())
        self.assertFalse(torch.equal(moved["roi"], label.bool()))
        untouched = warp_label_and_origin(label, ())
        self.assertTrue(torch.equal(untouched["roi"], label.bool()))

    def test_deltas_stay_paired_by_source(self) -> None:
        delta = paired_source_delta({"a": 1.0, "b": 4.0}, {"b": 5.0, "c": 9.0})
        self.assertEqual(delta, {"b": 1.0})


if __name__ == "__main__":
    unittest.main()
