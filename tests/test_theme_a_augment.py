"""Lightweight checks for the 3-view plan. No bank is fit and no image is scored."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from robustad.theme_a.action import (
    AUGMENT_CALIBRATION,
    AUGMENT_DECISION,
    AUGMENT_EXECUTED,
    AUGMENT_FIT_ENABLED,
    AUGMENT_MAIN_ZERO_IS_RESAMPLED,
    AUGMENT_OUTPUT_NAME,
    AUGMENT_QUANTILE,
    AUGMENT_REUSE_OLD_THRESHOLDS,
    AUGMENT_TEST_CONDITIONS,
    AUGMENT_ZERO_PATH,
    BASELINE_CORESET,
    OLD_IMAGE_THRESHOLD,
    OLD_PATCH_THRESHOLD,
    PROVENANCE_FIELDS,
    SPATIAL_CONSTRAINT_IN_AUGMENT,
    assert_integer_coreset,
    augment_storage,
    augment_views,
    candidate_index,
    cell_origin_flags,
    coreset_count,
    embedding_count,
    ratio_matching_coreset,
    reference_cell_center,
    required_free_ram_bytes,
    view_selection_breakdown,
    zero_only_control_agreement,
)
from robustad.theme_a.board_roi import auxiliary_board_metrics, load_confirmed_board_masks
from robustad.theme_a.paired_scores import _sign_counts
from robustad.theme_a.run_augment import refuse_augment_run, write_view_features

ROOT = Path(__file__).resolve().parents[1]


class AugmentPlanTests(unittest.TestCase):
    def test_zero_view_stays_untransformed(self) -> None:
        views = augment_views()
        self.assertFalse(AUGMENT_MAIN_ZERO_IS_RESAMPLED)
        self.assertEqual(views[0]["path"], AUGMENT_ZERO_PATH)
        self.assertFalse(views[0]["reflect_pad"])
        self.assertTrue(views[1]["reflect_pad"])
        self.assertTrue(views[2]["reflect_pad"])
        self.assertEqual([view["angle_degrees"] for view in views], [0.0, -5.0, 5.0])
        self.assertIn("未変換", str(views[0]["reason"]))

    def test_saved_patch_count_is_exact(self) -> None:
        storage = augment_storage()
        n_candidates = embedding_count(144, 3)
        self.assertEqual(n_candidates, 1_769_472)
        self.assertEqual(storage["n_candidates"], n_candidates)
        ratio = ratio_matching_coreset(n_candidates, BASELINE_CORESET)
        assert_integer_coreset(n_candidates, ratio, BASELINE_CORESET)
        self.assertEqual(coreset_count(n_candidates, ratio), 58982)
        self.assertEqual(storage["coreset_patches"], 58982)
        self.assertEqual(storage["projection_components"], 355)
        self.assertEqual(storage["candidate_bytes"], 10_871_635_968)
        self.assertEqual(storage["projected_bytes"], 2_512_650_240)
        self.assertEqual(storage["bank_bytes"], 362_385_408)
        self.assertEqual(storage["candidate_device"], "cpu")
        self.assertEqual(required_free_ram_bytes(), 17_679_253_504)
        self.assertFalse(storage["quota_enforced"])
        self.assertFalse(storage["fit_enabled"])
        with self.assertRaises(AssertionError):
            assert_integer_coreset(n_candidates, 0.1, BASELINE_CORESET)

    def test_view_counts_are_not_quotas(self) -> None:
        indices = [candidate_index(0, 0, 0, 0), candidate_index(1, 0, 2, 3)]
        report = view_selection_breakdown(indices)
        self.assertEqual(report["counts"], [2, 0, 0])
        self.assertFalse(report["quota_enforced"])
        self.assertEqual(candidate_index(0, 1, 0, 0), 64 * 64)

    def test_zero_cell_is_not_reflected_and_corner_of_plus5_is(self) -> None:
        center = reference_cell_center(0, 0, ())
        self.assertEqual(center["reference_x"], 4.0)
        self.assertEqual(center["reference_y"], 4.0)
        self.assertTrue(center["inside_original_square"])
        origin = cell_origin_flags(32, 32, ())
        self.assertFalse(origin["from_reflect_pad"])
        self.assertTrue(origin["geometrically_valid"])
        corner = cell_origin_flags(0, 0, (5.0,))
        middle = cell_origin_flags(32, 32, (5.0,))
        self.assertTrue(corner["from_reflect_pad"])
        self.assertFalse(middle["from_reflect_pad"])
        mapped = reference_cell_center(32, 32, ())
        self.assertEqual(mapped["reference_x"], 32 * 8 + 4)

    def test_thresholds_and_conditions_stay_narrow(self) -> None:
        self.assertFalse(AUGMENT_REUSE_OLD_THRESHOLDS)
        self.assertEqual(OLD_IMAGE_THRESHOLD, 36.28824043273926)
        self.assertEqual(OLD_PATCH_THRESHOLD, 24.215826511383057)
        self.assertEqual(AUGMENT_CALIBRATION, "untransformed_36")
        self.assertEqual(AUGMENT_QUANTILE, "np.quantile(scores, 0.95, method='linear')")
        self.assertEqual(AUGMENT_DECISION, "score >= threshold")
        self.assertEqual(
            AUGMENT_TEST_CONDITIONS,
            (("sd_original", ()), ("pad_warp_0", (0.0,)), ("pad_synthetic_m5", (-5.0,)), ("pad_synthetic_p5", (5.0,))),
        )
        self.assertFalse(AUGMENT_EXECUTED)
        self.assertFalse(AUGMENT_FIT_ENABLED)
        self.assertFalse(SPATIAL_CONSTRAINT_IN_AUGMENT)
        self.assertIn("fit_image_id", PROVENANCE_FIELDS)
        self.assertIn("coreset_index", PROVENANCE_FIELDS)
        self.assertIn("from_reflect_pad", PROVENANCE_FIELDS)

    def test_run_guard_opens_only_for_this_approved_run(self) -> None:
        with self.assertRaises(RuntimeError):
            refuse_augment_run(Path("results_a0"), confirmed=True, capacity_ok=True, manifest_saved=True)
        with self.assertRaises(RuntimeError):
            refuse_augment_run(Path(AUGMENT_OUTPUT_NAME), confirmed=False, capacity_ok=True, manifest_saved=True)
        with self.assertRaises(RuntimeError):
            refuse_augment_run(Path(AUGMENT_OUTPUT_NAME), confirmed=True)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / AUGMENT_OUTPUT_NAME
            output.mkdir()
            refuse_augment_run(output, confirmed=True, capacity_ok=True, manifest_saved=True)
            (output / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                refuse_augment_run(output, confirmed=True, capacity_ok=True, manifest_saved=True)

    def test_unconfirmed_roi_produces_no_score(self) -> None:
        with self.assertRaises(RuntimeError):
            load_confirmed_board_masks(ROOT / "roi_pending" / "pcb_sd_normal8")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "masks").mkdir()
            stem = "Test_PCB_resized_None_None_Normal_Set2_181"
            image = np.zeros((512, 512), dtype=np.uint8)
            image[16:32, 16:32] = 255
            Image.fromarray(image).save(root / "masks" / f"{stem}.png")
            manifest = {"status": "unconfirmed", "masks_created": False, "stems": [{"stem": stem, "status": "unconfirmed"}]}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_confirmed_board_masks(root)
            manifest["status"] = "confirmed"
            manifest["masks_created"] = True
            manifest["stems"][0]["status"] = "confirmed"
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            masks = load_confirmed_board_masks(root)
            scores = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
            summary = auxiliary_board_metrics(scores, masks[stem], threshold=10.0, frame="sd_original")
            self.assertGreater(summary["n_cells"], 0)
            self.assertTrue(np.isfinite(summary["median"]))
            with self.assertRaises(ValueError):
                auxiliary_board_metrics(scores, masks[stem], threshold=10.0, frame="feature_m5")
            reflection = np.zeros((64, 64), dtype=bool)
            reflection[2, 2] = True
            excluded = auxiliary_board_metrics(
                scores,
                masks[stem],
                threshold=10.0,
                frame="feature_m5",
                feature_reflection=reflection,
            )
            self.assertLess(excluded["n_cells"], summary["n_cells"])

    def test_zero_only_control_is_not_the_old_bank_and_is_not_fit(self) -> None:
        report = zero_only_control_agreement()
        self.assertFalse(report["fit_executed"])
        self.assertEqual(report["n_candidates"], 589824)
        self.assertTrue(report["matches"]["saved_patch_count"])
        self.assertTrue(report["matches"]["projection_components_match_baseline_328"])
        self.assertEqual(report["matches"]["projection_components_jl"], 328)
        self.assertEqual(report["matches"]["projection_components_forced"], 355)
        self.assertIn("shuffle", report["mismatches"]["image_order"])
        self.assertIn("seed", report["mismatches"]["seed_state"])
        self.assertGreater(report["estimate"]["candidate_bytes"], report["estimate"]["bank_bytes"])

    def test_candidate_write_does_not_change_dtype_or_leave_the_slice_empty(self) -> None:
        storage = torch.full((4096, 1536), float("nan"))
        embedding = torch.zeros((1, 1536, 64, 64), dtype=torch.float32)
        embedding[0, 0, 1, 2] = 7
        write_view_features(storage, [0], 0, embedding)
        self.assertEqual(float(storage[1 * 64 + 2, 0]), 7.0)
        self.assertTrue(torch.isfinite(storage).all())
        self.assertEqual(storage.device.type, "cpu")
        with self.assertRaises(RuntimeError):
            write_view_features(storage, [0], 0, embedding.double())

    def test_sign_counts_keep_image_and_component_apart(self) -> None:
        counts = _sign_counts(np.asarray([-1.0, 1.0, -0.5]), np.asarray([-0.2, 0.4, 0.3]))
        self.assertEqual(counts["both_lower"], 1)
        self.assertEqual(counts["both_higher"], 1)
        self.assertEqual(counts["image_lower_component_higher"], 1)
