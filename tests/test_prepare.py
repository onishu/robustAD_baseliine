"""Synthetic checks for the preparation-phase score and region rules."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from robustad.benchmark import _resize_mask
from robustad.data import split_train_val
from robustad.prepare.cache import CacheMismatch, read_header, write_header
from robustad.prepare.calibration import (
    clopper_pearson,
    detection_counts,
    empirical_threshold,
    fit_calibration_overlap,
    realized_fpr,
)
from robustad.prepare.names import parse_anomaly_stem
from robustad.prepare.regions import (
    attribution_masks,
    border_band,
    component_areas,
    display_labels,
    iter_original_components,
    label_components,
    max_pixel_membership,
    resize_binary_mask,
    restrict_labels,
)
from robustad.prepare.scores import max_reduction, nearest_upsample, patchcore_reweight, quantile_reduction
from robustad.prepare.spec import SMOOTH_KERNEL, SMOOTH_RADIUS


class ScoreTests(unittest.TestCase):
    def test_reweight_matches_anomalib(self) -> None:
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        torch.manual_seed(0)
        bank = torch.randn(12, 5)
        embedding = torch.randn(2 * 4, 5)
        distances = PatchcoreModel.euclidean_dist(embedding, bank)
        patch_scores, locations = distances.reshape(2, 4, -1).min(dim=2)
        # Build the support distances the way compute_anomaly_score does.
        dummy = type("Dummy", (), {})()
        dummy.num_neighbors = 4
        dummy.memory_bank = bank
        dummy.euclidean_dist = PatchcoreModel.euclidean_dist
        dummy.nearest_neighbors = lambda embedding, n_neighbors: PatchcoreModel.nearest_neighbors(
            dummy, embedding, n_neighbors
        )
        expected = PatchcoreModel.compute_anomaly_score(dummy, patch_scores, locations, embedding)
        max_index = torch.argmax(patch_scores, dim=1)
        score = patch_scores[torch.arange(2), max_index]
        nn_index = locations[torch.arange(2), max_index]
        _, support = dummy.nearest_neighbors(bank[nn_index], n_neighbors=4)
        support_distances = PatchcoreModel.euclidean_dist(
            embedding.reshape(2, 4, -1)[torch.arange(2), max_index].unsqueeze(1),
            bank[support],
        ).squeeze(1)
        got = patchcore_reweight(score, support_distances)
        self.assertTrue(torch.allclose(got, expected))

    def test_reweight_is_not_the_patch_maximum(self) -> None:
        score = torch.tensor([3.0, 5.0])
        distances = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.2, 0.4]])
        got = patchcore_reweight(score, distances)
        self.assertFalse(torch.allclose(got, score))

    def test_nearest_upsample_matches_generator(self) -> None:
        from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

        torch.manual_seed(0)
        patch_scores = torch.randn(1, 1, 4, 4)
        generator = AnomalyMapGenerator(sigma=4)
        self.assertEqual(generator.blur.kernel.shape[-1], SMOOTH_KERNEL)
        self.assertEqual(SMOOTH_RADIUS, 16)
        upsampled = nearest_upsample(patch_scores, (8, 8))
        # Generator blur changes values, so compare against the unsmoothed nearest map.
        manual = torch.nn.functional.interpolate(patch_scores, size=(8, 8))
        self.assertTrue(torch.equal(upsampled, manual))

    def test_reductions_handle_empty_and_nonfinite(self) -> None:
        self.assertTrue(np.isnan(max_reduction(np.array([]))))
        self.assertTrue(np.isnan(max_reduction(np.array([1.0, np.nan]))))
        self.assertTrue(np.isnan(quantile_reduction(np.array([np.inf]), 0.9)))
        self.assertEqual(quantile_reduction(np.array([1.0, 2.0, 3.0, 4.0]), 1.0), 4.0)


class RegionTests(unittest.TestCase):
    def test_resize_matches_benchmark_helper(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 2:5] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(mask, mode="L").save(path)
            expected = _resize_mask(str(path), 4, 4).numpy()
        got = resize_binary_mask(mask, 4, 4)
        self.assertTrue(np.array_equal(got, expected))

    def test_components_small_large_and_multiple(self) -> None:
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[0, 0] = 1
        mask[5:9, 5:9] = 1
        mask[2, 8] = 1
        mask[2, 9] = 1
        labels = label_components(mask)
        areas = component_areas(labels)
        self.assertEqual(len(areas), 3)
        self.assertIn(1, areas.values())
        self.assertIn(16, areas.values())
        self.assertIn(2, areas.values())

    def test_hole_is_not_a_component(self) -> None:
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 1
        mask[3, 3] = 0
        labels = label_components(mask)
        self.assertEqual(len(component_areas(labels)), 1)
        self.assertEqual(int((labels == 0).sum()), 7 * 7 - 24)

    def test_normal_mask_has_no_component(self) -> None:
        self.assertEqual(component_areas(label_components(np.zeros((5, 5), dtype=np.uint8))), {})

    def test_overlap_and_display_priority(self) -> None:
        defect = np.zeros((6, 6), dtype=bool)
        defect[0:2, 0:2] = True
        border = border_band(6, 6, 1)
        masks = attribution_masks(defect, border)
        self.assertTrue(masks["defect_and_border"].any())
        self.assertEqual(int(masks["defect"].sum() + masks["nondefect_unpartitioned"].sum()), 36)
        labels = display_labels(defect, border)
        self.assertTrue(np.all(labels[defect] == 1))

    def test_restriction_keeps_id_when_split(self) -> None:
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[0, 0] = 4
        labels[0, 4] = 4
        valid = np.ones((5, 5), dtype=bool)
        valid[0, 2] = False
        restricted, stats = restrict_labels(labels, valid)
        self.assertEqual(set(np.unique(restricted)) - {0}, {4})
        self.assertEqual(stats[4]["area_inside"], 2)
        self.assertFalse(stats[4]["outside_some"])
        valid[0, 0] = False
        _, partial = restrict_labels(labels, valid)
        self.assertTrue(partial[4]["outside_some"])

    def test_vanished_small_component(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0, 0] = 1
        records = iter_original_components(mask, 2, 2)
        self.assertEqual(len(records), 1)
        self.assertGreaterEqual(records[0].area_resized_alone, 0)

    def test_max_membership_reports_a_plateau(self) -> None:
        score = np.zeros((4, 4))
        score[0, 0] = 3
        score[3, 3] = 3
        defect = np.zeros((4, 4), dtype=bool)
        defect[0, 0] = True
        border = np.zeros((4, 4), dtype=bool)
        border[3, 3] = True
        shares = max_pixel_membership(score, attribution_masks(defect, border))
        self.assertEqual(shares["defect"], 0.5)
        self.assertEqual(shares["border"], 0.5)


class CalibrationTests(unittest.TestCase):
    def test_threshold_uses_only_normals(self) -> None:
        normals = np.array([0.0, 1.0, 2.0, 3.0])
        threshold = empirical_threshold(normals, target_fpr=0.25)
        self.assertEqual(threshold, np.quantile(normals, 0.75))
        caught = realized_fpr(normals, threshold)
        self.assertEqual(caught["n"], 4)
        hits = detection_counts(np.array([10.0, 0.0]), threshold)
        self.assertEqual(hits["n_detected"], 1)

    def test_overlap_and_interval(self) -> None:
        self.assertEqual(fit_calibration_overlap(["a", "b"], ["b", "c"]), ["b"])
        self.assertEqual(fit_calibration_overlap(["a"], ["b"]), [])
        lower, upper = clopper_pearson(0, 10)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)
        self.assertTrue(all(np.isnan(clopper_pearson(0, 0))))

    def test_split_is_deterministic_and_disjoint(self) -> None:
        first, second = split_train_val(180, 0.2, 0)
        again, _ = split_train_val(180, 0.2, 0)
        self.assertEqual(first, again)
        self.assertEqual(len(first), 144)
        self.assertEqual(len(second), 36)
        self.assertEqual(set(first) & set(second), set())

    def test_cache_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header.json"
            write_header(path, {"seed": 0, "sigma": 4})
            read_header(path, {"sigma": 4, "seed": 0})
            with self.assertRaises(CacheMismatch):
                read_header(path, {"seed": 1, "sigma": 4})


class NameTests(unittest.TestCase):
    def test_known_stems(self) -> None:
        pcb = parse_anomaly_stem("Test_PCB_Position_-10%_Missing Component_2")
        self.assertEqual(pcb["defect"], "Missing Component")
        self.assertEqual(pcb["condition"], "Position_-10%")
        metal = parse_anomaly_stem("Live_Chipping_&_Porosity_Scale_+5%_to_-5%_16")
        self.assertEqual(metal["defect"], "Chipping_&_Porosity")
        self.assertEqual(metal["condition"], "Scale_+5%_to_-5%")
        shadow = parse_anomaly_stem("Test_PCB_Shadow_Light Alternate Position_Scratch_1")
        self.assertEqual(shadow["defect"], "Scratch")
        self.assertEqual(parse_anomaly_stem("not_a_defect")["parsed"], 0)


class CorrespondenceTests(unittest.TestCase):
    def test_vanished_component_stays_in_the_count(self) -> None:
        from robustad.prepare.regions import UNEVALUABLE_INPUT_TRANSFORM, component_correspondence

        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0, 1] = 255
        mask[4, 4] = 255
        rows = component_correspondence(mask, 2, 2)
        self.assertEqual(len(rows), 2)
        statuses = {row["status"] for row in rows}
        self.assertIn(UNEVALUABLE_INPUT_TRANSFORM, statuses)
        self.assertEqual(sum(row["status"] == UNEVALUABLE_INPUT_TRANSFORM for row in rows), 1)

    def test_region_names_do_not_split_workpiece(self) -> None:
        from robustad.prepare.diagnose_rows import score_row_regions

        score = np.zeros((8, 8), dtype=np.float32)
        score[0, 0] = 5
        defect = np.zeros((8, 8), dtype=bool)
        defect[4, 4] = True
        patch = np.zeros((2, 2), dtype=np.float32)
        patch[0, 0] = 5
        row = score_row_regions({"sigma0": score, "sigma4": score, "sigma8": score}, defect, patch)
        self.assertIn("sigma4_m16_maxfrac_nondefect_unpartitioned", row)
        self.assertNotIn("sigma4_m16_maxfrac_background", row)
        self.assertNotIn("sigma4_m16_maxfrac_workpiece", row)
        self.assertGreater(row["sigma4_m16_maxfrac_border"], 0)


class SpecificationTests(unittest.TestCase):
    def test_component_maximum_and_denominator(self) -> None:
        from robustad.prepare.calibration import classify_component
        from robustad.prepare.regions import UNEVALUABLE_INPUT_TRANSFORM, component_patch_maximum

        patch = np.zeros((4, 4), dtype=np.float32)
        patch[1, 2] = 3.5
        component = np.zeros((32, 32), dtype=bool)
        component[8:16, 16:24] = True
        self.assertEqual(component_patch_maximum(patch, component), np.float32(3.5))
        empty = np.zeros((32, 32), dtype=bool)
        self.assertTrue(np.isnan(component_patch_maximum(patch, empty)))
        kept = classify_component(UNEVALUABLE_INPUT_TRANSFORM, float("nan"), threshold=1.0)
        scored = classify_component("evaluable", 3.5, threshold=1.0)
        missed = classify_component("evaluable", 0.5, threshold=1.0)
        self.assertTrue(kept["in_denominator"] and scored["in_denominator"] and missed["in_denominator"])
        self.assertFalse(kept["detected"])
        self.assertTrue(scored["detected"])
        self.assertFalse(missed["detected"])
        self.assertEqual(int(kept["in_denominator"]) + int(scored["in_denominator"]) + int(missed["in_denominator"]), 3)

    def test_coordinates_mask_and_full_valid_region(self) -> None:
        from robustad.prepare.regions import full_valid_region, pixel_to_patch, resize_binary_mask
        from robustad.prepare.spec import PRIMARY_VALID_REGION

        self.assertEqual(pixel_to_patch(8, 16, stride=8), (1, 2))
        self.assertEqual(pixel_to_patch(4, 7, stride=4), (1, 1))
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 255
        resized = resize_binary_mask(mask, 2, 2)
        self.assertEqual(resized.dtype, np.bool_)
        self.assertTrue(resized.any())
        valid = full_valid_region(8, 8)
        self.assertTrue(valid.all())
        self.assertEqual(PRIMARY_VALID_REGION, "full_512")
        border = np.zeros((8, 8), dtype=bool)
        border[0, :] = True
        self.assertTrue(valid[0, 0])
        self.assertTrue(border[0, 0])

    def test_calibration_reports_achieved_fpr_without_a_verdict(self) -> None:
        from robustad.prepare.calibration import fpr_report, image_threshold, pool_patch_scores, rate_comparison
        from robustad.prepare.spec import TARGET_IMAGE_FPR

        scores = np.arange(36, dtype=np.float32)
        threshold = image_threshold(scores)
        self.assertEqual(threshold, float(np.quantile(scores, 1.0 - TARGET_IMAGE_FPR, method="linear")))
        self.assertAlmostEqual(threshold, float(np.quantile(scores, 0.95, method="linear")))
        report = fpr_report(scores, threshold, TARGET_IMAGE_FPR)
        self.assertEqual(report["target_fpr"], 0.05)
        self.assertFalse(report["target_match_claimed"])
        tied = np.array([0.0, 1.0, 1.0, 1.0])
        self.assertEqual(realized_fpr(tied, 1.0)["n_false"], 3)
        below = realized_fpr(np.array([0.0, 0.5, 1.0]), 0.75)
        self.assertEqual(below["n_false"], 1)
        self.assertGreaterEqual(report["achieved_fpr"], 0.0)
        patches = pool_patch_scores([np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)])
        self.assertEqual(patches.shape, (8,))
        comparison = rate_comparison(2, 36, 4, 80)
        self.assertEqual(comparison["practical_improvement"], "未判定")
        self.assertEqual(comparison["equivalence"], "未判定")
        self.assertLess(comparison["difference_interval_low"], comparison["difference"])
        self.assertGreater(comparison["difference_interval_high"], comparison["difference"])

    def test_research_split_identity_does_not_share_indexes(self) -> None:
        from robustad.prepare.research_split import summarize_split

        paths = [Path(f"normal/Set1_{index}.jpg") for index in range(10)]
        summary = summarize_split(paths, seed=0, val_ratio=0.2)
        self.assertEqual(summary["n_overlap_files"], 0)
        self.assertEqual(summary["duplicate_indexes"], [])
        self.assertEqual(summary["bodies"], ["Set1"])


if __name__ == "__main__":
    unittest.main()
