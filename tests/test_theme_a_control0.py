"""Checks for the 0°-only control bank. No full fit and no full score."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from robustad.theme_a.action import (
    BASELINE_CORESET,
    CONTROL0_FORBIDDEN_THREE_VIEW_INDEX_AS_CANDIDATE,
    CONTROL0_INITIAL_COL,
    CONTROL0_INITIAL_IMAGE_ID,
    CONTROL0_INITIAL_IMAGE_INDEX,
    CONTROL0_INITIAL_INDEX,
    CONTROL0_INITIAL_ROW,
    CONTROL0_N_CANDIDATES,
    CONTROL0_OUTPUT_NAME,
    CONTROL0_PROJECTION_COMPONENTS,
    CONTROL0_RATIO,
    CONTROL0_THREE_VIEW_INITIAL_INDEX,
    candidate_index,
    control0_storage,
    projection_components,
    resolve_control0_initial_center,
)
from robustad.theme_a.run_augment import write_view_features
from robustad.theme_a.run_control0 import assert_initial_center, refuse_control0_run


class Control0PlanTests(unittest.TestCase):
    def test_initial_center_is_zero_view_cell_not_three_view_integer(self) -> None:
        resolved = resolve_control0_initial_center()
        self.assertTrue(resolved["view_is_zero"])
        self.assertEqual(resolved["three_view_candidate_index"], 430694)
        self.assertEqual(CONTROL0_INITIAL_INDEX, 143974)
        self.assertEqual(resolved["control_candidate_index"], 143974)
        self.assertNotEqual(CONTROL0_INITIAL_INDEX, CONTROL0_THREE_VIEW_INITIAL_INDEX)
        self.assertEqual(
            candidate_index(CONTROL0_INITIAL_IMAGE_INDEX, 0, CONTROL0_INITIAL_ROW, CONTROL0_INITIAL_COL, n_views=1),
            143974,
        )
        self.assertEqual(CONTROL0_FORBIDDEN_THREE_VIEW_INDEX_AS_CANDIDATE, 430694)

    def test_storage_forces_355_and_exact_patch_count(self) -> None:
        storage = control0_storage()
        self.assertEqual(CONTROL0_N_CANDIDATES, 589824)
        self.assertEqual(storage["n_candidates"], 589824)
        self.assertEqual(int(CONTROL0_N_CANDIDATES * CONTROL0_RATIO), BASELINE_CORESET)
        self.assertEqual(storage["coreset_patches"], 58982)
        self.assertEqual(projection_components(CONTROL0_N_CANDIDATES), 328)
        self.assertEqual(storage["projection_components_jl"], 328)
        self.assertEqual(storage["projection_components"], 355)
        self.assertEqual(CONTROL0_PROJECTION_COMPONENTS, 355)
        self.assertEqual(storage["candidate_bytes"], 3_623_878_656)
        self.assertEqual(storage["projected_bytes"], 837_550_080)
        self.assertEqual(storage["candidate_device"], "cpu")
        self.assertEqual(storage["initial_index"], 143974)

    def test_refuse_overwrite_and_wrong_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / "results_a0_augment"
            wrong.mkdir()
            with self.assertRaises(RuntimeError):
                refuse_control0_run(wrong, confirmed=True, capacity_ok=True, manifest_saved=True)
            good = Path(tmp) / CONTROL0_OUTPUT_NAME
            good.mkdir()
            (good / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                refuse_control0_run(good, confirmed=True, capacity_ok=True, manifest_saved=True)

    def test_write_view_features_one_view_order(self) -> None:
        storage = torch.full((2 * 64 * 64, 1536), float("nan"), dtype=torch.float32)
        embedding = torch.zeros(1, 1536, 64, 64, dtype=torch.float32)
        embedding[0, 0, 1, 2] = 7.0
        write_view_features(storage, [1], 0, embedding, n_views=1)
        self.assertEqual(float(storage[candidate_index(1, 0, 1, 2, n_views=1), 0]), 7.0)
        self.assertTrue(torch.isnan(storage[candidate_index(0, 0, 1, 2, n_views=1), 0]))

    def test_assert_initial_center_checks_stem(self) -> None:
        ids = [f"img_{i}" for i in range(144)]
        ids[CONTROL0_INITIAL_IMAGE_INDEX] = CONTROL0_INITIAL_IMAGE_ID
        resolved = assert_initial_center(ids)
        self.assertEqual(resolved["fit_image_id"], CONTROL0_INITIAL_IMAGE_ID)
        ids[CONTROL0_INITIAL_IMAGE_INDEX] = "wrong"
        with self.assertRaises(RuntimeError):
            assert_initial_center(ids)


if __name__ == "__main__":
    unittest.main()
