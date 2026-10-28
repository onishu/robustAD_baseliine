"""Choose one normal fit image without looking at detection scores."""

from __future__ import annotations

import json
from pathlib import Path

from robustad.catalog import SUBSETS
from robustad.data import list_images

MANIFEST = Path("results_p0/research_split_manifest.json")


def fit_stems(manifest_path: Path = MANIFEST) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stems = []
    for key in payload["datasets"]["pcb"]["fit"]:
        body, index = str(key).rsplit("#", 1)
        stems.append(f"{body}_{index}")
    if len(stems) != 144:
        raise RuntimeError(f"fit は 144 枚です: {len(stems)}")
    return sorted(stems)


def reference_stem(manifest_path: Path = MANIFEST) -> str:
    """First fit filename in lexicographic order. Calibration and test are absent."""

    return fit_stems(manifest_path)[0]


def reference_record(manifest_path: Path = MANIFEST) -> dict[str, object]:
    stem = reference_stem(manifest_path)
    return {
        "dataset": "pcb",
        "split": "fit",
        "rule": "results_p0 の pcb fit 144 をファイル名の辞書順に並べ、最初の1枚",
        "source_id": stem,
        "n_fit": 144,
        "uses_calibration": False,
        "uses_test": False,
        "uses_anomaly_label": False,
        "uses_detection_score": False,
        "manifest": str(manifest_path).replace("\\", "/"),
    }


def resolve_reference_path(data_root: Path, manifest_path: Path = MANIFEST) -> Path:
    stem = reference_stem(manifest_path)
    directory = SUBSETS["pcb"].train_dir(data_root) / "normal"
    matches = [path for path in list_images(directory) if path.stem == stem]
    if len(matches) != 1:
        raise RuntimeError(f"参照画像が一意ではありません: {stem} ({len(matches)})")
    return matches[0]
