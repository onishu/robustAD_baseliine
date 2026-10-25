"""Environment and reproduction-configuration manifest.

The coreset cardinality is not recorded. No memory bank is stored in the
existing results, so the count is unknown until a model is built again.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from robustad.benchmark import IMAGE_NET_MEAN, IMAGE_NET_STD, WEIGHT_SETS
from robustad.prepare.spec import RUN_ID, SMOOTH_KERNEL, SMOOTH_RADIUS, SMOOTH_SIGMA_DEFAULT


def _git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"未確認: {exc}"
    if completed.returncode != 0:
        return completed.stderr.strip() or "未確認"
    return completed.stdout.strip()


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "未導入"
    return str(getattr(module, "__version__", "不明"))


def build_manifest(repo_root: Path) -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "run_id": RUN_ID,
        "status": "awaiting_researcher_decision",
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status": _git(["status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "anomalib": _package_version("anomalib"),
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "weights": WEIGHT_SETS["timm"],
        "image_size": 512,
        "normalize_mean": IMAGE_NET_MEAN,
        "normalize_std": IMAGE_NET_STD,
        "center_crop": False,
        "preprocessor_padding": "なし。Resize のみ。GaussianBlur2d は出力サイズを保つ reflect padding。",
        "padim": {
            "backbone": "resnet18",
            "layers": ["layer1", "layer2", "layer3"],
            "n_features": 100,
            "image_score": "平滑化後異常マップの最大値",
            "smooth_sigma": SMOOTH_SIGMA_DEFAULT,
        },
        "patchcore": {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.1,
            "stored_patch_count": "未測定",
            "num_neighbors": 9,
            "num_neighbors_applies_to": "最悪パッチの再重み付け。異常マップの近傍数は 1。",
            "image_score": "再重み付けスコア。平滑化マップの最大値ではない。",
            "smooth_sigma": SMOOTH_SIGMA_DEFAULT,
            "smooth_kernel": SMOOTH_KERNEL,
            "smooth_radius": SMOOTH_RADIUS,
            "upsample": "F.interpolate の既定。mode 引数なし。",
        },
        "fit": "val_ratio 0。公開 train/normal 180 枚をすべて fit。独立した calibration は未作成。",
        "threshold_in_existing_results": "各テスト条件の正解で F1 を最大化。",
        "caches_reused": "results*/runs/*.json の集計値と vis の cases.json。パッチ距離、平滑化前マップ、メモリバンクは保存されていない。",
        "repo_root": str(repo_root),
    }


def write_manifest(repo_root: Path, output_dir: Path) -> dict:
    payload = build_manifest(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
