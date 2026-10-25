"""Heatmaps for the experiment report, drawn from saved patch grids.

Images are the fixed audit stems, not images chosen for a high or low score.
The color scale is shared. The marker is the pre-smoothing patch maximum.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import Resize

from robustad.catalog import SUBSETS
from robustad.data import find_mask, list_images
from robustad.prepare.regions import label_components, resize_binary_mask
from robustad.theme_a.boundary import MARGIN, content_probe, extend_and_warp
from robustad.theme_a.run_synthetic import CONDITIONS, _load_rgb
from robustad.theme_a.warp import REFERENCE_SIZE, apply_angles, valid_after_angles

STEMS = (
    "Test_PCB_resized_None_None_Normal_Set2_181",
    "Test_PCB_resized_None_None_Normal_Set2_241",
    "Test_PCB_None_None_Missing Component_1",
    "Test_PCB_None_None_Scratch_20",
)
PANELS = (
    ("original", "原画像", False),
    ("synthetic_m5", "−5°", False),
    ("restore_m5", "−5°往復", False),
    ("pad_warp_0", "折り返し 0°", True),
    ("pad_synthetic_m5", "折り返し −5°", True),
    ("pad_restore_m5", "折り返し −5°往復", True),
)
SIGN_PANELS = (
    ("synthetic_m5", "−5°", False),
    ("synthetic_p5", "+5°", False),
    ("pad_synthetic_m5", "折り返し −5°", True),
    ("pad_synthetic_p5", "折り返し +5°", True),
)
VMIN = 18.0
VMAX = 52.0
IMAGE_THRESHOLD = 36.28824043273926


def _scores(path: Path) -> dict[tuple[str, str], float]:
    rows = csv.DictReader(path.open(encoding="utf-8"))
    return {(row["source_id"], row["condition"]): float(row["reproduction_score"]) for row in rows}


def _maps(map_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    found: dict[tuple[str, str], np.ndarray] = {}
    for path in map_dir.glob("*.npz"):
        with np.load(path) as payload:
            grid = np.asarray(payload["patch_map"], dtype=np.float32)
            if grid.ndim == 3:
                grid = grid[0]
            found[(str(payload["image_id"]), str(payload["condition"]))] = grid
    return found


def _smooth(grid: np.ndarray) -> np.ndarray:
    from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

    generator = AnomalyMapGenerator(sigma=4)
    tensor = torch.from_numpy(grid)[None, None]
    with torch.no_grad():
        smoothed = generator(tensor, image_size=(REFERENCE_SIZE, REFERENCE_SIZE))
    return smoothed[0, 0].numpy()


def _warp_rgb(image: torch.Tensor, condition: str, padded: bool) -> torch.Tensor:
    angles = dict(CONDITIONS)[condition] if not padded else {
        "pad_warp_0": (0.0,),
        "pad_synthetic_m5": (-5.0,),
        "pad_synthetic_p5": (5.0,),
        "pad_restore_m5": (-5.0, 5.0),
        "pad_restore_p5": (5.0, -5.0),
    }[condition]
    if not padded:
        return apply_angles(image, angles, nearest=False)[0]
    return extend_and_warp(image, angles, nearest=False, margin=MARGIN)[0]


def _warp_mask(mask: torch.Tensor, condition: str, padded: bool) -> np.ndarray:
    angles = dict(CONDITIONS)[condition] if not padded else {
        "pad_warp_0": (0.0,),
        "pad_synthetic_m5": (-5.0,),
        "pad_synthetic_p5": (5.0,),
        "pad_restore_m5": (-5.0, 5.0),
        "pad_restore_p5": (5.0, -5.0),
    }[condition]
    if not padded:
        return apply_angles(mask, angles, nearest=True)[0].numpy() > 0
    return extend_and_warp(mask, angles, nearest=True, margin=MARGIN, pad_mode="constant")[0].numpy() > 0


def _boundary(condition: str, padded: bool) -> np.ndarray:
    angles = dict(CONDITIONS)[condition] if not padded else {
        "pad_warp_0": (0.0,),
        "pad_synthetic_m5": (-5.0,),
        "pad_synthetic_p5": (5.0,),
        "pad_restore_m5": (-5.0, 5.0),
        "pad_restore_p5": (5.0, -5.0),
    }[condition]
    if not padded:
        valid = valid_after_angles(REFERENCE_SIZE, REFERENCE_SIZE, angles).numpy()
        return ~valid
    origin = content_probe(angles, MARGIN).numpy()
    return origin < 1.0 - 1e-5


def _mask(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    with Image.open(path) as handle:
        original = np.asarray(handle.convert("L"))
    labels = label_components(original)
    joint = np.zeros((REFERENCE_SIZE, REFERENCE_SIZE), dtype=bool)
    for component_id in np.unique(labels):
        if component_id == 0:
            continue
        joint |= resize_binary_mask(labels == component_id, REFERENCE_SIZE, REFERENCE_SIZE)
    return joint


def _draw(stem: str, panels: tuple, rgb: torch.Tensor, defect: np.ndarray | None, scores: dict, grids: dict, destination: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams

    for font in (Path(r"C:\Windows\Fonts\YuGothM.ttc"), Path(r"C:\Windows\Fonts\meiryo.ttc")):
        if font.is_file():
            font_manager.fontManager.addfont(str(font))
            rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
            break
    rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, len(panels), figsize=(3.15 * len(panels), 6.6))
    for column, (condition, title, padded) in enumerate(panels):
        image = _warp_rgb(rgb, condition, padded).detach().clamp(0, 1).permute(1, 2, 0).numpy()
        grid = grids[(stem, condition)]
        heat = _smooth(grid)
        row, col = np.unravel_index(int(np.argmax(grid)), grid.shape)
        score = scores[(stem, condition)]
        decision = "画像しきい値以上" if score >= IMAGE_THRESHOLD else "画像しきい値未満"
        show = axes[0, column].imshow(image)
        del show
        axes[0, column].set_title(f"{title}\nscore {score:.2f}\n{decision}", fontsize=9)
        overlay = axes[1, column].imshow(heat, cmap="turbo", vmin=VMIN, vmax=VMAX)
        del overlay
        if defect is not None:
            warped_defect = _warp_mask(torch.from_numpy(defect.astype(np.float32)), condition, padded)
            axes[1, column].contour(warped_defect.astype(float), levels=[0.5], colors="white", linewidths=0.6)
        edge = _boundary(condition, padded)
        if edge.any():
            axes[1, column].contour(edge.astype(float), levels=[0.5], colors="cyan", linewidths=0.7)
        axes[1, column].add_patch(
            plt.Rectangle((col * 8 - 0.5, row * 8 - 0.5), 8, 8, fill=False, edgecolor="black", linewidth=1.2)
        )
        axes[1, column].set_title(f"平滑化前パッチ最大 ({row}, {col})", fontsize=8)
        for axis in (axes[0, column], axes[1, column]):
            axis.set_xticks([])
            axis.set_yticks([])
    figure.colorbar(axes[1, -1].images[0], ax=axes[1, :].tolist(), fraction=0.02, pad=0.01, label="sigma 4 異常マップ")
    short = stem.removeprefix("Test_PCB_").removeprefix("resized_")
    figure.suptitle(
        f"{short}\n上段は変換後の画像。下段はsigma 4。黒枠は平滑化前の最大パッチ。白線は欠陥。水色は、黒で埋まった領域か、端の折り返しを参照した領域。色範囲は全図で {VMIN:.0f}–{VMAX:.0f}。",
        fontsize=10,
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def main() -> None:
    subset = SUBSETS["pcb"]
    domain = next(item for item in subset.domains if item.name == "SD")
    domain_dir = subset.domain_dir(Path(r"K:\RobustAD"), domain)
    paths = {path.stem: path for path in list_images(domain_dir / "normal") + list_images(domain_dir / "anomaly")}
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    scores = _scores(Path("results_a0/tables/images.csv"))
    scores.update(_scores(Path("results_a0_boundary/tables/images.csv")))
    grids = _maps(Path("results_a0/maps"))
    grids.update(_maps(Path("results_a0_boundary/maps")))
    out = Path("report_figures/theme_a")
    for stem in STEMS:
        image = _load_rgb(paths[stem], resize)
        defect = None
        if "Normal" not in stem:
            defect = _mask(Path(find_mask(paths[stem], domain_dir / "masks")))
        _draw(stem, PANELS, image, defect, scores, grids, out / f"{stem.replace(' ', '_')}.png")
    _draw(STEMS[0], SIGN_PANELS, _load_rgb(paths[STEMS[0]], resize), None, scores, grids, out / "sign_normal_181.png")
    print(out)


if __name__ == "__main__":
    main()
