"""Charts and detection-location panels from saved patch grids. No new scoring."""

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
from robustad.theme_a.boundary import extend_and_warp
from robustad.theme_a.run_synthetic import _load_rgb
from robustad.theme_a.warp import REFERENCE_SIZE
from robustad.visualize import configure_font

OUT = Path("report_figures/theme_a")
# Fixed review stems. Not chosen by anomaly score or heatmap.
STEMS = (
    ("Test_PCB_resized_None_None_Normal_Set2_181", "正常 Set2_181"),
    ("Test_PCB_None_None_Scratch_20", "傷 Scratch_20"),
    ("Test_PCB_None_None_Missing Component_1", "欠損 Missing_1"),
)
OUTER = 4
VMIN = 18.0
VMAX = 52.0
OLD_IMAGE = 36.28824043273926
OLD_PATCH = 24.215826511383057
CONTROL_IMAGE = 37.03678894042969
CONTROL_PATCH = 24.219616413116455
THREE_IMAGE = 37.689165115356445
THREE_PATCH = 25.81197500228882

# condition key, title, map directory, npz condition, angles, image threshold, patch threshold, frame
# frame "scored" overlays the heatmap on the image that was scored.
# frame "original" is the feature-inverse grid, which stays in the unrotated square.
PANELS = (
    ("uncorrected", "補正なし", "results_a0_boundary/maps", "pad_synthetic_m5", (-5.0,), OLD_IMAGE, OLD_PATCH, "scored"),
    ("control", "回転ビューなし", "results_a0_control0/maps", "pad_synthetic_m5", (-5.0,), CONTROL_IMAGE, CONTROL_PATCH, "scored"),
    ("three", "0°と±5°入り", "results_a0_augment/maps", "pad_synthetic_m5", (-5.0,), THREE_IMAGE, THREE_PATCH, "scored"),
    ("feature", "③特徴を戻す", "results_a0_feature/maps", "feature_m5", (), OLD_IMAGE, OLD_PATCH, "original"),
    ("restore", "②画像を戻す", "results_a0_boundary/maps", "pad_restore_m5", (-5.0, 5.0), OLD_IMAGE, OLD_PATCH, "scored"),
)
PANELS_P5 = (
    ("uncorrected", "補正なし", "results_a0_boundary/maps", "pad_synthetic_p5", (5.0,), OLD_IMAGE, OLD_PATCH, "scored"),
    ("control", "回転ビューなし", "results_a0_control0/maps", "pad_synthetic_p5", (5.0,), CONTROL_IMAGE, CONTROL_PATCH, "scored"),
    ("three", "0°と±5°入り", "results_a0_augment/maps", "pad_synthetic_p5", (5.0,), THREE_IMAGE, THREE_PATCH, "scored"),
    ("feature", "③特徴を戻す", "results_a0_feature/maps", "feature_p5", (), OLD_IMAGE, OLD_PATCH, "original"),
    ("restore", "②画像を戻す", "results_a0_boundary/maps", "pad_restore_p5", (5.0, -5.0), OLD_IMAGE, OLD_PATCH, "scored"),
)


def _load_index(map_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    found: dict[tuple[str, str], np.ndarray] = {}
    for path in map_dir.glob("*.npz"):
        with np.load(path) as payload:
            grid = np.asarray(payload["patch_map"], dtype=np.float32)
            found[(str(payload["image_id"]), str(payload["condition"]))] = grid
    return found


def _argmax(grid: np.ndarray) -> tuple[int, int]:
    flat = int(np.argmax(grid))
    return divmod(flat, grid.shape[1])


def _outer(row: int, col: int, width: int = 64) -> bool:
    return row < OUTER or col < OUTER or row >= width - OUTER or col >= width - OUTER


def _smooth(grid: np.ndarray) -> np.ndarray:
    from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

    generator = AnomalyMapGenerator(sigma=4)
    tensor = torch.from_numpy(np.ascontiguousarray(grid))[None, None]
    with torch.no_grad():
        smoothed = generator(tensor, image_size=(REFERENCE_SIZE, REFERENCE_SIZE))
    return smoothed[0, 0].numpy()


def _defect_mask(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    with Image.open(path) as handle:
        original = np.asarray(handle.convert("L"))
    labels = label_components(original)
    joint = np.zeros((REFERENCE_SIZE, REFERENCE_SIZE), dtype=bool)
    for component_id in np.unique(labels):
        if int(component_id) == 0:
            continue
        joint |= resize_binary_mask(labels == component_id, REFERENCE_SIZE, REFERENCE_SIZE)
    return joint


def _warp_mask(mask: np.ndarray, angles: tuple[float, ...]) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
    if not angles:
        return tensor.numpy() > 0
    return extend_and_warp(tensor, angles, nearest=True, pad_mode="constant")[0].numpy() > 0


def _cell_hits_mask(row: int, col: int, mask: np.ndarray) -> bool:
    block = mask[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8]
    return bool(block.any())


def _bars(path: Path) -> None:
    import matplotlib.pyplot as plt

    configure_font()
    labels = ["原画像", "−5°", "+5°"]
    x = np.arange(len(labels))
    width = 0.25
    series = {
        "正常パッチFPRの中央値": ([0.064, 0.440, 0.433], [0.064, 0.440, 0.436], [0.062, 0.054, 0.057], 0.5),
        "正常画像の検出数 / 120": ([12, 81, 77], [9, 54, 59], [7, 5, 8], 90),
        "異常画像の検出数 / 75": ([69, 75, 75], [62, 75, 75], [64, 62, 66], 80),
    }
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    colors = ("#4C78A8", "#F58518", "#54A24B")
    names = ("先に作った正常モデル", "回転ビューなし", "0°と±5°入り")
    for axis, (title, (old, control, three, ymax)) in zip(axes, series.items()):
        axis.bar(x - width, old, width, label=names[0], color=colors[0])
        axis.bar(x, control, width, label=names[1], color=colors[1])
        axis.bar(x + width, three, width, label=names[2], color=colors[2])
        axis.set_xticks(x, labels)
        axis.set_ylim(0, ymax)
        axis.set_title(title, fontsize=11)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("各モデル固有のしきい値。パッチFPRは正常画像ごとの割合の中央値。", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(2)
    width = 0.24
    axis.bar(x - width, [81, 77], width, label="回転したまま", color="#4C78A8")
    axis.bar(x, [52, 55], width, label="③特徴を戻す", color="#E45756")
    axis.bar(x + width, [11, 11], width, label="②画像を戻す", color="#72B7B2")
    axis.set_xticks(x, ["−5°", "+5°"])
    axis.set_ylim(0, 90)
    axis.set_ylabel("正常画像の検出数 / 120")
    axis.set_title("先に作った正常モデルでの②と③。0°と±5°入りとは別の集合。")
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(path.with_name("bars_oldbank_normal.png"), dpi=140)
    plt.close(figure)


def _location_summary(indexes: dict[str, dict[tuple[str, str], np.ndarray]], masks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    specs = (
        ("補正なし −5°", "boundary", "pad_synthetic_m5", (-5.0,), False),
        ("補正なし +5°", "boundary", "pad_synthetic_p5", (5.0,), False),
        ("回転ビューなし −5°", "control", "pad_synthetic_m5", (-5.0,), False),
        ("回転ビューなし +5°", "control", "pad_synthetic_p5", (5.0,), False),
        ("0°と±5°入り −5°", "three", "pad_synthetic_m5", (-5.0,), False),
        ("0°と±5°入り +5°", "three", "pad_synthetic_p5", (5.0,), False),
        ("③ −5°", "feature", "feature_m5", (), True),
        ("③ +5°", "feature", "feature_p5", (), True),
        ("② −5°", "boundary", "pad_restore_m5", (-5.0, 5.0), False),
        ("② +5°", "boundary", "pad_restore_p5", (5.0, -5.0), False),
        ("回転ビューなし 原画像", "control", "sd_original", (), False),
        ("0°と±5°入り 原画像", "three", "sd_original", (), False),
    )
    summary = {}
    for title, pool, condition, angles, original_frame in specs:
        grids = indexes[pool]
        normal_outer = []
        anomaly_on_defect = []
        for (stem, cond), grid in grids.items():
            if cond != condition:
                continue
            row, col = _argmax(grid)
            if "Normal" in stem:
                normal_outer.append(_outer(row, col))
            elif stem in masks:
                mask = masks[stem] if original_frame or not angles else _warp_mask(masks[stem], angles)
                anomaly_on_defect.append(_cell_hits_mask(row, col, mask))
        summary[title] = {
            "normal_n": len(normal_outer),
            "normal_outer_argmax": int(sum(normal_outer)),
            "anomaly_n": len(anomaly_on_defect),
            "anomaly_argmax_on_defect": int(sum(anomaly_on_defect)),
        }
    return summary


def _location_bars(summary: dict[str, dict[str, float]], path: Path) -> None:
    import matplotlib.pyplot as plt

    configure_font()
    labels = ["回していない原画像", "−5°", "+5°"]
    keys = ("原画像", "−5°", "+5°")
    models = (("回転ビューなし", "#F58518"), ("0°と±5°入り", "#54A24B"))
    panels = (
        ("anomaly_argmax_on_defect", 75, "欠陥がある75枚", "判定を動かした1マスが\n正解の欠陥の上にあった枚数"),
        ("normal_outer_argmax", 120, "欠陥がない120枚", "その1マスが\n画像の端4マス以内にあった枚数"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    x = np.arange(len(labels))
    width = 0.36
    for axis, (field, denominator, heading, title) in zip(axes, panels):
        for offset, (model, color) in zip((-width / 2, width / 2), models):
            counts = [int(summary[f"{model} {key}"][field]) for key in keys]
            bars = axis.bar(x + offset, counts, width, label=model, color=color)
            for bar, count in zip(bars, counts):
                axis.text(bar.get_x() + bar.get_width() / 2, count + 1.2, f"{count}/{denominator}", ha="center", va="bottom", fontsize=8)
        axis.set_xticks(x, labels)
        axis.set_ylim(0, denominator)
        axis.set_ylabel("枚数")
        axis.set_title(f"{heading}\n{title}", fontsize=11)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("1枚につき、異常度が最も高い8×8画素を1マスだけ取り、原画像および正解の欠陥と比べた。", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _density(indexes: dict[str, dict[tuple[str, str], np.ndarray]], path: Path) -> None:
    import matplotlib.pyplot as plt

    configure_font()
    rows = (("control", "回転ビューなし"), ("three", "0°と±5°入り"))
    columns = (("sd_original", "回していない原画像"), ("pad_synthetic_m5", "同じ画像を −5° に傾けたあと"))
    figure, axes = plt.subplots(len(rows), len(columns), figsize=(7.2, 7.2))
    for row_index, (pool, model) in enumerate(rows):
        for column_index, (condition, angle) in enumerate(columns):
            axis = axes[row_index, column_index]
            counts = np.zeros((64, 64), dtype=np.float32)
            outer = 0
            for (stem, cond), grid in indexes[pool].items():
                if cond != condition or "Normal" not in stem:
                    continue
                cell_row, cell_col = _argmax(grid)
                counts[cell_row, cell_col] += 1
                outer += int(_outer(cell_row, cell_col))
            axis.imshow(np.ones((64, 64, 3)), vmin=0, vmax=1)
            shown = np.ma.masked_where(counts == 0, counts)
            axis.imshow(shown, cmap="turbo", vmin=1, vmax=max(3, float(counts.max())))
            axis.add_patch(plt.Rectangle((3.5, 3.5), 56, 56, fill=False, edgecolor="black", linewidth=1.0))
            axis.set_title(f"{model}\n{angle}\n端にある枚数 {outer}/120", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("欠陥がない120枚。点は判定を動かした1マス。左が原画像、右が−5°。黒枠の外が端。", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _draw_ground_truth(axis, mask: np.ndarray, *, fill: bool) -> None:
    if fill:
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = (1.0, 0.0, 1.0, 0.45)
        axis.imshow(overlay)
    axis.contour(mask.astype(float), levels=[0.5], colors="#FF00FF", linewidths=1.6)


def _panel(stem: str, caption: str, rgb: torch.Tensor, defect: np.ndarray | None, indexes: dict, panels: tuple, path: Path, angle_label: str) -> None:
    import matplotlib.pyplot as plt

    configure_font()
    figure, axes = plt.subplots(2, len(panels), figsize=(3.05 * len(panels), 6.5))
    for column, (_key, title, _directory, condition, angles, image_thr, patch_thr, frame) in enumerate(panels):
        pool = {"uncorrected": "boundary", "control": "control", "three": "three", "feature": "feature", "restore": "boundary"}[_key]
        grid = indexes[pool][(stem, condition)]
        row, col = _argmax(grid)
        heat = _smooth(grid)
        if frame == "original" or not angles:
            image = rgb.detach().clamp(0, 1).permute(1, 2, 0).numpy()
            mask = defect
        else:
            image = extend_and_warp(rgb, angles, nearest=False)[0].detach().clamp(0, 1).permute(1, 2, 0).numpy()
            mask = None if defect is None else _warp_mask(defect, angles)
        axes[0, column].imshow(image)
        if mask is not None:
            _draw_ground_truth(axes[0, column], mask, fill=True)
        hit = _cell_hits_mask(row, col, mask) if mask is not None else None
        where = {True: "欠陥上", False: "欠陥外", None: "正常"}[hit]
        above = "画像しきい値以上" if float(np.max(grid)) >= 0 and _image_score(indexes, pool, stem, condition) >= image_thr else "画像しきい値未満"
        axes[0, column].set_title(f"{title}\n{above}\n最大パッチは{where}", fontsize=8)
        axes[1, column].imshow(heat, cmap="turbo", vmin=VMIN, vmax=VMAX)
        detected = grid >= patch_thr
        if detected.any():
            axes[1, column].contour(
                np.repeat(np.repeat(detected, 8, axis=0), 8, axis=1).astype(float),
                levels=[0.5],
                colors="red",
                linewidths=0.4,
            )
        if mask is not None:
            _draw_ground_truth(axes[1, column], mask, fill=False)
        axes[1, column].add_patch(plt.Rectangle((col * 8 - 0.5, row * 8 - 0.5), 8, 8, fill=False, edgecolor="black", linewidth=1.1))
        axes[1, column].set_title(f"黒枠 ({row}, {col})  赤線はパッチしきい値以上", fontsize=7)
        for axis in (axes[0, column], axes[1, column]):
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        f"{caption}  {angle_label}\n上段は見ている枠の画像。マゼンタ線は正解の欠陥。下段は sigma 4。③の枠は回転前、他は採点した画像。色範囲は {VMIN:.0f}–{VMAX:.0f}。",
        fontsize=10,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _image_score(indexes: dict, pool: str, stem: str, condition: str) -> float:
    # Stored only as the map max is not the image score. Read from the csv cache attached on indexes.
    return float(indexes["scores"][(pool, stem, condition)])


def _scores() -> dict[tuple[str, str, str], float]:
    files = {
        "boundary": Path("results_a0_boundary/tables/images.csv"),
        "control": Path("results_a0_control0/tables/images.csv"),
        "three": Path("results_a0_augment/tables/images.csv"),
        "feature": Path("results_a0_feature/tables/images.csv"),
    }
    found = {}
    for pool, path in files.items():
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                found[(pool, row["source_id"], row["condition"])] = float(row["reproduction_score"])
    return found


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _bars(OUT / "bars_builder.png")
    indexes = {
        "boundary": _load_index(Path("results_a0_boundary/maps")),
        "control": _load_index(Path("results_a0_control0/maps")),
        "three": _load_index(Path("results_a0_augment/maps")),
        "feature": _load_index(Path("results_a0_feature/maps")),
        "scores": _scores(),
    }
    _density(indexes, OUT / "location_normal_argmax.png")
    subset = SUBSETS["pcb"]
    domain = next(item for item in subset.domains if item.name == "SD")
    data_root = Path(r"K:\RobustAD")
    domain_dir = subset.domain_dir(data_root, domain)
    paths = {path.stem: path for path in list_images(domain_dir / "normal") + list_images(domain_dir / "anomaly")}
    masks = {}
    for stem, _caption in STEMS:
        if "Normal" in stem:
            continue
        masks[stem] = _defect_mask(Path(find_mask(paths[stem], domain_dir / "masks")))
    # All anomaly masks for the overlap summary.
    for path in list_images(domain_dir / "anomaly"):
        masks[path.stem] = _defect_mask(Path(find_mask(path, domain_dir / "masks")))
    summary = _location_summary(indexes, masks)
    import json

    (OUT / "location_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _location_bars(summary, OUT / "bars_location.png")
    resize = Resize((REFERENCE_SIZE, REFERENCE_SIZE), antialias=True)
    for stem, caption in STEMS:
        rgb = _load_rgb(paths[stem], resize)
        defect = masks.get(stem)
        safe = stem.replace(" ", "_")
        _panel(stem, caption, rgb, defect, indexes, PANELS, OUT / f"loc_m5_{safe}.png", "−5°")
        _panel(stem, caption, rgb, defect, indexes, PANELS_P5, OUT / f"loc_p5_{safe}.png", "+5°")
    split = "Test_PCB_resized_None_None_Normal_Set2_185"
    _panel(
        split,
        "正常 Set2_185（−5°で補正なしと回転ビューなしだけが検出）",
        _load_rgb(paths[split], resize),
        None,
        indexes,
        PANELS,
        OUT / "loc_m5_split_Normal_Set2_185.png",
        "−5°",
    )
    # First anomaly filename where, at −5°, the uncorrected, no-rotation, and
    # 0°±5° columns all place the max patch on the defect and sit at or above
    # their own image thresholds. Not chosen by looking at the heatmap.
    detected_on_defect = "Test_PCB_None_None_Missing Component_15"
    _panel(
        detected_on_defect,
        "欠損 Missing_15（−5°で3列とも最大が欠陥上、画像しきい値以上）",
        _load_rgb(paths[detected_on_defect], resize),
        masks[detected_on_defect],
        indexes,
        PANELS,
        OUT / "loc_m5_detected_Missing_Component_15.png",
        "−5°",
    )
    print(OUT)
    for key, value in summary.items():
        print(key, value)


if __name__ == "__main__":
    main()
