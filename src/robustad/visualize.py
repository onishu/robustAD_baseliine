"""Save anomaly heatmaps, pixel-detection panels, and result heatmaps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from robustad.catalog import SUBSETS

ROLES = (
    ("anomaly_high", "異常・高スコア（検出）", 1, True),
    ("anomaly_low", "異常・低スコア（見逃し）", 1, False),
    ("normal_high", "正常・高スコア（誤検知）", 0, True),
    ("normal_low", "正常・低スコア", 0, False),
)


def configure_font() -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib import font_manager, rcParams

    for path in (
        Path(r"C:\Windows\Fonts\YuGothM.ttc"),
        Path(r"C:\Windows\Fonts\meiryo.ttc"),
        Path(r"C:\Windows\Fonts\msgothic.ttc"),
    ):
        if not path.is_file():
            continue
        font_manager.fontManager.addfont(str(path))
        rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
        break
    rcParams["axes.unicode_minus"] = False


def select_samples(samples: list[dict], per_role: int = 2) -> list[dict]:
    """Keep the extreme normal and anomalous images for the image log."""

    chosen: list[dict] = []
    used: set[int] = set()
    for role, caption, label, highest in ROLES:
        pool = [sample for sample in samples if sample["label"] == label and id(sample) not in used]
        pool.sort(key=lambda sample: (sample["score"], sample["image_path"]), reverse=highest)
        count = per_role if label == 1 else 1
        for sample in pool[:count]:
            used.add(id(sample))
            picked = dict(sample)
            picked["role"] = role
            picked["caption"] = caption
            chosen.append(picked)
    return chosen


def render_domain(
    samples: list[dict],
    destination: Path,
    *,
    title: str,
    metrics_line: str,
    threshold: float | None,
) -> None:
    """Write one panel per selected image and a contact sheet for the domain."""

    selected = select_samples(samples)
    if not selected:
        return
    configure_font()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, sample in enumerate(selected, start=1):
        filename = f"{index:02d}_{sample['role']}.png"
        _render_panel(sample, threshold).save(destination / filename)
        manifest.append(
            {
                "file": filename,
                "role": sample["role"],
                "caption": sample["caption"],
                "image_path": sample["image_path"],
                "label": sample["label"],
                "score": sample["score"],
                "pixel": _pixel_counts(sample, threshold),
            }
        )
    _contact_sheet(selected, destination / "montage.png", title, metrics_line, threshold)
    _write_manifest(destination / "cases.json", {"title": title, "metrics": metrics_line, "cases": manifest})


def save_result_figures(summary: dict, destination: Path, baseline: dict | None = None) -> None:
    """Draw annotated heatmaps of image AUROC and pixel metrics."""

    if not summary:
        return
    configure_font()
    destination.mkdir(parents=True, exist_ok=True)
    rows = _rows(summary)
    _draw_heatmap(
        destination / "image_auroc.png",
        rows,
        _image_columns(summary, baseline),
        "Image AUROC（%）  torchvision ImageNet-1K 重み",
        diverging=False,
    )
    _draw_heatmap(
        destination / "image_auroc_delta.png",
        rows,
        _delta_columns(summary, baseline),
        "Image AUROC の差（ポイント）  正は今回の方が高い",
        diverging=True,
    )
    pixel_columns = _pixel_columns(summary)
    if pixel_columns:
        _draw_heatmap(
            destination / "pixel_metrics.png",
            [row for row in rows if row[0] != "piled_bags"],
            pixel_columns,
            "画素レベルの評価（%）  F1 は同一テスト集合で最大になるしきい値",
            diverging=False,
        )


def _rows(summary: dict) -> list[tuple[str, str, str]]:
    rows = []
    for key, subset in SUBSETS.items():
        if key not in summary:
            continue
        for domain in subset.domains:
            rows.append((key, domain.name, f"{subset.title} / {domain.name}"))
    return rows


def _value(summary: dict, dataset: str, model: str, domain: str, field: str) -> float:
    stats = summary.get(dataset, {}).get(model, {})
    domain_stats = stats.get("domains", {}).get(domain, {})
    value = domain_stats.get(field, float("nan"))
    return float(value)


def _image_columns(summary: dict, baseline: dict | None) -> list[tuple[str, list[float]]]:
    columns = []
    for model, label in (("padim", "PaDiM"), ("patchcore", "PatchCore")):
        if not any(model in models for models in summary.values()):
            continue
        columns.append(
            (
                f"{label} 今回",
                [_value(summary, dataset, model, domain, "auroc_mean") for dataset, domain, _ in _rows(summary)],
            )
        )
        columns.append(
            (
                f"{label} 論文",
                [_value(summary, dataset, model, domain, "paper_mean") for dataset, domain, _ in _rows(summary)],
            )
        )
        if baseline:
            columns.append(
                (
                    f"{label} 旧重み",
                    [
                        _value(baseline, dataset, model, domain, "auroc_mean")
                        for dataset, domain, _ in _rows(summary)
                    ],
                )
            )
    return columns


def _delta_columns(summary: dict, baseline: dict | None) -> list[tuple[str, list[float]]]:
    columns = []
    rows = _rows(summary)
    for model, label in (("padim", "PaDiM"), ("patchcore", "PatchCore")):
        if not any(model in models for models in summary.values()):
            continue
        ours = np.array([_value(summary, dataset, model, domain, "auroc_mean") for dataset, domain, _ in rows])
        paper = np.array([_value(summary, dataset, model, domain, "paper_mean") for dataset, domain, _ in rows])
        columns.append((f"{label} − 論文", (ours - paper).tolist()))
        if baseline:
            previous = np.array(
                [_value(baseline, dataset, model, domain, "auroc_mean") for dataset, domain, _ in rows]
            )
            columns.append((f"{label} − 旧重み", (ours - previous).tolist()))
    return columns


def _pixel_columns(summary: dict) -> list[tuple[str, list[float]]]:
    rows = [row for row in _rows(summary) if row[0] != "piled_bags"]
    if not rows:
        return []
    if not any("pixel_auroc_mean" in domain for stats in summary.values() for model in stats.values() for domain in model["domains"].values()):
        return []
    columns = []
    for model, label in (("padim", "PaDiM"), ("patchcore", "PatchCore")):
        if not any(model in models for models in summary.values()):
            continue
        for field, suffix in (
            ("pixel_auroc_mean", "pixel AUROC"),
            ("pixel_aupr_mean", "pixel AUPR"),
            ("pixel_f1_mean", "pixel F1"),
        ):
            columns.append(
                (
                    f"{label} {suffix}",
                    [_value(summary, dataset, model, domain, field) for dataset, domain, _ in rows],
                )
            )
    return columns


def _draw_heatmap(
    path: Path,
    rows: list[tuple[str, str, str]],
    columns: list[tuple[str, list[float]]],
    title: str,
    *,
    diverging: bool,
) -> None:
    import matplotlib.pyplot as plt

    if not rows or not columns:
        return
    matrix = np.array([column for _, column in columns], dtype=np.float64).T
    if diverging:
        span = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
        span = max(float(span), 5.0)
        vmin, vmax = -span, span
        cmap = plt.get_cmap("RdBu_r")
    else:
        vmin, vmax = 0.0, 100.0
        cmap = plt.get_cmap("viridis")
    fig_w = max(8.0, 1.55 * matrix.shape[1] + 3.2)
    fig_h = max(4.0, 0.42 * matrix.shape[0] + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), [name for name, _ in columns], rotation=30, ha="right")
    ax.set_yticks(range(matrix.shape[0]), [label for _, _, label in rows])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                text = "—"
                color = "black"
            else:
                text = f"{value:.1f}"
                color = _annotation_color(cmap, value, vmin, vmax)
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _annotation_color(cmap, value: float, vmin: float, vmax: float) -> str:
    point = 0.5 if vmax == vmin else (value - vmin) / (vmax - vmin)
    red, green, blue, _ = cmap(np.clip(point, 0.0, 1.0))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.55 else "black"


def _render_panel(sample: dict, threshold: float | None) -> Image.Image:
    import io

    import matplotlib.pyplot as plt

    rgb, heat, overlay, gt, detection = _views(sample, threshold)
    show_pixels = gt is not None
    views = [rgb, heat, overlay] if not show_pixels else [rgb, heat, overlay, gt, detection]
    names = ["原画像", "異常ヒートマップ", "重畳"]
    if show_pixels:
        names += ["正解マスク", "検出マスク（黄枠=予測 水色枠=正解）"]
    fig, axes = plt.subplots(1, len(views), figsize=(3.2 * len(views), 3.8))
    for axis, view, name in zip(axes, views, names, strict=True):
        axis.imshow(view)
        axis.set_title(name, fontsize=9)
        axis.axis("off")
    counts = _pixel_counts(sample, threshold)
    detail = f"{sample['caption']}  score {sample['score']:.3f}  {Path(sample['image_path']).name}"
    if counts:
        detail += (
            f"  IoU {counts['iou'] * 100:.1f}%"
            f"  precision {counts['precision'] * 100:.1f}%"
            f"  recall {counts['recall'] * 100:.1f}%"
        )
    fig.suptitle(detail, fontsize=10)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB").copy()


def _contact_sheet(
    samples: list[dict],
    path: Path,
    title: str,
    metrics_line: str,
    threshold: float | None,
) -> None:
    import matplotlib.pyplot as plt

    has_pixels = samples[0].get("mask") is not None and threshold is not None and np.isfinite(threshold)
    columns = 4 if has_pixels else 3
    fig, axes = plt.subplots(len(samples), columns, figsize=(3.1 * columns, 3.1 * len(samples)))
    if len(samples) == 1:
        axes = np.array([axes])
    headers = ["原画像", "異常ヒートマップ", "重畳", "検出"]
    for row, sample in enumerate(samples):
        rgb, heat, overlay, _gt, detection = _views(sample, threshold)
        views = [rgb, heat, overlay] if detection is None else [rgb, heat, overlay, detection]
        counts = _pixel_counts(sample, threshold)
        caption = f"{sample['caption']}\nscore {sample['score']:.3f}"
        if counts:
            caption += f"\nIoU {counts['iou'] * 100:.1f}%"
        for col, view in enumerate(views):
            axes[row, col].imshow(view)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(headers[col], fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(caption, fontsize=8)
    fig.suptitle(f"{title}\n{metrics_line}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _views(
    sample: dict, threshold: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    anomaly_map = np.asarray(sample["anomaly_map"], dtype=np.float32)
    height, width = anomaly_map.shape[-2:]
    rgb = _load_rgb(sample["image_path"], width, height)
    low, high = np.percentile(anomaly_map, [1, 99])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((anomaly_map - low) / (high - low), 0.0, 1.0)
    import matplotlib.pyplot as plt

    heat = (plt.get_cmap("turbo")(normalized)[..., :3] * 255).astype(np.uint8)
    overlay = (0.42 * rgb + 0.58 * heat).astype(np.uint8)
    mask = sample.get("mask")
    if mask is None or threshold is None or not np.isfinite(threshold):
        return rgb, heat, overlay, None, None
    gt = (np.asarray(mask) > 0)
    prediction = anomaly_map >= threshold
    gt_view = np.zeros_like(rgb)
    gt_view[gt] = (255, 255, 255)
    detection = rgb.copy()
    true_pos = prediction & gt
    false_pos = prediction & ~gt
    false_neg = ~prediction & gt
    detection[true_pos] = (40, 190, 70)
    detection[false_pos] = (220, 40, 40)
    detection[false_neg] = (30, 90, 230)
    # Yellow traces the predicted mask. Cyan traces the ground-truth mask.
    detection[_outline(prediction)] = (255, 220, 0)
    detection[_outline(gt)] = (0, 220, 255)
    return rgb, heat, overlay, gt_view, detection


def _outline(mask: np.ndarray) -> np.ndarray:
    """One-pixel boundary of a binary mask."""

    up = np.zeros_like(mask)
    down = np.zeros_like(mask)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    up[1:] = mask[:-1]
    down[:-1] = mask[1:]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]
    return mask & ~(up & down & left & right)


def _pixel_counts(sample: dict, threshold: float | None) -> dict[str, float] | None:
    mask = sample.get("mask")
    if mask is None or threshold is None or not np.isfinite(threshold):
        return None
    gt = np.asarray(mask) > 0
    prediction = np.asarray(sample["anomaly_map"], dtype=np.float32) >= threshold
    true_pos = int((prediction & gt).sum())
    false_pos = int((prediction & ~gt).sum())
    false_neg = int((~prediction & gt).sum())
    union = true_pos + false_pos + false_neg
    iou = 1.0 if union == 0 else true_pos / union
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    return {"iou": iou, "precision": precision, "recall": recall}


def _load_rgb(path: str, width: int, height: int) -> np.ndarray:
    if path and path != "None" and Path(path).is_file():
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            return np.asarray(resized)
    return np.zeros((height, width, 3), dtype=np.uint8)


def _write_manifest(path: Path, payload: dict) -> None:
    import json

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
