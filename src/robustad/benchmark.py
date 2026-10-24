"""Train PaDiM and PatchCore on RobustAD and compare with Table 2."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import Normalize, Resize

from robustad.catalog import DOMAIN_FACTOR, FACTOR_ORDER, PAPER_IMAGE_AUROC, PAPER_SUMMARY, SUBSETS
from robustad.data import build_domain_module, build_train_module
from robustad.metrics import average_relative_drop, best_detection, mean_std, safe_aupr, safe_auroc
from robustad.visualize import render_domain, save_result_figures

LOGGER = logging.getLogger("robustad")

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = {"padim": 8, "patchcore": 2}
# The first reproduction uses anomalib 2.7's bare names, which load
# resnet18.a1_in1k and wide_resnet50_2.racm_in1k. torchvision is the
# ImageNet-1K V1 comparison stored in results_tv.
WEIGHT_SETS = {
    "timm": {
        "padim": "resnet18",
        "patchcore": "wide_resnet50_2",
        "note": "anomalib 2.7 の既定（resnet18.a1_in1k / wide_resnet50_2.racm_in1k）",
        "tags": ("a1_in1k", "racm_in1k"),
    },
    "torchvision": {
        "padim": "resnet18.tv_in1k",
        "patchcore": "wide_resnet50_2.tv_in1k",
        "note": "torchvision ImageNet-1K V1（timm の tv_in1k）",
        "tags": ("tv_in1k",),
    },
}


def paper_pre_processor(image_size: int):
    """Resize every sample to 512 and normalize only the RGB image.

    anomalib's stock ``PreProcessor`` applies one transform to the image and
    the mask together. ImageNet normalization would shift the mask, so the
    mask is resized and then binarized again.
    """

    from anomalib.pre_processing import PreProcessor

    class PaperPreProcessor(PreProcessor):
        def __init__(self) -> None:
            super().__init__(transform=Resize((image_size, image_size), antialias=True))
            self._normalize = Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

        def _transform_batch(self, batch) -> None:
            if batch.gt_mask is not None:
                batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)
                batch.gt_mask = (batch.gt_mask > 0).to(dtype=torch.uint8)
            else:
                batch.image = self.transform(batch.image)
            batch.image = self._normalize(batch.image)

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
            del trainer, pl_module, batch_idx
            self._transform_batch(batch)

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
            del trainer, pl_module, batch_idx
            self._transform_batch(batch)

        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0) -> None:
            del trainer, pl_module, batch_idx, dataloader_idx
            self._transform_batch(batch)

        def on_predict_batch_start(
            self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0
        ) -> None:
            del trainer, pl_module, batch_idx, dataloader_idx
            self._transform_batch(batch)

    return PaperPreProcessor()


def accelerate_patchcore_coreset() -> None:
    """Run the same k-center loop on GPU.

    anomalib's coreset selection is a Python loop over every selected center.
    At 512 input and a 0.1 sampling ratio that loop is hundreds of thousands of
    iterations. Moving the already-projected features to CUDA does not change
    the selection rule.
    """

    from tqdm import tqdm

    import anomalib.models.components.sampling.k_center_greedy as kcenter

    if getattr(kcenter.KCenterGreedy, "_robustad_gpu", False):
        return

    def select_coreset_idxs(self) -> list[int]:
        # Same selection rule as anomalib 2.7, including the random start
        # center. Distance updates run on CUDA when it is available.
        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
        self.reset_distances()
        index = torch.randint(
            high=self.n_observations, size=(1,), device=self.features.device
        ).squeeze()
        if torch.cuda.is_available() and self.features.device.type != "cuda":
            self.features = self.features.cuda(non_blocking=True)
            index = index.cuda()
        selected = [index]
        for _ in tqdm(
            range(self.coreset_size - 1),
            desc="Selecting Coreset Indices.",
            mininterval=30,
        ):
            self.update_distances(cluster_center=index)
            index = self.get_new_idx()
            self.min_distances.scatter_(0, index.unsqueeze(0).unsqueeze(1), 0.0)
            selected.append(index)
        return torch.stack(selected).cpu().tolist()

    kcenter.KCenterGreedy.select_coreset_idxs = select_coreset_idxs
    kcenter.KCenterGreedy._robustad_gpu = True


def build_model(model_name: str, image_size: int, weights: str = "timm"):
    pre_processor = paper_pre_processor(image_size)
    spec = WEIGHT_SETS[weights]
    backbone = spec[model_name]
    if model_name == "padim":
        from anomalib.models import Padim

        model = Padim(
            backbone=backbone,
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
            n_features=100,
            pre_processor=pre_processor,
            evaluator=False,
            visualizer=False,
        )
    elif model_name == "patchcore":
        from anomalib.models import Patchcore

        accelerate_patchcore_coreset()
        model = Patchcore(
            backbone=backbone,
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
            pre_processor=pre_processor,
            evaluator=False,
            visualizer=False,
        )
    else:
        raise ValueError(f"未知のモデルです: {model_name}")
    tag, url = _pretrained_fields(model)
    LOGGER.info("backbone %s tag=%s url=%s", backbone, tag, url)
    if not any(expected in tag or expected in url for expected in spec["tags"]):
        raise RuntimeError(f"想定した重みではありません: tag={tag} url={url} expected={spec['tags']}")
    return model


def _pretrained_fields(model) -> tuple[str, str]:
    extractor = model.model.feature_extractor.feature_extractor
    cfg = getattr(extractor, "pretrained_cfg", None)
    if isinstance(cfg, dict):
        return str(cfg.get("tag", "")), str(cfg.get("url", ""))
    return str(getattr(cfg, "tag", "")), str(getattr(cfg, "url", ""))


def make_engine(output_dir: Path):
    from anomalib.callbacks import ModelCheckpoint
    from anomalib.engine import Engine

    kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "max_epochs": 1,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "logger": False,
        "enable_model_summary": False,
        "default_root_dir": str(output_dir / "lightning"),
        # Keep Engine from writing a full memory-bank checkpoint on every run.
        "callbacks": [ModelCheckpoint(save_top_k=0)],
    }
    accepted = set(inspect.signature(Engine.__init__).parameters)
    if "self" in accepted:
        accepted.remove("self")
    # Trainer-style engines accept **kwargs even when they are not listed.
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in inspect.signature(Engine.__init__).parameters.values()):
        return Engine(**kwargs)
    return Engine(**{key: value for key, value in kwargs.items() if key in accepted})


def predict(engine, model, datamodule):
    signature = inspect.signature(engine.predict)
    kwargs = {"model": model}
    if "datamodule" in signature.parameters:
        kwargs["datamodule"] = datamodule
    if "return_predictions" in signature.parameters:
        kwargs["return_predictions"] = True
    outputs = engine.predict(**kwargs)
    if not outputs:
        raise RuntimeError("予測結果が空です。")
    return outputs


def _as_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return None


def _resize_mask(path: str | None, height: int, width: int) -> torch.Tensor:
    if not path or path == "None":
        return torch.zeros((height, width), dtype=torch.int32)
    with Image.open(path) as image:
        array = np.array(image.convert("L"))
    array = torch.as_tensor(array, dtype=torch.float32)
    resized = F.interpolate(array[None, None], size=(height, width), mode="nearest")
    return (resized[0, 0] > 0).to(torch.int32)


def score_domain(outputs, *, localization: bool, keep_samples: bool) -> tuple[dict[str, float], list[dict]]:
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    maps: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    samples: list[dict] = []
    for batch in outputs:
        score = _as_tensor(getattr(batch, "pred_score", None))
        label = _as_tensor(getattr(batch, "gt_label", None))
        if score is None or label is None:
            raise RuntimeError(f"pred_score または gt_label がバッチにありません: {type(batch)!r}")
        score = score.reshape(-1).float()
        label = label.reshape(-1).to(torch.int32)
        scores.append(score)
        labels.append(label)
        anomaly_map = _as_tensor(getattr(batch, "anomaly_map", None))
        if anomaly_map is None:
            raise RuntimeError("anomaly_map がバッチにありません。")
        if anomaly_map.ndim == 4:
            anomaly_map = anomaly_map[:, 0]
        height, width = anomaly_map.shape[-2:]
        pixel_mask = _batch_masks(batch, anomaly_map.shape[0], height, width) if localization else None
        if localization:
            if pixel_mask is None:
                raise RuntimeError("マスクパスも gt_mask もありません。")
            maps.append(anomaly_map.reshape(-1).float())
            masks.append(pixel_mask.reshape(-1))
        if keep_samples:
            paths = [str(path) for path in list(getattr(batch, "image_path", []) or [])]
            for index in range(score.shape[0]):
                samples.append(
                    {
                        "image_path": paths[index] if index < len(paths) else "",
                        "label": int(label[index]),
                        "score": float(score[index]),
                        "anomaly_map": anomaly_map[index].float().numpy().astype(np.float16),
                        "mask": None if pixel_mask is None else pixel_mask[index].numpy().astype(np.uint8),
                    }
                )

    image_score = torch.cat(scores).numpy()
    image_label = torch.cat(labels).numpy()
    image_detection = best_detection(image_label, image_score)
    result = {
        "image_auroc": safe_auroc(image_label, image_score),
        "image_aupr": safe_aupr(image_label, image_score),
        "image_f1": image_detection["f1"],
        "image_precision": image_detection["precision"],
        "image_recall": image_detection["recall"],
        "image_threshold": image_detection["threshold"],
    }
    if localization:
        pixel_score = torch.cat(maps).numpy()
        pixel_label = torch.cat(masks).numpy()
        result["pixel_auroc"] = safe_auroc(pixel_label, pixel_score)
        result["pixel_aupr"] = safe_aupr(pixel_label, pixel_score)
        detection = best_detection(pixel_label, pixel_score)
        result["pixel_f1"] = detection["f1"]
        result["pixel_precision"] = detection["precision"]
        result["pixel_recall"] = detection["recall"]
        result["pixel_threshold"] = detection["threshold"]
    return result, samples


def _batch_masks(batch, count: int, height: int, width: int) -> torch.Tensor | None:
    mask_paths = list(getattr(batch, "mask_path", []) or [])
    if len(mask_paths) == count:
        return torch.stack([_resize_mask(path, height, width) for path in mask_paths])
    gt_mask = _as_tensor(getattr(batch, "gt_mask", None))
    if gt_mask is None:
        return None
    if gt_mask.ndim == 4:
        gt_mask = gt_mask[:, 0]
    if gt_mask.shape[-2:] != (height, width):
        gt_mask = F.interpolate(gt_mask[:, None].float(), size=(height, width), mode="nearest")[:, 0]
    return (gt_mask > 0).to(torch.int32)


def run_path(output_dir: Path, dataset: str, model_name: str, seed: int) -> Path:
    return output_dir / "runs" / f"{dataset}__{model_name}__seed{seed}.json"


def run_once(
    *,
    dataset: str,
    model_name: str,
    seed: int,
    data_root: Path,
    output_dir: Path,
    image_size: int,
    val_ratio: float,
    num_workers: int,
    save_checkpoint: bool,
    force: bool,
    save_gallery: bool,
    weights: str,
) -> dict:
    destination = run_path(output_dir, dataset, model_name, seed)
    if destination.exists() and not force:
        LOGGER.info("既存の結果を使います: %s", destination.name)
        return json.loads(destination.read_text(encoding="utf-8"))

    from lightning.pytorch import seed_everything

    seed_everything(seed, workers=True)
    subset = SUBSETS[dataset]
    batch_size = BATCH_SIZE[model_name]
    train_module, train_counts = build_train_module(
        subset,
        data_root,
        seed=seed,
        val_ratio=val_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    LOGGER.info(
        "%s / %s / seed %d : 正常画像 %d 枚のうち %d 枚で学習（検証 %d 枚はメモリバンクに入れない）",
        subset.title,
        model_name,
        seed,
        train_counts["n_normal_pool"],
        train_counts["n_train"],
        train_counts["n_val"],
    )
    model = build_model(model_name, image_size, weights)
    engine = make_engine(output_dir)
    engine.fit(model=model, datamodule=train_module)

    domains: dict[str, dict] = {}
    for domain in subset.domains:
        domain_module, counts = build_domain_module(
            subset,
            data_root,
            domain,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        outputs = predict(engine, model, domain_module)
        metrics, samples = score_domain(
            outputs,
            localization=subset.localization,
            keep_samples=save_gallery,
        )
        metrics.update(counts)
        domains[domain.name] = metrics
        pixel_note = ""
        if "pixel_auroc" in metrics:
            pixel_note = (
                f"  pixel AUROC {metrics['pixel_auroc'] * 100.0:6.2f}"
                f"  F1 {metrics['pixel_f1'] * 100.0:6.2f}"
            )
        LOGGER.info(
            "  %-12s image AUROC %6.2f  F1 %6.2f%s",
            domain.name,
            metrics["image_auroc"] * 100.0,
            metrics["image_f1"] * 100.0,
            pixel_note,
        )
        if save_gallery and samples:
            gallery = output_dir / "vis" / f"{dataset}__{model_name}__seed{seed}" / domain.name
            metrics_line = (
                f"image AUROC {metrics['image_auroc'] * 100.0:.2f}"
                f"   image F1 {metrics['image_f1'] * 100.0:.2f}"
            )
            if "pixel_auroc" in metrics:
                metrics_line += (
                    f"   pixel AUROC {metrics['pixel_auroc'] * 100.0:.2f}"
                    f"   pixel AUPR {metrics['pixel_aupr'] * 100.0:.2f}"
                    f"   pixel F1 {metrics['pixel_f1'] * 100.0:.2f}"
                    f"（precision {metrics['pixel_precision'] * 100.0:.2f}"
                    f" / recall {metrics['pixel_recall'] * 100.0:.2f}）"
                )
            render_domain(
                samples,
                gallery,
                title=f"{subset.title} / {model_name} / {domain.name} / seed {seed}",
                metrics_line=metrics_line,
                threshold=metrics.get("pixel_threshold"),
            )
            LOGGER.info("  画像ログ: %s", gallery)
        del samples

    payload = {
        "dataset": dataset,
        "model": model_name,
        "backbone": WEIGHT_SETS[weights][model_name],
        "weights": WEIGHT_SETS[weights]["note"],
        "seed": seed,
        "image_size": image_size,
        "val_ratio": val_ratio,
        "train": train_counts,
        "domains": domains,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if save_checkpoint:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            model.state_dict(),
            checkpoint_dir / f"{dataset}__{model_name}__seed{seed}.pt",
        )
    return payload


def summarize(output_dir: Path) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output_dir / "runs").glob("*.json"))]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["dataset"], run["model"]), []).append(run)

    summary: dict[str, dict] = {}
    for (dataset, model_name), items in grouped.items():
        subset = SUBSETS[dataset]
        domain_stats = {}
        for domain in subset.domains:
            scores = [
                item["domains"][domain.name]["image_auroc"] * 100.0
                for item in items
                if domain.name in item["domains"]
            ]
            mean, std = mean_std(scores)
            paper_mean, paper_std = PAPER_IMAGE_AUROC[dataset][model_name][domain.name]
            entry = {
                "n": len(scores),
                "auroc_mean": mean,
                "auroc_std": std,
                "paper_mean": paper_mean,
                "paper_std": paper_std,
                "delta": mean - paper_mean if not math.isnan(mean) else float("nan"),
            }
            for metric in (
                "image_f1",
                "image_precision",
                "image_recall",
                "pixel_auroc",
                "pixel_aupr",
                "pixel_f1",
                "pixel_precision",
                "pixel_recall",
            ):
                pixel_scores = [
                    item["domains"][domain.name][metric] * 100.0
                    for item in items
                    if metric in item["domains"].get(domain.name, {})
                ]
                if pixel_scores:
                    pixel_mean, pixel_std = mean_std(pixel_scores)
                    entry[f"{metric}_mean"] = pixel_mean
                    entry[f"{metric}_std"] = pixel_std
            domain_stats[domain.name] = entry
        domain_means = [domain_stats[domain.name]["auroc_mean"] for domain in subset.domains]
        overall_mean, overall_std = mean_std(domain_means)
        source = domain_stats["SD"]["auroc_mean"]
        targets = [domain_stats[domain.name]["auroc_mean"] for domain in subset.domains if domain.name != "SD"]
        ard = average_relative_drop(source, targets)
        source_pixel = domain_stats["SD"].get("pixel_auroc_mean", float("nan"))
        pixel_targets = [
            domain_stats[domain.name]["pixel_auroc_mean"]
            for domain in subset.domains
            if domain.name != "SD" and "pixel_auroc_mean" in domain_stats[domain.name]
        ]
        ard_pixel = average_relative_drop(source_pixel, pixel_targets)
        for domain in subset.domains:
            entry = domain_stats[domain.name]
            if domain.name == "SD":
                entry["image_drop"] = 0.0
                entry["pixel_drop"] = 0.0 if "pixel_auroc_mean" in entry else float("nan")
                continue
            entry["image_drop"] = min(0.0, entry["auroc_mean"] - source)
            if "pixel_auroc_mean" in entry and not math.isnan(source_pixel):
                entry["pixel_drop"] = min(0.0, entry["pixel_auroc_mean"] - source_pixel)
        paper = PAPER_SUMMARY[dataset][model_name]
        summary.setdefault(dataset, {})[model_name] = {
            "seeds": sorted(item["seed"] for item in items),
            "backbone": items[0].get("backbone", ""),
            "weights": items[0].get("weights", ""),
            "domains": domain_stats,
            "mean": overall_mean,
            "std": overall_std,
            "source": source,
            "ard": ard,
            "source_pixel": source_pixel,
            "ard_pixel": ard_pixel,
            "paper_mean": paper["mean"],
            "paper_std": paper["std"],
            "paper_ard": paper["ard"],
        }
    destination = output_dir / "summary.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _points(value: float, width: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{'—':>{width}}"
    return f"{value:{width}.2f}"


def _metric_cell(domain: dict, name: str) -> str:
    mean = domain.get(f"{name}_mean", domain.get(name, float("nan")))
    std = domain.get(f"{name}_std", 0.0)
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return f"{'—':>16}"
    if f"{name}_std" not in domain and f"{name}_mean" not in domain:
        return f"{mean:16.2f}"
    return f"{mean:6.2f} ± {std:5.2f}"


def format_summary(summary: dict, baseline: dict | None = None) -> str:
    lines = [
        "各撮影条件の image AUROC / image F1 / pixel AUROC / pixel F1（%）。",
        "F1 は、そのテスト条件の中で F1 が最大になるしきい値。画像と画素で別々に決める。",
        "ARD はターゲットについて min(0, T - S) の平均。image drop / pixel drop はその各項。",
        "Pile of Packets にマスクはないので、画素指標と画素 ARD はない。",
        "",
    ]
    for dataset, models in summary.items():
        title = SUBSETS[dataset].title
        for model_name, stats in models.items():
            lines.append(
                f"## {title} / {model_name}   seeds={stats['seeds']}   {stats.get('backbone', '')}"
            )
            if stats.get("weights"):
                lines.append(stats["weights"])
            header = f"{'domain':<12} {'ours':>16} {'paper':>16} {'delta':>8}"
            if baseline:
                header += f" {'vs old':>8}"
            lines.append(header)
            for domain_name, domain in stats["domains"].items():
                ours = f"{domain['auroc_mean']:6.2f} ± {domain['auroc_std']:5.2f}"
                paper = f"{domain['paper_mean']:6.2f} ± {domain['paper_std']:5.2f}"
                delta = f"{domain['delta']:+6.2f}"
                row = f"{domain_name:<12} {ours:>16} {paper:>16} {delta:>8}"
                if baseline:
                    previous = _baseline_auroc(baseline, dataset, model_name, domain_name)
                    row += f" {domain['auroc_mean'] - previous:+8.2f}" if not math.isnan(previous) else f" {'':>8}"
                lines.append(row)
            lines.append(
                f"{'mean':<12} {stats['mean']:6.2f} ± {stats['std']:5.2f}   "
                f"paper {stats['paper_mean']:.2f} ± {stats['paper_std']:.2f}"
            )
            lines.append(
                f"{'SD, ARD':<12} ({stats['source']:.2f}, {stats['ard']:.2f})   "
                f"paper ({PAPER_IMAGE_AUROC[dataset][model_name]['SD'][0]:.2f}, {stats['paper_ard']:.2f})"
            )
            lines.append(
                f"{'domain':<12} {'factor':<10} {'image AUROC':>16} {'image F1':>16} {'image drop':>10}"
                f" {'pixel AUROC':>16} {'pixel F1':>16} {'pixel drop':>10}"
            )
            for domain_name, domain in stats["domains"].items():
                lines.append(
                    f"{domain_name:<12} {DOMAIN_FACTOR.get(domain_name, ''):<10} "
                    f"{_metric_cell(domain, 'auroc'):>16} {_metric_cell(domain, 'image_f1'):>16} "
                    f"{_points(domain.get('image_drop', float('nan')), 10)} "
                    f"{_metric_cell(domain, 'pixel_auroc'):>16} {_metric_cell(domain, 'pixel_f1'):>16} "
                    f"{_points(domain.get('pixel_drop', float('nan')), 10)}"
                )
            if not math.isnan(stats.get("ard_pixel", float("nan"))):
                lines.append(
                    f"{'pixel ARD':<12} {stats['ard_pixel']:.2f}   "
                    f"source pixel AUROC {stats['source_pixel']:.2f}"
                )
            lines.append("")
    lines.extend(_factor_lines(summary))
    return "\n".join(lines)


def _factor_lines(summary: dict) -> list[str]:
    grouped: dict[str, list[tuple[str, str, str, dict]]] = {factor: [] for factor in FACTOR_ORDER}
    for dataset, models in summary.items():
        for model_name, stats in models.items():
            for domain_name, domain in stats["domains"].items():
                factor = DOMAIN_FACTOR.get(domain_name, "")
                if factor in grouped:
                    grouped[factor].append((SUBSETS[dataset].title, model_name, domain_name, domain))
    lines = ["## 変動要因", ""]
    for factor, rows in grouped.items():
        if not rows:
            continue
        lines.append(f"### {factor}")
        lines.append(
            f"{'dataset':<18} {'model':<12} {'domain':<12} "
            f"{'imgAUROC':>8} {'imgF1':>8} {'imgDrop':>8} {'pixAUROC':>8} {'pixF1':>8} {'pixDrop':>8}"
        )
        for title, model_name, domain_name, domain in rows:
            def cell(name: str) -> str:
                value = domain.get(f"{name}_mean", float("nan"))
                return f"{value:8.2f}" if not math.isnan(value) else f"{'—':>8}"

            lines.append(
                f"{title:<18} {model_name:<12} {domain_name:<12} "
                f"{cell('auroc')} {cell('image_f1')} {_points(domain.get('image_drop', float('nan')), 8)} "
                f"{cell('pixel_auroc')} {cell('pixel_f1')} {_points(domain.get('pixel_drop', float('nan')), 8)}"
            )
        lines.append("")
    return lines


def _baseline_auroc(baseline: dict, dataset: str, model_name: str, domain_name: str) -> float:
    try:
        return float(baseline[dataset][model_name]["domains"][domain_name]["auroc_mean"])
    except KeyError:
        return float("nan")


def load_baseline(output_dir: Path, baseline_dir: Path | None) -> dict | None:
    if baseline_dir is None:
        candidate = output_dir.parent / "results"
        if candidate.resolve() == output_dir.resolve():
            return None
        baseline_dir = candidate
    path = baseline_dir / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_reports(output_dir: Path, baseline_dir: Path | None) -> str:
    summary = summarize(output_dir)
    baseline = load_baseline(output_dir, baseline_dir)
    save_result_figures(summary, output_dir / "figures", baseline)
    text = format_summary(summary, baseline)
    (output_dir / "summary.txt").write_text(text, encoding="utf-8")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"K:\RobustAD"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="+", choices=["padim", "patchcore"], default=["padim", "patchcore"])
    parser.add_argument("--datasets", nargs="+", choices=sorted(SUBSETS), default=["piled_bags", "metal_parts", "pcb"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--weights",
        choices=sorted(WEIGHT_SETS),
        default="timm",
        help="timm は1つ目の実験（anomalib 既定）。torchvision は ImageNet-1K V1。",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-vis", action="store_true", help="異常ヒートマップの画像ログを書かない")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="旧重みの summary.json があるディレクトリ。省略時は隣の results/",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        if not args.data_root.is_dir():
            raise FileNotFoundError(f"データセットが見つかりません: {args.data_root}")
        # Seed 0 completes a full table before the remaining repeats.
        for seed in args.seeds:
            for model_name in args.models:
                for dataset in args.datasets:
                    run_once(
                        dataset=dataset,
                        model_name=model_name,
                        seed=seed,
                        data_root=args.data_root,
                        output_dir=output_dir,
                        image_size=args.image_size,
                        val_ratio=args.val_ratio,
                        num_workers=args.num_workers,
                        save_checkpoint=args.save_checkpoint,
                        force=args.force,
                        save_gallery=seed == args.seeds[0] and not args.no_vis,
                        weights=args.weights,
                    )
                    text = write_reports(output_dir, args.baseline_dir)
    else:
        text = write_reports(output_dir, args.baseline_dir)
    print(text)


if __name__ == "__main__":
    main()
