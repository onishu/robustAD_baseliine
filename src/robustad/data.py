"""RobustAD folders as anomalib datasets.

The Hugging Face release stores source-domain train and validation images
together. Appendix Table 9 splits that pool 144/36 for PCB and Metal Parts
(and 148/32 for piled bags), which is an 80/20 split of the 180 normal
images in each train folder. Unsupervised models fit only the normal
training images. Every released test folder is evaluated in full.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.base.image import AnomalibDataset

from robustad.catalog import Domain, Subset

MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"画像ディレクトリがありません: {directory}")
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"画像がありません: {directory}")
    return sorted(images, key=lambda path: path.name)


def find_mask(image_path: Path, mask_dir: Path) -> str:
    for extension in MASK_EXTENSIONS:
        candidate = mask_dir / f"{image_path.stem}{extension}"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"マスクがありません: {image_path.name} in {mask_dir}")


def split_train_val(n_items: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Match anomalib's ``random_split`` chunking, including seed 0.

    anomalib treats a seed of 0 as unset (``if seed``). This helper always
    seeds the generator so run 0 is reproducible.
    """

    ratios = [1.0 - val_ratio, val_ratio]
    lengths = [math.floor(n_items * ratio) for ratio in ratios]
    for index in range(n_items - sum(lengths)):
        lengths[index % len(lengths)] += 1
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_items, generator=generator)
    train_index, val_index = torch.split(permutation, lengths)
    return train_index.tolist(), val_index.tolist()


def _frame(records: list[dict[str, object]], task: str) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    frame.attrs["task"] = task
    return frame


class RobustADDataset(AnomalibDataset):
    def __init__(self, frame: pd.DataFrame, task: str) -> None:
        super().__init__()
        self.samples = frame
        # ``sort_values`` inside the setter can drop ``attrs``.
        self._samples.attrs["task"] = task


def _records_for_split(
    images: list[Path],
    *,
    label_index: int,
    split: str,
    mask_dir: Path | None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for image_path in images:
        mask_path = ""
        if label_index == 1 and mask_dir is not None:
            mask_path = find_mask(image_path, mask_dir)
        records.append(
            {
                "image_path": str(image_path),
                "label_index": label_index,
                "mask_path": mask_path,
                "split": split,
            }
        )
    return records


def make_dataset(frame: pd.DataFrame, task: str) -> RobustADDataset:
    return RobustADDataset(frame, task)


class RobustADDataModule(AnomalibDataModule):
    """DataModule that does not resample the official test folders."""

    def __init__(
        self,
        train_data: RobustADDataset,
        val_data: RobustADDataset,
        test_data: RobustADDataset,
        batch_size: int,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
            val_split_mode="none",
            test_split_mode="none",
        )
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = self._train_data
        self.val_data = self._val_data
        self.test_data = self._test_data

    def setup(self, stage: str | None = None) -> None:
        # Skip AnomalibDataModule's from_test / from_train resampling.
        # Resize from the model pre-processor is still attached here.
        self._setup(stage)
        self._update_augmentations()


def build_train_module(
    subset: Subset,
    data_root: Path,
    *,
    seed: int,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
) -> tuple[RobustADDataModule, dict[str, int]]:
    train_dir = subset.train_dir(data_root)
    normal_images = list_images(train_dir / "normal")
    task = "segmentation" if subset.localization else "classification"
    if val_ratio <= 0:
        train_images = normal_images
        val_images = normal_images[:1]
        n_train = len(train_images)
        n_val = 0
    else:
        train_indices, val_indices = split_train_val(len(normal_images), val_ratio, seed)
        train_images = [normal_images[index] for index in train_indices]
        val_images = [normal_images[index] for index in val_indices]
        n_train = len(train_indices)
        n_val = len(val_indices)
    train_frame = _frame(
        _records_for_split(train_images, label_index=0, split="train", mask_dir=None),
        task,
    )
    val_frame = _frame(
        _records_for_split(val_images, label_index=0, split="val", mask_dir=None),
        task,
    )
    # Fit does not read the test loader. Keep a one-image placeholder.
    module = RobustADDataModule(
        train_data=make_dataset(train_frame, task),
        val_data=make_dataset(val_frame, task),
        test_data=make_dataset(val_frame.iloc[[0]].copy(), task),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    counts = {
        "n_normal_pool": len(normal_images),
        "n_train": n_train,
        "n_val": n_val,
    }
    return module, counts


def build_domain_module(
    subset: Subset,
    data_root: Path,
    domain: Domain,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[RobustADDataModule, dict[str, int]]:
    domain_dir = subset.domain_dir(data_root, domain)
    normal_images = list_images(domain_dir / "normal")
    anomaly_images = list_images(domain_dir / "anomaly")
    mask_dir = domain_dir / "masks" if subset.localization else None
    task = "segmentation" if subset.localization else "classification"
    records = _records_for_split(
        normal_images, label_index=0, split="test", mask_dir=None
    ) + _records_for_split(
        anomaly_images, label_index=1, split="test", mask_dir=mask_dir
    )
    frame = _frame(records, task)
    dataset = make_dataset(frame, task)
    module = RobustADDataModule(
        train_data=dataset,
        val_data=dataset,
        test_data=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    counts = {"n_normal": len(normal_images), "n_anomaly": len(anomaly_images)}
    return module, counts
