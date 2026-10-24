"""RobustAD subset layout and the domain names used in the paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Domain:
    """One evaluation domain. ``SD`` is the source-domain test split."""

    name: str
    dirname: str


@dataclass(frozen=True)
class Subset:
    key: str
    title: str
    root_name: str
    train_dirname: str
    localization: bool
    domains: tuple[Domain, ...]

    def root(self, data_root: Path) -> Path:
        return data_root / self.root_name

    def train_dir(self, data_root: Path) -> Path:
        return self.root(data_root) / self.train_dirname

    def domain_dir(self, data_root: Path, domain: Domain) -> Path:
        return self.root(data_root) / domain.dirname


# Folder names follow the Hugging Face release. Domain labels follow Table 2
# of Pemula et al., CVPRW 2025, matched by the counts in the appendix.
SUBSETS: dict[str, Subset] = {
    "metal_parts": Subset(
        key="metal_parts",
        title="Metal Parts",
        root_name="MetalParts",
        train_dirname="metal_parts_data_dir_train",
        localization=True,
        domains=(
            Domain("SD", "metal_parts_data_dir_test0"),
            Domain("Lighting", "metal_parts_data_dir_test1"),
            Domain("Position", "metal_parts_data_dir_test2"),
            Domain("Rotation", "metal_parts_data_dir_test3"),
            Domain("Scale", "metal_parts_data_dir_test4"),
            Domain("BG1", "metal_parts_data_dir_test5"),
            Domain("BG2", "metal_parts_data_dir_test6"),
        ),
    ),
    "piled_bags": Subset(
        key="piled_bags",
        title="Pile of Packets",
        root_name="PiledBags",
        train_dirname="piled_bags_data_dir_train",
        localization=False,
        domains=(
            Domain("SD", "piled_bags_data_dir_test0"),
            Domain("Lighting", "piled_bags_data_dir_test1"),
            Domain("Background", "piled_bags_data_dir_test2"),
            Domain("Rot+pos", "piled_bags_data_dir_test3"),
            Domain("Scale", "piled_bags_data_dir_test4"),
            Domain("Shadows", "piled_bags_data_dir_test5"),
        ),
    ),
    "pcb": Subset(
        key="pcb",
        title="PCB",
        root_name="PCB",
        train_dirname="pcb_data_dir_train",
        localization=True,
        domains=(
            Domain("SD", "pcb_data_dir_test0"),
            Domain("Lighting", "pcb_data_dir_test1"),
            Domain("AWB", "pcb_data_dir_test2"),
            Domain("Rotation", "pcb_data_dir_test3"),
            Domain("Position", "pcb_data_dir_test4"),
            Domain("Shadows", "pcb_data_dir_test5"),
        ),
    ),
}

# Image-level AUROC (%) reported in Table 2. Cell values are mean ± std over 5 runs.
PAPER_IMAGE_AUROC: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "metal_parts": {
        "padim": {
            "SD": (85.67, 2.03),
            "Lighting": (96.31, 0.85),
            "Position": (63.28, 6.61),
            "Rotation": (82.28, 6.41),
            "Scale": (41.01, 3.89),
            "BG1": (69.43, 9.62),
            "BG2": (68.15, 11.97),
        },
        "patchcore": {
            "SD": (87.92, 0.46),
            "Lighting": (96.59, 0.09),
            "Position": (89.46, 0.32),
            "Rotation": (88.27, 0.18),
            "Scale": (80.88, 0.17),
            "BG1": (74.87, 0.60),
            "BG2": (83.50, 1.16),
        },
    },
    "piled_bags": {
        "padim": {
            "SD": (98.80, 0.70),
            "Lighting": (65.70, 10.66),
            "Background": (47.67, 1.78),
            "Rot+pos": (53.41, 3.60),
            "Scale": (51.34, 4.79),
            "Shadows": (77.36, 12.68),
        },
        "patchcore": {
            "SD": (99.98, 0.01),
            "Lighting": (95.89, 0.15),
            "Background": (51.23, 0.22),
            "Rot+pos": (61.74, 0.75),
            "Scale": (75.34, 0.24),
            "Shadows": (84.31, 0.58),
        },
    },
    "pcb": {
        "padim": {
            "SD": (99.99, 0.02),
            "Lighting": (10.41, 7.40),
            "AWB": (6.96, 7.07),
            "Rotation": (31.79, 12.98),
            "Position": (40.88, 9.32),
            "Shadows": (60.68, 10.13),
        },
        "patchcore": {
            "SD": (96.32, 0.14),
            "Lighting": (73.93, 0.65),
            "AWB": (81.59, 0.49),
            "Rotation": (68.53, 0.87),
            "Position": (67.18, 0.49),
            "Shadows": (46.69, 0.50),
        },
    },
}

PAPER_SUMMARY: dict[str, dict[str, dict[str, float]]] = {
    "metal_parts": {
        "padim": {"mean": 72.30, "std": 17.99, "ard": -17.37},
        "patchcore": {"mean": 85.93, "std": 6.94, "ard": -4.09},
    },
    "piled_bags": {
        "padim": {"mean": 65.72, "std": 19.56, "ard": -39.70},
        "patchcore": {"mean": 78.08, "std": 19.14, "ard": -26.28},
    },
    "pcb": {
        "padim": {"mean": 41.79, "std": 34.76, "ard": -69.85},
        "patchcore": {"mean": 72.38, "std": 16.50, "ard": -28.74},
    },
}
