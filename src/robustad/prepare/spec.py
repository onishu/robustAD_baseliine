"""Identifiers for preparation phase P after the diagnostic close-out.

Improvement and equivalence margins stay unset. They are reported as
undecided and do not block the written rules below.
"""

from __future__ import annotations

RUN_ID = "p0"
STATUS = "p_closed"

# Image-level operating rule. PatchCore keeps its reweighted score.
# PaDiM keeps the maximum of the sigma-4 map. Both are pred_score.
PRIMARY_IMAGE_SCORE = "reproduction_image"
PRIMARY_SMOOTH_SIGMA = 4
PRIMARY_COMPONENT_STATISTIC = "max"
TARGET_IMAGE_FPR = 0.05
TARGET_PATCH_FPR = 0.05
IMPROVEMENT_MARGIN = None
EQUIVALENCE_MARGIN = None
VERDICT_UNDECIDED = "未判定"
RESEARCH_SPLIT = {"seed": 0, "val_ratio": 0.2, "n_fit": 144, "n_calibration": 36}
PRIMARY_VALID_REGION = "full_512"

# Reproduction-compatible names. These describe the existing benchmark, not a
# new model.
REPRODUCTION_IMAGE_SCORE = "reproduction_image"
DIAG_MAX_PATCH = "diag_max_patch"
DIAG_MAX_UPSAMPLED = "diag_max_upsampled"
DIAG_MAX_SMOOTHED = "diag_max_smoothed"

# anomalib AnomalyMapGenerator default. Kernel radius is (kernel-1)/2.
SMOOTH_SIGMA_DEFAULT = 4
SMOOTH_KERNEL = 2 * int(4.0 * SMOOTH_SIGMA_DEFAULT + 0.5) + 1
SMOOTH_RADIUS = (SMOOTH_KERNEL - 1) // 2

# Sensitivity set. Not a search over test AUROC.
SMOOTH_SIGMAS = (0, 4, 8)
BORDER_MARGINS = (0, 16, 32, 64)

PROTECTED_OUTPUTS = ("results", "results_a1", "results_tv")

# Domains named in the preparation plan. Folder names come from the catalog.
DIAGNOSTIC_DOMAINS = (
    {"dataset": "metal_parts", "domain": "SD", "role": "source"},
    {"dataset": "metal_parts", "domain": "Scale", "role": "geometry"},
    {"dataset": "pcb", "domain": "SD", "role": "source"},
    {"dataset": "pcb", "domain": "Rotation", "role": "geometry"},
    {"dataset": "pcb", "domain": "Position", "role": "geometry"},
    {"dataset": "pcb", "domain": "Shadows", "role": "photometric_control"},
    {"dataset": "pcb", "domain": "Lighting", "role": "photometric_control"},
)

PCB_DEFECTS = ("Missing Component", "Scratch", "Soldering Melt")
METAL_DEFECTS = (
    "Chipping_&_Porosity",
    "Multiple_Chipping",
    "Multiple_Porosity",
    "Chipping",
    "Porosity",
)
