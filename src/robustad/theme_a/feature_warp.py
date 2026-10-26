"""Inverse-rotate the fused PatchCore embedding. CNN weights stay frozen.

The correction is inserted on the tensor ``embedding`` returned by
``PatchcoreModel.generate_embedding``, before ``PatchcoreModel.reshape_embedding``.
That tensor is ``(N, 1536, 64, 64)``: layer2 and layer3 after ``AvgPool2d(3, stride=1, padding=1)``,
layer3 bilinear-resized onto layer2 (``align_corners=False``, the library default),
then concatenated. ``score_fused_feature_inverse`` warps this tensor and then calls
``nearest_neighbors`` and ``compute_anomaly_score`` on the warped rows.
Stored image scores and reweight coefficients are not read.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from robustad.theme_a.warp import sampling_matrix

FEATURE_GRID = 64
FEATURE_CHANNELS = 1536
FEATURE_MODE = "bilinear"
FEATURE_PADDING = "reflection"
FEATURE_ALIGN_CORNERS = False
INSIDE_SLACK = 1e-4


def feature_sample_grid(degrees: float, size: int = FEATURE_GRID, *, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized sampling grid for one rotation around the feature-grid center.

    The 2×3 matrix is the same construction as torchvision's ``rotate``
    (positive degrees are counter-clockwise). The pixel lattice matches
    ``torchvision.transforms._functional_tensor._gen_affine_grid``, which
    ``grid_sample`` consumes with ``align_corners=False``.
    """

    if size < 2:
        raise ValueError(f"特徴格子は 2 以上です: {size}")
    matrix = sampling_matrix(float(degrees), size, size)
    theta = torch.tensor(matrix[:2, :], dtype=dtype, device=device).reshape(1, 2, 3)
    half = 0.5
    base = torch.empty(1, size, size, 3, dtype=dtype, device=device)
    axis = torch.linspace(-size * 0.5 + half, size * 0.5 + half - 1, steps=size, device=device, dtype=dtype)
    base[..., 0] = axis
    base[..., 1] = axis.unsqueeze(-1)
    base[..., 2] = 1
    scale = torch.tensor([0.5 * size, 0.5 * size], dtype=dtype, device=device)
    output = base.view(1, size * size, 3).bmm(theta.transpose(1, 2) / scale)
    return output.view(1, size, size, 2)


def source_pixels(degrees: float, size: int = FEATURE_GRID) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous source ``(x, y)`` of each output cell. ``align_corners=False``.

    ``x`` is the column and ``y`` is the row. Pixel centers of an identity grid
    are the integer indices ``0 .. size-1``.
    """

    grid = feature_sample_grid(degrees, size)
    gx = grid[0, ..., 0]
    gy = grid[0, ..., 1]
    px = (gx + 1) * size / 2 - 0.5
    py = (gy + 1) * size / 2 - 0.5
    return px, py


def reflection_reference_mask(degrees: float, size: int = FEATURE_GRID) -> torch.Tensor:
    """True where bilinear reads outside the input feature grid.

    A cell is inside only when its sample point lies in ``[0, size-1]`` on both
    axes, with ``INSIDE_SLACK`` for floating-point noise. Outside that square,
    ``grid_sample`` padding ``reflection`` supplies the value. The mask depends
    on the angle, not on the image.
    """

    px, py = source_pixels(degrees, size)
    inside = (
        (px >= -INSIDE_SLACK)
        & (px <= (size - 1) + INSIDE_SLACK)
        & (py >= -INSIDE_SLACK)
        & (py <= (size - 1) + INSIDE_SLACK)
    )
    return ~inside


def warp_fused_features(embedding: torch.Tensor, degrees: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear rotation of ``(N, 1536, 64, 64)`` with reflection outside the grid."""

    if embedding.ndim != 4 or tuple(embedding.shape[1:]) != (FEATURE_CHANNELS, FEATURE_GRID, FEATURE_GRID):
        raise ValueError(f"融合特徴は (N, 1536, 64, 64) です: {tuple(embedding.shape)}")
    grid = feature_sample_grid(float(degrees), FEATURE_GRID, device=embedding.device, dtype=embedding.dtype)
    grid = grid.expand(embedding.shape[0], -1, -1, -1)
    warped = F.grid_sample(
        embedding,
        grid,
        mode=FEATURE_MODE,
        padding_mode=FEATURE_PADDING,
        align_corners=FEATURE_ALIGN_CORNERS,
    )
    return warped, reflection_reference_mask(float(degrees)).to(device="cpu")


def score_embedding(torch_model, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
    """Nearest neighbor and a fresh reweighted image score. No stored coefficient is used."""

    batch, _channels, grid_h, grid_w = embedding.shape
    flat = torch_model.reshape_embedding(embedding)
    patch_scores, locations = torch_model.nearest_neighbors(embedding=flat, n_neighbors=1)
    patch_scores = patch_scores.reshape((batch, -1))
    locations = locations.reshape((batch, -1))
    pred_score = torch_model.compute_anomaly_score(patch_scores, locations, flat)
    return {
        "pred_score": pred_score,
        "patch_map": patch_scores.reshape((batch, grid_h, grid_w)),
    }


def fused_embedding(torch_model, normalized: torch.Tensor) -> torch.Tensor:
    """Run the frozen extractor, the existing pooler, and ``generate_embedding``."""

    images = normalized.type(torch_model.memory_bank.dtype)
    features = torch_model.feature_extractor(images)
    pooled = {layer: torch_model.feature_pooler(feature) for layer, feature in features.items()}
    embedding = torch_model.generate_embedding(pooled)
    if tuple(embedding.shape[1:]) != (FEATURE_CHANNELS, FEATURE_GRID, FEATURE_GRID):
        raise RuntimeError(f"融合特徴の形状が 1536×64×64 ではありません: {tuple(embedding.shape)}")
    return embedding


def score_fused_feature_inverse(torch_model, normalized: torch.Tensor, inverse_degrees: float) -> dict[str, torch.Tensor]:
    """Score a normalized batch after inverse-rotating the fused embedding.

    ``normalized`` is ``(N, 3, 512, 512)`` after ImageNet normalization.
    ``inverse_degrees`` is the feature-grid rotation, the opposite of the image
    rotation. Image decisions later use every cell, including reflection references.
    """

    embedding = fused_embedding(torch_model, normalized)
    warped, reflection = warp_fused_features(embedding, inverse_degrees)
    scored = score_embedding(torch_model, warped)
    scored["reflection"] = reflection
    scored["unwarped_embedding"] = embedding
    return scored
