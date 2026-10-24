"""Threshold-free metrics and the paper's Average Relative Drop."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = _as_1d(y_true).astype(np.int32)
    y_score = _as_1d(y_score).astype(np.float64)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = _as_1d(y_true).astype(np.int32)
    y_score = _as_1d(y_score).astype(np.float64)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def average_relative_drop(source: float, targets: list[float]) -> float:
    """ARD = mean_i min(0, T_i - S), as in Equation (1) of the paper.

    The published tables report this value in percentage points, so ``source``
    and ``targets`` must already be on that scale (0-100). Improvements over
    the source domain contribute 0, which keeps a gain in one domain from
    hiding a drop in another.
    """

    if not targets or math.isnan(source):
        return float("nan")
    drops = [min(0.0, target - source) for target in targets if not math.isnan(target)]
    if not drops:
        return float("nan")
    return float(sum(drops) / len(drops))


def best_detection(y_true: np.ndarray, y_score: np.ndarray, chunk: int = 1_000_000) -> dict[str, float]:
    """Pick the score threshold with the highest pixel F1.

    Every rank is considered, in chunks, so a defect of a few pixels is not
    skipped by a coarse grid. The search uses the same images that are being
    scored, so the F1 is an inspection operating point. AUROC and AUPR stay
    the threshold-free numbers to compare with the paper.
    """

    labels = _as_1d(y_true).astype(np.uint8, copy=False)
    scores = _as_1d(y_score).astype(np.float32, copy=False)
    total_pos = int(labels.sum())
    empty = {
        "threshold": float("nan"),
        "f1": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
    }
    if total_pos == 0 or total_pos == labels.size:
        return empty
    order = np.argsort(scores, kind="quicksort")
    labels = labels[order]
    scores = scores[order]
    count = labels.size
    best_f1 = -1.0
    best_k = 1
    best_precision = 0.0
    best_recall = 0.0
    true_positives = 0
    end = count
    while end > 0:
        start = max(0, end - chunk)
        # Highest scores in this slice come last in the ascending array.
        block = np.cumsum(labels[start:end][::-1], dtype=np.int64)
        block += true_positives
        predicted = np.arange(count - end + 1, count - start + 1, dtype=np.int64)
        precision = block / predicted
        recall = block / total_pos
        f1 = np.zeros(block.shape, dtype=np.float64)
        valid = (precision + recall) > 0
        f1[valid] = 2.0 * precision[valid] * recall[valid] / (precision[valid] + recall[valid])
        local = int(np.argmax(f1))
        if f1[local] > best_f1:
            best_f1 = float(f1[local])
            best_k = int(predicted[local])
            best_precision = float(precision[local])
            best_recall = float(recall[local])
        true_positives = int(block[-1])
        end = start
    return {
        "threshold": float(scores[count - best_k]),
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray([value for value in values if not math.isnan(value)], dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1))
