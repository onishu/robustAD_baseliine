"""Thresholds from normal scores only.

A threshold estimated at a target false-positive rate on normal images is not
assumed to produce that rate on a shifted test set. Patch-level rates are not
reported as connected-component rates.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta

from robustad.prepare.spec import (
    TARGET_IMAGE_FPR,
    TARGET_PATCH_FPR,
    VERDICT_UNDECIDED,
)


def empirical_threshold(normal_scores: np.ndarray, target_fpr: float) -> float:
    """Quantile at ``1 - target_fpr``. For 0.05 this is the 95% quantile.

    ``np.quantile(..., method="linear")`` interpolates between order
    statistics, so the threshold need not equal a sample. A later comparison
    treats every score greater than or equal to that threshold as anomalous,
    including exact ties. With ``n`` normals, rates below ``1/n`` are not
    identifiable; the caller decides whether that candidate is usable.
    """

    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"目標 FPR は (0, 1) です: {target_fpr}")
    scores = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("正常スコアが空です。")
    if not np.isfinite(scores).all():
        raise ValueError("正常スコアに NaN または Inf があります。")
    return float(np.quantile(scores, 1.0 - target_fpr, method="linear"))


def realized_fpr(normal_scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Count scores greater than or equal to ``threshold``. Ties are positives."""

    scores = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return {"n": 0, "n_false": 0, "fpr": float("nan")}
    n_false = int((finite >= threshold).sum())
    return {"n": int(finite.size), "n_false": n_false, "fpr": float(n_false / finite.size)}


def detection_counts(anomaly_scores: np.ndarray, threshold: float) -> dict[str, float]:
    scores = np.asarray(anomaly_scores, dtype=np.float64).reshape(-1)
    finite = scores[np.isfinite(scores)]
    n_hit = int((finite >= threshold).sum()) if finite.size else 0
    return {
        "n": int(finite.size),
        "n_detected": n_hit,
        "rate": float(n_hit / finite.size) if finite.size else float("nan"),
        "n_nonfinite": int(scores.size - finite.size),
    }


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval. Zero trials has no interval."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"不正な計数です: {successes}/{trials}")
    if trials == 0:
        return float("nan"), float("nan")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def proportion_halfwidth(trials: int, rate: float = 0.5) -> float:
    """Normal-approximation half-width of a single proportion.

    This is a planning figure, not a test result. ``rate=0.5`` is the widest
    binomial variance.
    """

    if trials <= 0:
        return float("nan")
    variance = rate * (1.0 - rate) / trials
    return float(1.96 * math.sqrt(variance))


def fit_calibration_overlap(fit_ids: list[str], calibration_ids: list[str]) -> list[str]:
    return sorted(set(fit_ids) & set(calibration_ids))


def identifiable_fprs(n_normal: int, numerators: tuple[int, ...] = (1, 2, 5)) -> list[dict[str, float]]:
    rows = []
    for numerator in numerators:
        if numerator >= n_normal:
            continue
        rows.append(
            {
                "false_positives_on_calibration": numerator,
                "target_fpr": numerator / n_normal,
                "expected_events": float(numerator),
            }
        )
    return rows


def pool_patch_scores(patch_maps: list[np.ndarray]) -> np.ndarray:
    """Concatenate every native-grid cell from calibration images.

    The unit is one float32 score per cell, including cells on the image
    border. Fit images are not part of this pool. A smoothed pixel is not a
    calibration unit.
    """

    if not patch_maps:
        raise ValueError("校正用のパッチスコアが空です。")
    pieces = []
    for patch_map in patch_maps:
        array = np.asarray(patch_map, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"パッチ格子は2次元です: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("パッチスコアに NaN または Inf があります。")
        pieces.append(array.reshape(-1))
    return np.concatenate(pieces)


def image_threshold(normal_image_scores: np.ndarray, target_fpr: float = TARGET_IMAGE_FPR) -> float:
    """Threshold for the reproduction image score. The default target is 5%."""

    return empirical_threshold(np.asarray(normal_image_scores, dtype=np.float32), target_fpr)


def patch_threshold(normal_patch_scores: np.ndarray, target_fpr: float = TARGET_PATCH_FPR) -> float:
    """Threshold for native patch scores. This is not a component-level FPR."""

    return empirical_threshold(np.asarray(normal_patch_scores, dtype=np.float32), target_fpr)


def fpr_report(normal_scores: np.ndarray, threshold: float, target_fpr: float) -> dict[str, object]:
    """Realized FPR and its interval. Equality with the target is not claimed."""

    realized = realized_fpr(normal_scores, threshold)
    lower, upper = clopper_pearson(int(realized["n_false"]), int(realized["n"]))
    return {
        "target_fpr": target_fpr,
        "threshold": threshold,
        "n": realized["n"],
        "n_false": realized["n_false"],
        "achieved_fpr": realized["fpr"],
        "interval_low": lower,
        "interval_high": upper,
        "target_match_claimed": False,
    }


def classify_component(status: str, statistic: float, threshold: float) -> dict[str, object]:
    """Keep every component in the denominator.

    ``入力変換による評価不能`` is not a detection and is not removed.
    """

    unevaluable = status == "入力変換による評価不能"
    finite = statistic is not None and math.isfinite(float(statistic))
    detected = (not unevaluable) and finite and float(statistic) >= threshold
    return {
        "status": status,
        "statistic": float(statistic) if finite else float("nan"),
        "detected": detected,
        "in_denominator": True,
        "unevaluable": unevaluable,
    }


def rate_comparison(successes_a: int, trials_a: int, successes_b: int, trials_b: int) -> dict[str, object]:
    """Difference of two rates and their separate intervals.

    The result never classifies the difference as a practical improvement or
    as equivalence. Those margins are unset.
    """

    if min(trials_a, trials_b) <= 0:
        difference = float("nan")
    else:
        difference = successes_b / trials_b - successes_a / trials_a
    low_a, high_a = clopper_pearson(successes_a, trials_a)
    low_b, high_b = clopper_pearson(successes_b, trials_b)
    if min(trials_a, trials_b) <= 0:
        difference_low, difference_high = float("nan"), float("nan")
    else:
        rate_a = successes_a / trials_a
        rate_b = successes_b / trials_b
        width = 1.96 * math.sqrt(
            rate_a * (1.0 - rate_a) / trials_a + rate_b * (1.0 - rate_b) / trials_b
        )
        difference_low = difference - width
        difference_high = difference + width
    return {
        "difference": difference,
        "difference_interval_low": difference_low,
        "difference_interval_high": difference_high,
        "interval_a": (low_a, high_a),
        "interval_b": (low_b, high_b),
        "practical_improvement": VERDICT_UNDECIDED,
        "equivalence": VERDICT_UNDECIDED,
    }
