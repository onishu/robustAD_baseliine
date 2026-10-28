"""Tables and charts from the saved theme B scores. This does not rescore."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from robustad.metrics import safe_auroc
from robustad.prepare.names import parse_anomaly_stem
from robustad.theme_b.estimate import diagnostic_added_angle

SOURCE = Path("results_b0")
OUTPUT = Path("results_b0_analysis")
ANGLE_ORDER = (
    ("series_m10", -10.0),
    ("series_m5", -5.0),
    ("series_m2_5", -2.5),
    ("series_0", 0.0),
    ("series_p2_5", 2.5),
    ("series_p5", 5.0),
    ("series_p10", 10.0),
)
BANKS = (
    ("stage1_control0", "回転ビューなし"),
    ("stage1_three", "0°と±5°入り"),
)
STAGE2_PAIRS = (
    ("series_m10", "known_m10", "estimated_m10", -10.0),
    ("series_p10", "known_p10", "estimated_p10", 10.0),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"真偽値が True/False ではありません: {value}")


def _float_or_none(value: str) -> float | None:
    if value == "":
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear"))


def decision_counts(reference: list[bool], other: list[bool]) -> dict[str, int]:
    """Pair decisions on the same images. Counts are not a claim of equivalence."""

    if len(reference) != len(other):
        raise ValueError("対応する判定の長さが違います。")
    left = np.asarray(reference, dtype=bool)
    right = np.asarray(other, dtype=bool)
    return {
        "n": int(left.size),
        "both_detected": int(np.sum(left & right)),
        "reference_only": int(np.sum(left & ~right)),
        "other_only": int(np.sum(~left & right)),
        "both_missed": int(np.sum(~left & ~right)),
        "same_decision": int(np.sum(left == right)),
    }


def tracking_rows(estimates: list[dict[str, str]], condition: str, added: float) -> dict[str, object]:
    """Follow the added angle only when both the 0° estimate and this one succeeded."""

    zero = {row["source_id"]: row for row in estimates if row["condition"] == "estimated_0"}
    rows = [row for row in estimates if row["condition"] == condition]
    errors = []
    absolute_success = []
    absolute_failed = []
    excluded = {"zero_failed": 0, "condition_failed": 0, "both_failed": 0}
    for row in rows:
        base = zero[row["source_id"]]
        angle = _float_or_none(row["estimated_degrees"])
        base_angle = _float_or_none(base["estimated_degrees"])
        failed = _bool(row["fallback"])
        base_failed = _bool(base["fallback"])
        if angle is not None and not failed:
            absolute_success.append(angle)
        elif angle is not None and failed:
            absolute_failed.append(angle)
        if failed and base_failed:
            excluded["both_failed"] += 1
            continue
        if base_failed:
            excluded["zero_failed"] += 1
            continue
        if failed:
            excluded["condition_failed"] += 1
            continue
        if angle is None or base_angle is None:
            raise RuntimeError(f"成功なのに角度がありません: {row['source_id']} {condition}")
        errors.append(diagnostic_added_angle(angle, base_angle) - added)
    return {
        "condition": condition,
        "added_degrees": added,
        "n_images": len(rows),
        "n_tracking": len(errors),
        "n_excluded": int(sum(excluded.values())),
        "excluded": excluded,
        "tracking_error_median": _quantile(errors, 0.5),
        "tracking_error_abs_median": _quantile([abs(value) for value in errors], 0.5),
        "absolute_success_median": _quantile(absolute_success, 0.5),
        "absolute_success_abs_median": _quantile([abs(value) for value in absolute_success], 0.5),
        "absolute_failed_stored_n": len(absolute_failed),
        "failures_are_not_zero_error": True,
    }


def _index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["source_id"], row["condition"]): row for row in rows}


def _series_block(stage: str, images: list[dict[str, str]], fpr_rows: list[dict[str, str]], components: list[dict[str, str]]) -> dict[str, object]:
    image_index = _index(images)
    fpr_index = _index(fpr_rows)
    by_angle = []
    for condition, degrees in ANGLE_ORDER:
        selected = [row for row in images if row["condition"] == condition]
        normal = [row for row in selected if row["label"] == "0"]
        anomaly = [row for row in selected if row["label"] == "1"]
        rates = [float(fpr_index[(row["source_id"], condition)]["fpr_full"]) for row in normal]
        labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int32)
        scores = np.asarray([float(row["reproduction_score"]) for row in selected], dtype=np.float64)
        kinds: dict[str, list[bool]] = {}
        for row in anomaly:
            kind = str(parse_anomaly_stem(row["source_id"])["defect"]) or "unparsed"
            kinds.setdefault(kind, []).append(_bool(row["detected"]))
        component_rows = [row for row in components if row["condition"] == condition]
        finite = [float(row["statistic_common"]) for row in component_rows if _float_or_none(row["statistic_common"]) is not None]
        by_angle.append(
            {
                "condition": condition,
                "degrees": degrees,
                "normal_detected": int(sum(_bool(row["detected"]) for row in normal)),
                "normal_n": len(normal),
                "normal_image_fpr": len(normal) and sum(_bool(row["detected"]) for row in normal) / len(normal),
                "anomaly_detected": int(sum(_bool(row["detected"]) for row in anomaly)),
                "anomaly_n": len(anomaly),
                "anomaly_detection_rate": len(anomaly) and sum(_bool(row["detected"]) for row in anomaly) / len(anomaly),
                "patch_exceedance_median": _quantile(rates, 0.5),
                "patch_exceedance_q25": _quantile(rates, 0.25),
                "patch_exceedance_q75": _quantile(rates, 0.75),
                "auxiliary_image_auroc": safe_auroc(labels, scores),
                "defect_types": {
                    kind: {"detected": int(sum(flags)), "n": len(flags)} for kind, flags in sorted(kinds.items())
                },
                "components_detected": int(sum(_bool(row["detected"]) for row in component_rows)),
                "components_n": len(component_rows),
                "component_statistic_median_finite": _quantile(finite, 0.5),
                "component_statistic_finite_n": len(finite),
            }
        )
    if len(image_index) != len(images):
        raise RuntimeError(f"{stage} の画像行が元画像IDと条件で一意ではありません。")
    return {"stage": stage, "angles": by_angle}


def _pair_block(left_name: str, right_name: str, left_rows: dict, right_rows: dict, ids: list[str]) -> dict[str, object]:
    grouped = {}
    for label, title in (("0", "normal"), ("1", "anomaly")):
        chosen = [source_id for source_id in ids if left_rows[(source_id, left_name)]["label"] == label]
        grouped[title] = decision_counts(
            [_bool(left_rows[(source_id, left_name)]["detected"]) for source_id in chosen],
            [_bool(right_rows[(source_id, right_name)]["detected"]) for source_id in chosen],
        )
    return {"left": left_name, "right": right_name, "by_label": grouped}


def build_report(source: Path = SOURCE, output: Path = OUTPUT) -> dict[str, object]:
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise RuntimeError(f"解析は {source} の中へ書きません: {output}")
    output.mkdir(parents=True, exist_ok=True)
    identifiers = {}
    for relative in (
        "run_status.json",
        "series_status.json",
        "stage1_control0/summary.json",
        "stage1_control0/status.json",
        "stage1_three/summary.json",
        "stage1_three/status.json",
        "stage2_control0/summary.json",
        "stage2_control0/status.json",
    ):
        path = source / relative
        identifiers[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    stages = {}
    image_tables = {}
    fpr_tables = {}
    for stage, _title in BANKS:
        images = _read_csv(source / stage / "tables" / "images.csv")
        fpr_rows = _read_csv(source / stage / "tables" / "image_patch_fpr.csv")
        components = _read_csv(source / stage / "tables" / "components.csv")
        stages[stage] = _series_block(stage, images, fpr_rows, components)
        image_tables[stage] = _index(images)
        fpr_tables[stage] = _index(fpr_rows)
    stage2_images = _index(_read_csv(source / "stage2_control0" / "tables" / "images.csv"))
    stage2_fpr = _index(_read_csv(source / "stage2_control0" / "tables" / "image_patch_fpr.csv"))
    estimates = _read_csv(source / "stage2_control0" / "tables" / "estimates.csv")
    ids = sorted({source_id for source_id, _condition in image_tables["stage1_control0"]})
    pairs = []
    for uncorrected, known, estimated, degrees in STAGE2_PAIRS:
        pairs.append(
            {
                "degrees": degrees,
                "uncorrected_vs_known": _pair_block(uncorrected, known, image_tables["stage1_control0"], stage2_images, ids),
                "uncorrected_vs_estimated": _pair_block(
                    uncorrected, estimated, image_tables["stage1_control0"], stage2_images, ids
                ),
                "known_vs_estimated": _pair_block(known, estimated, stage2_images, stage2_images, ids),
            }
        )
    score_delta = []
    patch_delta = []
    zero_flags = {"normal": [], "anomaly": []}
    for source_id in ids:
        before = image_tables["stage1_control0"][(source_id, "series_0")]
        after = stage2_images[(source_id, "estimated_0")]
        score_delta.append(float(after["reproduction_score"]) - float(before["reproduction_score"]))
        key = "normal" if before["label"] == "0" else "anomaly"
        zero_flags[key].append((_bool(before["detected"]), _bool(after["detected"])))
        if key == "normal":
            patch_delta.append(
                float(stage2_fpr[(source_id, "estimated_0")]["fpr_full"])
                - float(fpr_tables["stage1_control0"][(source_id, "series_0")]["fpr_full"])
            )
    failures = []
    for condition, added in (("estimated_0", 0.0), ("estimated_m10", -10.0), ("estimated_p10", 10.0)):
        selected = [row for row in estimates if row["condition"] == condition and _bool(row["fallback"])]
        by_label: dict[str, dict[str, int]] = {}
        for row in selected:
            image = stage2_images[(row["source_id"], condition)]
            label = "normal" if image["label"] == "0" else "anomaly"
            kind = "normal" if label == "normal" else str(parse_anomaly_stem(row["source_id"])["defect"]) or "unparsed"
            bucket = by_label.setdefault(kind, {})
            reason = row["reason"] or "failed"
            bucket[reason] = bucket.get(reason, 0) + 1
        failures.append(
            {
                "condition": condition,
                "added_degrees": added,
                "n_failed": len(selected),
                "n_images": sum(row["condition"] == condition for row in estimates),
                "by_kind": by_label,
            }
        )
    payload = {
        "source": str(source),
        "inputs": identifiers,
        "overwrote_results_b0": False,
        "thresholds_recomputed": False,
        "transformed_copies_are_not_new_subjects": True,
        "series": stages,
        "paired_decisions": pairs,
        "zero_input_normalization": {
            "description": "未補正0°へ、固定参照への絶対角の逆向きを接続した影響。追加回転の補正効果ではない。",
            "score_delta_median": _quantile(score_delta, 0.5),
            "score_delta_mean": float(np.mean(score_delta)),
            "normal_patch_exceedance_delta_median": _quantile(patch_delta, 0.5),
            "decisions": {
                key: decision_counts([item[0] for item in flags], [item[1] for item in flags])
                for key, flags in zero_flags.items()
            },
        },
        "estimation_failures": failures,
        "angles": {
            "rule": "0°推定は診断であり、補正角から引いていない。片方でも失敗した画像は追従誤差の分母に入れない。",
            "estimated_0_absolute": tracking_rows(estimates, "estimated_0", 0.0),
            "estimated_m10": tracking_rows(estimates, "estimated_m10", -10.0),
            "estimated_p10": tracking_rows(estimates, "estimated_p10", 10.0),
        },
        "auxiliary_auroc_not_used_for_selection": True,
        "verdict_scope": "実用上の最良と同等性は未判定。観測した件数は未判定にしない。",
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_series_csv(output / "angle_series.csv", stages)
    _write_figures(output / "figures", stages)
    return payload


def _write_series_csv(path: Path, stages: dict[str, dict]) -> None:
    rows = []
    for stage, block in stages.items():
        for angle in block["angles"]:
            row = {"stage": stage, **{key: value for key, value in angle.items() if key != "defect_types"}}
            for kind, counts in angle["defect_types"].items():
                row[f"defect_{kind}_detected"] = counts["detected"]
                row[f"defect_{kind}_n"] = counts["n"]
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(directory: Path, stages: dict[str, dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    from robustad.visualize import configure_font

    configure_font()
    import matplotlib.pyplot as plt

    panels = (
        ("normal_image_fpr", "正常画像の誤検出割合", "枚数 / 120"),
        ("anomaly_detection_rate", "異常画像の検出割合", "枚数 / 75"),
        ("patch_exceedance_median", "正常パッチ超過割合の中央値", "セル割合"),
        ("auxiliary_image_auroc", "補助 Image AUROC", "AUROC"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), constrained_layout=True)
    for axis, (key, title, ylabel) in zip(axes.ravel(), panels, strict=True):
        for stage, title_bank in BANKS:
            angles = stages[stage]["angles"]
            axis.plot(
                [item["degrees"] for item in angles],
                [item[key] for item in angles],
                marker="o",
                label=title_bank,
            )
        axis.set_title(title)
        axis.set_xlabel("加えた角度（度）")
        axis.set_ylabel(ylabel)
        axis.set_xticks([-10, -5, -2.5, 0, 2.5, 5, 10])
        axis.grid(True, linewidth=0.4)
        axis.legend()
    figure.suptitle("41 px 系列。しきい値は角度ごとに変えていない。変換画像は別個体にしていない。")
    figure.savefig(directory / "angle_series.png", dpi=140)
    plt.close(figure)


def main() -> None:
    payload = build_report()
    print(json.dumps({"output": str(OUTPUT), "angles": payload["angles"]["estimated_m10"]["n_tracking"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
