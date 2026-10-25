"""Component rows for one source image under several warps.

A component lost by the 512 resize and a component lost by a later warp stay
in the denominator under different statuses.
"""

from __future__ import annotations

import numpy as np

from robustad.prepare.regions import UNEVALUABLE_INPUT_TRANSFORM, component_patch_maximum

WARP_UNEVALUABLE = "ワープによる評価不能"
COMMON_UNEVALUABLE = "共通有効領域による評価不能"
NOT_APPLICABLE = "not_applicable"


def component_record(
    *,
    component_id: int,
    status_input: str,
    area_original: int,
    area_512: int,
    warped: np.ndarray | None,
    condition_valid: np.ndarray,
    common_valid: np.ndarray,
    patch_map: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Score one original component. Empty results stay in the denominator."""

    record: dict[str, object] = {
        "component_id": int(component_id),
        "area_original": int(area_original),
        "area_512": int(area_512),
        "status_input": status_input,
        "in_denominator": True,
    }
    if status_input == UNEVALUABLE_INPUT_TRANSFORM or area_512 <= 0 or warped is None:
        record.update(
            {
                "status_warp": NOT_APPLICABLE,
                "status_common": NOT_APPLICABLE,
                "area_condition": 0,
                "area_common": 0,
                "remaining_ratio": float("nan"),
                "remaining_ratio_common": float("nan"),
                "statistic_common": float("nan"),
                "detected": False,
            }
        )
        return record
    mask = np.asarray(warped, dtype=bool)
    area_condition = int((mask & np.asarray(condition_valid, dtype=bool)).sum())
    area_common = int((mask & np.asarray(common_valid, dtype=bool)).sum())
    if int(mask.sum()) == 0 or area_condition == 0:
        status_warp = WARP_UNEVALUABLE
        statistic = float("nan")
        status_common = NOT_APPLICABLE
    else:
        status_warp = "evaluable"
        if area_common == 0:
            status_common = COMMON_UNEVALUABLE
            statistic = float("nan")
        else:
            status_common = "evaluable"
            statistic = component_patch_maximum(patch_map, mask & np.asarray(common_valid, dtype=bool))
    detected = status_warp == "evaluable" and status_common == "evaluable" and statistic >= threshold
    record.update(
        {
            "status_warp": status_warp,
            "status_common": status_common,
            "area_condition": area_condition,
            "area_common": area_common,
            "remaining_ratio": float(area_condition / area_512),
            "remaining_ratio_common": float(area_common / area_512),
            "statistic_common": statistic,
            "detected": bool(detected),
        }
    )
    return record
