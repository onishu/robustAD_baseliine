"""Print the 3-view augmentation estimate. Nothing is fit or scored."""

from __future__ import annotations

import json

from robustad.theme_a.action import (
    AUGMENT_FIT_ENABLED,
    AUGMENT_OUTPUT_NAME,
    AUGMENT_ZERO_PATH_REASON,
    augment_storage,
    augment_views,
)


def main() -> None:
    payload = {
        "fit_enabled": AUGMENT_FIT_ENABLED,
        "output_dir": AUGMENT_OUTPUT_NAME,
        "zero_path_reason": AUGMENT_ZERO_PATH_REASON,
        "views": augment_views(),
        "storage": augment_storage(),
        "command_after_confirmation": (
            ".\\.venv\\Scripts\\python -m robustad.theme_a.run_augment "
            "--confirm-augment-fit --data-root K:\\RobustAD "
            "--output-dir results_a0_augment "
            "--manifest results_p0\\research_split_manifest.json"
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
