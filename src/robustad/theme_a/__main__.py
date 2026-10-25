"""Run the authorized synthetic rotation experiment."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from robustad.theme_a.run_synthetic import run_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"K:\RobustAD"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_a0"))
    parser.add_argument("--manifest", type=Path, default=Path("results_p0/research_split_manifest.json"))
    parser.add_argument("--confirm-fit144", action="store_true")
    parser.add_argument("--reuse-bank", action="store_true")
    args = parser.parse_args()
    if not args.confirm_fit144:
        raise SystemExit("144枚 fit には --confirm-fit144 が必要です。")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = run_synthetic(args.data_root, args.output_dir, args.manifest, reuse_bank=args.reuse_bank)
    print(json.dumps({"conditions": len(payload["conditions"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
