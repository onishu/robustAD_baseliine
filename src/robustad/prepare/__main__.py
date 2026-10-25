"""Commands for preparation phase P.

``diagnose --confirm-fit180`` refits the approved seed-0 diagnostic runs.
It writes under ``results_p0`` and does not touch ``results_a1``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from robustad.prepare.infer_diagnostic import run_diagnostic
from robustad.prepare.inventory import build_inventory
from robustad.prepare.research_split import write_research_split
from robustad.prepare.manifest import write_manifest
from robustad.prepare.proposal import write_proposal
from robustad.prepare.smoke import run_patchcore_smoke
from robustad.prepare.spec import PROTECTED_OUTPUTS


def _guard(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if resolved.name in PROTECTED_OUTPUTS:
        raise SystemExit(f"既存の再現出力には書きません: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory")
    inventory.add_argument("--data-root", type=Path, default=Path(r"K:\RobustAD"))
    inventory.add_argument("--output-dir", type=Path, default=Path("results_p0"))

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--repo-root", type=Path, default=Path("."))
    manifest.add_argument("--output-dir", type=Path, default=Path("results_p0"))

    proposal = sub.add_parser("proposal")
    proposal.add_argument("--data-root", type=Path, default=Path(r"K:\RobustAD"))
    proposal.add_argument("--output-dir", type=Path, default=Path("results_p0"))

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--image-size", type=int, default=64)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--output-dir", type=Path, default=Path("results_p0"))

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--data-root", type=Path, default=Path(r"K:\RobustAD"))
    diagnose.add_argument("--output-dir", type=Path, default=Path("results_p0"))
    diagnose.add_argument("--reference-dir", type=Path, default=Path("results_a1"))
    diagnose.add_argument("--repo-root", type=Path, default=Path("."))
    diagnose.add_argument("--confirm-fit180", action="store_true")

    freeze_split = sub.add_parser("freeze-split")
    freeze_split.add_argument("--data-root", type=Path, default=Path(r"K:\RobustAD"))
    freeze_split.add_argument("--output-dir", type=Path, default=Path("results_p0"))

    args = parser.parse_args()
    output_dir = _guard(args.output_dir)
    if args.command == "inventory":
        payload = build_inventory(args.data_root, output_dir)
        print(json.dumps({"domains": len(payload["domains"]), "unparsed": payload["n_unparsed_anomaly_names"]}, ensure_ascii=False))
    elif args.command == "manifest":
        payload = write_manifest(args.repo_root, output_dir)
        print(payload["git_commit"])
    elif args.command == "proposal":
        payload = write_proposal(args.data_root, output_dir)
        for key, item in payload["datasets"].items():
            print(key, item["n_fit"], item["n_calibration"], "overlap", len(item["overlap"]))
    elif args.command == "smoke":
        payload = run_patchcore_smoke(output_dir, image_size=args.image_size, seed=args.seed)
        print(json.dumps(payload, ensure_ascii=False))
    elif args.command == "freeze-split":
        payload = write_research_split(args.data_root, output_dir)
        for key, item in payload["datasets"].items():
            print(key, item["n_fit"], item["n_calibration"], "overlap", item["n_overlap_files"], "bodies", len(item["bodies"]))
    elif args.command == "diagnose":
        if not args.confirm_fit180:
            raise SystemExit("180枚 fit の診断再推論には --confirm-fit180 が必要です。")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        payload = run_diagnostic(args.data_root, output_dir, args.reference_dir, args.repo_root)
        print(json.dumps({"domains": len(payload["domains"])}, ensure_ascii=False))
    else:
        raise SystemExit(f"未知のコマンドです: {args.command}")


if __name__ == "__main__":
    main()
