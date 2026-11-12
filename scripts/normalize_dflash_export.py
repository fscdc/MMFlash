"""Normalize an exported DFlash / MMFlash draft for SGLang and vLLM.

``specforge export --to hf`` writes the trained draft's own architecture name (``MMFlashDraftModel``). Both serving
engines load every block-diffusion draft through their DFLASH implementation, which resolves the class by the name
``DFlashDraftModel``; this rewrites ``architectures`` (and drops a local ``auto_map``) after checking the block size.

    python scripts/normalize_dflash_export.py --config <export dir>/config.json --block-size 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def normalize_export(config_path: str, expected_block_size: int) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    block_size = config.get("block_size")
    if block_size != expected_block_size:
        raise ValueError(f"exported block_size={block_size!r}, expected {expected_block_size}")
    method_config = config.get("dflash_config") or {}
    projector_type = method_config.get("projector_type", "dflash")
    if projector_type not in {"dflash", "mmflash"}:
        raise ValueError(f"export is not a DFlash-family draft: dflash_config.projector_type={projector_type!r}")

    config["architectures"] = ["DFlashDraftModel"]
    config.pop("auto_map", None)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="config.json of the exported draft")
    parser.add_argument("--block-size", type=int, required=True)
    args = parser.parse_args()
    normalize_export(args.config, args.block_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
