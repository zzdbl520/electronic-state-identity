#!/usr/bin/env python3
"""Collect the small, submission-relevant outputs from a completed rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--work-root", type=Path, required=True)
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parser().parse_args()
    work = args.work_root.resolve()
    if read_status(work / "acceptance_report.json") != "PASS":
        raise RuntimeError("Refusing to collect a run that has not passed acceptance.")
    selected: list[Path] = []
    selected.extend(path for path in work.glob("*.json") if path.is_file())
    selected.extend(path for path in work.glob("*.md") if path.is_file())
    selected.extend(path for path in (work / "logs").glob("*.stdout") if path.is_file())
    selected.extend(path for path in (work / "figures").glob("BeH2_B3R_*.*") if path.is_file())
    for stage, prefix in (("B0", "B0"), ("B1", "B1"), ("B2", "B2"), ("B3R", "B3R")):
        directory = work / stage
        selected.extend(path for path in directory.glob(f"{prefix}_*.csv") if path.is_file())
        selected.extend(path for path in directory.glob(f"{prefix}_*.json") if path.is_file())
    selected = sorted(set(selected))
    manifest = {
        str(path.relative_to(work)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in selected
    }
    manifest_path = work / "RETURN_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    selected.append(manifest_path)
    archive = work / "BeH2_ccpVTZ_final_return.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in selected:
            stream.write(path, arcname=str(path.relative_to(work)))
    print(archive)
    return 0


def read_status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return str(json.loads(path.read_text(encoding="utf-8")).get("status", "UNKNOWN"))


if __name__ == "__main__":
    raise SystemExit(main())
