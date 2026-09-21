#!/usr/bin/env python3
"""Replace machine-specific path prefixes in archival text records.

This is a release-preparation utility, not part of the scientific analysis. It
does not alter numerical values or stored object digests. Hash fields embedded in
the source records continue to identify the original, unsanitized artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS = (
    ("/home/zdb/2026_JCTC_0916/LiH", "${LIH_ARCHIVE_ROOT}"),
    ("/mnt/BeH2_ccpvtz", "${BEH2_WORK_ROOT}"),
    ("/mnt/BeH2", "${BEH2_WORK_ROOT}"),
    ("/mnt/LiH", "${LIH_WORK_ROOT}"),
    ("/scr/zdb", "${CPDC_ARCHIVE_ROOT}"),
)

TEXT_SUFFIXES = {".json", ".md", ".stdout", ".txt"}


def sanitize(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    updated = text
    for source, replacement in REPLACEMENTS:
        updated = updated.replace(source, replacement)
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    changed = [
        path
        for path in sorted(args.root.rglob("*"))
        if path.is_file() and path.suffix in TEXT_SUFFIXES and sanitize(path)
    ]
    for path in changed:
        print(path.relative_to(args.root))
    print(f"SANITIZED_FILES={len(changed)}")


if __name__ == "__main__":
    main()
