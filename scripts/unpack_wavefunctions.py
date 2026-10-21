#!/usr/bin/env python3
"""Restore deposited LiH NPZ files, checking part/archive/per-file SHA-256 hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

DATA = Path(__file__).resolve().parents[1] / 'data/revision-2026-10-21'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination', type=Path, default=DATA,
                    help='Data root to restore into (default: deposited data directory)')
    dest = ap.parse_args().destination.resolve()
    parts = DATA / 'wavefunctions'
    manifest = json.loads((parts/'MANIFEST.json').read_text())
    provenance = json.loads((DATA/'SOURCE_PROVENANCE.json').read_text())
    expected = {r['path']:r['sha256'] for r in provenance['files'] if r['path'].endswith('.npz')}
    if len(expected) != manifest['npz_files']:
        raise RuntimeError('NPZ count does not match the source manifest')
    with tempfile.TemporaryDirectory(prefix='lih-wavefunctions-') as tmp:
        archive = Path(tmp)/'wavefunctions.tar.gz'
        with archive.open('wb') as out:
            for part in manifest['parts']:
                p = parts / part['name']
                if p.stat().st_size != part['bytes'] or sha(p) != part['sha256']:
                    raise RuntimeError('Archive-part checksum mismatch: '+part['name'])
                with p.open('rb') as inp:
                    shutil.copyfileobj(inp, out)
        if archive.stat().st_size != manifest['archive_bytes'] or sha(archive) != manifest['archive_sha256']:
            raise RuntimeError('Combined archive checksum mismatch')
        seen = set()
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise RuntimeError('Unexpected archive member: '+member.name)
                target = (dest/member.name).resolve()
                if dest not in target.parents:
                    raise RuntimeError('Archive path escapes destination')
                content = tar.extractfile(member).read()
                if hashlib.sha256(content).hexdigest() != expected[member.name]:
                    raise RuntimeError('NPZ checksum mismatch: '+member.name)
                if target.exists():
                    if sha(target) != expected[member.name]:
                        raise RuntimeError('Refusing to replace modified local data: '+str(target))
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                        out.write(content); temporary = Path(out.name)
                    temporary.replace(target)
                seen.add(member.name)
        if seen != set(expected):
            raise RuntimeError('Archive is missing NPZ files')
    print(f'PASS: {len(seen)} wave-function files restored and verified')

if __name__ == '__main__':
    main()
