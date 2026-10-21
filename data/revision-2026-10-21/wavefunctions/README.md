# LiH wave-function archive

All 1,936 NPZ files are stored in numbered archive parts. From the repository root:

```bash
python scripts/unpack_wavefunctions.py
```

The script checks every part, the assembled archive, and each NPZ file against
`MANIFEST.json` and `../SOURCE_PROVENANCE.json`. Files are restored to their
original paths under `../records/`. Modified existing files are never overwritten.
Counts include optimizer candidates and duplicate baselines.
