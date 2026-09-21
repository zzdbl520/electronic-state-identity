#!/usr/bin/env python3
"""Preflight and immutable-model audit for the final BeH2 rerun."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--static-only", action="store_true")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_model_lock() -> dict:
    lock = json.loads((ROOT / "model_lock.json").read_text(encoding="utf-8"))
    assert lock["basis"] == "cc-pVTZ"
    assert lock["active_space"] == {"electrons": 6, "orbitals": 6}
    assert lock["freeze_core"] is True
    assert lock["retained_manifold_dimension"] == 3
    assert set(lock["protocols"]) == {"SA3", "SA4"}
    assert abs(sum(lock["protocols"]["SA3"]) - 1.0) < 1.0e-14
    assert abs(sum(lock["protocols"]["SA4"]) - 1.0) < 1.0e-14
    return lock


def check_sources() -> dict[str, str]:
    required = [
        SCRIPTS / "BeH2_B0_stationary_solutions.py",
        SCRIPTS / "BeH2_B0_analyze.py",
        SCRIPTS / "BeH2_B1_branch_support.py",
        SCRIPTS / "BeH2_B1_analyze.py",
        SCRIPTS / "BeH2_B2_refine.py",
        SCRIPTS / "BeH2_B2_plaquettes.py",
        SCRIPTS / "BeH2_B3R_scaling.py",
        SCRIPTS / "BeH2_B3R_analyze.py",
        ROOT / "plotting" / "plot_BeH2_B3R_validation.py",
        ROOT / "plotting" / "plot_BeH2_B3R_components.py",
    ]
    forbidden = ("6-31G", "CAS(6,12)", "SELECTED_M = 2")
    hashes: dict[str, str] = {}
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        hits = [token for token in forbidden if token in text]
        if hits:
            raise RuntimeError(f"Forbidden legacy model token(s) {hits} in {path}")
        compile(text, str(path), "exec")
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes


def run_self_tests() -> list[str]:
    tests = [
        ("BeH2_B0_stationary_solutions.py",),
        ("BeH2_B0_analyze.py",),
        ("BeH2_B1_branch_support.py",),
        ("BeH2_B1_analyze.py",),
        ("BeH2_B2_refine.py",),
        ("BeH2_B2_plaquettes.py",),
        ("BeH2_B3R_scaling.py",),
        ("BeH2_B3R_analyze.py",),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")
    outputs: list[str] = []
    for (name,) in tests:
        completed = subprocess.run(
            [sys.executable, str(SCRIPTS / name), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(completed.stdout.strip())
    return outputs


def dependency_versions(static_only: bool) -> dict[str, str]:
    if static_only:
        return {"mode": "static-only"}
    modules = ("numpy", "scipy", "pyscf", "pandas", "matplotlib", "PIL")
    versions: dict[str, str] = {}
    for name in modules:
        module = importlib.import_module(name)
        distribution = "pillow" if name == "PIL" else name
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = str(getattr(module, "__version__", "unknown"))
        versions[name] = version
    from pyscf import fci, mcscf

    if not hasattr(fci.addons, "overlap"):
        raise RuntimeError("PySCF fci.addons.overlap is required.")
    if not hasattr(mcscf, "project_init_guess"):
        raise RuntimeError("PySCF mcscf.project_init_guess is required.")
    return versions


def main() -> int:
    args = parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = check_model_lock()
    hashes = check_sources()
    tests = run_self_tests()
    versions = dependency_versions(args.static_only)
    report = {
        "status": "PASS",
        "static_only": args.static_only,
        "model_lock": lock,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": versions,
        "source_sha256": hashes,
        "self_tests": tests,
    }
    path = args.output_root / "preflight_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("BeH2 final preflight: PASS")
    print(f"Model: CAS(6,6)/cc-pVTZ; frozen Be 1s; SA3/SA4; m=3")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
