#!/usr/bin/env python3
"""Audit the October revision records and imported file integrity (NumPy only).

This does not optimize CASSCF or reconstruct AO overlaps. For independent
raw-wave-function reconstruction use the validators in the data directory.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
DATA = REPOSITORY / 'data/revision-2026-10-21'

def load(relative):
    return json.loads((DATA / relative).read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def close(a, b, message, atol=1e-12, rtol=1e-9):
    require(np.allclose(a, b, atol=atol, rtol=rtol), message)

def verify_lih():
    s = load('analysis/augccpvtz_summary.json')
    p = s['production']
    loops = load('analysis/augccpvtz_refinement_validation/recomputed_loops.json')
    final = max(loops, key=lambda r: r['number_of_edges'])
    require(p['number_of_edges'] == final['number_of_edges'] == 154, 'Final mesh changed')
    for field in ['C', 'D_mu', 'eta', 'leakage', 'assignment_margin']:
        close(p[field], final[field], 'Summary/source mismatch: '+field)
    require(p['qualified'] and all(p['checks'].values()), 'Final LiH gates fail')
    require(p['route1_permutation'] == p['route2_permutation'] == list(range(4)),
            'Final endpoint assignments are not both identity')
    a, b = np.array(p['route_R_then_F']), np.array(p['route_F_then_R'])
    close(np.linalg.norm(a-b)/4, p['C'], 'Transport matrices do not reproduce C')
    close(p['C'], 3.191653837014006e-4, 'Published C changed')
    close(p['D_mu'], 6.300611946495908e-4, 'Published D_mu changed')
    d = p['delta_dipole_in_start_frame']
    close(np.linalg.norm(np.array(d['real'])+1j*np.array(d['imag'])),
          p['delta_mu_au'], 'Transported dipole norm mismatch')
    require(not s['local_boundary'][0]['qualified'], 'Rejected uniform boundary lost')
    require(s['local_boundary'][0]['assignment_margin'] < 0.1,
            'Uniform-boundary failure qualification lost')
    require(s['complete_100_fidelity_pass'] == 0, 'Original-rectangle limitation lost')
    require(len(s['stride_distribution']) == 16, 'Incomplete locator stride groups')
    require(sum(r['count'] for r in s['stride_distribution']) == 100, 'Incomplete locator')
    nested = s['nested']
    require(len(nested) == 4 and all(r['qualified'] for r in nested), 'Nested-loop gates fail')
    slope = np.polyfit(np.log([r['area'] for r in nested]),
                       np.log([r['C8'] for r in nested]), 1)[0]
    close(slope, s['nested_fit']['log_log_slope'], 'Area-scaling slope mismatch')
    require(len(s['histories']) == 2 and all(h['history_pass'] for h in s['histories']),
            'Final opposed-history checks fail')
    audit = load('analysis/augccpvtz_refinement_validation/audit.json')
    require(audit['status'] == 'QUALIFIED_VERIFIED', 'Refinement audit status changed')
    require(all(audit['final_checks_independently_reassembled'].values()),
            'Refinement audit has failed checks')
    require(load('analysis/augccpvtz_production_validation/audit.json')['status'] ==
            'REVIEW_REQUIRED', 'Original production failure lost')
    for basis in ['augccpvdz', 'ccpvtz']:
        r = load(f'records/lih_revision_{basis}/revision_results.json')
        require(all(x['fidelity_pass'] and x['assignment_pass'] and x['all_converged']
                    and x['same_assignment'] for x in r['loops'] + r['histories']),
                basis+' loop/history failed')
        require(len(r['histories']) == 2, basis+' history missing')


def verify_beh2():
    r = load('records/beh2_model_comparison/G0C_forensic_summary.json')
    require(r['status'] == 'REVIEW_REQUIRED', 'BeH2 qualification lost')
    require(r['decision_checks']['all_stride1_upper_edges_locally_valid'] is False,
            'BeH2 failure must remain explicit')
    require(r['decision_checks']['main_figure_candidate'] is False,
            'BeH2 failed plaquettes promoted')


def verify_cyclopentadienyl_cation():
    s = load('analysis/cpdc_summary.json')
    require(s['status'] == 'RETAINED_RANK2_MANIFOLD_EVIDENCE', 'CPDC scope changed')
    require(s['independent_SA3_minimum_pair_sigma'] > 0.9998, 'External pair fidelity fails')
    require(s['independent_SA3_minimum_third_gap_Eh'] > 0.0277, 'External gap fails')
    phases = {r['sequence']: r['positive_phase_rad'] for r in s['histories']}
    diff = max(abs(phases[k]-phases['Reference']) for k in ['clockwise', 'counterclockwise'])
    close(diff, s['maximum_reference_history_phase_difference_rad'], 'CPDC history arithmetic')
    mesh = {r['npoints']: r['positive_phase_rad'] for r in s['mesh']}
    require(diff > abs(mesh[80]-mesh[40]), 'CPDC initialization sensitivity hidden')
    require(s['arithmetic_checks']['phase_comparisons_pass'], 'CPDC record arithmetic fails')
    for path, expected in s['source_sha256'].items():
        require(hashlib.sha256((DATA/path).read_bytes()).hexdigest() == expected,
                'CPDC source changed: '+path)


def verify_historical_controls():
    h = load('records/H3/H3_local_refinement_decision.json')
    # Historical stdout contains the decision JSON followed by further output.
    o, _ = json.JSONDecoder().raw_decode(
        (DATA/'records/O3/O3_weighted_SA2_pilot_record.txt').read_text().lstrip())
    require(h['status'] == 'STOP_POINTWISE_REFINEMENT_FAILURE', 'H3 failure hidden')
    require(o['status'] == 'STOP_EDGE_FIDELITY_FAILURE', 'O3 failure hidden')
    require(o['manuscript_claim_authorized'] is False, 'Failed O3 promoted')
    for path in ['records/H3/source_data/ring_profile.csv',
                 'records/O3/source_data/gate4A/O3_gate4A_calculations.csv',
                 'records/O3/source_data/gate4B/O3_gate4B_plaquettes.csv']:
        require((DATA/path).stat().st_size > 0, 'Missing historical table: '+path)


def verify_integrity():
    provenance = load('SOURCE_PROVENANCE.json')
    require(provenance['source_archive_checksums_verified'], 'Source archive not verified')
    imported_npz = set()
    for item in provenance['files']:
        path = DATA / item['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == item['sha256'], 'Imported file changed: '+item['path'])
        if item['path'].startswith('records/'):
            require(digest == item['source_sha256'], 'Raw record changed: '+item['path'])
        if path.suffix == '.npz':
            imported_npz.add(path.relative_to(DATA).as_posix())
    actual_npz = {p.relative_to(DATA).as_posix() for p in (DATA/'records').rglob('*.npz')}
    require(imported_npz == actual_npz and len(actual_npz) == 1936,
            'Incomplete LiH NPZ deposit')
    for name, count in [('lih_augccpvtz_production',736), ('lih_augccpvtz_local_refinement',644)]:
        manifest = load(f'records/{name}/wavefunction_sha256.json')
        require(len(manifest) == count, name+' manifest count changed')
        for path, expected in manifest.items():
            actual = hashlib.sha256((DATA/'records'/name/path).read_bytes()).hexdigest()
            require(actual == expected, 'Original NPZ checksum mismatch: '+name+'/'+path)
    # Historical archive containers exclude rendered figures; retained numerical bytes are unchanged.
    import io
    import tarfile
    repacked = load('NUMERICAL_ARCHIVES.json')['archives']
    for archive in repacked:
        if 'parts' in archive:
            chunks = []
            for part in archive['parts']:
                chunk = (DATA / part['path']).read_bytes()
                require(len(chunk) == part['bytes'] and
                        hashlib.sha256(chunk).hexdigest() == part['sha256'],
                        'Numerical archive part changed: ' + part['path'])
                chunks.append(chunk)
            content = b''.join(chunks)
        else:
            content = (DATA / archive['numerical_archive_path']).read_bytes()
        require(hashlib.sha256(content).hexdigest() == archive['numerical_archive_sha256'],
                'Numerical archive checksum mismatch')
        with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as tar:
            actual = {m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                      for m in tar if m.isfile()}
        require(actual == {m['path']: m['sha256'] for m in archive['included_members']},
                'Repacked numerical members changed')
    print(f'PASS  {len(provenance["files"])} imported files; {len(actual_npz)} LiH NPZ files')


def main():
    for name, func in [('LiH current revision',verify_lih), ('BeH2 limitations',verify_beh2),
                       ('C5H5+ record arithmetic and sensitivity',verify_cyclopentadienyl_cation),
                       ('H3/O3 historical controls',verify_historical_controls)]:
        func(); print('PASS  '+name)
    verify_integrity()
    print('REFERENCE_AUDIT=PASS (record/hash audit; no new CASSCF calculations)')

if __name__ == '__main__':
    main()
