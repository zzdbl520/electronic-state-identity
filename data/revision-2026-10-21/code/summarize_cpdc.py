#!/usr/bin/env python3
"""Reanalyze the supplied C5H5+ numerical records; no raw-overlap replay claimed."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'records/CPDC'
OUT = ROOT / 'analysis'

def load(name):
    return json.loads((DATA / f'CPDC_ccpVTZ_rank2_{name}_v1_decision.json').read_text())

def write_csv(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    p, h, o = load('projector'), load('history'), load('opposed_history')
    assert p['status'] == 'PASS_RANK2_PROJECTOR_QUALIFICATION'
    assert h['status'] == 'PASS_RANK2_OPPOSED_HISTORY'
    assert o['result_authorizes_inclusion_as_rank2_manifold_evidence'] is True
    leakage_errors, invariant_errors = [], []
    def check_pair(r, sv='pair_singular_values', leak='pair_leakage'):
        error = abs(1 - np.mean(np.square(r[sv])) - r[leak])
        leakage_errors.append(float(error))
        assert error < 2e-14
    def spectrum(r):
        w = r['wilson_invariants']
        ev = np.exp(1j * np.array(w['eigenphases_rad']))
        for k, value in [('trace', sum(ev)), ('determinant', np.prod(ev))]:
            error = abs(value - complex(w[k]['real'], w[k]['imag']))
            invariant_errors.append(float(error))
            assert error < 5e-13
        return max(w['eigenphases_rad'])

    mesh_rows = []
    for m in p['projector_meshes']:
        assert all(m[k] for k in ['all_points_pass', 'all_projector_edges_pass', 'gauge_pass', 'cyclic_start_pass', 'orientation_pass'])
        for e in m['edge_records']:
            check_pair(e)
        assert min(e['pair_sigma_min'] for e in m['edge_records']) == m['minimum_pair_sigma']
        assert max(e['pair_leakage'] for e in m['edge_records']) == m['maximum_pair_leakage']
        mesh_rows.append(dict(npoints=m['mesh_npoints'], positive_phase_rad=spectrum(m),
                              minimum_pair_sigma=m['minimum_pair_sigma'], maximum_pair_leakage=m['maximum_pair_leakage']))
    sa3_rows = []
    for r in p['SA3_wedge_arbitration']['records']:
        assert r['pass']
        check_pair(r, 'projector_singular_values', 'projector_leakage')
        md = r['SA3_metadata']
        es = md['state_energies_Eh']
        assert abs(es[2]-es[1]-md['third_singlet_gap_Eh']) < 1e-12
        sa3_rows.append(dict(angle_degrees=r['angle_degrees'], internal_gap_Eh=md['internal_pair_gap_Eh'],
                             third_gap_Eh=md['third_singlet_gap_Eh'], sigma_min=r['projector_sigma_min'],
                             leakage=r['projector_leakage'], third_root_weight=r['third_root_SA2_weight']))
    history_rows = [dict(sequence='Reference', **{k:v for k,v in mesh_rows[-1].items() if k!='npoints'})]
    for name, m in h['history_Wilson'].items():
        assert m['all_edges_pass'] and m['gauge_pass'] and m['cyclic_start_pass']
        for e in m['edge_records']:
            check_pair(e)
        history_rows.append(dict(sequence=name, positive_phase_rad=spectrum(m),
                                 minimum_pair_sigma=m['minimum_pair_sigma'], maximum_pair_leakage=m['maximum_pair_leakage']))
    for r in h['point_records']:
        assert r['point_quality_pass'] and r['projector_pass'] and r['energy_pass']
        for v in r['comparisons'].values():
            check_pair(v)
    for r in o['history_SA3_records']:
        assert r['quality_pass'] and r['energy_pass']
        check_pair(r['seed_SA2_to_SA3'])
        check_pair(r['reference_SA3_to_history_SA3'])
    for r in o['cross_history_records']:
        check_pair(r['projector_metrics'])
    phase_floor = max(r['errors']['eigenphase_set_error_rad'] for m in p['projector_meshes'] for r in m['random_U2_gauge_audit']['records'])
    references = [dict(comparison=f"{r['coarse_npoints']}/{r['fine_npoints']} mesh", phase_difference_rad=r['eigenphase_set_difference_rad']) for r in p['mesh_comparisons']]
    for name in ['clockwise_vs_reference', 'counterclockwise_vs_reference', 'clockwise_vs_counterclockwise']:
        references.append(dict(comparison=name, phase_difference_rad=h['Wilson_comparisons'][name]['eigenphase_set_difference_rad']))
    for a, b, name in [(1,0,'clockwise_vs_reference'),(2,0,'counterclockwise_vs_reference'),(1,2,'clockwise_vs_counterclockwise')]:
        assert abs(abs(history_rows[a]['positive_phase_rad']-history_rows[b]['positive_phase_rad'])-h['Wilson_comparisons'][name]['eigenphase_set_difference_rad']) < 2e-14
    for i, r in enumerate(p['mesh_comparisons']):
        assert abs(abs(mesh_rows[i]['positive_phase_rad']-mesh_rows[i+1]['positive_phase_rad'])-r['eigenphase_set_difference_rad']) < 2e-14
    m = p['projector_meshes'][-1]
    summary = dict(status='RETAINED_RANK2_MANIFOLD_EVIDENCE', source_statuses=[d['status'] for d in [p,h,o]],
        scientific_scope='Near-degenerate lowest-singlet pair, external isolation, rank-2 fidelity and Wilson spectrum; no scalar Berry phase, CI count, absolute-ground-state or individual-root identity claim.',
        reproducibility_scope='Record-level arithmetic, tables and figures can be regenerated. Cartesian geometry/surface coefficients and original orbital/CI arrays are unavailable; overlaps and Wilson products cannot be independently recomputed from these files.',
        internal_gap_range_Eh=[m['minimum_internal_pair_gap_Eh'],m['maximum_internal_pair_gap_Eh']],
        independent_SA3_minimum_third_gap_Eh=min(r['third_gap_Eh'] for r in sa3_rows),
        independent_SA3_minimum_pair_sigma=min(r['sigma_min'] for r in sa3_rows),
        mesh=mesh_rows, independent_SA3=sa3_rows, histories=history_rows,
        reference_levels=references, gauge_eigenphase_roundoff_rad=phase_floor,
        maximum_reference_history_phase_difference_rad=max(r['phase_difference_rad'] for r in references[2:4]),
        arithmetic_checks=dict(pair_leakage_comparisons=len(leakage_errors), max_leakage_discrepancy=max(leakage_errors),
                               max_spectrum_invariant_discrepancy=max(invariant_errors), phase_comparisons_pass=True),
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(DATA.glob('*_decision.json'))})
    (OUT/'cpdc_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name, rows in [('cpdc_mesh.csv',mesh_rows),('cpdc_sa3.csv',sa3_rows),('cpdc_histories.csv',history_rows),('cpdc_phase_reference_levels.csv',references)]:
        write_csv(name, rows)
    print(json.dumps({k:summary[k] for k in ['status','gauge_eigenphase_roundoff_rad','arithmetic_checks']},indent=2))

if __name__ == '__main__':
    main()
