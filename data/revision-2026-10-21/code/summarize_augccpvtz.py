"""Reanalyze supplied LiH records; performs no electronic-structure optimization."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
def read(p):
    return json.loads((ROOT / p).read_text())
def write_csv(name, rows):
    with (ROOT / 'analysis' / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
def fit(a, c):
    a, c = np.asarray(a), np.asarray(c)
    r = linregress(a, c)
    return dict(slope=float(r.slope), intercept=float(r.intercept),
                r_squared=float(r.rvalue**2),
                zero_intercept_slope=float(a @ c / (a @ a)),
                log_log_slope=float(linregress(np.log(a), np.log(c)).slope))

loops = read('records/lih_augccpvtz_production/loop_results.json')
local = read('analysis/augccpvtz_refinement_validation/recomputed_loops.json')
validation = read('analysis/augccpvtz_refinement_validation/audit.json')
fine = local[-1]
assert fine['qualified'] and all(validation['final_checks_independently_reassembled'].values())
nested = [v for v in loops if v['scale'] < 1 and v['segments_per_side'] == 8]
assert len(nested) == 4 and all(v['qualified'] for v in nested)
nested_rows = []
for v in nested:
    coarse = next(x for x in loops if x['scale'] == v['scale'] and x['segments_per_side'] == 4)
    nested_rows.append(dict(scale=v['scale'], area=v['area_Angstrom_au'], C4=coarse['C'],
        C8=v['C'], D_mu=v['D_mu'], curvature_norm=4*v['C']/v['area_Angstrom_au'],
        mesh_C_difference=abs(v['C']-coarse['C']), eta=v['eta'], leakage=v['leakage'],
        assignment_margin=v['assignment_margin'], qualified=v['qualified']))
write_csv('lih_augccpvtz_nested.csv', nested_rows)
keys = ['local_refinement_factor', 'number_of_edges', 'C', 'D_mu', 'eta', 'leakage',
        'assignment_margin', 'qualified', 'route1_permutation', 'route2_permutation']
write_csv('lih_augccpvtz_boundary.csv', [{k:v[k] for k in keys} for v in local])

with (ROOT/'analysis/lih_100_locator_plaquettes.csv').open() as f:
    locator = list(csv.DictReader(f))
stride_rows = []
for sr in range(1,5):
    for sf in range(1,5):
        rr = [v for v in locator if int(v['R_stride']) == sr and int(v['F_stride']) == sf]
        cc = np.array([float(v['C']) for v in rr])
        stride_rows.append(dict(R_stride=sr, F_stride=sf, count=len(rr),
            area=sr*sf*.05*.00025, minimum_C=float(cc.min()), median_C=float(np.median(cc)),
            maximum_C=float(cc.max()), active_only_fidelity_pass=sum(v['fidelity_pass_active_only']=='True' for v in rr),
            selection_candidate_count=sum(v['candidate']=='True' for v in rr)))
assert sum(v['count'] for v in stride_rows) == 100
write_csv('lih_100_stride_distribution.csv', stride_rows)
full = [v for v in read('analysis/lih_complete_overlap_audit.json')['plaquettes'] if v['mode']=='complete']
if not full:
    full = [v for v in read('analysis/lih_complete_overlap_audit.json')['plaquettes'] if v['mode']!='active_only']
assert len(full) == 100 and not any(v['fidelity_pass'] for v in full)
full_rows = [{k:v[k] for k in ['R0','R1','F0','F1','area','C','D_mu','eta','leakage','assignment_margin','fidelity_pass','assignment_pass']} for v in full]
write_csv('lih_100_complete_overlap_rectangles.csv', full_rows)
summary = dict(production={k:v for k,v in fine.items() if k not in ['edges','vertices']},
    nested=nested_rows, nested_fit=fit([v['area'] for v in nested_rows],[v['C8'] for v in nested_rows]),
    local_boundary=[{k:v[k] for k in keys} for v in local],
    histories=validation['histories'],
    empirical_C_sensitivity=max(v['C_difference_from_direct'] for v in validation['histories']),
    empirical_D_mu_sensitivity=max(v['D_mu_difference_from_direct'] for v in validation['histories']),
    last_mesh_C_change=abs(local[-1]['C']-local[-2]['C']),
    last_mesh_D_mu_change=abs(local[-1]['D_mu']-local[-2]['D_mu']),
    stride_distribution=stride_rows,
    complete_100_descriptive_fit=fit([v['area'] for v in full],[v['C'] for v in full]),
    complete_100_fidelity_pass=0)
(ROOT/'analysis/augccpvtz_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['nested_fit','last_mesh_C_change','empirical_C_sensitivity','complete_100_descriptive_fit']},indent=2))
