#!/usr/bin/env python3
"""Recompute tables and descriptive regressions from archived and fresh records."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import linregress

ROOT=Path(__file__).resolve().parents[1]
def fit(a,c):
    a=np.asarray(a);c=np.asarray(c);lr=linregress(a,c);log=linregress(np.log(a),np.log(c))
    k=float(a@c/(a@a));res=c-k*a
    return {'n':len(a),'slope_with_intercept':float(lr.slope),'intercept':float(lr.intercept),
            'R2_with_intercept':float(lr.rvalue**2),'zero_intercept_slope':k,
            'log_log_slope':float(log.slope),'log_log_R2':float(log.rvalue**2),
            'max_relative_residual_zero_intercept':float(np.max(abs(res)/c)),
            'note':'Descriptive fits. Shared vertices and overlapping loops preclude interpreting ordinary regression p-values as independent-sample significance.'}

def main():
    ps=json.loads((ROOT/'records/lih_legacy/aug-cc-pVDZ/LiH_S1_diagnostics.json').read_text())['plaquettes_ranked']
    rows=[]
    for p in ps:
        rows.append({'plaquette':p['plaquette'],'R_stride':p['R_stride_points'],'F_stride':p['F_stride_points'],
                     'R0':p['R0_angstrom'],'R1':p['R1_angstrom'],'F0':p['F0_au'],'F1':p['F1_au'],
                     'area':(p['R1_angstrom']-p['R0_angstrom'])*(p['F1_au']-p['F0_au']),
                     'C':p['C_qp'],'D_mu':p['dipole_path_difference_normalized'],
                     'fidelity_pass_active_only':p['all_four_direct_edges_strict'],'candidate':p['candidate_window'],
                     'eta':min(e['state_subspace']['sigma_min'] for e in p['edges'].values()),
                     'leakage':max(e['state_subspace']['leakage'] for e in p['edges'].values()),
                     'selected':p['plaquette']=='R0->3:F0->4'})
    with (ROOT/'analysis/lih_100_locator_plaquettes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'legacy_locator':{}}
    for label,ss in [('all',rows),('fidelity_qualified',[r for r in rows if r['fidelity_pass_active_only']]),('selection_candidates',[r for r in rows if r['candidate']])]:
        c=[r['C'] for r in ss];a=[r['area'] for r in ss]
        summary['legacy_locator'][label]={**fit(a,c),'quantiles_C':dict(zip(['min','q25','median','q75','max'],np.quantile(c,[0,.25,.5,.75,1]).tolist()))}
    selected=next(r for r in rows if r['selected'])
    summary['legacy_locator']['selected']=selected
    summary['legacy_locator']['selected_rank_descending_all']=1+sum(r['C']>selected['C'] for r in rows)
    summary['legacy_locator']['selected_rank_descending_qualified']=1+sum(r['C']>selected['C'] for r in rows if r['fidelity_pass_active_only'])
    summary['legacy_locator']['definition_warning']='All legacy locator statistics use active-only overlap and absolute-overlap assignments. They document original selection; they are not full-electron fidelity-qualified molecular curvature data.'
    for basis in ['augccpvdz','ccpvtz']:
        d=json.loads((ROOT/f'records/lih_revision_{basis}/revision_results.json').read_text())
        prod=next(r for r in d['loops'] if r['scale']==1 and r['segments_per_side']==16)
        simple=lambda r:{k:v for k,v in r.items() if k not in ['edges','vertices','direct_comparisons']}
        s={'production':simple(prod),'loops':[simple(r) for r in d['loops']],
           'histories':[simple(h) for h in d['histories']]}
        s['empirical_C_sensitivity']=max(abs(h['C']-prod['C']) for h in d['histories'])
        s['empirical_D_mu_sensitivity']=max(abs(h['D_mu']-prod['D_mu']) for h in d['histories'])
        s['mesh_C_difference_8_16']=abs(next(r['C'] for r in d['loops'] if r['scale']==1 and r['segments_per_side']==8)-prod['C'])
        nested=[r for r in d['loops'] if r['segments_per_side']==8]
        if len(nested)>1:
            s['nested_fit_all']=fit([r['area'] for r in nested],[r['C'] for r in nested])
            last=[r for r in nested if r['scale']<=.25]
            s['nested_fit_smallest3']=fit([r['area'] for r in last],[r['C'] for r in last])
            s['curvature_norm_smallest']=4*last[-1]['C']/last[-1]['area']
        summary[basis]=s
    full=json.loads((ROOT/'analysis/lih_complete_overlap_audit.json').read_text())
    summary['complete_overlap_comparison']={k:v for k,v in full.items() if k not in ['edge_records','plaquettes','wavefunction_hashes']}
    summary['complete_overlap_comparison']['full_electron_qualified_original_rectangles']=sum(p['fidelity_pass'] for p in full['plaquettes'] if p['mode']=='complete_four_electron')
    beh=json.loads((ROOT/'records/beh2_model_comparison/G0C_forensic_summary.json').read_text())
    summary['beh2_models']=beh['models']
    cp=json.loads((ROOT/'records/CPDC/CPDC_ccpVTZ_rank2_projector_v1_decision.json').read_text())
    summary['cpdc_gauge_eigenphase_floor_rad']=max(r['errors']['eigenphase_set_error_rad'] for m in cp['projector_meshes'] for r in m['random_U2_gauge_audit']['records'])
    (ROOT/'analysis/revision_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['beh2_models','complete_overlap_comparison']},indent=2))

if __name__=='__main__':main()
