#!/usr/bin/env python3
"""Offline raw-wavefunction audit; never calls a CASSCF optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    assert root != out and root not in out.parents
    expected_sources = {
        'lih_engine.py':'15f834a365312b4271efa3643595f7ec0baa62d2704fca25148ed7fe47545df4',
        'refine_lih.py':'0fb0676cebc3730db8c4b06298dce1d28569119e4278c6c273b6866dd8847882',
    }
    for f, sha in expected_sources.items():
        if digest(root/'code'/f) != sha:
            raise RuntimeError('Unexpected source: '+f)
    sys.path.insert(0, str(root/'code'))
    import refine_lih as r
    import numpy as np
    r.e.lib.num_threads(args.threads)
    plan = read(root/'plan.json')
    assert hashlib.sha256(json.dumps(plan['plan'],sort_keys=True).encode()).hexdigest() == plan['fingerprint']
    assert plan['plan']['driver_sha256'] == expected_sources['refine_lih.py']
    assert plan['plan']['engine_sha256'] == expected_sources['lih_engine.py']
    assert plan['plan']['thresholds'] == r.e.THRESHOLDS
    for f, sha in plan['plan']['parent_report_hashes'].items():
        assert digest(root/'baseline'/f) == sha
    parent_plan = read(root/'baseline/plan.json')
    assert parent_plan['fingerprint'] == plan['plan']['parent_fingerprint']
    assert hashlib.sha256(json.dumps(parent_plan['plan'],sort_keys=True).encode()).hexdigest() == parent_plan['fingerprint']
    manifest = read(root/'wavefunction_sha256.json')
    assert set(manifest) == {str(p.relative_to(root)) for p in root.rglob('*.npz')}
    states = {}
    for name, sha in manifest.items():
        assert digest(root/name) == sha
        s = r.e.load(root/name)
        expected = parent_plan['fingerprint'] if name.startswith('baseline/') else plan['fingerprint']
        assert s['metadata']['run_fingerprint'] == expected
        states[name] = s
    original_manifest = read(root/'baseline/wavefunction_sha256.json')
    baseline_sols = {}
    selected = {}
    counters = Counter()
    bad_states = []
    solver_bad_candidates = []
    selections = Counter()
    for name, s in states.items():
        group = name.split('/')[0]
        counters[group] += 1
        ck = r.e.state_checks(s)
        if group == 'candidates':
            if not all(v for k,v in ck.items() if k!='same_symmetry_isolation'):
                solver_bad_candidates.append({'file':name,'checks':ck})
        elif not all(ck.values()):
            bad_states.append({'file':name,'checks':ck})
        key = (s['metadata']['R_angstrom'], s['metadata']['field_z_au'])
        if group == 'baseline':
            assert manifest[name] == original_manifest[name.removeprefix('baseline/')]
            baseline_sols[key] = s
        if group in ('baseline','solutions'):
            selected[key] = s
        if group == 'solutions':
            routes = [states[f'candidates/{m}/{Path(name).name}'] for m in ('one_step','newton')]
            valid = [v for v in routes if all(x for k,x in r.e.state_checks(v).items() if k!='same_symmetry_isolation')]
            assert s['metadata']['average_energy_Eh'] == min(v['metadata']['average_energy_Eh'] for v in valid)
            choice = s['metadata']['candidate_selection']['selected_history'].removeprefix('candidate_')
            selections[choice] += 1
            chosen = states[f'candidates/{choice}/{Path(name).name}']
            for a in ('mo','ci','dipole'):
                assert np.array_equal(s[a],chosen[a])
    assert not bad_states
    out.mkdir(parents=True, exist_ok=True)
    factors = [1] + sorted(int(p.stem.split('_')[-1]) for p in (root/'analysis').glob('loop_factor_*.json'))
    loops, reports, comparisons = [], [], []
    for f in factors:
        ps = r.refined_path(f)
        new = r.analyze(ps, selected, f)
        fn = 'baseline_n16.json' if f==1 else f'loop_factor_{f:03d}.json'
        old = read(root/'analysis'/fn)
        errors = {k:abs(new[k]-old[k]) for k in ('C','D_mu','eta','leakage','assignment_margin')}
        errors['max_M'] = max(float(np.max(abs(a['M']-np.asarray(b['M'])))) for a,b in zip(new['edges'],old['edges']))
        assert max(errors.values()) < 1e-11
        assert new['checks'] == old['checks']
        if loops:
            comparisons.append(r.mesh_comparison(loops[-1],new))
        loops.append(new)
        reports.append({'factor':f,'recomputation_errors':errors})
    old_comparisons = read(root/'mesh_comparisons.json')
    for a,b in zip(comparisons, old_comparisons):
        assert a['checks']==b['checks'] and r.e._jsonable(a['changed_edge_permutations'])==b['changed_edge_permutations']
    fine = loops[-1]
    assert comparisons[-1]['pass']
    f = fine['local_refinement_factor']
    ps = r.refined_path(f)
    refs = []
    for ed in fine['edges']:
        a,b = selected[ed['a']],selected[ed['b']]
        _,x = r.e.overlaps(a,b,r.e.mol(a),r.e.mol(b))
        refs.append(float(np.max(abs(ed['M']-r.e.fci_reference(a,b,x)))))
    assert max(refs)<1e-12
    history_summaries=[]
    history_full=[]
    for old in read(root/'analysis'/f'histories_factor_{f:03d}.json'):
        name=old['history']
        seq=ps if name=='forward' else list(reversed(ps))
        hs={}
        previous=selected[seq[0]]
        for i,k in enumerate(seq):
            cur=states[f'histories/factor_{f:03d}/{name}/{i:04d}.npz']
            assert cur['metadata']['parent_sha256']==digest(previous['path'])
            assert (cur['metadata']['R_angstrom'],cur['metadata']['field_z_au'])==k
            assert cur['metadata']['history']==f'refine_{f:03d}_{name}'
            if i==len(seq)-1:
                closing=cur
            else:
                hs[k]=cur
            previous=cur
        new=r.analyze(ps,hs,f)
        cs=[r.e.compare_states(selected[k],hs[k]) for k in ps[:-1]]
        closure=r.e.compare_states(hs[ps[0]],closing)
        assert all(r.e.history_qualified(c) for c in cs) and r.e.history_qualified(closure)
        errors={k:abs(new[k]-old[k]) for k in ('C','D_mu','assignment_margin')}
        assert max(errors.values())<1e-11
        assert new['checks']==old['checks']
        c_diff=abs(new['C']-fine['C'])
        d_diff=abs(new['D_mu']-fine['D_mu'])
        checks={'loop_qualified':new['qualified'],
                'C_reproducibility':c_diff<=r.e.THRESHOLDS['history_max_C_difference'],
                'D_mu_reproducibility':d_diff<=r.e.THRESHOLDS['history_max_D_mu_difference'],
                'same_assignment_as_direct':new['route1_permutation']==fine['route1_permutation'] and new['route2_permutation']==fine['route2_permutation']}
        assert all(checks.values()) and old['history_pass']
        history_full.append({'history':name,'loop':new,'comparisons':cs,'closure':closure})
        history_summaries.append({'history':name,'C':new['C'],'D_mu':new['D_mu'],
            'recomputation_errors':errors,'checks':checks,'history_pass':True,
            'max_average_energy_difference_Eh':max(c['average_energy_difference_Eh'] for c in cs),
            'min_same_point_eta':min(c['eta'] for c in cs),
            'max_same_point_leakage':max(c['leakage'] for c in cs),
            'max_matched_root_energy_difference_Eh_diagnostic_only':max(c['matched_root_energy_max_difference_Eh'] for c in cs),
            'C_difference_from_direct':c_diff,'D_mu_difference_from_direct':d_diff,
            'minimum_assignment_margin':new['assignment_margin'],
            'parent_chain_verified_points':len(seq),'closure':closure})
    inherited = read(root/'baseline/LiH_augccpVTZ_decision.json')['checks']
    final_checks={
        'inherited_global_mesh_converged':inherited['main_mesh_converged'],
        'inherited_nested_loops_qualified':inherited['nested_loops_qualified'],
        'inherited_nested_mesh_converged':inherited['nested_mesh_converged'],
        'supplemental_finest_loop_qualified':fine['qualified'],
        'adjacent_refinement_levels_stable':comparisons[-1]['pass'],
        'C_consistent_with_original_n16':r.close_metric(fine,loops[0],'C')['pass'],
        'D_mu_consistent_with_original_n16':r.close_metric(fine,loops[0],'D_mu')['pass'],
        'opposed_histories_qualified_on_final_mesh':len(history_summaries)==2 and all(h['history_pass'] for h in history_summaries)}
    decision=read(root/'LiH_local_refine_decision.json')
    assert final_checks==decision['checks'] and all(final_checks.values())
    used=[s for n,s in states.items() if not n.startswith('candidates/')]
    metadata={
        'max_gradient':max(s['metadata']['maximum_orbital_gradient'] for s in used),
        'min_excluded_singlet_A1_gap_Eh':min(s['metadata']['excluded_root_gap_Eh'] for s in used),
        'max_probe_consistency_error_Eh':max(s['metadata']['probe_retained_energy_max_error_Eh'] for s in used),
        'max_abs_S2':max(abs(v) for s in used for v in s['metadata']['state_S2']),
        'max_mo_orthonormality_error':max(s['metadata']['mo_orthonormality_max_error'] for s in used),
        'all_selected_and_history_checks_pass':not bad_states,
    }
    result={'status':'QUALIFIED_VERIFIED','original_decision':decision,
        'integrity':{'wavefunctions':len(manifest),'group_counts':dict(counters),'all_hashes_match':True,
                     'all_code_and_plan_hashes_match':True,'baseline_matches_original_manifest':True,
                     'selected_candidate_arrays_match':True,'selected_routes':dict(selections)},
        'wavefunction_quality':metadata,'solver_invalid_candidates':solver_bad_candidates,
        'loop_recomputation':reports,'recomputed_mesh_comparisons':comparisons,
        'all_154_edges_embedded_full_determinant_max_error':max(refs),
        'histories':history_summaries,'final_checks_independently_reassembled':final_checks,
        'audit_runtime':r.runtime(), 'production_runtime':plan['plan']['runtime_versions'],
        'scope':'Same singlet A1 SA4-CASSCF(2e,5o) model. No new quantum-chemistry optimization. Nested-loop checks inherit the checksum-verified original evidence; new loops and both histories recomputed from raw wavefunctions.'}
    r.e.write_json(out/'audit.json',result)
    r.e.write_json(out/'recomputed_loops.json',loops)
    r.e.write_json(out/'recomputed_histories.json',history_full)
    print(json.dumps(r.e._jsonable({k:v for k,v in result.items() if k not in ('original_decision','recomputed_mesh_comparisons')}),indent=2))


if __name__=='__main__':
    main()
