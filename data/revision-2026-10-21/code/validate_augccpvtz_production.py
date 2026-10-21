#!/usr/bin/env python3
"""Read-only audit of a completed LiH aug-cc-pVTZ one-shot result archive."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    root, out = args.input.resolve(), args.out.resolve()
    if out == root or root in out.parents:
        ap.error('Output must be outside the original result directory.')
    os.environ['LIH_THREADS'] = str(args.threads)
    source = root / 'calculation_source.py'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    expected_source = '15f834a365312b4271efa3643595f7ec0baa62d2704fca25148ed7fe47545df4'
    if source_sha != expected_source:
        raise RuntimeError('Unexpected archived source; inspect it before executing.')
    spec = importlib.util.spec_from_file_location('lih_archived', source)
    s = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = s
    spec.loader.exec_module(s)
    import numpy as np
    s.lib.num_threads(args.threads)
    out.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((root / name).read_text())
    plan = read('plan.json')
    manifest = read('wavefunction_sha256.json')
    actual_paths = {str(p.relative_to(root)) for p in root.rglob('*.npz')}
    hash_errors = [name for name, sha in manifest.items()
                   if not (root / name).is_file() or s.sha256(root / name) != sha]
    archive_check = {
        'wavefunction_count': len(manifest),
        'hash_mismatches': hash_errors,
        'missing_or_unlisted_npz': sorted(set(manifest) ^ actual_paths),
        'source_sha256': source_sha,
        'source_matches_plan': source_sha == plan['plan']['source_sha256'],
        'plan_fingerprint_matches': hashlib.sha256(json.dumps(plan['plan'], sort_keys=True).encode()).hexdigest() == plan['fingerprint'],
    }
    if hash_errors or archive_check['missing_or_unlisted_npz']:
        raise RuntimeError('Input archive failed file integrity checks.')
    states = {name: s.load(root / name) for name in manifest}
    bad_states, solver_invalid, route_counts, selection_errors = [], [], Counter(), []
    candidates_by_method = {}
    for name, state in states.items():
        if state['metadata']['run_fingerprint'] != plan['fingerprint']:
            raise RuntimeError('Checkpoint fingerprint mismatch: ' + name)
        ck = s.state_checks(state)
        if name.startswith(('solutions/', 'histories/')) and not all(ck.values()):
            bad_states.append({'file': name, 'checks': ck})
        if name.startswith('candidates/'):
            method = name.split('/')[1]
            candidates_by_method.setdefault(method, []).append(state)
            if not all(v for k, v in ck.items() if k != 'same_symmetry_isolation'):
                solver_invalid.append({'file': name, 'checks': ck})
        if name.startswith('solutions/'):
            md = state['metadata']
            selected = md['candidate_selection']['selected_history']
            route_counts[selected] += 1
            routes = [states[f'candidates/{m}/{Path(name).name}'] for m in ('one_step', 'newton')]
            valid = [r for r in routes if all(v for k, v in s.state_checks(r).items() if k != 'same_symmetry_isolation')]
            if not valid or md['average_energy_Eh'] != min(r['metadata']['average_energy_Eh'] for r in valid):
                selection_errors.append(name)
            chosen = states[f'candidates/{selected.removeprefix("candidate_")}/{Path(name).name}']
            if any(not np.array_equal(state[k], chosen[k]) for k in ('mo', 'ci', 'dipole')):
                selection_errors.append(name + ': copied arrays differ')
    selected_states = [v for k, v in states.items() if k.startswith('solutions/')]
    used_states = [v for k, v in states.items() if k.startswith(('solutions/', 'histories/'))]
    qc = {
        'independent_selected': len(selected_states),
        'candidate_count': sum(len(v) for v in candidates_by_method.values()),
        'history_count': len(used_states) - len(selected_states),
        'invalid_selected_or_history': bad_states,
        'invalid_candidates': solver_invalid,
        'selected_routes': dict(route_counts),
        'selection_errors': selection_errors,
        'max_gradient_used': max(v['metadata']['maximum_orbital_gradient'] for v in used_states),
        'min_fifth_A1_gap_used_Eh': min(v['metadata']['excluded_root_gap_Eh'] for v in used_states),
        'max_spin_square_used': max(abs(x) for v in used_states for x in v['metadata']['state_S2']),
        'max_probe_energy_error_Eh': max(v['metadata']['probe_retained_energy_max_error_Eh'] for v in used_states),
        'max_mo_orth_error': max(v['metadata']['mo_orthonormality_max_error'] for v in used_states),
    }
    direct = {(v['metadata']['R_angstrom'], v['metadata']['field_z_au']): v for v in selected_states}
    loops, metrics = [], []
    for original in read('loop_results.json'):
        scale, n = original['scale'], original['segments_per_side']
        recomputed = s.analyze_loop(s.coords(scale, n), direct, scale)
        loops.append(recomputed)
        metrics.append({
            'scale': scale, 'n': n,
            'C_error': abs(recomputed['C'] - original['C']),
            'D_mu_error': abs(recomputed['D_mu'] - original['D_mu']),
            'max_M_error': max(float(np.max(abs(np.asarray(a['M']) - np.asarray(b['M'])))) for a, b in zip(recomputed['edges'], original['edges'])),
            'checks_match': recomputed['checks'] == original['checks'],
        })
        print('Recomputed loop', scale, n, 'C', recomputed['C'], flush=True)
    finest = next(l for l in loops if l['scale'] == 1 and l['segments_per_side'] == 16)
    # Independent determinant-space overlap verifies every edge of the finest loop.
    determinant_errors = []
    for e in finest['edges']:
        a, b = direct[e['a']], direct[e['b']]
        _, x = s.overlaps(a, b, s.mol(a), s.mol(b))
        determinant_errors.append(float(np.max(abs(e['M'] - s.fci_reference(a, b, x)))))
    ambiguous = []
    for i, e in enumerate(finest['edges']):
        if e['squared_overlap_margin'] >= s.THRESHOLDS['assignment_margin_min']:
            continue
        ranks = sorted([(sum(abs(e['M'][j, p[j]])**2 for j in range(4)), p)
                        for p in itertools.permutations(range(4))], reverse=True)
        ambiguous.append({
            'edge_index_zero_based': i, **e,
            'best_and_runnerup_scores': ranks[:2],
            'endpoint_energies_Eh': [direct[e[k]]['metadata']['state_energies_Eh'] for k in ('a', 'b')],
            'endpoint_internal_gaps_Eh': [np.diff(direct[e[k]]['metadata']['state_energies_Eh']) for k in ('a', 'b')],
        })
    histories = []
    ps = s.coords(1., 16)
    for old in read('history_results.json'):
        name = old['history']
        seq = ps if name == 'forward' else list(reversed(ps))
        hs = {}
        parent_errors = []
        previous = s.outpath(root / 'solutions', seq[0])
        for i, key in enumerate(seq):
            path = root / 'histories' / name / f'{i:03d}.npz'
            cur = states[str(path.relative_to(root))]
            if cur['metadata']['parent_sha256'] != s.sha256(previous):
                parent_errors.append(i)
            if i == len(seq)-1:
                closing = cur
            else:
                hs[key] = cur
            previous = path
        loop = s.analyze_loop(ps, hs, 1.)
        comparisons = [s.compare_states(direct[k], hs[k]) for k in ps[:-1]]
        closure = s.compare_states(hs[ps[0]], closing)
        deltas = {k: abs(loop[k] - finest[k]) for k in ('C', 'D_mu')}
        numeric_checks = {
            'same_point': all(s.history_qualified(c) for c in comparisons),
            'closure': s.history_qualified(closure),
            'C': deltas['C'] <= s.THRESHOLDS['history_max_C_difference'],
            'D_mu': deltas['D_mu'] <= s.THRESHOLDS['history_max_D_mu_difference'],
        }
        histories.append({
            'history': name, 'C': loop['C'], 'D_mu': loop['D_mu'],
            'C_error_vs_archived_report': abs(loop['C'] - old['C']),
            'D_mu_error_vs_archived_report': abs(loop['D_mu'] - old['D_mu']),
            'loop_checks': loop['checks'], 'original_history_pass': old['history_pass'],
            'numerical_reproducibility_checks_only': numeric_checks,
            'numerical_reproducibility_pass_only': all(numeric_checks.values()),
            'parent_hash_errors': parent_errors,
            'C_difference_from_direct': deltas['C'], 'D_mu_difference_from_direct': deltas['D_mu'],
            'max_average_energy_difference_Eh': max(c['average_energy_difference_Eh'] for c in comparisons),
            'max_same_point_leakage': max(c['leakage'] for c in comparisons),
            'min_same_point_eta': min(c['eta'] for c in comparisons),
            'closure': closure,
        })
    report = {
        'status': 'REVIEW_REQUIRED',
        'purpose': 'Read-only verification. Original qualification thresholds and decisions are unchanged.',
        'input_archive': root.name,
        'audit_runtime': {'pyscf': __import__('pyscf').__version__, 'numpy': np.__version__, 'scipy': __import__('scipy').__version__},
        'integrity': archive_check, 'wavefunction_quality': qc,
        'loop_recomputation_errors': metrics,
        'max_full_determinant_overlap_error_all_64_edges': max(determinant_errors),
        'ambiguous_edges': ambiguous, 'histories': histories,
        'original_decision': read('LiH_augccpVTZ_decision.json'),
        'remaining_issue': 'One third/fourth-state assignment margin is below the original 0.10 threshold; this also makes both original history_pass values false. The subspace and numerical history checks pass separately.',
        'scope': 'Retained singlet A1 SA4-CASSCF(2e,5o) model only. Dipole metric is a transport-dependent representation comparison, not different physical endpoint dipoles.',
    }
    s.write_json(out / 'audit.json', report)
    s.write_json(out / 'recomputed_loops.json', loops)
    print(json.dumps(s._jsonable({'integrity': archive_check, 'quality': qc, 'histories': histories}), indent=2))


if __name__ == '__main__':
    main()
