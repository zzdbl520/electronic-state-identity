"""Replay complete-overlap loop results from NPZ arrays; perform no new SCF."""
from pathlib import Path
import hashlib,json
import numpy as np
from pyscf import lib
from lih_complete_overlap import load,mol,overlaps,fci_reference
from run_lih_revision import analyze,outpath
ROOT=Path(__file__).resolve().parents[1]
lib.num_threads(1)
report={'hashes_checked':0,'loop_replays':0,'maximum_C_replay_error':0.,'maximum_D_mu_replay_error':0.,'basis_results':{}}
for basis in ['augccpvdz','ccpvtz']:
 root=ROOT/'records'/f'lih_revision_{basis}';d=json.loads((root/'revision_results.json').read_text())
 for file,sha in d['files'].items():
  assert hashlib.sha256((root/file).read_bytes()).hexdigest()==sha,file
  report['hashes_checked']+=1
 for rec in d['loops']:
  ps=[tuple(p) for p in rec['vertices']];sols={p:load(outpath(root/'solutions',p)) for p in set(ps)}
  new=analyze(ps,sols)
  for key in ['C','D_mu','eta','leakage','assignment_margin','delta_mu']:
   assert abs(new[key]-rec[key])<2e-11,(basis,key,new[key],rec[key])
  assert new['fidelity_pass'] and new['assignment_pass'] and new['same_assignment'] and new['all_converged']
  assert new['max_orbital_gradient']<1e-4 and new['min_excluded_gap']>.01
  report['maximum_C_replay_error']=max(report['maximum_C_replay_error'],abs(new['C']-rec['C']))
  report['maximum_D_mu_replay_error']=max(report['maximum_D_mu_replay_error'],abs(new['D_mu']-rec['D_mu']))
  report['loop_replays']+=1
 for rec in d['histories']:
  ps=[tuple(p) for p in rec['vertices']];sols={}
  for f in sorted((root/rec['history']).glob('*.npz')):
   ss=load(f);md=ss['metadata'];key=(round(md['R_angstrom'],12),round(md['field_z_au'],12))
   if key not in sols:sols[key]=ss
  new=analyze(ps,sols)
  assert abs(new['C']-rec['C'])<2e-11 and abs(new['D_mu']-rec['D_mu'])<2e-11
  assert new['fidelity_pass'] and new['assignment_pass'] and new['same_assignment']
  report['loop_replays']+=1
 report['basis_results'][basis]={'direct_loops':len(d['loops']),'history_loops':len(d['histories']),'all_pass':True}

audit=json.loads((ROOT/'analysis/lih_complete_overlap_audit.json').read_text())
assert len([r for r in audit['plaquettes'] if r['mode']=='complete_four_electron'])==100
assert not any(r['fidelity_pass'] for r in audit['plaquettes'] if r['mode']=='complete_four_electron')
files=list((ROOT/'records/lih_25point_replay/solutions').rglob('*.npz'))
assert len(files)==25
a,b=map(load,files[:2]);m,x=overlaps(a,b,mol(a),mol(b));err=float(np.max(abs(m['complete_four_electron']-fci_reference(a,b,x))))
assert err<1e-12
report['independent_embedded_fci_element_error']=err
report['status']='PASS'
(ROOT/'analysis/validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
