#!/usr/bin/env python3
"""LiH CAS(2e,5o): compare two-active-electron and complete four-electron overlaps.

For one doubly occupied inactive orbital, the alpha (and beta) determinant
overlap for active orbitals a,c is X00*Xac-Xa0*X0c. No core normalization
or scalar division is performed. Coordinates and electronic states are fixed.
"""
import argparse, csv, hashlib, itertools, json
from pathlib import Path
import numpy as np
from pyscf import gto, lib, fci

def load(path):
    with np.load(path, allow_pickle=False) as d:
        md=json.loads(str(d['metadata_json'].item()))
        return {'metadata':md,'path':str(path),'mo':d['mo_coeff'],
                'ci':np.stack([d[f'ci_root_{i}'] for i in range(4)]),
                'dipole':d['dipole_au']}

def mol(s):
    m=s['metadata'];r=m['R_angstrom']
    return gto.M(atom=[('Li',(0,0,-r/2)),('H',(0,0,r/2))],unit='Angstrom',
                 basis=m['basis'],symmetry='C2v',spin=0,verbose=0)

def overlaps(a,b,ma,mb):
    x=a['mo'][:,:6].conj().T @ gto.intor_cross('int1e_ovlp',ma,mb) @ b['mo'][:,:6]
    xa=x[1:,1:]
    determinants=x[0,0]*xa-np.outer(x[1:,0],x[0,1:])
    def contract(d):return np.einsum('iab,jcd,ac,bd->ij',a['ci'].conj(),b['ci'],d,d,optimize=True)
    return {'active_only':contract(xa),'complete_four_electron':contract(determinants)},x

def polar(m):
    u,s,vh=np.linalg.svd(m);return u@vh,s

def assign(m,power=2):
    scores=sorted([(sum(abs(m[i,p[i]])**power for i in range(4)),p)
                   for p in itertools.permutations(range(4))],reverse=True)
    return list(scores[0][1]),float((scores[0][0]-scores[1][0])/4)

def edge(m):
    t,s=polar(m);p,margin=assign(m)
    return {'T':t,'sigma_min':float(s.min()),'leakage':float(1-np.mean(s*s)),
            'squared_overlap_permutation':p,'squared_overlap_margin':margin,
            'absolute_overlap_permutation':assign(m,1)[0],
            'absolute_overlap_margin':assign(m,1)[1]}

def compose(ps):
    p=np.arange(4)
    for step in ps:p=np.asarray(step)[p]
    return p.tolist()

def route(ms,dipole):
    es=[edge(m) for m in ms]
    k1=es[0]['T']@es[1]['T'];k2=es[2]['T']@es[3]['T']
    diff=np.array([k1@o@k1.conj().T-k2@o@k2.conj().T for o in dipole])
    p1=compose([es[0]['squared_overlap_permutation'],es[1]['squared_overlap_permutation']])
    p2=compose([es[2]['squared_overlap_permutation'],es[3]['squared_overlap_permutation']])
    out={'C':float(np.linalg.norm(k1-k2)/4),'delta_mu':float(np.linalg.norm(diff)),
         'D_mu':float(np.linalg.norm(diff)/np.linalg.norm(dipole)),
         'eta':min(e['sigma_min'] for e in es),'leakage':max(e['leakage'] for e in es),
         'assignment_margin':min(e['squared_overlap_margin'] for e in es),
         'route1_permutation':p1,'route2_permutation':p2,'same_assignment':p1==p2,
         'eigenphases':sorted(np.angle(np.linalg.eigvals(k1@k2.conj().T)).tolist())}
    out['fidelity_pass']=out['eta']>=.98 and out['leakage']<=.01
    out['assignment_pass']=out['assignment_margin']>=.1
    return out

def fci_reference(a,b,x):
    strings=fci.cistring.make_strings(range(6),2)
    idx={int(s):i for i,s in enumerate(strings)}
    def embed(cs):
        answer=[]
        for c in cs:
            v=np.zeros((15,15))
            for i in range(5):
                for j in range(5):v[idx[1|(1<<(i+1))],idx[1|(1<<(j+1))]]=c[i,j]
            answer.append(v)
        return answer
    aa,bb=embed(a['ci']),embed(b['ci'])
    return np.array([[fci.addons.overlap(c,d,6,(2,2),s=x) for d in bb] for c in aa])

def jsonable(x):
    if isinstance(x,np.ndarray):return {'real':x.real.tolist(),'imag':x.imag.tolist()} if np.iscomplexobj(x) else x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();lib.num_threads(1)
    ss=[load(f) for f in sorted(args.root.glob('solutions/**/*.npz'))]
    sols={(s['metadata']['R_angstrom'],s['metadata']['field_z_au']):s for s in ss}
    rs=sorted({r for r,f in sols});fs=sorted({f for r,f in sols});mols={r:mol(sols[r,fs[0]]) for r in rs}
    matrices={};records=[];self_error=0.;reciprocity=0.;fci_error=0.
    for key,s in sols.items():
        mm,x=overlaps(s,s,mols[key[0]],mols[key[0]])
        self_error=max(self_error,max(np.max(abs(m-np.eye(4))) for m in mm.values()))
    for dim in [0,1]:
        for a,b in itertools.combinations(sols,2):
            if a[1-dim]!=b[1-dim]:continue
            mm,x=overlaps(sols[a],sols[b],mols[a[0]],mols[b[0]])
            rev,_=overlaps(sols[b],sols[a],mols[b[0]],mols[a[0]])
            reciprocity=max(reciprocity,max(np.max(abs(mm[k]-rev[k].conj().T)) for k in mm))
            if len(records)<3:
                fci_error=max(fci_error,float(np.max(abs(mm['complete_four_electron']-fci_reference(sols[a],sols[b],x)))))
            matrices[a,b]=mm;matrices[b,a]={k:v.conj().T for k,v in mm.items()}
            records.append({'a':a,'b':b,'matrices':mm,'metrics':{k:edge(v) for k,v in mm.items()}})
    rows=[]
    for r0,r1 in itertools.combinations(rs,2):
        for f0,f1 in itertools.combinations(fs,2):
            a,b,c,d=(r0,f0),(r1,f0),(r0,f1),(r1,f1)
            for mode in ['active_only','complete_four_electron']:
                out=route([matrices[v,w][mode] for v,w in [(a,b),(b,d),(a,c),(c,d)]],sols[d]['dipole'])
                rows.append({'mode':mode,'R0':r0,'R1':r1,'F0':f0,'F1':f1,
                             'area':(r1-r0)*(f1-f0),**out})
    fixed=[r for r in rows if np.allclose([r['R0'],r['R1'],r['F0'],r['F1']],[5.8,5.95,.0105,.0115],rtol=0,atol=1e-10)]
    result={'overlap_definition':'complete determinants include optimized inactive core and core-active cross terms',
            'assignment_definition':'sum of squared absolute overlaps; legacy absolute-overlap results retained for comparison',
            'pyscf_version':__import__('pyscf').__version__,'wavefunctions':len(ss),
            'max_orbital_gradient':max(s['metadata']['maximum_orbital_gradient'] for s in ss),
            'all_converged':all(s['metadata']['converged'] for s in ss),
            'self_overlap_error':float(self_error),'reverse_overlap_error':float(reciprocity),
            'pyscf_embedded_fci_check_max_error':fci_error,'fixed_window_direct':fixed,'plaquettes':rows,
            'wavefunction_hashes':{str(Path(s['path']).relative_to(args.root)):hashlib.sha256(Path(s['path']).read_bytes()).hexdigest() for s in ss},
            'edge_records':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,default=jsonable,indent=2)+'\n')
    with args.output.with_suffix('.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ['plaquettes','wavefunction_hashes','edge_records']},default=jsonable,indent=2))

if __name__=='__main__':main()
