#!/usr/bin/env python3
"""Fixed-window full-overlap validation, mesh refinement and nested-loop scaling.

The plan is written before any new SCF calculation. No signal-based selection,
threshold changes, core division or post hoc center optimization are used.
The original PySCF 2.13.1 electronic-state solver is supplied via --workflow.
"""
import argparse, hashlib, importlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
from pyscf import lib
from lih_complete_overlap import load, mol, overlaps, edge, compose, jsonable

def coords(scale,n):
    rc,fc=5.875,.011;dr,df=.15*scale,.001*scale
    r0,r1=rc-dr/2,rc+dr/2;f0,f1=fc-df/2,fc+df/2
    sides=[[(r0+dr*k/n,f0) for k in range(n)],[(r1,f0+df*k/n) for k in range(n)],
           [(r1-dr*k/n,f1) for k in range(n)],[(r0,f1-df*k/n) for k in range(n)]]
    ps=[(round(r,12),round(f,12)) for s in sides for r,f in s]
    return ps+[ps[0]]

def outpath(root,key):return root/f'R_{key[0]:.12f}_F_{key[1]:.12f}.npz'

def analyze(ps,sols):
    ms=[];es=[];ma={};reverse_error=0.;self_error=0.
    for a,b in zip(ps[:-1],ps[1:]):
        if a not in ma:ma[a]=mol(sols[a])
        if b not in ma:ma[b]=mol(sols[b])
        mm,_=overlaps(sols[a],sols[b],ma[a],ma[b]);m=mm['complete_four_electron']
        rev,_=overlaps(sols[b],sols[a],ma[b],ma[a])
        reverse_error=max(reverse_error,float(np.max(abs(m-rev['complete_four_electron'].conj().T))))
        ms.append(m);es.append(edge(m))
    mid=len(ms)//2;k1=np.eye(4);k2=np.eye(4)
    for e in es[:mid]:k1=k1@e['T']
    for e in reversed(es[mid:]):k2=k2@e['T'].conj().T
    o=sols[ps[mid]]['dipole'];dif=np.array([k1@v@k1.conj().T-k2@v@k2.conj().T for v in o])
    p1=compose([e['squared_overlap_permutation'] for e in es[:mid]])
    p2=compose([np.argsort(e['squared_overlap_permutation']).tolist() for e in reversed(es[mid:])])
    eta=min(e['sigma_min'] for e in es);leak=max(e['leakage'] for e in es)
    margin=min(e['squared_overlap_margin'] for e in es)
    rng=np.random.default_rng(20261013)
    gs={p:np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0] for p in ps[:-1]}
    wg=np.eye(4,dtype=complex);w=np.eye(4)
    for (a,b),m,e in zip(zip(ps[:-1],ps[1:]),ms,es):
        wg=wg@edge(gs[a].conj().T@m@gs[b])['T'];w=w@e['T']
    C=float(np.linalg.norm(k1-k2)/4)
    result={'segments_per_side':len(ms)//4,'C':C,'delta_mu':float(np.linalg.norm(dif)),
       'D_mu':float(np.linalg.norm(dif)/np.linalg.norm(o)),'eta':eta,'leakage':leak,
       'assignment_margin':margin,'fidelity_pass':eta>=.98 and leak<=.01,
       'assignment_pass':margin>=.1,'same_assignment':p1==p2,'route1_permutation':p1,'route2_permutation':p2,
       'eigenphases':sorted(np.angle(np.linalg.eigvals(w)).tolist()),
       'gauge_C_abs_error':abs(float(np.linalg.norm(np.eye(4)-wg)/4)-C),
       'gauge_W_covariance_error':float(np.linalg.norm(wg-gs[ps[0]].conj().T@w@gs[ps[0]])),
       'reverse_overlap_error':reverse_error,
       'max_orbital_gradient':max(sols[p]['metadata']['maximum_orbital_gradient'] for p in set(ps)),
       'min_excluded_gap':min(sols[p]['metadata']['excluded_root_gap_Eh'] for p in set(ps)),
       'all_converged':all(sols[p]['metadata']['converged'] for p in set(ps)),
       'vertices':ps,'edges':[{'a':a,'b':b,'M':m,**e} for (a,b),m,e in zip(zip(ps[:-1],ps[1:]),ms,es)]}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--workflow',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--basis',required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--nested',action='store_true');p.add_argument('--histories',action='store_true')
    p.add_argument('--threads',type=int,default=2);args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    lib.num_threads(args.threads)
    if not Path(f'/proc/{os.getpid()}/statm').exists():lib.current_memory=lambda:(0.,0.)
    sys.path.insert(0,str(args.workflow));s1=importlib.import_module('LiH_S1_physical_field_locator')
    selector='original S0 target-AO selector'
    if args.basis.lower() in ['cc-pvtz','aug-cc-pvtz']:
        s3=importlib.import_module('LiH_S3_fixed_window_basis_robustness')
        s1.s0.build_initial_mos=s3.build_initial_mos_pvtz
        selector='archived S3 occupied-valence-locked target-AO selector'
    specs=[(1.,n) for n in [4,8,16]]
    if args.nested:specs += [(s,n) for s in [.5,.25,.125,.0625] for n in [4,8]]
    plan={'created_unix':time.time(),'basis':args.basis,'center':[5.875,.011],
          'full_widths':[.15,.001],'loop_scales_and_segments':specs,
          'thresholds':{'eta_min':.98,'leakage_max':.01,'squared_assignment_margin_min':.10,
                        'max_orbital_gradient':1e-4,'min_excluded_gap':.01},
          'overlap':'complete four-electron determinant contraction','seed_selector':selector,
          'histories':args.histories,'source_solver_sha256':hashlib.sha256((args.workflow/'LiH_S1_physical_field_locator.py').read_bytes()).hexdigest()}
    if not (root/'plan.json').exists():(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    elif 'seed_selector' not in json.loads((root/'plan.json').read_text()):
        (root/'seed_selector_clarification.json').write_text(json.dumps({'time':time.time(),'seed_selector':selector,
            'reason':'match the original basis-specific S0/S3 initial-orbital protocol'},indent=2)+'\n')
    (root/'solutions').mkdir(exist_ok=True)
    sols={}
    if args.cache:
        for f in args.cache.glob('solutions/**/*.npz'):
            s=load(f);md=s['metadata'];key=(round(md['R_angstrom'],12),round(md['field_z_au'],12))
            if md['basis'].lower()!=args.basis.lower():raise ValueError('cached basis mismatch')
            out=outpath(root/'solutions',key)
            if not out.exists():shutil.copy2(f,out)
    protocol=s1.Protocol(args.basis,(5.8,5.95),(.0105,.0115))
    loop_results=[]
    for scale,n in specs:
        ps=coords(scale,n)
        for k in ps[:-1]:
            if k in sols:continue
            path=outpath(root/'solutions',k)
            if not path.exists():
                print('RUN',args.basis,k,flush=True);s1.casscf_job(*k,protocol,path)
            sols[k]=load(path)
        r=analyze(ps,sols);r['scale']=scale;r['area']=.15*.001*scale**2;loop_results.append(r)
        (root/'loop_results.json').write_text(json.dumps(loop_results,default=jsonable,indent=2)+'\n')
        print('LOOP',json.dumps({k:v for k,v in r.items() if k not in ['vertices','edges']}),flush=True)
    history_results=[]
    if args.histories:
        ps=coords(1.,16)
        for name,seq in [('clockwise',ps),('counterclockwise',list(reversed(ps)))]:
            hroot=root/name;hroot.mkdir(exist_ok=True);hs={};prev=sols[seq[0]]
            # A reoptimized first vertex and closing vertex are kept distinct on disk.
            for i,k in enumerate(seq):
                path=hroot/f'{i:03d}.npz'
                if not path.exists():
                    print('HISTORY',args.basis,name,i,k,flush=True)
                    s1.casscf_job(*k,protocol,path,initial_from=(mol(prev),prev['mo']),history=name)
                now=load(path)
                if i==len(seq)-1:closing=now
                else:hs[k]=now
                prev=now
            # Closed-loop transport uses the single first-frame at both endpoints;
            # the independently reoptimized closure is separately compared.
            hr=analyze(ps,hs);hr['history']=name
            m,_=overlaps(hs[ps[0]],closing,mol(closing),mol(closing));ce=edge(m['complete_four_electron'])
            comparisons=[]
            for k,s in hs.items():
                m,_=overlaps(s,sols[k],mol(s),mol(s));e=edge(m['complete_four_electron'])
                comparisons.append({'vertex':k,'delta_E':abs(s['metadata']['average_energy_Eh']-sols[k]['metadata']['average_energy_Eh']),
                                    'leakage':e['leakage'],'eta':e['sigma_min']})
            hr['closure_eta']=ce['sigma_min'];hr['closure_leakage']=ce['leakage'];hr['direct_comparisons']=comparisons
            hr['history_direct_max_delta_E']=max(x['delta_E'] for x in comparisons)
            hr['history_direct_max_leakage']=max(x['leakage'] for x in comparisons)
            history_results.append(hr)
            (root/'history_results.json').write_text(json.dumps(history_results,default=jsonable,indent=2)+'\n')
            print('HISTORY_DONE',name,'C',hr['C'],'max dE',hr['history_direct_max_delta_E'],flush=True)
    final={'plan':plan,'loops':loop_results,'histories':history_results,
           'environment':{'pyscf':__import__('pyscf').__version__,'numpy':np.__version__},
           'files':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.npz')}}
    (root/'revision_results.json').write_text(json.dumps(final,default=jsonable,indent=2)+'\n')

if __name__=='__main__':main()
