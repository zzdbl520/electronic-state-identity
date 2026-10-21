"""Generate manuscript figures directly from the accompanying numerical records."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
S=json.loads((ROOT/'analysis/revision_summary.json').read_text())
A=json.loads((ROOT/'analysis/augccpvtz_summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':True,
 'axes.spines.right':True,'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42,'savefig.bbox':'tight','axes.labelsize':9,
 'legend.fontsize':7.5,'axes.titlesize':10,'lines.linewidth':1.4})
colors=['#176488','#BD6530','#5E7360']
def save(fig,name):
 for ext in ['pdf','png']:
  target=ROOT/'figures'/f'{name}.{ext}'
  temporary=target.with_name(f'{name}_temporary.{ext}')
  fig.savefig(temporary,dpi=600)
  assert temporary.stat().st_size>1000
  temporary.replace(target)
 plt.close(fig)
def panel(ax,title):ax.set_title(title,loc='left',fontweight='bold',pad=9)
def csvrows(path):
 with open(ROOT/path) as f:return list(csv.DictReader(f))

fig,axs=plt.subplots(2,2,figsize=(7.4,6.1),layout='constrained')
ax=axs[0,0];panel(ax,'a  Fixed rectangle, resolved boundary')
x=np.arange(2)
for i,(result,label) in enumerate([(S['augccpvdz']['production'],'aug-cc-pVDZ'),(S['ccpvtz']['production'],'cc-pVTZ'),(A['production'],'aug-cc-pVTZ')]):
 ax.bar(x+(i-1)*.24,[result['C']*1e4,result['D_mu']*1e4],width=.22,color=colors[i],label=label)
ax.set_xticks(x,[r'$\mathcal{C}_{RF}$',r'$D_\mu$']);ax.set_ylabel(r'Value ($10^{-4}$)');ax.set_ylim(0,10.2);ax.legend(frameon=False,loc='upper left',ncol=1)
ax=axs[0,1];panel(ax,'b  aug-cc-pVTZ: molecular area law')
rr=A['nested'];a=np.array([r['area'] for r in rr]);c=np.array([r['C8'] for r in rr])
ax.loglog(a,c,'o',color=colors[2],label='Complete overlap')
ax.loglog(a,2.11862*a,'--',color='0.3',label=r'$C=2.11862 A$')
ax.set_xlabel('Area (Å × a.u. field)');ax.set_ylabel(r'$\mathcal{C}_{RF}$');ax.legend(frameon=False)
ax.text(.05,.73,'Log–log slope: 1.00024',transform=ax.transAxes,fontsize=8)
ax=axs[1,0];panel(ax,'c  Local refinement and reference levels')
rr=A['local_boundary'][:-1]
x=[r['local_refinement_factor'] for r in rr];y=[abs(r['C']-A['production']['C']) for r in rr]
ax.loglog(x,y,'-',color=colors[2])
for v,xx,yy in zip(rr,x,y):ax.plot(xx,yy,'o',color=colors[2],markerfacecolor=colors[2] if v['qualified'] else 'white',markersize=6)
ax.axhline(A['empirical_C_sensitivity'],ls='--',color=colors[1],label=r'History: $6.40\times10^{-10}$')
ax.axhline(A['production']['algebra_errors']['gauge_C'],ls=':',color='0.35',label=r'Gauge: $3.41\times10^{-14}$')
ax.set_ylim(1e-14,2e-8);ax.set_xlim(.8,10)
ax.set_xticks([1,2,4,8],[1,2,4,8]);ax.set_xlabel('Subdivision factor on six selected edges')
ax.set_ylabel(r'$|\mathcal{C}_{RF}(f)-\mathcal{C}_{RF}(16)|$');ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,.14),fontsize=7)
ax.text(.03,.91,'Open: assignment criterion fails',transform=ax.transAxes,fontsize=7.5)
ax=axs[1,1];panel(ax,'d  aug-cc-pVTZ: curvature norm')
for n,m,col in [(4,'s',colors[0]),(8,'o',colors[2])]:
 ax.semilogx([r['area'] for r in A['nested']],[4*r[f'C{n}']/r['area'] for r in A['nested']],m+'-',color=col,markerfacecolor='white' if n==4 else col,label=f'{n} segments / side')
ax.ticklabel_format(axis='y',style='plain',useOffset=False)
ax.set_xlabel('Area (Å × a.u. field)');ax.set_ylabel(r'$4\mathcal{C}_{RF}/A$ (Å$^{-1}$ × a.u. field$^{-1}$)');ax.legend(frameon=False,loc='upper left')
save(fig,'Figure_LiH_revision')

rows=csvrows('analysis/lih_100_locator_plaquettes.csv')
fig,axs=plt.subplots(1,2,figsize=(7.4,3.0),layout='constrained')
ax=axs[0];panel(ax,'a  Initial grid: 100 rectangles')
for cond,col,lab in [(False,'0.65','Above fidelity limits'),(True,colors[0],'Within fidelity limits')]:
 rr=[r for r in rows if (r['fidelity_pass_active_only']=='True')==cond]
 ax.scatter([float(r['area'])*1e4 for r in rr],[float(r['C'])*1e4 for r in rr],s=23,facecolors='none',edgecolors=col,label=lab)
selected=S['legacy_locator']['selected'];ax.scatter(selected['area']*1e4,selected['C']*1e4,marker='*',s=100,color=colors[1],zorder=5,label='Largest value meeting all criteria')
ax.set_xlabel(r'Area ($10^{-4}$ Å × a.u. field)');ax.set_ylabel(r'Active-electron $\mathcal{C}_{RF}$ ($10^{-4}$)');ax.legend(frameon=False,fontsize=6.6)
ax=axs[1];panel(ax,'b  Selection in the full distribution')
for rr,col,lab in [(rows,'0.4','All 100'),([r for r in rows if r['fidelity_pass_active_only']=='True'],colors[0],'90 within fidelity limits')]:
 v=np.sort([float(r['C'])*1e4 for r in rr]);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=col,label=lab)
ax.axvline(1,color='0.45',ls=':',label='Selection threshold')
ax.axvline(selected['C']*1e4,color=colors[1],ls='--',label='Selected value')
ax.set_xlabel(r'Active-electron $\mathcal{C}_{RF}$ ($10^{-4}$)');ax.set_ylabel('Cumulative fraction');ax.legend(frameon=False,fontsize=6.6,loc='lower right')
save(fig,'Figure_LiH_locator')

fig,axs=plt.subplots(1,2,figsize=(7.4,3.1),layout='constrained')
ax=axs[0];panel(ax,'a  SA3 scans in opposite directions')
rr=[r for r in csvrows('records/BeH2/source_data/B3R/B3R_branch_map.csv') if r['protocol']=='SA3']
ax.semilogy([float(r['lambda_bohr']) for r in rr],[float(r['delta_e_absolute']) for r in rr],'o-',markersize=3.5,color=colors[0])
ax.ticklabel_format(axis='x',useOffset=False);ax.set_xlabel(r'$\lambda$ (Bohr)');ax.set_ylabel(r'$|\Delta\bar E|$ ($E_h$)')
ax.text(.03,.18,'21 points; solver floor not calibrated',transform=ax.transAxes,fontsize=7.5)
ax=axs[1];panel(ax,'b  Subspace change: SA3 to SA4')
rr=list(S['beh2_models'].values());v=[r['upper_history_stride1']['maximum_edge_leakage'] for r in rr]
ax.bar(np.arange(3),v,color=[colors[1],colors[0],colors[2]],width=.65)
ax.axhline(.01,color='0.3',ls='--',label='Acceptance ceiling 0.01')
for i,val in enumerate(v):ax.text(i,val+.003,f'{val:.5f}',ha='center',fontsize=8)
ax.set_xticks(range(3),['Restricted\nCAS(6e,6o)','All-electron\nCAS(6e,6o)','Valence\nCAS(4e,6o)'],fontsize=7.5)
ax.set_ylabel('Maximum edge leakage');ax.set_ylim(0,.13);ax.legend(frameon=False,loc='upper left',fontsize=7)
save(fig,'Figure_BeH2_revision')

rr=csvrows('records/O3/source_data/gate4B/O3_gate4B_scaling.csv')
pr=csvrows('records/O3/source_data/gate4B/O3_gate4B_plaquettes.csv')
floor=max(float(r['gauge_test_abs_error']) for r in pr)
fig,axs=plt.subplots(2,2,figsize=(7.4,5.5),layout='constrained')
for i,(direction,unit) in enumerate([('delta_r','Å'),('theta','degree')]):
 r=[r for r in rr if r['direction']==direction];ax=axs[0,i];panel(ax,['a  Antisymmetric stretch','b  Equal-bond bend'][i])
 for k,m in [('C_minus','s'),('C_plus','o')]:ax.loglog([float(v['h']) for v in r],[float(v[k]) for v in r],m+'-',label=['Negative','Positive'][k=='C_plus'])
 ax.set_xlabel(f'Absolute step ({unit})');ax.set_ylabel(r'$\mathcal{C}_{qp}$');ax.legend(frameon=False)
 if i==1:
  vs=[float(v[k]) for v in r for k in ['C_minus','C_plus']];ax.axhspan(min(vs),max(vs),color='0.9',zorder=0)
  ax.text(.02,.02,'Shaded: observed control range',transform=ax.transAxes,fontsize=7)
 ax=axs[1,i];panel(ax,['c  Nuclear-edge leakage','d  Worst singular value'][i])
 for direction,lab in [('delta_r','Stretch'),('theta','Bend')]:
  r=[r for r in rr if r['direction']==direction];hh=np.array([float(v['h']) for v in r]);hh/=hh.max()
  key='lambda_nuclear_max' if i==0 else 'eta_edge_min'
  getattr(ax,'loglog' if i==0 else 'semilogx')(hh,[float(v[key]) for v in r],'o-',label=lab)
 ax.set_xlabel('Step / largest step');ax.set_ylabel('Nuclear leakage' if i==0 else r'$\eta_{\min}$');ax.legend(frameon=False)
 if i==0:ax.axhline(.01,color='0.3',ls=':',label='Leakage limit')
save(fig,'Figure_O3_revision')
print('Created four figures from the supplied numerical data.')
