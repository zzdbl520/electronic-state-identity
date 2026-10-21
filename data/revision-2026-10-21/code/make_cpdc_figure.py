#!/usr/bin/env python3
"""Figure 6 from recorded CPDC quantities, without reconstructing missing geometry."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
S=json.loads((ROOT/'analysis/cpdc_summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'xtick.direction':'in','ytick.direction':'in',
 'pdf.fonttype':42,'axes.labelsize':9,'legend.fontsize':7.2,'axes.titlesize':10,'lines.linewidth':1.4})
blue,orange,green='#176488','#BD6530','#5E7360'
fig,axs=plt.subplots(2,2,figsize=(7.4,6.0),layout='constrained')
def panel(ax,s): ax.set_title(s,loc='left',fontweight='bold',pad=9)
ax=axs[0,0];panel(ax,'a  Independent three-state calculations')
r=S['independent_SA3'];x=[v['angle_degrees'] for v in r]
ax.semilogy(x,[v['third_gap_Eh'] for v in r],'o-',color=blue,label=r'$E_3-E_2$: external separation')
ax.semilogy(x,[v['internal_gap_Eh'] for v in r],'s-',color=orange,label=r'$E_2-E_1$: internal splitting')
ax.axhline(.002,ls=':',color='0.35',label='External-gap threshold')
ax.set_xticks(x);ax.set_xlabel('Loop angle (degree)');ax.set_ylabel(r'Energy gap ($E_h$)')
ax.set_ylim(5e-6,.08);ax.legend(frameon=False,loc='center left',fontsize=7)
ax=axs[0,1];panel(ax,'b  Wilson spectrum: positive eigenphase')
r=S['mesh'];ax.plot([v['npoints'] for v in r],[v['positive_phase_rad']*1e4 for v in r],'o-',color=blue,label='Independent-point reference')
for v,marker,col,label in zip(S['histories'][1:],['s','x'],[orange,green],['Clockwise initialization','Counterclockwise initialization']):
 ax.plot(80,v['positive_phase_rad']*1e4,marker,color=col,ms=7,markerfacecolor='none',label=label)
ax.set_xticks([20,40,80]);ax.set_xlabel('Number of loop points');ax.set_ylabel(r'Positive eigenphase ($10^{-4}$ rad)')
ax.set_ylim(7.3,10.1);ax.legend(frameon=False,loc='upper left',fontsize=7)
ax=axs[1,0];panel(ax,'c  Fidelity of the retained pair')
ax.semilogy([v['npoints'] for v in r],[v['maximum_pair_leakage'] for v in r],'o-',color=blue,label='Reference meshes')
ax.axhline(.01,ls='--',color='0.35',label='Acceptance ceiling 0.01')
ax.set_xticks([20,40,80]);ax.set_xlabel('Number of loop points');ax.set_ylabel('Maximum edge leakage')
ax.set_ylim(5e-5,.02);ax.legend(frameon=False,loc='center right')
ax=axs[1,1];panel(ax,'d  Phase differences and numerical floor')
refs=S['reference_levels']; vals=[refs[0]['phase_difference_rad'],refs[1]['phase_difference_rad'],S['maximum_reference_history_phase_difference_rad'],refs[4]['phase_difference_rad']]
ax.semilogy(range(4),vals,'o',color=blue)
floor=S['gauge_eigenphase_roundoff_rad']; ax.axhline(floor,color='0.35',ls=':',label=f'Gauge roundoff: {floor:.2e} rad')
ax.set_xticks(range(4),['20/40\nmesh','40/80\nmesh','History /\nreference','Opposed\nhistories'],fontsize=7)
ax.set_xlim(-.45,3.45);ax.set_ylim(1e-16,2e-3);ax.set_ylabel('Eigenphase-set difference (rad)')
ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,.13),fontsize=6.8)
ax.text(.03,.96,'Reference levels, not uncertainty bounds',transform=ax.transAxes,va='top',fontsize=7)
for ext in ['pdf','png']:
 fig.savefig(ROOT/'figures'/f'Figure_CPDC_revision.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)
print('Figure 6 regenerated from CPDC numerical records.')
