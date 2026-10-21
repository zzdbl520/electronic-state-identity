"""Draw the state-assignment/path-comparison figure as vector artwork."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(9.3,3.35),gridspec_kw={'width_ratios':[1,1.35]})
blue='#176488'; orange='#b85e26'; grey='#64717b'
for ax in axs:
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax=axs[0];ax.text(.02,.98,'(a) Local root assignment',va='top',weight='bold',fontsize=12)
for x in [.13,.77]:
    ax.add_patch(FancyBboxPatch((x-.085,.41),.17,.38,boxstyle='round,pad=.02',fc='#edf5fa',ec=blue,lw=1.2))
    for y,c in zip([.49,.60,.71],[grey,orange,blue]):ax.plot([x-.06,x+.06],[y,y],lw=2.5,color=c)
for y1,y2,c in [(.49,.49,grey),(.6,.71,orange),(.71,.6,blue)]:
    ax.plot([.23,.66],[y1,y2],ls='--',lw=1.15,color=c)
ax.text(.13,.33,r'$\Psi(X)$',ha='center',fontsize=12)
ax.text(.77,.33,r'$\Psi(Y)$',ha='center',fontsize=12)
ax.text(.45,.20,r'$M_{IJ}^{XY}=\langle\Psi_I^X|\Psi_J^Y\rangle$',ha='center',fontsize=12)
ax.text(.45,.065,'Which root follows which?',ha='center',weight='bold',fontsize=10.5)
ax=axs[1];ax.text(.015,.98,'(b) Transport along two paths',va='top',weight='bold',fontsize=12)
positions={'A':(.13,.70),'B':(.83,.70),'C':(.13,.35),'D':(.83,.35)}
for start,end,c in [('A','B',blue),('B','D',blue),('A','C',orange),('C','D',orange)]:
    ax.add_patch(FancyArrowPatch(positions[start],positions[end],arrowstyle='-|>',mutation_scale=14,lw=2,color=c,shrinkA=17,shrinkB=17))
for label,(x,y) in positions.items():
    ax.add_patch(Circle((x,y),.055,ec=grey,fc='white',lw=1.2));ax.text(x,y,label,ha='center',va='center',weight='bold',fontsize=11)
for label,t in [('A',r'$(q,p)$'),('B',r'$(q+\Delta q,p)$'),('C',r'$(q,p+\Delta p)$'),('D',r'$(q+\Delta q,p+\Delta p)$')]:
    x,y=positions[label];ax.text(x,y+(.10 if label in ['A','B'] else -.10),t,ha='center',va='center',fontsize=10)
ax.text(.48,.525,r'$K_{q\rightarrow p}$  vs.  $K_{p\rightarrow q}$',ha='center',fontsize=13)
ax.text(.48,.065,'Do the two paths give the same transport?',ha='center',weight='bold',fontsize=10.5)
fig.subplots_adjust(left=.025,right=.985,top=.98,bottom=.02,wspace=.15)
for ext in ['pdf','png']:fig.savefig(ROOT/'figures'/f'Figure1_scientific.{ext}',dpi=600,bbox_inches='tight')
