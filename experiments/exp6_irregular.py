"""Independent analytic nonlinear VIE validation on a strongly irregular hybrid time scale.
Run: cd code; python experiments/exp6_irregular.py
All reference values are from a closed-form primitive, not a numerical solver.
"""
from pathlib import Path
import sys, csv, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsvolterra.timescale import TimeScale, Partition
from tsvolterra import nystrom
from tsvolterra.problems import VolterraProblem

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; OUT.mkdir(parents=True,exist_ok=True)
FIG=ROOT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
T=TimeScale([(0,0.45),(.70,.70),(1.03,1.03),(1.48,1.48),(1.90,2.30),(2.75,2.75),(3.20,3.55)])

def poly(s):return .2+.04*s+.003*s*s

def true_x(t):return np.arcsin(poly(np.asarray(t)))

def primitive(t,s):
    return .2*s+(.04+.014*t)*s*s/2+(.003+.0028*t)*s**3/3+.0000525*t*s**4

def exact_integral(t):
    total=0.
    for kind,u,v in T.pieces(T.a,float(t)):
        if kind=='d': total+=primitive(t,v)-primitive(t,u)
        else: total+=v*(1+.07*t*u)*poly(u)
    return .18*total

def f(t):
    t=np.atleast_1d(np.asarray(t,float))
    return true_x(t)-np.array([exact_integral(float(z)) for z in t])

def K(t,s,sig,x,xp=np): return .18*(1+.07*t*s)*xp.sin(x)
def Kx(t,s,sig,x,xp=np): return .18*(1+.07*t*s)*xp.cos(x)
PBL=VolterraProblem(T,K,Kx,.18*(1+.07*T.b**2),x_exact=true_x,f=f,name='irregular_exact')
rows=[]; snapshots=[]
for M in [4,8,16,32,64]:
    counts=[int(round(M*(b-a)/.05)) if b>a else 0 for a,b in T.comps]
    part=Partition.gap_conforming(T,counts=counts)
    t0=time.perf_counter(); y=nystrom.solve(PBL,part); sec=time.perf_counter()-t0
    exact=true_x(part.t)
    err=float(np.max(abs(y-exact)))
    residual_exact=float(np.max(abs(nystrom.residual(PBL,part,exact))))
    S,gamma=nystrom.stability_constant(part,PBL.L)
    rows.append(dict(M=M,nodes=len(y),dense_step=part.h_dense(),max_graininess=max(b-a for a,b in T.gaps()),
                     max_error=err,exact_solution_discrete_residual=residual_exact,
                     stability_bound=float(S[-1]*residual_exact),seconds=sec))
    snapshots.append((part,y))
    print('M',M,'N',len(y),'h',part.h_dense(),'error',err,'truncation',residual_exact)
for i in range(1,len(rows)):
    rows[i]['observed_order']=float(np.log(rows[i-1]['max_error']/rows[i]['max_error'])/np.log(rows[i-1]['dense_step']/rows[i]['dense_step']))
with (OUT/'irregular_hybrid_convergence.csv').open('w',newline='') as fo:
    wr=csv.DictWriter(fo,fieldnames=list(rows[-1]));wr.writeheader();wr.writerows(rows)
with (OUT/'irregular_hybrid_profile.csv').open('w',newline='') as fo:
    wr=csv.writer(fo); wr.writerow(['t','x_exact','nystrom_approx','abs_error','component'])
    part,y=snapshots[-1]
    for ti,yi in zip(part.t,y):
        wr.writerow([float(ti),float(true_x(ti)),float(yi),float(abs(yi-true_x(ti))),T.component_of(ti)])
with (OUT/'irregular_hybrid_metadata.json').open('w') as fo:
    json.dump({'components':T.comps,'kernel':'0.18(1+0.07ts)sin(x)','exact':'arcsin(0.2+0.04t+0.003t^2)',
               'jump_lengths':[b-a for a,b in T.gaps()],'rows':rows},fo,indent=2)
# Each continuous component drawn separately: never connect across a gap.
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
part,y=snapshots[-1]
for i,(a,b) in enumerate(T.comps):
    mask=(part.t>=a-1e-12)&(part.t<=b+1e-12)
    if b>a:
        ax.plot(part.t[mask],true_x(part.t[mask]),c='#263c78',lw=2.0,label='exact on dense components' if i==0 else None)
        ax.plot(part.t[mask],y[mask],c='#cc6b32',lw=1.2,ls='--',label='Nyström on dense components' if i==0 else None)
    else:
        ax.scatter([a],true_x(a),c='#263c78',marker='o',s=38,zorder=3)
        ax.scatter([a],[y[mask][0]],c='#cc6b32',marker='+',s=55,zorder=4)
ax.set(xlabel='t in the time scale',ylabel='solution x(t)',title='Nonlinear Volterra equation on an irregular hybrid time scale')
ax.grid(alpha=.16);ax.legend(loc='best')
fig.savefig(FIG/'irregular_hybrid_profile.png',dpi=260,facecolor='white')
fig.savefig(FIG/'irregular_hybrid_profile.pdf',facecolor='white');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.3,4.4),layout='constrained')
ax.loglog([r['dense_step'] for r in rows],[r['max_error'] for r in rows],'-o',label='Nyström error')
xs=np.array([r['dense_step'] for r in rows]);ax.loglog(xs,rows[0]['max_error']*(xs/xs[0])**2,'--',label='reference slope 2')
ax.set(xlabel='dense-component step h (jump sizes remain fixed)',ylabel='maximum true nodal error',title='Observable dense-mesh convergence')
ax.grid(alpha=.2,which='both');ax.legend()
fig.savefig(FIG/'irregular_hybrid_convergence.png',dpi=260,facecolor='white')
fig.savefig(FIG/'irregular_hybrid_convergence.pdf',facecolor='white');plt.close(fig)
# animated lines are component-only; gaps always remain blank
from matplotlib.animation import FuncAnimation, PillowWriter
fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.set_xlim(T.a-.1,T.b+.12);ax.set_ylim(.15,.46)
ax.set(xlabel='t',ylabel='x(t)',title='Hybrid solution revealed on its domain')
ax.grid(alpha=.12)
for a,b in T.gaps():ax.axvspan(a,b,color='#eceff4',alpha=.65,zorder=-3)
draw=[]; palette=['#2a5aa8','#267b76','#8a5aa7'];
for idx,(a,b) in enumerate(T.comps):
    if b>a:
        xx=np.linspace(a,b,70); yy=true_x(xx)
    else:
        xx=np.array([a]);yy=true_x(xx)
    draw.extend([(float(q),float(v),idx,a==b) for q,v in zip(xx,yy)])
lines=[]
for idx,(a,b) in enumerate(T.comps):
    line,=ax.plot([],[],color=palette[idx%len(palette)],lw=2.4,marker='o' if a==b else None,markersize=6)
    lines.append(line)
def update(frame):
    for idx,line in enumerate(lines):
        elems=[z for z in draw[:frame] if z[2]==idx]
        if elems:line.set_data([z[0] for z in elems],[z[1] for z in elems])
    return lines
ani=FuncAnimation(fig,update,frames=list(range(0,len(draw)+1,4))+[len(draw)],interval=55,blit=False)
ani.save(FIG/'irregular_hybrid_motion.gif',writer=PillowWriter(fps=16),dpi=95)
plt.close(fig)
print('outputs:',OUT,FIG)
