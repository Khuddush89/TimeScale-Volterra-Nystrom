"""Independent checks added during the second mathematical revision.

Run from code/: python experiments/revision_checks.py
Uses NumPy, SciPy and Matplotlib only. No neural training is required.
Archived neural runs are audited separately; they are not rerun by this file.
"""
from pathlib import Path
import sys, json, time, platform
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsvolterra.timescale import TimeScale, Partition
from tsvolterra.problems import (VolterraProblem, kernel_exp_sin, kernel_pendulum,
                                ts_hybrid, ts_interval, ts_qscale, ts_pendulum, x_star)
from tsvolterra import nystrom
from tsvolterra.quadrature import kappa_closed_form, peano_P2

OUT = ROOT / "results"
FIG = ROOT / "figures"
TABLE = ROOT / "tables"
M2_ANALYTIC = 19.5 + np.sqrt(37.0)


def scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def irregular_problem():
    T = TimeScale([(0, .45), (.70, .70), (1.03, 1.03), (1.48, 1.48),
                   (1.90, 2.30), (2.75, 2.75), (3.20, 3.55)])
    p = lambda s: .2 + .04 * np.asarray(s) + .003 * np.asarray(s)**2
    x = lambda t: np.arcsin(p(t))
    def A(t, s):
        return .2*s + (.04+.014*t)*s*s/2 + (.003+.0028*t)*s**3/3 + .0000525*t*s**4
    def integral(t):
        dense, atom = 0., 0.
        for kind, u, v in T.pieces(T.a, float(t)):
            if kind == "d":
                dense += A(t, v)-A(t, u)
            else:
                atom += v * (1+.07*t*u) * p(u)
        return .18*dense, .18*atom
    def f(t):
        t = np.atleast_1d(np.asarray(t, float))
        return x(t)-np.array([sum(integral(ti)) for ti in t])
    K = lambda t, s, sig, y, xp=np: .18*(1+.07*t*s)*xp.sin(y)
    Kx = lambda t, s, sig, y, xp=np: .18*(1+.07*t*s)*xp.cos(y)
    L = .18*(1+.07*T.b**2)
    prob = VolterraProblem(T, K, Kx, L, x_exact=x, f=f, name="irregular_exact")
    return prob, p, x, integral


def exact_defect(P):
    """I-Q for the manufactured cubic integrand; no numerical differences."""
    defect = np.zeros(P.N+1)
    for n, tn in enumerate(P.t):
        for j in range(n):
            if P.kind[j] == "dense":
                h = P.t[j+1]-P.t[j]
                mid = (P.t[j+1]+P.t[j])/2
                defect[n] -= .18*h**3/12*(.006+.0056*tn+.00126*tn*mid)
    return defect


def irregular_study():
    prob, p, x, integral = irregular_problem()
    T = prob.T
    ell = lambda t, s: .18*(1+.07*t*s)
    M2 = .18*(.006+.0056*T.b+.00126*T.b*T.b)
    rows, solutions = [], []
    for M in [1, 2, 4, 8, 16, 32, 64]:
        counts = [round((q-p0)/.05)*M if q>p0 else 0 for p0, q in T.comps]
        P = Partition.gap_conforming(T, counts=counts)
        begin = time.perf_counter()
        Y = nystrom.solve(prob, P)
        sec = time.perf_counter()-begin
        error = np.max(np.abs(Y-x(P.t)))
        defect = exact_defect(P)
        U = nystrom.majorant_stability(P, ell)
        z_float = nystrom.residual(prob, P, x(P.t))
        assert np.max(np.abs(defect-z_float)) < 2e-15
        row = dict(M=M, h=P.h_dense(), nodes=P.N+1, error=float(error),
                   kappa=P.kappa_total, defect=float(np.max(np.abs(defect))),
                   U=float(U[-1]), bound=float(U[-1]*P.kappa_total*M2),
                   defect_bound=float(U[-1]*np.max(np.abs(defect))), seconds=sec)
        if rows:
            row["eoc"] = float(np.log2(rows[-1]["error"]/error))
        assert error <= row["bound"] + 2e-15
        rows.append(row)
        solutions.append((P, Y))
    for k in range(len(rows)-1):
        P, Y = solutions[k]
        Pf, Yf = solutions[k+1]
        R = nystrom.richardson(Y, nystrom.restrict(Pf, Yf, P))
        rows[k]["richardson"] = float(np.max(np.abs(R-x(P.t))))
    atoms = [dict(q=q, sigma=r, mu=r-q,
                  contribution=float(.18*(r-q)*(1+.07*T.b*q)*p(q))) for q, r in T.gaps()]
    # Nonlinear solve by an independently assembled all-at-once root system.
    P, Y = solutions[0]
    Z = root(lambda z: nystrom.residual(prob, P, z), prob.f(P.t), tol=1e-11)
    assert Z.success and np.max(np.abs(Z.x-Y)) < 2e-12
    # Picard baseline for the same discrete system.
    Yp, iterations, hist = nystrom.picard(prob, P, tol=1e-15)
    assert np.max(np.abs(Yp-Y)) < 3e-14
    data = dict(rows=rows, L=prob.L, M2=M2, dense_length=T.dense_length(),
                max_graininess=max(r-q for q, r in T.gaps()), atoms=atoms,
                endpoint=dict(x=float(x(T.b)), f=scalar(prob.f(T.b)),
                              dense=integral(T.b)[0], atoms=integral(T.b)[1]),
                picard=dict(iterations=iterations, difference=float(np.max(abs(Yp-Y)))),
                independent_root_difference=float(np.max(abs(Z.x-Y))),
                eL=T.exp_const(prob.L, T.b))
    table = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$h_d$ & nodes & error & EOC & $\|\zeta\|_\infty$ & $\bar U_N\kappa M_2$ & Richardson\\",
             r"\midrule"]
    def fmt(v):
        if v is None: return "--"
        a, b = f"{v:.3e}".split("e")
        return "$"+a+r"\times10^{"+str(int(b))+"}$"
    for r in rows:
        table.append(f"{r['h']:.8f} & {r['nodes']} & {fmt(r['error'])} & "+
                     (f"{r['eoc']:.2f}" if "eoc" in r else "--")+
                     f" & {fmt(r['defect'])} & {fmt(r['bound'])} & {fmt(r.get('richardson'))}"+r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (TABLE/"tab_irregular.tex").write_text("\n".join(table))
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axs = plt.subplots(1,2,figsize=(10.2,3.7),layout="constrained")
    P, Y = solutions[1]
    for i, (a,b) in enumerate(T.comps):
        mask=(P.t>=a-1e-12)&(P.t<=b+1e-12)
        if b>a:
            tt=np.linspace(a,b,100)
            axs[0].plot(tt,x(tt),c="#183e71",lw=1.8,label="exact" if i==0 else None)
            axs[0].plot(P.t[mask],Y[mask],"o",mfc="white",mec="#b3472b",ms=3,
                        label=r"Nystr$\ddot{o}$m" if i==0 else None)
        else:
            axs[0].plot([a],x(a),"s",color="#183e71",ms=5)
    axs[0].set(xlabel=r"$t\in\mathbb{T}_{\rm irr}$",ylabel=r"$x(t)$",title="(a) Nonlinear hybrid solution")
    axs[0].legend(frameon=False)
    hh=np.array([r["h"] for r in rows])
    axs[1].loglog(hh,[r['error'] for r in rows],"o-",color="#183e71",label="nodal error")
    axs[1].loglog(hh,[r['bound'] for r in rows],"s--",color="#b3472b",label="analytic bound")
    axs[1].loglog(hh[:-1],[r['richardson'] for r in rows[:-1]],"^-",color="#27725b",label="Richardson")
    axs[1].set(xlabel=r"dense step $h_d$",ylabel="maximum error",title="(b) Fixed jumps, refined dense parts")
    axs[1].legend(frameon=False,fontsize=8)
    for ax in axs: ax.grid(alpha=.15)
    fig.savefig(FIG/"fig_irregular_check.pdf",facecolor="white")
    fig.savefig(FIG/"fig_irregular_check.png",dpi=400,facecolor="white")
    plt.close(fig)
    return data


def reproduce_standard():
    """Recompute main convergence rows with a proved analytic derivative bound."""
    from exp3_convergence import study, pendulum_reference, counts_for
    archived=json.loads((OUT/"exp3_convergence.json").read_text())
    comparisons={}
    K,Kx,L=kernel_exp_sin(1.,1.)
    for name,T in [("E1",ts_interval(0,2)),("E2",ts_hybrid())]:
        prob=VolterraProblem(T,K,Kx,L,x_exact=x_star)
        tr,ri=study(prob,x_star,list(range(2,11)),M2=M2_ANALYTIC,
                    ell=lambda t,s: np.exp(-(t-s)))
        old=archived[name]
        comparisons[name]=dict(max_error_change=max(abs(r['err']-o['err']) for r,o in zip(tr,old['trap'])),
                               max_richardson_change=max(abs(r['err']-o['err']) for r,o in zip(ri,old['rich'])))
        archived[name].update(trap=tr,rich=ri,M2=M2_ANALYTIC,M2_source="analytic: 39/2+sqrt(37)")
    # q-scale reference from the exponential-kernel memory recurrence, O(N).
    T=ts_qscale();t=T.p;f=1+.5*np.sin(2*t);memory=0.;xx=np.empty_like(t);xx[0]=f[0]
    for j in range(len(t)-1):
        step=t[j+1]-t[j]
        memory=np.exp(-.5*step)*(memory+step*np.sin(xx[j]))
        xx[j+1]=f[j+1]+memory
    K,Kx,L=kernel_exp_sin(1.,.5)
    prob=VolterraProblem(T,K,Kx,L,f=lambda s:1+.5*np.sin(2*np.asarray(s)))
    yy=nystrom.solve(prob,Partition(T,t))
    comparisons['q_reference_difference']=float(np.max(abs(xx-yy)))
    assert comparisons['q_reference_difference']<2e-14
    # Independent pendulum ODE + exact scattered updates at representative levels.
    T=ts_pendulum();K,Kx,L=kernel_pendulum(2.,6.)
    prob=VolterraProblem(T,K,Kx,L,f=lambda s:1+0*np.asarray(s))
    ref=pendulum_reference(T,2.,1.,0.)
    pend=[]
    for lev in [4,5,6,7,8]:
        P=Partition.gap_conforming(T,counts=counts_for(T,lev));y=nystrom.solve(prob,P)
        yn=nystrom.solve(prob,P,naive=True)
        pend.append(dict(h=P.h_dense(),error=float(np.max(abs(y-ref(P.t)))),
                         naive_error=float(np.max(abs(yn-ref(P.t))))))
    comparisons['pendulum']=pend
    (OUT/"exp3_convergence.json").write_text(json.dumps(archived,indent=1))
    return comparisons


def identities():
    rng=np.random.default_rng(618)
    discrepancies=[]
    for _ in range(200):
        z=np.sort(rng.uniform(0,1,8));T=TimeScale([(0,z[0]),(z[1],z[2]),(z[3],z[4]),(z[5],z[6]),(z[7],1)])
        g=np.array([v-u for u,v in T.gaps()]);kap=kappa_closed_form(1,g)
        discrepancies.append(abs(kap-(T.h2(1,0)**2-T.h3(1,0))))
        assert 0<=kap<1/12
        for c in T.sample(10):assert peano_P2(T,0,1,c)<1e-14
    # A deliberately nonconforming step: exact polynomial error, order one.
    interface=[]
    mu=.4
    for h in [.1,.05,.025]:
        T=TimeScale([(0,h),(h+mu,h+mu)])
        P=Partition(T,np.array([0,h+mu]));q=P.wm[0]*0+P.wp[0]*(h+mu)**2
        actual=h**3/3+mu*h*h-q
        formula=-h*(h*h+3*h*mu+6*mu*mu)/6
        assert abs(actual-formula)<1e-15
        interface.append(dict(h=h,error=actual,formula=formula))
    return dict(max_kappa_difference=max(discrepancies),interface=interface)


def main():
    begin=time.perf_counter()
    for p in [OUT,FIG,TABLE]:p.mkdir(exist_ok=True,parents=True)
    data=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                               machine=platform.machine()),
              identities=identities(),irregular=irregular_study(),standard=reproduce_standard())
    data['seconds']=time.perf_counter()-begin
    (OUT/"revision_validation.json").write_text(json.dumps(data,indent=2))
    print(json.dumps({"seconds":data['seconds'],"standard":data['standard'],
                      "irregular":data['irregular']},indent=2))


if __name__=="__main__":main()
