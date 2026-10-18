"""Fresh, small residual-trained networks for the exact irregular hybrid test.

Run: OPENBLAS_NUM_THREADS=1 python experiments/revision_neural.py
All fitted parameters are saved. Errors/indicators are sampled, not validated.
No exact values beyond the initial condition enter training.
"""
from pathlib import Path
import json, time, sys, math
from fractions import Fraction as F
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from revision_checks import irregular_problem
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tsvolterra.timescale import Partition
from tsvolterra.quadrature import GaussVolterraOperator
from tsvolterra import nystrom


def analytic_certificate():
    """Exact rational inequalities for the perturbed-arcsine trial in the text."""
    b=F(71,20); eps=F(1,10000); L=F(18,100)*(1+F(7,100)*b*b)
    pmax=F(1,5)+F(1,25)*b+F(3,1000)*b*b
    zbound=F(381,1000); slope=F(1083,1000)
    assert pmax+eps*b < zbound
    assert slope*slope*(1-zbound*zbound)>1
    e_upper=slope*eps*b
    d_upper=L*eps*b*b/2
    assert d_upper<e_upper
    z=L*F(6,5)
    # Positive Taylor sum with an exact geometric remainder bound.
    exp_upper=sum((z**k/F(math.factorial(k)) for k in range(13)),F(0))
    exp_upper+=(z**13/F(math.factorial(13)))/(1-z/F(14))
    product=F(1)
    for g in [F(1,4),F(33,100),F(45,100),F(42,100),F(45,100),F(45,100)]:
        product*=1+L*g
    gamma_upper=exp_upper*product
    assert gamma_upper<F(3167,1000)
    certificate=F(3167,1000)*e_upper
    return dict(epsilon=float(eps),error_lower=float(eps*b),
                error_upper=float(e_upper),residual_upper=float(e_upper),
                integral_perturbation_upper=float(d_upper),gamma_upper=float(gamma_upper),
                validated_error_upper=float(certificate),
                exact_bound_numerator=certificate.numerator,
                exact_bound_denominator=certificate.denominator,
                effectivity_upper=float(certificate/(eps*b)),validated=True)


def network(t,theta,m,b,random_features=None):
    t=np.asarray(t); xi=2*t/b-1; scale=t/b
    if random_features is None:
        a0=theta[0]; beta=theta[1:m+1];w=theta[m+1:2*m+1];c=theta[2*m+1:]
    else:
        a0=theta[0];beta=theta[1:];w,c=random_features
    z=np.tanh(xi[:,None]*w[None,:]+c[None,:])
    y=scale*(a0+z@beta)
    J=np.column_stack([np.ones(len(t)),z])
    if random_features is None:
        zz=(1-z*z)*beta[None,:]
        J=np.column_stack([J,zz*xi[:,None],zz])
    return y,scale[:,None]*J


def train(prob,m,seed,rf=False):
    T=prob.T; rng=np.random.default_rng(seed)
    P=Partition.gap_conforming(T,h=.025)
    op=GaussVolterraOperator(T,P.t,n_gauss=8)
    # delta masses are included in W exactly as in the manuscript formula
    W=op.mask*op.w[None,:]
    A=.18*W*(1+.07*op.t[:,None]*op.s[None,:])
    f=prob.f(op.t); initial=float(np.arcsin(.2))
    w=rng.uniform(-2,2,m);c=rng.uniform(-1,1,m)
    feature=(w,c) if rf else None
    theta=np.concatenate([[0.],rng.normal(0,.01,m)])
    if not rf:theta=np.concatenate([theta,w,c])
    def residual_and_jac(params):
        yt,Jt=network(op.t,params,m,T.b,feature)
        ys,Js=network(op.s,params,m,T.b,feature)
        return initial+yt-f-A@np.sin(initial+ys), Jt-A@(np.cos(initial+ys)[:,None]*Js)
    start=time.perf_counter()
    sol=least_squares(lambda q:residual_and_jac(q)[0],theta,
                      jac=lambda q:residual_and_jac(q)[1],method='trf',
                      ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=1200)
    sec=time.perf_counter()-start
    return dict(method="random features" if rf else "MLP",features=m,seed=seed,
                parameters=sol.x.tolist(),feature_w=w.tolist() if rf else None,
                feature_c=c.tolist() if rf else None,seconds=sec,nfev=sol.nfev,
                termination_status=int(sol.status),cost=float(sol.cost),
                initial=initial,validated=False)


def main():
    prob,poly,exact,integ=irregular_problem();T=prob.T
    E=T.sample(500)
    op=GaussVolterraOperator(T,E,n_gauss=12)
    W=op.mask*op.w[None,:]
    A=.18*W*(1+.07*op.t[:,None]*op.s[None,:])
    fe=prob.f(op.t)
    rows=[]
    for rf,m in [(True,24),(False,8)]:
        for seed in range(5):
            row=train(prob,m,seed,rf)
            theta=np.array(row['parameters'])
            feat=(np.array(row['feature_w']),np.array(row['feature_c'])) if rf else None
            yt,_=network(op.t,theta,m,T.b,feat);ys,_=network(op.s,theta,m,T.b,feat)
            yt+=row['initial'];ys+=row['initial']
            res=yt-fe-A@np.sin(ys)
            row.update(error=float(np.max(abs(yt-exact(op.t)))),
                       residual=float(np.max(abs(res))),
                       indicator=float(T.exp_const(prob.L,T.b)*np.max(abs(res))))
            rows.append(row)
            print(row['method'],seed,'error',row['error'],'seconds',row['seconds'],flush=True)
    classic=[]
    for h in [.05,.0125]:
        P=Partition.gap_conforming(T,h=h);start=time.perf_counter();Y=nystrom.solve(prob,P)
        sec=time.perf_counter()-start
        Pf=Partition.gap_conforming(T,h=h/2);start=time.perf_counter();Yf=nystrom.solve(prob,Pf)
        R=nystrom.richardson(Y,nystrom.restrict(Pf,Yf,P));sr=time.perf_counter()-start+sec
        def interp(v):
            out=np.empty(len(E))
            for a,b in T.comps:
                mask=(P.t>=a-1e-12)&(P.t<=b+1e-12);dest=(E>=a-1e-12)&(E<=b+1e-12)
                out[dest]=CubicSpline(P.t[mask],v[mask])(E[dest]) if b>a else v[mask][0]
            return out
        for name,v,secs in [('Nystrom',Y,sec),('Richardson',R,sr)]:
            classic.append(dict(method=name,h=h,parameters=len(v),seconds=secs,
                                error=float(np.max(abs(interp(v)-exact(E)))),
                                nodal_error=float(np.max(abs(v-exact(P.t))))))
    data=dict(runs=rows,classical=classic,analytic_certificate=analytic_certificate(),
              evaluation_points=len(E),training_dense_step=.025,gauss_training=8,gauss_validation=12,
              note="Fresh runs. Sampled errors and indicators; parameters retained. No interval validation of the networks.")
    (ROOT/'results/revision_neural.json').write_text(json.dumps(data,indent=2))
    print('validated analytic example',data['analytic_certificate'])


if __name__=='__main__':main()
