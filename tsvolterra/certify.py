"""A posteriori certificates.

Continuous certificate (Theorem in the paper): for any rd-continuous v on [a,b]_T,
    sup_{s<=t} |v(s)-x(s)| <= e_L(t,a) * sup_{s<=t} |r_v(s)|,
    r_v(t) = v(t) - f(t) - int_a^t K(t,s,sigma(s),v(s)) Delta s,
and conversely |r_v(t)| <= (1 + L (t-a)) sup_{s<=t}|v(s)-x(s)|.

Discrete certificate: for grid functions Y on a partition P,
    max_{k<=n} |Y_k - X_k| <= S_n * max_{k<=n} |Y_k - Phi(Y)_k|.
"""
from __future__ import annotations
import numpy as np
from .quadrature import GaussVolterraOperator


def continuous_residual(prob, v, t_eval, n_gauss=12, max_panel=0.02):
    """r_v on an evaluation set t_eval subset T (component endpoints are added)."""
    op = GaussVolterraOperator(prob.T, t_eval, n_gauss=n_gauss, max_panel=max_panel)
    I = op.apply_chunked(prob.K, v(op.s))
    r = v(op.t) - prob.f(op.t) - I
    return op.t, r


def certificate(prob, v, t_eval, **kw):
    """Sampled residual indicator, not a validated uniform error certificate.

    Use the manuscript's validation proposition to add a proved sampling
    enclosure, quadrature enclosure, and roundoff enclosure before treating
    the returned eta as a rigorous uniform bound.
    """
    t, r = continuous_residual(prob, v, t_eval, **kw)
    rr = np.maximum.accumulate(np.abs(r))
    eL = np.array([prob.T.exp_const(prob.L, ti) for ti in t])
    bound = eL * rr
    return dict(t=t, r=r, bound=bound, sup_r=float(rr[-1]), eL=float(eL[-1]),
                validated=False,
                eta=float(bound[-1]), lower=float(rr[-1] / (1 + prob.L * (prob.b - prob.a))))
