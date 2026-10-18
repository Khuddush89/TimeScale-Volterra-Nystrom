"""Time-scale trapezoidal rule (Georgiev-Erhan Newton-Cotes rule with n = 1),
its Peano kernel and error constant, and Gauss-type operators on time scales."""
from __future__ import annotations
import numpy as np
from .timescale import TimeScale


def trapezoid_weights(T: TimeScale, a, b):
    """weights (w_minus, w_plus) of  int_a^b f Delta t ~ w_minus f(a) + w_plus f(b)"""
    D = b - a
    wp = T.h2(b, a) / D
    return D - wp, wp


def kappa_step(T: TimeScale, a, b):
    """Peano constant kappa = h_2(b,a)^2/(b-a) - h_3(b,a)"""
    D = b - a
    return T.h2(b, a) ** 2 / D - T.h3(b, a)


def kappa_closed_form(D, gap_lengths):
    """kappa = D^3/12 - (1/3) sum g^3 + (sum g^2)^2/(4D)  (Proposition in the paper)"""
    g = np.asarray(gap_lengths, float)
    return D ** 3 / 12.0 - np.sum(g ** 3) / 3.0 + np.sum(g ** 2) ** 2 / (4.0 * D)


def peano_P2(T: TimeScale, a, b, t):
    """P_2(t) = int_a^t (sigma(s) - a - w_minus) Delta s  (<= 0 on [a,b]_T)"""
    wm, _ = trapezoid_weights(T, a, b)
    return (t - a) ** 2 - T.h2(t, a) - wm * (t - a)


def trapezoid(T: TimeScale, F, a, b):
    """apply the rule to F(tau, sigma_tau); right node uses the left limit if the
    step is a real interval (sigma argument = b)."""
    wm, wp = trapezoid_weights(T, a, b)
    pcs = T.pieces(a, b)
    dense = len(pcs) == 1 and pcs[0][0] == "d"
    sb = b if dense else T.sigma(b)
    return wm * F(np.array([a]), np.array([T.sigma(a)]))[0] + wp * F(np.array([b]), np.array([sb]))[0]


# ----------------------------------------------------------------------------
# Gauss-type quadrature operators for continuous residuals
# ----------------------------------------------------------------------------
class GaussVolterraOperator:
    """Operator v -> I_v(t_i) = int_a^{t_i} K(t_i, s, sigma(s), v(s)) Delta s
    for a fixed ordered set of evaluation points t_i in T that contains every
    component endpoint.  Dense pieces between consecutive evaluation points are
    integrated by n_gauss-point Gauss-Legendre (split into sub-panels of length
    <= max_panel); right-scattered points contribute mu*K exactly.

    Data:  t (n_eval,), s (n_nodes,), w (n_nodes,), sig (n_nodes,),
           owner (n_nodes,) index k such that node lies in [t_k, t_{k+1}),
           so node q contributes to t_i iff owner[q] < i.
    """

    def __init__(self, T: TimeScale, t_eval, n_gauss=10, max_panel=None):
        t_eval = np.unique(np.asarray(t_eval, float))
        # make sure all component endpoints are present
        ends = np.concatenate([T.p, T.q])
        t_eval = np.unique(np.concatenate([t_eval, ends]))
        self.T, self.t = T, t_eval
        xg, wg = np.polynomial.legendre.leggauss(n_gauss)
        S, W, SIG, OWN = [], [], [], []
        for k in range(len(t_eval) - 1):
            a, b = t_eval[k], t_eval[k + 1]
            pcs = T.pieces(a, b)
            for kind, u, v in pcs:
                if kind == "d":
                    npan = 1 if max_panel is None else max(1, int(np.ceil((v - u) / max_panel)))
                    e = np.linspace(u, v, npan + 1)
                    for e0, e1 in zip(e[:-1], e[1:]):
                        S.append(0.5 * (e1 - e0) * xg + 0.5 * (e1 + e0))
                        W.append(0.5 * (e1 - e0) * wg)
                        SIG.append(S[-1].copy())
                        OWN.append(np.full(n_gauss, k))
                else:
                    S.append(np.array([u])); W.append(np.array([v]))
                    SIG.append(np.array([u + v])); OWN.append(np.array([k]))
        self.s = np.concatenate(S); self.w = np.concatenate(W)
        self.sig = np.concatenate(SIG); self.owner = np.concatenate(OWN)
        self.mask = (self.owner[None, :] < np.arange(len(t_eval))[:, None]).astype(float)

    def apply(self, K, vs, t_sub=None, xp=np):
        """vs: values v(s_q) at the nodes. returns I_v(t_i) for all evaluation points"""
        tt = self.t[:, None]
        Kmat = K(tt, self.s[None, :], self.sig[None, :], vs[None, :], xp=xp)
        return xp.sum(self.mask * self.w[None, :] * Kmat, axis=1)

    def apply_chunked(self, K, vs, chunk=256):
        out = np.empty(len(self.t))
        for i0 in range(0, len(self.t), chunk):
            i1 = min(i0 + chunk, len(self.t))
            tt = self.t[i0:i1, None]
            msk = self.owner[None, :] < np.arange(i0, i1)[:, None]
            Kmat = K(tt, self.s[None, :], self.sig[None, :], vs[None, :], xp=np)
            out[i0:i1] = np.sum(np.where(msk, self.w[None, :] * Kmat, 0.0), axis=1)
        return out
