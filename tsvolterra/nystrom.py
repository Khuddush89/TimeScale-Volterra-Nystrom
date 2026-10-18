"""Time-scale Nystrom method (trapezoidal rule on time scales) for
    x(t) = f(t) + int_a^t K(t, s, sigma(s), x(s)) Delta s ,  t in [a,b]_T,
solved by forward substitution with Newton's method on implicit (dense) steps;
global Picard iteration; discrete residuals; stability constants; Richardson.
Kernels have signature K(t, s, sig, x, xp=np) and derivative Kx(...)."""
from __future__ import annotations
import numpy as np
from .timescale import Partition


def _sig_right(P: Partition, naive: bool):
    return P.sigR_naive if naive else P.sigR


def solve(prob, P: Partition, newton_tol=1e-14, maxit=50, naive=False, f_nodes=None, fp_maxit=100000):
    """forward substitution; returns X at the partition nodes.
    Implicit rows: scalar Newton iteration; if it fails, the row fixed-point iteration (a contraction when gamma<1)."""
    t = P.t
    N = P.N
    sR = _sig_right(P, naive)
    f = prob.f(t) if f_nodes is None else f_nodes
    X = np.empty(N + 1)
    X[0] = f[0]
    for n in range(1, N + 1):
        tn = t[n]
        # left contributions j = 0..n-1 ; right contributions j = 0..n-2 (nodes 1..n-1)
        c = f[n] + np.sum(P.wm[:n] * prob.K(tn, t[:n], P.sigL[:n], X[:n]))
        if n >= 2:
            c += np.sum(P.wp[:n - 1] * prob.K(tn, t[1:n], sR[:n - 1], X[1:n]))
        w = P.wp[n - 1]
        if w == 0.0:
            X[n] = c
            continue
        z = X[n - 1]
        sr = sR[n - 1]
        converged = False
        for _ in range(maxit):
            g = z - c - w * prob.K(tn, tn, sr, z)
            dg = 1.0 - w * prob.Kx(tn, tn, sr, z)
            dz = g / dg
            z -= dz
            if abs(dz) <= newton_tol * (1.0 + abs(z)):
                converged = True
                break
        if not converged or not np.isfinite(z):
            # fallback: the row map z -> c + w K(t_n, t_n, z) is a contraction when gamma < 1 (Theorem 5.2)
            z = X[n - 1]
            for _ in range(fp_maxit):
                z_new = c + w * prob.K(tn, tn, sr, z)
                if abs(z_new - z) <= newton_tol * (1.0 + abs(z_new)):
                    z = z_new; converged = True; break
                z = z_new
            if not converged or not np.isfinite(z):
                raise RuntimeError(f"implicit solve did not converge at node {n} (Newton and fixed-point fallback)")
        X[n] = z
    return X


def apply_Phi(prob, P: Partition, Y, naive=False, f_nodes=None):
    """discrete Volterra operator Phi(Y)_n (vectorised, O(N^2) memory in chunks)"""
    t = P.t
    N = P.N
    sR = _sig_right(P, naive)
    f = prob.f(t) if f_nodes is None else f_nodes
    out = np.empty(N + 1)
    out[0] = f[0]
    chunk = 512
    for n0 in range(1, N + 1, chunk):
        n1 = min(n0 + chunk, N + 1)
        tn = t[n0:n1, None]
        j = np.arange(N)[None, :]
        nn = np.arange(n0, n1)[:, None]
        KL = prob.K(tn, t[None, :N], P.sigL[None, :], Y[None, :N])
        KR = prob.K(tn, t[None, 1:], sR[None, :], Y[None, 1:])
        msk = j < nn
        out[n0:n1] = f[n0:n1] + np.sum(np.where(msk, P.wm[None, :] * KL + P.wp[None, :] * KR, 0.0), axis=1)
    return out


def residual(prob, P: Partition, Y, naive=False, f_nodes=None):
    return Y - apply_Phi(prob, P, Y, naive=naive, f_nodes=f_nodes)


def picard(prob, P: Partition, tol=1e-13, maxit=10000, Y0=None, f_nodes=None):
    f = prob.f(P.t) if f_nodes is None else f_nodes
    Y = np.full(P.N + 1, f[0]) if Y0 is None else Y0.copy()
    hist = []
    for k in range(maxit):
        Yn = apply_Phi(prob, P, Y, f_nodes=f)
        d = np.max(np.abs(Yn - Y))
        hist.append(d)
        Y = Yn
        if d <= tol * (1 + np.max(np.abs(Y))):
            break
    return Y, k + 1, np.array(hist)


def stability_constant(P: Partition, L):
    """S_P = (1/(1-g)) prod_j (1 + q c_j), q = L/(1-g), g = L max w_plus,
    c_j = w_minus[j] + w_plus[j-1]  (node weights without the implicit term).
    Returns array S_n for every node n (S_0 = 1/(1-g))."""
    gam = L * float(np.max(P.wp)) if P.N > 0 else 0.0
    if gam >= 1:
        raise ValueError("step condition L*max(w_plus) < 1 violated")
    q = L / (1.0 - gam)
    c = P.wm.copy()
    c[1:] += P.wp[:-1]
    A = np.concatenate([[1.0], np.cumprod(1.0 + q * c)])
    return A / (1.0 - gam), gam


def richardson(X_h, X_h2):
    """(4 X_{h/2} - X_h)/3 at the nodes of the coarse partition (X_h2 restricted)"""
    return (4.0 * X_h2 - X_h) / 3.0


def restrict(P_fine: Partition, X_fine, P_coarse: Partition):
    idx = np.searchsorted(P_fine.t, P_coarse.t)
    assert np.allclose(P_fine.t[idx], P_coarse.t)
    return X_fine[idx]


def majorant_stability(P: Partition, ell):
    """Sharper discrete stability constants U_n from the discrete majorant equation
        U_n = 1 + sum_{j<n} [wm_j ell(t_n,t_j) U_j + wp_j ell(t_n,t_{j+1}) U_{j+1}],
    where |K(t,s,u)-K(t,s,v)| <= ell(t,s)|u-v|.  Then max_{k<=n}|Y_k-Z_k| <= U_n max_k|delta_k|
    (discrete stability theorem).  Requires wp_{n-1} ell(t_n,t_n) < 1."""
    t = P.t; N = P.N
    U = np.empty(N + 1); U[0] = 1.0
    for n in range(1, N + 1):
        tn = t[n]
        c = 1.0 + np.sum(P.wm[:n] * ell(tn, t[:n]) * U[:n])
        if n >= 2:
            c += np.sum(P.wp[:n - 1] * ell(tn, t[1:n]) * U[1:n])
        d = P.wp[n - 1] * ell(tn, np.array([tn]))[0]
        if d >= 1:
            raise ValueError("diagonal condition violated")
        U[n] = c / (1.0 - d)
    return np.maximum.accumulate(U)
