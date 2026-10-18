"""Independent verification of the manuscript (written from scratch; does NOT import tsvolterra).

Part S: symbolic checks with SymPy (monomials h2,h3, Peano constant, interface defect,
        analytic constants M2, closed-form irregular example).
Part N: an independent time-scale Nystrom implementation reproducing Tables 1, 2, 4.
Part R: data-perturbation robustness (discrete stability, Theorem thm:stab) at noise levels 0,1,5,10 %.
Run:  python verify_independent.py  -> results/verification.json + results/csv/*.csv
"""
import json, time, csv, os, sys, platform
import numpy as np
import sympy as sp

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(os.path.join(OUT, "csv"), exist_ok=True)
R = {}

# ------------------------------------------------------------------ Part S
def part_symbolic():
    s, t, g, h, mu, H, a1, g1, g2, u = sp.symbols('s t g h mu H a1 g1 g2 u', positive=True)
    out = {}
    # time scale [s, s+a1] U [s+a1+g1, s+a1+g1+u] U ... with two gaps; compute h2,h3 by the delta-integral decomposition
    # pieces: dense [0,a1], gap g1 at a1, dense [a1+g1, a1+g1+u], gap g2 at a1+g1+u, end point t=a1+g1+u+g2 (isolated)
    tau = sp.symbols('tau')
    def h2_at(r):  # h2(r,0) for r on the scale (s=0)
        return r
    # build piecewise h2(r,0)
    A = a1; B = a1+g1; C = a1+g1+u; Tend = C+g2
    h2_dense1 = lambda r: r**2/2
    h2_B = h2_dense1(A) + g1*A
    h2_dense2 = lambda r: h2_B + (r**2-B**2)/2
    h2_end = h2_dense2(C) + g2*C
    S2 = g1**2+g2**2; S3 = g1**3+g2**3
    out['h2_formula'] = sp.simplify(h2_end - (Tend**2 - S2)/2) == 0
    # h3(Tend,0) = int h2(r,0) Delta r
    h3 = sp.integrate(h2_dense1(tau), (tau, 0, A)) + g1*h2_dense1(A) \
        + sp.integrate(h2_dense2(tau), (tau, B, C)) + g2*h2_dense2(C)
    h3_formula = Tend**3/6 - Tend*S2/2 + S3/3
    out['h3_formula'] = sp.simplify(h3 - h3_formula) == 0
    kappa = sp.simplify(h2_end**2/Tend - h3)
    kappa_closed = Tend**3/12 - S3/3 + S2**2/(4*Tend)
    out['kappa_closed_form'] = sp.simplify(kappa - kappa_closed) == 0
    # uniform discrete step
    k_uni = sp.simplify((H**3/12 - (H*mu**2)/3 + (H*mu)**2/(4*H)) - H*(H**2-mu**2)/12)
    out['kappa_uniform'] = k_uni == 0
    # interface defect on [0,h] U {h+mu}, F=r^2
    I = h**3/3 + mu*h**2
    h2ts = h**2/2 + mu*h
    Q = h2ts/(h+mu)*(h+mu)**2
    E_ = sp.simplify(I - Q)
    out['interface_defect'] = sp.simplify(E_ + h*(h**2+3*h*mu+6*mu**2)/6) == 0
    wm = ((h+mu)**2 - h2ts)/(h+mu)
    Pi_q = h**2 - h**2/2 - wm*h
    out['Pi_q_formula'] = sp.simplify(Pi_q + h*mu*(h+2*mu)/(2*(h+mu))) == 0
    # M2 for P1/P2
    x = sp.symbols('x')
    xs = 1 + sp.exp(-x/2)*sp.sin(3*x)
    d1 = sp.diff(xs, x); d2 = sp.diff(xs, x, 2)
    out['xstar_d1'] = sp.simplify(d1 - sp.exp(-x/2)*(3*sp.cos(3*x) - sp.sin(3*x)/2)) == 0
    out['xstar_d2'] = sp.simplify(d2 - sp.exp(-x/2)*(-sp.Rational(35, 4)*sp.sin(3*x) - 3*sp.cos(3*x))) == 0
    out['amp_d1'] = str(sp.sqrt(sp.Integer(9) + sp.Rational(1, 4)))
    out['amp_d2'] = str(sp.sqrt(sp.Rational(35, 4)**2 + 9))
    V1 = sp.sqrt(37)/2; V2 = sp.Rational(37, 4)
    M2 = 1 + 2*V1 + V1**2 + V2
    out['M2_P1P2'] = str(sp.nsimplify(M2)); out['M2_P1P2_value'] = float(M2)
    # irregular example
    tt, ss = sp.symbols('t s')
    p = sp.Rational(1, 5) + sp.Rational(4, 100)*ss + sp.Rational(3, 1000)*ss**2
    prod = sp.expand((1 + sp.Rational(7, 100)*tt*ss)*p)
    target = sp.Rational(1, 5) + (sp.Rational(4, 100) + sp.Rational(14, 1000)*tt)*ss + (sp.Rational(3, 1000) + sp.Rational(28, 10000)*tt)*ss**2 + sp.Rational(21, 100000)*tt*ss**3
    out['irr_product'] = sp.simplify(prod - target) == 0
    A_t = sp.Rational(1, 5)*ss + (sp.Rational(4, 100) + sp.Rational(14, 1000)*tt)/2*ss**2 + (sp.Rational(3, 1000) + sp.Rational(28, 10000)*tt)/3*ss**3 + sp.Rational(525, 10**7)*tt*ss**4
    out['irr_primitive'] = sp.simplify(sp.diff(A_t, ss) - prod) == 0
    d2K = sp.diff(sp.Rational(18, 100)*prod, ss, 2)
    out['irr_d2K'] = sp.simplify(d2K - sp.Rational(18, 100)*(sp.Rational(6, 1000) + sp.Rational(56, 10000)*tt + sp.Rational(126, 100000)*tt*ss)) == 0
    b = sp.Rational(355, 100)
    out['irr_M2'] = float(d2K.subs({tt: b, ss: b}))
    Lb = sp.Rational(18, 100)*(1 + sp.Rational(7, 100)*b**2)
    out['irr_L'] = float(Lb)
    gaps = [sp.Rational(x, 100) for x in (25, 33, 45, 42, 45, 45)]
    eL = sp.exp(sp.Rational(6, 5)*Lb)*sp.prod([1 + Lb*gg for gg in gaps])
    out['irr_eL_b'] = float(sp.N(eL, 30)); out['irr_eL_b_lt_3167'] = bool(sp.N(eL, 30) < sp.Rational(3167, 1000))
    zmax = float(p.subs(ss, b) + sp.Rational(1, 10000)*b)
    out['irr_z_max'] = zmax; out['irr_arcsin_slope'] = 1/np.sqrt(1 - zmax**2)
    out['irr_bound'] = float(sp.Rational(3167, 1000)*sp.Rational(1083, 1000)*sp.Rational(1, 10000)*b)
    # e_L(b,a) for T_hyb with L=1
    out['eL_Thyb'] = float(sp.exp(sp.Rational(5, 2))*sp.Rational(5, 4)**4*sp.Rational(3, 2)**2)
    # majorant 1+lambda(t-a) on an interval solves u = 1 + int lam e^{-lam(t-s)} u ds
    lam = sp.symbols('lam', positive=True)
    out['explicit_majorant'] = sp.simplify(1 + sp.integrate(lam*sp.exp(-lam*(t - s))*(1 + lam*s), (s, 0, t)) - (1 + lam*t)) == 0
    return out

# ------------------------------------------------------------------ Part N: independent solver
GL = {n: np.polynomial.legendre.leggauss(n) for n in (8, 12, 16, 20)}

class TS:
    """[a,b]_T = union of closed components (p,q), p<=q, ordered."""
    def __init__(self, comps):
        self.c = [(float(p), float(q)) for p, q in comps]
    def gaps(self):
        return [(self.c[i][1], self.c[i+1][0]) for i in range(len(self.c)-1)]
    def nodes(self, h):
        pts = []
        for p, q in self.c:
            if q > p:
                n = int(round((q-p)/h)); pts += list(np.linspace(p, q, n+1))
            else:
                pts.append(p)
        return np.array(sorted(set(np.round(pts, 14))))
    def step_info(self, t0, t1):
        """return (H, list of gap lengths inside (t0,t1)), dense flag"""
        G = [g1-g0 for g0, g1 in self.gaps() if g0 >= t0-1e-13 and g1 <= t1+1e-13]
        return t1-t0, G
    def delta_int(self, F, t, n=16, panel=0.05):
        tot = 0.0
        for p, q in self.c:
            if p >= t: break
            d = min(q, t)
            if d > p:
                m = max(1, int(np.ceil((d-p)/panel))); e = np.linspace(p, d, m+1)
                x, w = GL[n]
                for e0, e1 in zip(e[:-1], e[1:]):
                    s = 0.5*(e1-e0)*x + 0.5*(e1+e0); tot += 0.5*(e1-e0)*np.sum(w*F(s))
        for g0, g1 in self.gaps():
            if g0 < t: tot += (g1-g0)*F(np.array([g0]))[0]
        return tot

def weights(T, t):
    N = len(t)-1; wm = np.zeros(N); wp = np.zeros(N); kap = np.zeros(N); dense = np.zeros(N, bool)
    for j in range(N):
        H, G = T.step_info(t[j], t[j+1]); S2 = sum(g*g for g in G); S3 = sum(g**3 for g in G)
        wp[j] = (H*H - S2)/(2*H); wm[j] = H - wp[j]
        kap[j] = H**3/12 - S3/3 + S2**2/(4*H); dense[j] = (len(G) == 0)
    return wm, wp, kap, dense

def nystrom(T, t, f, K, Kx, Kleft=None):
    """forward substitution; Kleft(t,s,x) = left-limit kernel used at the right node of dense steps"""
    wm, wp, _, dense = weights(T, t); Kr = Kleft or K
    X = np.empty(len(t)); X[0] = f(t[:1])[0]
    for n in range(1, len(t)):
        c = f(t[n:n+1])[0]
        for j in range(n):
            c += wm[j]*K(t[n], t[j], X[j])
            if j < n-1:
                c += wp[j]*(Kr(t[n], t[j+1], X[j+1]) if dense[j] else K(t[n], t[j+1], X[j+1]))
        w = wp[n-1]
        if w == 0: X[n] = c; continue
        z = X[n-1]
        for _ in range(60):
            g = z - c - w*K(t[n], t[n], z); dz = g/(1 - w*Kx(t[n], t[n], z)); z -= dz
            if abs(dz) < 1e-15*(1+abs(z)): break
        X[n] = z
    return X

def thyb():
    return TS([(0, 1), (1.25, 1.25), (1.5, 1.5), (1.75, 1.75), (2, 3), (3.5, 3.5), (4, 4.5)])

def part_numeric():
    out = {}
    K = lambda t, s, x: np.exp(-(t-s))*np.sin(x); Kx = lambda t, s, x: np.exp(-(t-s))*np.cos(x)
    xs = lambda t: 1 + np.exp(-np.asarray(t)/2)*np.sin(3*np.asarray(t))
    rows = []
    for name, T in [("interval", TS([(0, 2)])), ("hybrid", thyb())]:
        cache = {}
        def f(tt, T=T, cache=cache):
            o = []
            for ti in tt:
                k = round(float(ti), 13)
                if k not in cache:
                    cache[k] = xs(ti) - T.delta_int(lambda s: np.exp(-(ti-s))*np.sin(xs(s)), ti)
                o.append(cache[k])
            return np.array(o)
        prev = None; sols = {}
        for k in range(2, 10):
            h = 2.0**-k; t = T.nodes(h); X = nystrom(T, t, f, K, Kx); err = np.max(np.abs(X - xs(t)))
            sols[k] = (t, X)
            row = dict(scale=name, h=h, N=len(t)-1, error=err, eoc=None if prev is None else np.log2(prev/err))
            prev = err; rows.append(row)
        for k in range(2, 9):
            t, X = sols[k]; tf, Xf = sols[k+1]; idx = np.searchsorted(tf, t - 1e-13)
            Rr = (4*Xf[idx] - X)/3
            for r in rows:
                if r['scale'] == name and r['h'] == 2.0**-k: r['richardson'] = float(np.max(np.abs(Rr - xs(t))))
    out['table1_independent'] = rows
    # q-scale (Table 2)
    q = 1.002; pts = q**np.arange(0, 550); pts = pts[pts <= 3 + 1e-12]
    T = TS([(p, p) for p in pts]); fq = lambda tt: 1 + 0.5*np.sin(2*np.asarray(tt))
    Kq = lambda t, s, x: np.exp(-0.5*(t-s))*np.sin(x); Kqx = lambda t, s, x: np.exp(-0.5*(t-s))*np.cos(x)
    Xfull = nystrom(T, pts, fq, Kq, Kqx)
    # independent check: brute recursion x(t_n) = f(t_n) + sum mu_j K
    xr = np.empty(len(pts)); xr[0] = fq(pts[:1])[0]
    for n in range(1, len(pts)):
        xr[n] = fq(pts[n:n+1])[0] + np.sum(np.diff(pts[:n+1])*np.exp(-0.5*(pts[n]-pts[:n]))*np.sin(xr[:n]))
    out['qscale_full_vs_recursion'] = float(np.max(np.abs(Xfull - xr)))
    qrows = []
    for m in [1, 2, 4, 8, 16, 32, 64, 128]:
        idx = list(range(0, len(pts), m)); idx = idx if idx[-1] == len(pts)-1 else idx + [len(pts)-1]
        tc = pts[idx]; Xc = nystrom(T, tc, fq, Kq, Kqx); _, _, kap, _ = weights(T, tc)
        qrows.append(dict(m=m, N=len(tc)-1, H=float(np.max(np.diff(tc))), error=float(np.max(np.abs(Xc - xr[idx]))), kappa=float(kap.sum())))
    out['table2_independent'] = qrows
    # irregular hybrid T_irr (Table 4) with closed-form forcing
    Ti = TS([(0, .45), (.70, .70), (1.03, 1.03), (1.48, 1.48), (1.90, 2.30), (2.75, 2.75), (3.20, 3.55)])
    p = lambda s: .2 + .04*s + .003*s*s
    xi = lambda t: np.arcsin(p(np.asarray(t)))
    A = lambda t, s: .2*s + (.04+.014*t)*s*s/2 + (.003+.0028*t)*s**3/3 + .0000525*t*s**4
    def fi(tt):
        o = []
        for ti in np.atleast_1d(tt):
            I = 0.0
            for pp, qq in Ti.c:
                if pp < ti and qq > pp: I += A(ti, min(qq, ti)) - A(ti, pp)
            for g0, g1 in Ti.gaps():
                if g0 < ti: I += (g1-g0)*(1+.07*ti*g0)*p(g0)
            o.append(xi(ti) - .18*I)
        return np.array(o)
    Ki = lambda t, s, x: .18*(1+.07*t*s)*np.sin(x); Kix = lambda t, s, x: .18*(1+.07*t*s)*np.cos(x)
    irows = []; prev = None
    for M in [1, 2, 4, 8, 16, 32, 64]:
        t = Ti.nodes(.05/M); X = nystrom(Ti, t, fi, Ki, Kix); err = float(np.max(np.abs(X - xi(t))))
        irows.append(dict(h=.05/M, nodes=len(t), error=err, eoc=None if prev is None else float(np.log2(prev/err)))); prev = err
    out['table4_independent'] = irows
    out['irr_f_b'] = float(fi(3.55)[0]); out['irr_x_b'] = float(xi(3.55))
    return out, (Ti, fi, xi, Ki, Kix)

# ------------------------------------------------------------------ Part R: robustness to data noise
def discrete_majorant(T, t, ell):
    wm, wp, _, dense = weights(T, t); U = np.empty(len(t)); U[0] = 1
    for n in range(1, len(t)):
        c = 1 + sum(wm[j]*ell(t[n], t[j])*U[j] for j in range(n)) + sum(wp[j]*ell(t[n], t[j+1])*U[j+1] for j in range(n-1))
        U[n] = c/(1 - wp[n-1]*ell(t[n], t[n]))
    return np.maximum.accumulate(U)

def part_noise(ctx):
    Ti, fi, xi, Ki, Kix = ctx
    t = Ti.nodes(.0125); f0 = fi(t); X0 = nystrom(Ti, t, lambda tt: fi(tt), Ki, Kix)
    U = discrete_majorant(Ti, t, lambda a, b: .18*(1+.07*a*b))
    rows = []
    for lvl in [0.0, 0.01, 0.05, 0.10]:
        for seed in range(10):
            rng = np.random.default_rng(1000+seed)
            delta = lvl*np.abs(f0)*rng.standard_normal(len(t))
            fmap = dict(zip(np.round(t, 13), f0 + delta))
            Xn = nystrom(Ti, t, lambda tt: np.array([fmap[round(float(x), 13)] for x in np.atleast_1d(tt)]), Ki, Kix)
            d = Xn - X0; ex = Xn - xi(t)
            rows.append(dict(noise=lvl, seed=seed, max_abs_dev=float(np.max(np.abs(d))),
                             bound=float(U[-1]*np.max(np.abs(delta))), ratio=float(np.max(np.abs(d))/max(U[-1]*np.max(np.abs(delta)), 1e-300)) if lvl > 0 else 0.0,
                             rel_L2_error=float(np.linalg.norm(ex)/np.linalg.norm(xi(t))), rmse=float(np.sqrt(np.mean(ex**2))),
                             max_abs_error=float(np.max(np.abs(ex)))))
    return dict(U=float(U[-1]), rows=rows)

def write_csv(name, rows):
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT, "csv", name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]

if __name__ == "__main__":
    t0 = time.perf_counter()
    R['env'] = dict(python=platform.python_version(), numpy=np.__version__, sympy=sp.__version__)
    R['symbolic'] = part_symbolic(); print(json.dumps(R['symbolic'], indent=1), flush=True)
    num, ctx = part_numeric(); R['numeric'] = num
    R['noise'] = part_noise(ctx)
    R['seconds'] = time.perf_counter() - t0
    with open(os.path.join(OUT, "verification.json"), "w") as fh: json.dump(R, fh, indent=1, default=float)
    write_csv("indep_table1.csv", num['table1_independent']); write_csv("indep_table2.csv", num['table2_independent'])
    write_csv("indep_table4.csv", num['table4_independent']); write_csv("noise_robustness.csv", R['noise']['rows'])
    print("seconds", R['seconds'])
