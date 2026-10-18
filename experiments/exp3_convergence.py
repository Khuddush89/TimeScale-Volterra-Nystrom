"""Experiment 3 (Section 7.3): convergence of the time-scale Nystrom method.
E1: T=[0,2]; E2: hybrid scale; E3: q-scale with coarse partitions;
E4: pendulum x^{DD} = -omega^2 sin x on a hybrid scale (sigma-dependent kernel)."""
from common import *
from scipy.integrate import solve_ivp
from tsvolterra.timescale import TimeScale, Partition
from tsvolterra import nystrom
from tsvolterra.problems import *
from tsvolterra import plotting as pl


def counts_for(T, lev):
    return [int(round(2 ** lev * (q - p))) if q > p else 0 for (p, q) in T.comps]


def M2_estimate(prob, x_fun, xpp_fun=None, nodes=None, n_fine=4000):
    """max over nodes t_n and dense s<t_n of |d^2/ds^2 K(t_n, s, s, x(s))| (finite differences)"""
    T = prob.T
    best = 0.0
    tn_list = nodes if nodes is not None else T.sample(20)
    for (p, q) in T.comps:
        if q <= p: continue
        s = np.linspace(p, q, n_fine)
        hs = s[1] - s[0]
        xs = x_fun(s)
        for tn in tn_list:
            if tn < p: continue
            msk = s <= tn + 1e-14
            if msk.sum() < 5: continue
            Fv = prob.K(tn, s[msk], s[msk], xs[msk])
            d2 = (Fv[2:] - 2 * Fv[1:-1] + Fv[:-2]) / hs ** 2
            best = max(best, float(np.max(np.abs(d2))))
    return best


def pendulum_reference(T, om, x0, v0):
    """x on T via the dynamic equation: ODE on dense components, recursion across gaps"""
    x, v = x0, v0
    sols = []
    for i, (p, q) in enumerate(T.comps):
        if q > p:
            sol = solve_ivp(lambda t, y: [y[1], -om ** 2 * np.sin(y[0])], (p, q), [x, v], method="DOP853",
                            rtol=1e-13, atol=1e-15, dense_output=True)
            sols.append(("d", p, q, sol.sol))
            x, v = sol.sol(q)
        else:
            sols.append(("p", p, p, (x, v)))
        if i < T.m - 1:
            mu = T.p[i + 1] - q
            x, v = x + mu * v, v - mu * om ** 2 * np.sin(x)

    def xfun(t):
        t = np.atleast_1d(np.asarray(t, float)); out = np.empty_like(t)
        for kind, p, q, obj in sols:
            msk = (t >= p - 1e-13) & (t <= q + 1e-13)
            if not msk.any():
                continue
            if kind == "d":
                out[msk] = obj(np.clip(t[msk], p, q))[0]
            else:
                out[msk] = obj[0]
        return out
    return xfun


def study(prob, x_ref, levels, naive=False, richardson=True, M2=None, ell=None):
    rows = []
    sols = {}
    for lev in levels:
        P = Partition.gap_conforming(prob.T, counts=counts_for(prob.T, lev))
        t0 = time.time(); X = nystrom.solve(prob, P, naive=naive); dt = time.time() - t0
        xr = x_ref(P.t)
        err = float(np.max(np.abs(X - xr)))
        tau = nystrom.residual(prob, P, xr, naive=naive)
        try:
            S, gam = nystrom.stability_constant(P, prob.L)
        except ValueError:
            S = np.array([np.nan])
        row = dict(lev=lev, h=P.h_dense(), N=P.N, err=err, time=dt, kappa=P.kappa_total, S=float(S[-1]),
                   tau=float(np.max(np.abs(tau))), B1=float(S[-1] * np.max(np.abs(tau))))
        if M2 is not None:
            row["B2"] = float(S[-1] * P.kappa_total * M2)
        if ell is not None:
            U = nystrom.majorant_stability(P, ell)
            row["U"] = float(U[-1]); row["B1m"] = float(U[-1] * np.max(np.abs(tau)))
            if M2 is not None:
                row["B2m"] = float(U[-1] * P.kappa_total * M2)
        rows.append(row); sols[lev] = (P, X)
    for i in range(1, len(rows)):
        rows[i]["eoc"] = float(np.log2(rows[i - 1]["err"] / rows[i]["err"]))
    rich = []
    if richardson:
        for i in range(len(levels) - 1):
            Pc, Xc = sols[levels[i]]; Pf, Xf = sols[levels[i + 1]]
            R = nystrom.richardson(Xc, nystrom.restrict(Pf, Xf, Pc))
            rich.append(dict(lev=levels[i], h=Pc.h_dense(), err=float(np.max(np.abs(R - x_ref(Pc.t)))),
                             time=rows[i]["time"] + rows[i + 1]["time"]))
        for i in range(1, len(rich)):
            rich[i]["eoc"] = float(np.log2(rich[i - 1]["err"] / max(rich[i]["err"], 1e-300)))
    return rows, rich


def run():
    out = {}
    K, Kx, L = kernel_exp_sin(1.0, 1.0)
    # E1
    prob = VolterraProblem(ts_interval(0.0, 2.0), K, Kx, L, x_exact=x_star)
    M2 = 19.5 + np.sqrt(37.0)  # analytic bound from the manuscript
    ell = lambda t, s: np.exp(-(t - s))
    out["E1"] = dict(zip(["trap", "rich"], study(prob, x_star, list(range(2, 11)), M2=M2, ell=ell)), M2=M2,
                     eL=prob.T.exp_const(L, prob.b))
    # E2
    prob = VolterraProblem(ts_hybrid(), K, Kx, L, x_exact=x_star)
    M2 = 19.5 + np.sqrt(37.0)  # analytic bound from the manuscript
    out["E2"] = dict(zip(["trap", "rich"], study(prob, x_star, list(range(2, 11)), M2=M2, ell=ell)), M2=M2,
                     eL=prob.T.exp_const(L, prob.b))
    # Picard vs forward substitution on E2
    pic = []
    for lev in [3, 5, 7, 9]:
        P = Partition.gap_conforming(prob.T, counts=counts_for(prob.T, lev))
        t0 = time.time(); X = nystrom.solve(prob, P); t_fs = time.time() - t0
        t0 = time.time(); Y, its, hist = nystrom.picard(prob, P, tol=1e-15); t_pi = time.time() - t0
        pic.append(dict(lev=lev, N=P.N, t_fs=t_fs, t_picard=t_pi, its=its, diff=float(np.max(np.abs(X - Y)))))
    out["picard"] = pic
    # E3: q-scale, exact solution is the recursion on the full time scale
    TQ = ts_qscale(1.002, 1.0, 3.0)
    Kq, Kxq, Lq = kernel_exp_sin(1.0, 0.5)
    fq = lambda t: 1.0 + 0.5 * np.sin(2.0 * np.asarray(t))
    probq = VolterraProblem(TQ, Kq, Kxq, Lq, f=fq)
    Pfull = Partition(TQ, TQ.p.copy())
    Xex = nystrom.solve(probq, Pfull)
    # discrete M2 on the q-scale: second delta derivatives of F_n(s) = K(t_n,s,sigma(s),x(s))
    s = TQ.p; mu = np.diff(s); M2q = 0.0
    for n in range(2, len(s)):
        Fv = Kq(s[n], s[:n + 1], np.append(s[1:n + 1], s[n]), Xex[:n + 1])
        D1 = np.diff(Fv) / mu[:n]
        D2 = np.diff(D1) / mu[:n - 1]
        M2q = max(M2q, float(np.max(np.abs(D2))))
    rowsq = []
    for m in [1, 2, 4, 8, 16, 32, 64, 128]:
        P = Partition.coarse(TQ, m)
        t0 = time.time(); X = nystrom.solve(probq, P); dt = time.time() - t0
        idx = np.searchsorted(TQ.p, P.t)
        err = float(np.max(np.abs(X - Xex[idx])))
        S, gam = nystrom.stability_constant(P, Lq)
        U = nystrom.majorant_stability(P, lambda t, s: np.exp(-0.5 * (t - s)))
        rowsq.append(dict(m=m, N=P.N, H=float(np.max(np.diff(P.t))), err=err, kappa=P.kappa_total,
                          S=float(S[-1]), B2=float(S[-1] * P.kappa_total * M2q), U=float(U[-1]),
                          B2m=float(U[-1] * P.kappa_total * M2q), time=dt))
    out["E3"] = dict(rows=rowsq, n_points=len(TQ.p), M2=M2q, mu_min=float(mu.min()), mu_max=float(mu.max()))
    # 3^N0 sanity check of exactness (example of the earlier literature): x = 10 e_{1/100}(t,1)
    T3 = TimeScale.from_points(3.0 ** np.arange(5))
    K3 = lambda t, s, sg, x, xp=np: 0.01 * x + 0 * t
    K3x = lambda t, s, sg, x, xp=np: 0.01 + 0 * x
    p3 = VolterraProblem(T3, K3, K3x, 0.01, f=lambda t: 10.0 + 0 * np.asarray(t))
    X3 = nystrom.solve(p3, Partition(T3, T3.p.copy()))
    ex3 = 10 * np.array([T3.exp_const(0.01, t) for t in T3.p])
    out["E3b"] = dict(err=float(np.max(np.abs(X3 - ex3))), X=X3, exact=ex3)
    # E4 pendulum
    T4 = ts_pendulum(); om, x0, v0 = 2.0, 1.0, 0.0
    K4, K4x, L4 = kernel_pendulum(om, T4.b - T4.a)
    prob4 = VolterraProblem(T4, K4, K4x, L4, f=lambda t: x0 + v0 * (np.asarray(t) - T4.a))
    xref = pendulum_reference(T4, om, x0, v0)
    M24 = M2_estimate(prob4, xref)
    tr4, ri4 = study(prob4, xref, list(range(3, 12)), M2=M24)
    trn, _ = study(prob4, xref, list(range(3, 12)), naive=True, richardson=False)
    out["E4"] = dict(trap=tr4, rich=ri4, naive=trn, M2=M24)
    save_json("exp3_convergence", out)
    return out


def plot(out=None):
    out = load_json("exp3_convergence") if out is None else out
    pl.setup()
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.9))
    axs = axs.ravel()

    def series(ax, rows, key, xkey="h", **kw):
        x = np.array([r[xkey] for r in rows]); y = np.array([r.get(key, np.nan) for r in rows], float)
        m = np.isfinite(y) & (y > 0)
        ax.loglog(x[m], y[m], **kw)
    for ax, key, title in zip(axs[:2], ["E1", "E2"], [r"(a) $\mathbb{T}=[0,2]$", r"(b) $\mathbb{T}_{\rm hyb}$"]):
        d = out[key]
        series(ax, d["trap"], "B2m", color=pl.C[0], lw=0.9, alpha=0.5, label=r"a priori bound $\bar U_N\,\kappa(\mathcal{P})\,M_2$")
        series(ax, d["trap"], "err", color=pl.C[0], marker=pl.MARK[0], label="time-scale trapezoid")
        series(ax, d["rich"], "err", color=pl.C[1], marker=pl.MARK[1], label="Richardson extrapolation")
        ax.set_title(title, loc="left"); ax.set_xlabel(r"largest dense step $h$")
        ax.set_ylim(5e-14, 3)
        pl.slope_triangle(ax, 2.2e-3, 1e-10, 2, 0.4, "2")
        pl.slope_triangle(ax, 3.2e-2, 1.5e-12, 4, 0.33, "4")
        ax.set_ylabel(r"$\max_n|x(t_n)-X_n|$")
    axs[0].legend(loc="upper left", fontsize=6.6)
    ax = axs[2]; d = out["E3"]["rows"]
    H = np.array([r["H"] for r in d]); e = np.array([r["err"] for r in d]); B = np.array([r["B2m"] for r in d])
    ax.loglog(H[1:], e[1:], color=pl.C[0], marker=pl.MARK[0], label=r"coarse partitions, $m=2,\dots,128$")
    ax.text(0.97, 0.06, r"full partition $m=1$: exact to roundoff", transform=ax.transAxes, fontsize=6.8,
            color=pl.INK2, ha="right", va="bottom")
    pl.slope_triangle(ax, 0.2, 2e-5, 2, 0.3, "2")
    ax.set_title(r"(c) $\mathbb{T}=q^{\mathbb{N}_0}\cap[1,3]$, $q=1.002$ (550 points)", loc="left")
    ax.set_xlabel(r"largest step $H$ (every $m$-th point of $\mathbb{T}$)")
    ax.set_ylabel(r"$\max_n|x(t_n)-X_n|$")
    ax.legend(loc="upper left", fontsize=6.6)
    ax = axs[3]; d = out["E4"]
    series(ax, d["naive"], "err", color=pl.C[2], marker=pl.MARK[2], label=r"naive value $K(t,q_i,\cdot)$")
    series(ax, d["trap"], "err", color=pl.C[0], marker=pl.MARK[0], label=r"left limit $K(t,q_i^-,\cdot)$")
    series(ax, d["rich"], "err", color=pl.C[1], marker=pl.MARK[1], label="left limit + Richardson")
    pl.slope_triangle(ax, 1.2e-3, 3e-2, 1, 0.5, "1", below=False)
    pl.slope_triangle(ax, 6e-4, 3e-9, 2, 0.4, "2")
    ax.set_ylim(3e-13, 3)
    ax.set_title(r"(d) $y^{\Delta\Delta}=-\omega^2\sin y$ on $\mathbb{T}_{\rm pend}$", loc="left")
    ax.set_xlabel(r"largest dense step $h$")
    ax.legend(loc="lower right", fontsize=6.3, bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    pl.save(fig, os.path.join(FIG, "fig_convergence"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        t0 = time.time(); o = run(); plot(o); print("exp3 done", time.time() - t0)
        for key in ["E1", "E2", "E4"]:
            print(key, "M2", o[key]["M2"])
            for r in o[key]["trap"]:
                print("  trap", {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in r.items()})
            for r in o[key]["rich"]:
                print("  rich", {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in r.items()})
        for r in o["E4"]["naive"]:
            print("  naive", r["h"], r["err"], r.get("eoc"))
        for r in o["E3"]["rows"]:
            print("  E3", r)
        print("E3b exactness error", o["E3b"]["err"])
        print("picard", o["picard"])
