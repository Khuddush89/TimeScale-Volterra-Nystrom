"""Experiment 2 (Section 7.2): the time-scale trapezoidal rule.
(a) Peano profiles, (b) closed form and bound for kappa, (c) composite errors."""
from common import *
from tsvolterra.timescale import TimeScale, Partition
from tsvolterra.quadrature import trapezoid_weights, kappa_step, kappa_closed_form, peano_P2
from tsvolterra.problems import ts_hybrid, ts_qscale
from tsvolterra import plotting as pl

rng = np.random.default_rng(2026)
F = lambda t, sg: np.exp(np.sin(3.0 * t))


def peano_profile(T, s, t, n=400):
    """u -> G(pi^+(u)) on [s,t], G = -P2, pi^+(u) = right end of the gap containing u (or u)"""
    u = np.linspace(s, t, n)
    out = np.empty_like(u)
    for i, uu in enumerate(u):
        # pi^+(u): if uu in T -> uu, else right end of gap
        if T.contains(uu, tol=1e-12):
            r = uu
        else:
            k = int(np.searchsorted(T.p, uu))
            r = T.p[k]
        out[i] = -peano_P2(T, s, t, r)
    return u, out


def run():
    res = {}
    # (a) Peano profiles on four unit steps
    steps = {
        "interval $[0,1]$": TimeScale([(0, 1)]),
        "uniform $\\{0,\\frac{1}{8},\\dots,1\\}$": TimeScale.from_points(np.linspace(0, 1, 9)),
        "nonuniform $\\{0,.1,.25,.45,.7,1\\}$": TimeScale.from_points([0, .1, .25, .45, .7, 1]),
        "mixed $\\{0,.15,.3\\}\\cup[.45,1]$": TimeScale([(0, 0), (.15, .15), (.3, .3), (.45, 1)]),
    }
    prof = {}
    for name, T in steps.items():
        u, g = peano_profile(T, 0.0, 1.0)
        gaps = [p - q for (q, p) in T.gaps()]
        prof[name] = dict(u=u, g=g, kappa=kappa_step(T, 0.0, 1.0), kappa_cf=kappa_closed_form(1.0, gaps))
    res["profiles"] = prof
    # (b) random steps
    rows = []
    for trial in range(1500):
        D = rng.uniform(0.2, 3.0)
        k = rng.integers(0, 8)
        cuts = np.sort(rng.uniform(0, D, 2 * k))
        comps = []; cur = 0.0
        for i in range(k):
            comps.append((cur, cuts[2 * i])); cur = cuts[2 * i + 1]
        comps.append((cur, D))
        comps = [(p, q) if rng.random() > 0.35 else (p, p) for (p, q) in comps]
        if comps[-1][1] < D: comps.append((D, D))
        clean = [comps[0]]
        for c in comps[1:]:
            if c[0] > clean[-1][1]: clean.append(c)
        T = TimeScale(clean)
        gaps = np.array([p - q for (q, p) in T.gaps()])
        kap = kappa_step(T, 0.0, D); cf = kappa_closed_form(D, gaps)
        rows.append([D, float(np.sum(gaps ** 3)), kap, cf, len(gaps)])
    rows = np.array(rows)
    res["random"] = dict(D=rows[:, 0], g3=rows[:, 1], kappa=rows[:, 2], kappa_cf=rows[:, 3], ngaps=rows[:, 4],
                         maxdev=float(np.max(np.abs(rows[:, 2] - rows[:, 3]) / rows[:, 0] ** 3)))
    # (c) composite rule errors
    def delta_int(T):
        return T.delta_integral(F, T.a, T.b, n_gauss=30, panel=0.05)

    def composite(P):
        T = P.T; t = P.t
        val = 0.0
        for j in range(P.N):
            val += P.wm[j] * F(t[j], None) + P.wp[j] * F(t[j + 1], None)
        return val

    comp = {}
    TR = TimeScale([(0, 2)]); I_R = delta_int(TR)
    hs = 2.0 ** -np.arange(2, 11)
    comp["R"] = dict(h=hs, err=[abs(I_R - composite(Partition.gap_conforming(TR, h))) for h in hs],
                     kappa=[Partition.gap_conforming(TR, h).kappa_total for h in hs])
    TH = ts_hybrid(); I_H = delta_int(TH)
    comp["hyb"] = dict(h=hs, err=[abs(I_H - composite(Partition.gap_conforming(TH, h))) for h in hs],
                       kappa=[Partition.gap_conforming(TH, h).kappa_total for h in hs])
    # non-conforming: drop right endpoints q_i (left-dense, right-scattered) of dense components
    errs = []
    for h in hs:
        P = Partition.gap_conforming(TH, h)
        drop = [q for (p, q) in TH.comps[:-1] if q > p]
        keep = np.array([not np.any(np.isclose(tt, drop)) for tt in P.t])
        Pn = Partition(TH, P.t[keep])
        errs.append(abs(I_H - composite(Pn)))
    comp["hyb_nc"] = dict(h=hs, err=errs)
    TQ = ts_qscale(1.002, 1.0, 3.0); I_Q = sum(TQ.mu(t) * F(t, None) for t in TQ.p[:-1])
    ms = [1, 2, 4, 8, 16, 32, 64, 128]
    Hs, eq, kq = [], [], []
    for m in ms:
        P = Partition.coarse(TQ, m)
        Hs.append(float(np.max(np.diff(P.t)))); eq.append(abs(I_Q - composite(P))); kq.append(P.kappa_total)
    comp["q"] = dict(m=ms, H=Hs, err=eq, kappa=kq, mu_max=float(TQ.mu(TQ.p[-2])))
    res["composite"] = comp
    save_json("exp2_quadrature", res)
    return res


def plot(res=None):
    res = load_json("exp2_quadrature") if res is None else res
    pl.setup()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6.9, 4.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.35], hspace=0.55, wspace=0.55)
    names = list(res["profiles"].keys())
    for k, name in enumerate(names):
        ax = fig.add_subplot(gs[0, k])
        d = res["profiles"][name]
        u, g = np.array(d["u"]), np.array(d["g"])
        ax.fill_between(u, 0, g, color=pl.C[0], alpha=0.12, lw=0)
        ax.plot(u, g, color=pl.C[0], lw=1.2)
        ax.plot(u, u * (1 - u) / 2, color=pl.MUTED, lw=0.7)
        ax.set_ylim(0, 0.17); ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.05, 0.1])
        if k > 0: ax.set_yticklabels([])
        ax.set_title(name, fontsize=7.2, pad=3)
        ax.text(0.5, 0.152, r"$\kappa=%.4f$" % d["kappa"], ha="center", fontsize=7.2, color=pl.INK)
        if k == 0: ax.set_ylabel(r"$-\Pi(\pi^+(u))$")
        ax.set_xlabel(r"$u$", labelpad=1)
    # (b)
    ax = fig.add_subplot(gs[1, 0:2])
    r = res["random"]
    D, g3, kap = np.array(r["D"]), np.array(r["g3"]), np.array(r["kappa"])
    xb = 1 - g3 / D ** 3; yb = kap / (D ** 3 / 12)
    ax.plot([0, 1], [0, 1], color=pl.MUTED, lw=0.8, zorder=1)
    ax.scatter(xb, yb, s=7, color=pl.C[0], edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_xlabel(r"$1-\sum_i g_i^3/\Delta^3$")
    ax.set_ylabel(r"$\kappa\,/\,(\Delta^3/12)$")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.text(0.05, 0.93, r"upper bound $\kappa\leq(\Delta^3-\sum_i g_i^3)/12$", fontsize=7, color=pl.INK2)
    ax.set_title("(b) error constants of 1500 random steps", fontsize=8, loc="left")
    # (c)
    ax = fig.add_subplot(gs[1, 2:4])
    c = res["composite"]
    ax.loglog(c["R"]["h"], c["R"]["err"], marker=pl.MARK[0], color=pl.C[0], label=r"$[0,2]$")
    ax.loglog(c["hyb"]["h"], c["hyb"]["err"], marker=pl.MARK[1], color=pl.C[1], label=r"$\mathbb{T}_{\rm hyb}$, gap-conforming")
    ax.loglog(c["hyb_nc"]["h"], c["hyb_nc"]["err"], marker=pl.MARK[2], color=pl.C[2], label=r"$\mathbb{T}_{\rm hyb}$, not conforming")
    H = np.array(c["q"]["H"]); e = np.array(c["q"]["err"])
    ax.loglog(H[1:], e[1:], marker=pl.MARK[3], color=pl.C[3], label=r"$q^{\mathbb{N}_0}$, coarse ($H$); exact if $H=\mu$")
    pl.slope_triangle(ax, 1.3e-3, 2.5e-9, 2, 0.45, label="2")
    pl.slope_triangle(ax, 2.5e-3, 4e-4, 1, 0.6, label="1", below=False)
    ax.set_xlabel(r"largest non-gap step $h$ (or $H$)"); ax.set_ylabel("quadrature error")
    ax.legend(loc="lower right", fontsize=6.6)
    ax.set_title("(c) composite rule", fontsize=8, loc="left")
    fig.text(0.07, 0.965, "(a) Peano profiles on unit steps; shaded area $=\\kappa$, grey: $u(1-u)/2$", fontsize=8)
    pl.save(fig, os.path.join(FIG, "fig_quadrature"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        t0 = time.time(); r = run(); plot(r); print("exp2 done", time.time() - t0)
        print("max closed-form deviation:", r["random"]["maxdev"])
        for k, v in r["composite"].items():
            print(k, ["%.2e" % x for x in v["err"]])
