"""Experiment 5 (Section 7.5): one network for a continuum of time scales
    T_nu = U_{k=0}^{3} [k, k+nu] U {4},   nu in [0,1]   (nu=0: {0,...,4}; nu=1: [0,4]),
for x(t) = 1 + 0.5 sin 2t + int_0^t e^{-(t-s)} sin x(s) Delta s."""
from common import *
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from tsvolterra.timescale import TimeScale, Partition
from tsvolterra import nystrom, certify
from tsvolterra.quadrature import GaussVolterraOperator
from tsvolterra.problems import *
from tsvolterra.neural import init_mlp, mlp_apply, train_adam, train_lbfgs
from tsvolterra import plotting as pl

KP = 4
K, Kx, L = kernel_exp_sin(1.0, 1.0)
fnat = lambda t: 1.0 + 0.5 * np.sin(2.0 * np.asarray(t))
fnat_j = lambda t: 1.0 + 0.5 * jnp.sin(2.0 * t)
X0 = 1.0


def problem(nu):
    return VolterraProblem(ts_family(nu, KP), K, Kx, L, f=fnat)


# minimal majorant u_ell for ell(t,s) = exp(-(t-s)):  u = 1 + int_0^t exp(-(t-s)) u(s) Delta s
Kmaj = lambda t, s, sg, x, xp=np: xp.exp(-(t - s)) * x
Kmajx = lambda t, s, sg, x, xp=np: xp.exp(-(t - s)) + 0 * x


def majorant(nu, lev0=6, levs=(3, 4)):
    """u_ell on T_nu at the nodes of the reference partition (Richardson extrapolation, enlarged by 0.1%)"""
    T = ts_family(nu, KP)
    pm = VolterraProblem(T, Kmaj, Kmajx, 1.0, f=lambda t: 1.0 + 0 * np.asarray(t))
    c0 = [max(1, int(np.ceil((q - p) * 2 ** lev0))) if q > p else 0 for (p, q) in T.comps]
    if all(c == 0 for c in c0):
        P = Partition(T, T.p.copy())
        return P, nystrom.solve(pm, P)          # exact on a finite time scale
    Ps = [Partition.gap_conforming(T, counts=[c * 2 ** l for c in c0]) for l in levs]
    Us = [nystrom.solve(pm, P) for P in Ps]
    return Ps[0], 1.001 * nystrom.richardson(Us[0], nystrom.restrict(Ps[1], Us[1], Ps[0]))


def add_majorant(out):
    """eta_maj = max u_ell * sup |r_v|  (two-sided a posteriori estimate with Gamma = u_ell); equals max_t u_ell(t) rho(t)
    when u_ell is nondecreasing, which is checked and recorded."""
    for r in out["rows"]:
        P, U = majorant(r["nu"])
        r["Gamma_max"] = float(np.max(U)); r["Gamma_monotone"] = bool(np.all(np.diff(U) >= -1e-14))
        r["eta_maj"] = r["Gamma_max"] * r["sup_r"]
    return out


def G_clock(t, nu, xp=np):
    """graininess clock of T_nu, continuous in (t,nu) on [0,4]x[0,1]"""
    g = 0.0 * t
    for k in range(KP):
        g = g + xp.minimum(xp.maximum(t - k - nu, 0.0), 1.0 - nu)
    return g


def net_v(params, t, nu):
    xi = t / (KP / 2.0) - 1.0
    gam = G_clock(t, nu, jnp) / (KP / 2.0) - 1.0
    eta = 2.0 * nu - 1.0
    z = jnp.stack([xi, gam, eta * jnp.ones_like(t)], axis=-1)
    return X0 + (t / KP) * mlp_apply(params, z.reshape(-1, 3)).reshape(t.shape)


def colloc_points(nu, spacing=0.05):
    pts = []
    for k in range(KP):
        if nu > 0:
            n = max(1, int(np.ceil(nu / spacing)))
            pts.append(np.linspace(k, k + nu, n + 1))
        else:
            pts.append(np.array([float(k)]))
    pts.append(np.array([float(KP)]))
    return np.unique(np.concatenate(pts))


def build_batch(nus, spacing=0.05, n_gauss=8):
    ops = [GaussVolterraOperator(ts_family(nu, KP), colloc_points(nu, spacing), n_gauss=n_gauss) for nu in nus]
    nt = max(len(o.t) for o in ops); ns = max(len(o.s) for o in ops)
    B = len(nus)
    Tt = np.zeros((B, nt)); Ss = np.zeros((B, ns)); W = np.zeros((B, ns)); SG = np.zeros((B, ns))
    M = np.zeros((B, nt, ns)); Fv = np.zeros((B, nt)); RM = np.zeros((B, nt))
    for b, o in enumerate(ops):
        n1, n2 = len(o.t), len(o.s)
        Tt[b, :n1] = o.t; Tt[b, n1:] = o.t[-1]
        Ss[b, :n2] = o.s; Ss[b, n2:] = o.s[-1] if n2 else 0.0
        W[b, :n2] = o.w; SG[b, :n2] = o.sig; SG[b, n2:] = Ss[b, n2:]
        M[b, :n1, :n2] = o.mask
        Fv[b, :n1] = fnat(o.t); RM[b, :n1] = 1.0
    return dict(T=jnp.asarray(Tt), S=jnp.asarray(Ss), W=jnp.asarray(W), SG=jnp.asarray(SG), M=jnp.asarray(M),
                F=jnp.asarray(Fv), RM=jnp.asarray(RM), NU=jnp.asarray(np.asarray(nus, float)))


def residual_batch(params, bt):
    nu = bt["NU"][:, None]
    vt = net_v(params, bt["T"], nu)
    vs = net_v(params, bt["S"], nu)
    Km = K(bt["T"][:, :, None], bt["S"][:, None, :], bt["SG"][:, None, :], vs[:, None, :], xp=jnp)
    I = jnp.sum(bt["M"] * bt["W"][:, None, :] * Km, axis=2)
    return (vt - bt["F"] - I) * bt["RM"]


def reference(nu, lev0=6, levs=(3, 4)):
    prob = problem(nu); T = prob.T
    c0 = [max(1, int(np.ceil((q - p) * 2 ** lev0))) if q > p else 0 for (p, q) in T.comps]
    if all(c == 0 for c in c0):
        P = Partition(T, T.p.copy()); X = nystrom.solve(prob, P)
        return prob, P, X, 0.0
    Ps = [Partition.gap_conforming(T, counts=[c * 2 ** l for c in c0]) for l in levs]
    t0 = time.time(); Xs = [nystrom.solve(prob, P) for P in Ps]; dt = time.time() - t0
    R = nystrom.richardson(Xs[0], nystrom.restrict(Ps[1], Xs[1], Ps[0]))
    return prob, Ps[0], R, dt


def cheapest_direct(nu, tol):
    """time of the cheapest trapezoid Nystrom solve with error <= tol (vs the reference)"""
    prob, Pr, Xr, _ = reference(nu)
    T = prob.T
    if all(q == p for (p, q) in T.comps):
        t0 = time.time(); nystrom.solve(prob, Partition(T, T.p.copy())); return time.time() - t0, 0
    for lev in range(1, 12):
        P = Partition.gap_conforming(T, counts=[max(1, int(np.ceil((q - p) * 2 ** lev))) if q > p else 0 for (p, q) in T.comps])
        t0 = time.time(); X = nystrom.solve(prob, P); dt = time.time() - t0
        # compare at common nodes
        common = np.intersect1d(np.round(P.t, 12), np.round(Pr.t, 12))
        ia = np.searchsorted(np.round(P.t, 12), common); ib = np.searchsorted(np.round(Pr.t, 12), common)
        if np.max(np.abs(X[ia] - Xr[ib])) <= tol:
            return dt, lev
    return dt, lev


def run(n_train=25, width=48, depth=3, adam_steps=8000, lbfgs_steps=4000, n_test=41, name="exp5_family"):
    nus_tr = 0.5 * (1 - np.cos(np.pi * np.arange(n_train) / (n_train - 1)))
    bt = build_batch(nus_tr)
    print("batch shapes", {k: tuple(v.shape) for k, v in bt.items()}, flush=True)
    params0 = init_mlp(jax.random.PRNGKey(7), width, depth, d_in=3)
    scale = float(np.sqrt(np.asarray(bt["RM"]).size / np.sum(np.asarray(bt["RM"]))))
    rf = lambda p: residual_batch(p, bt).ravel() * scale
    t0 = time.time()
    p, h1 = train_adam(rf, params0, steps=adam_steps, lr=3e-3, lr_end=1e-4, log_every=200)
    ta = time.time() - t0
    p, h2 = train_lbfgs(rf, p, maxiter=lbfgs_steps, log_every=200, it0=adam_steps, t_offset=ta)
    t_train = time.time() - t0
    print("training time", t_train, "final loss", (h2 or h1)[-1]["loss"], flush=True)
    np.save(os.path.join(RES, name + "_params.npy"), np.asarray(ravel_pytree(p)[0]))
    vj = jax.jit(lambda t, nu: net_v(p, t, nu))
    vnp = lambda nu: (lambda t: np.asarray(vj(jnp.asarray(np.atleast_1d(np.asarray(t, float))), float(nu))))
    gradx4 = jax.jit(jax.grad(lambda nu: net_v(p, jnp.array([float(KP)]), nu)[0]))
    # evaluation
    nus_te = np.linspace(0.0, 1.0, n_test)
    rows = []
    for nu in nus_te:
        prob, Pr, Xr, t_ref = reference(nu)
        vf = vnp(nu)
        _ = vf(Pr.t)  # warm-up / compile for this shape
        t0 = time.time(); V = vf(Pr.t); t_eval = time.time() - t0
        err = float(np.max(np.abs(V - Xr)))
        tc0 = time.time()
        cert = certify.certificate(prob, vf, Pr.t, n_gauss=8, max_panel=1.0)
        t_cert = time.time() - tc0
        rows.append(dict(nu=float(nu), err=err, sup_r=cert["sup_r"], eL=cert["eL"], eta=cert["eta"],
                         x4_ref=float(Xr[-1]), x4_net=float(V[-1]), dx4_net=float(gradx4(float(nu))),
                         t_eval=t_eval, t_cert=t_cert, t_ref=t_ref, n_eval=len(Pr.t),
                         in_train=bool(np.any(np.isclose(nu, nus_tr)))))
        print(f"nu={nu:.3f} err={err:.2e} r={cert['sup_r']:.2e} eta={cert['eta']:.2e} x4={Xr[-1]:.6f}", flush=True)
    # reference derivative of x(4;nu) by central differences of reference solutions
    dnu = 1e-4
    dref = []
    for nu in nus_te[1:-1]:
        xp_ = reference(nu + dnu)[2][-1]; xm_ = reference(nu - dnu)[2][-1]
        dref.append(dict(nu=float(nu), dx4=float((xp_ - xm_) / (2 * dnu))))
    # cost of the cheapest direct solve matching the median surrogate error
    med = float(np.median([r["err"] for r in rows]))
    costs = []
    for nu in [0.25, 0.5, 0.75, 1.0]:
        dt, lev = cheapest_direct(nu, med)
        costs.append(dict(nu=nu, t_direct=dt, lev=lev))
    # heat map data on a grid
    tg = np.linspace(0, KP, 801); ng = np.linspace(0, 1, 201 if n_test > 10 else 11)
    Z = np.array([vnp(nu)(tg) for nu in ng])
    inT = np.array([[ts_family(nu, KP).contains(t, tol=1e-9) for t in tg] for nu in ng])
    out = dict(rows=rows, dref=dref, costs=costs, median_err=med, t_train=t_train, hist=h1 + h2,
               n_params=int(ravel_pytree(p)[0].size), nus_train=nus_tr, heat=dict(t=tg, nu=ng, Z=Z, inT=inT),
               width=width, depth=depth)
    add_majorant(out)
    save_json(name, out)
    return out




def plot(out=None):
    out = load_json("exp5_family") if out is None else out
    pl.setup()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fig = plt.figure(figsize=(6.9, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], hspace=0.5, wspace=0.27)
    # (a) heat map of the network on {(t,nu): t in T_nu}
    ax = fig.add_subplot(gs[0, :])
    H = out["heat"]; tg = np.array(H["t"]); ng = np.array(H["nu"]); Z = np.array(H["Z"]); inT = np.array(H["inT"], bool)
    Zm = np.ma.masked_where(~inT, Z)
    ax.set_facecolor(pl.GAP)
    norm = Normalize(Z[inT].min(), Z[inT].max())
    im = ax.pcolormesh(tg, ng, Zm, cmap="turbo", norm=norm, shading="nearest", rasterized=True)
    for k in range(KP + 1):   # the isolated points of T_0 = {0,...,4}
        ax.plot([k], [0.0], "s", ms=3.0, color=plt.get_cmap("turbo")(norm(Z[0, np.argmin(abs(tg - k))])), mec=pl.INK, mew=0.3, clip_on=False, zorder=5)
    nus_tr = np.array(out["nus_train"])
    for nv in nus_tr:          # training values of nu
        ax.plot([3.955, 4.0], [nv, nv], color=pl.INK, lw=0.8, solid_capstyle="butt", zorder=4)
    ax.grid(False)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\nu$")
    ax.set_xlim(0, 4.0); ax.set_ylim(-0.012, 1.0)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cb.set_label(r"$v_\theta(t;\nu)$", fontsize=8); cb.outline.set_linewidth(0.4)
    ax.set_title(r"(a) one network for all $\mathbb{T}_\nu$: $v_\theta(t;\nu)$ for $t\in\mathbb{T}_\nu$ (grey: gaps; ticks at $t=4$: training values of $\nu$)", loc="left")
    # (b) x(4;nu) and its derivative
    ax = fig.add_subplot(gs[1, 0])
    rows = out["rows"]
    nu = np.array([r["nu"] for r in rows]); x4r = np.array([r["x4_ref"] for r in rows]); x4n = np.array([r["x4_net"] for r in rows])
    ax.plot(nu, x4r, color=pl.C[0], lw=1.6, label=r"reference $x(4;\nu)$", zorder=2)
    ax.plot(nu[::2], x4n[::2], ls="none", marker="o", ms=3.6, mfc="none", mec=pl.C[1], mew=0.9, label=r"network $v_\theta(4;\nu)$", zorder=3)
    ax.set_xlabel(r"$\nu$"); ax.set_ylabel(r"$x(4;\nu)$")
    ax.set_title(r"(b) value and sensitivity at $t=4$", loc="left")
    ax.legend(loc="upper left", fontsize=6.4)
    ax2 = ax.inset_axes([0.52, 0.13, 0.44, 0.42])
    dref = out["dref"]
    ax2.plot([d["nu"] for d in dref], [d["dx4"] for d in dref], color=pl.C[0], lw=1.4, zorder=2)
    dn = np.array([r["dx4_net"] for r in rows])
    ax2.plot(nu[1::2], dn[1::2], ls="none", marker="o", ms=2.6, mfc="none", mec=pl.C[1], mew=0.7, zorder=3)
    ax2.set_title(r"$\partial_\nu x(4;\nu)$", fontsize=6.6, pad=2)
    ax2.tick_params(labelsize=5.8, length=2, pad=1)
    for sp in ax2.spines.values(): sp.set_linewidth(0.5)
    # (c) error and certificates
    ax = fig.add_subplot(gs[1, 1])
    err = np.array([r["err"] for r in rows]); eta = np.array([r["eta"] for r in rows]); intr = np.array([r["in_train"] for r in rows])
    ax.semilogy(nu, eta, color=pl.C[1], lw=1.0, ls=(0, (3, 1.5)), label=r"indicator, $\Gamma=e_L(\cdot,0)$")
    if "eta_maj" in rows[0]:
        etam = np.array([r["eta_maj"] for r in rows])
        ax.semilogy(nu, etam, color=pl.C[1], lw=1.5, label=r"indicator, $\Gamma=u_\ell$")
    ax.semilogy(nu, err, color=pl.C[0], lw=1.5, label=r"true error on $\mathbb{T}_\nu$")
    ax.semilogy(nu[intr], err[intr], ls="none", marker="o", ms=3.4, color=pl.C[0], mec="white", mew=0.5, label=r"training values of $\nu$")
    ax.set_xlabel(r"$\nu$"); ax.set_title(r"(c) sampled accuracy at 41 values of $\nu$", loc="left")
    ax.legend(loc="upper left", fontsize=6.2, ncol=1)
    lo = min(err.min(), eta.min()) / 2.5; hi = max(eta.max(), err.max()) * 60
    ax.set_ylim(lo, hi)
    pl.save(fig, os.path.join(FIG, "fig_family"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    elif len(sys.argv) > 1 and sys.argv[1] == "majorant":
        o = add_majorant(load_json("exp5_family")); save_json("exp5_family", o); plot(o)
    else:
        t0 = time.time(); o = run(); print("exp5 done", time.time() - t0); plot(o)
