"""Animated GIFs for the repository README; every frame is computed from an actual run.
Run: python experiments/make_gifs.py   (writes ../assets/*.gif)"""
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import least_squares
from tsvolterra.timescale import TimeScale, Partition
from tsvolterra import nystrom
from tsvolterra.problems import VolterraProblem, kernel_exp_sin, ts_hybrid, ts_family
from tsvolterra.quadrature import GaussVolterraOperator
from tsvolterra.tsplot import DARK, plot_on_scale, component_colors, scale_strip
from revision_checks import irregular_problem
from revision_neural import network

plt.rcParams.update(DARK)
ASSETS = os.path.join(ROOT, "assets", "gifs"); os.makedirs(ASSETS, exist_ok=True)
LOG = {}


def save(anim, name, fps):
    t0 = time.time(); anim.save(os.path.join(ASSETS, name), writer=PillowWriter(fps=fps), dpi=90)
    LOG[name] = time.time() - t0; print(name, f"{LOG[name]:.1f}s", flush=True)


def gif_forward_substitution():
    T = ts_hybrid(); K, Kx, L = kernel_exp_sin(1, 1)
    prob = VolterraProblem(T, K, Kx, L, f=lambda t: 1 + 0.5 * np.sin(2 * np.asarray(t)))
    P = Partition.gap_conforming(T, h=2 ** -4); X = nystrom.solve(prob, P)
    Pr = Partition.gap_conforming(T, h=2 ** -9); Xr = nystrom.solve(prob, Pr)
    cols = component_colors(T.m, "turbo")
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    def draw(n):
        ax.clear(); ax.set_xlim(-0.1, 4.6); ax.set_ylim(0.95, 2.45)
        plot_on_scale(ax, T.comps, Pr.t, Xr, colors=[(*c[:3], 0.35) for c in cols], lw=1.2)
        for (p, q), c in zip(T.comps, cols):
            m = (P.t[:n + 1] >= p - 1e-12) & (P.t[:n + 1] <= q + 1e-12)
            ax.plot(P.t[:n + 1][m], X[:n + 1][m], "o-" if q > p else "o", color=c, ms=4, lw=1.6)
        ax.plot(P.t[n], X[n], "o", ms=11, mfc="none", mec="w", mew=1.5)
        scale_strip(ax, T.comps, 1.0, colors=cols)
        kind = P.kind[n - 1] if n > 0 else "start"
        ax.set_title(f"forward substitution on $\\mathbb{{T}}_{{hyb}}$: node {n}/{P.N}, t={P.t[n]:.3f} ({kind} step)")
        ax.set_xlabel("t ∈ T"); ax.set_ylabel("X_n")
    save(FuncAnimation(fig, draw, frames=range(P.N + 1)), "forward_substitution_hybrid.gif", 8); plt.close(fig)


def gif_refinement_irregular():
    prob, p, x, _ = irregular_problem(); T = prob.T
    sols = []
    for M in [1, 2, 4, 8, 16, 32, 64]:
        counts = [round((q - p0) / .05) * M if q > p0 else 0 for p0, q in T.comps]
        P = Partition.gap_conforming(T, counts=counts); X = nystrom.solve(prob, P)
        sols.append((P, X, float(np.max(abs(X - x(P.t))))))
    tt = np.concatenate([np.linspace(a, b, 200) if b > a else [a] for a, b in T.comps])
    cols = component_colors(T.m, "plasma")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    def draw(i):
        for a in ax: a.clear()
        P, X, e = sols[i]
        plot_on_scale(ax[0], T.comps, tt, x(tt), colors=[(1, 1, 1, .5)] * T.m, lw=1)
        plot_on_scale(ax[0], T.comps, P.t, X, colors=cols, dense_markers=True, ms=5)
        ax[0].set(title=f"$\\mathbb{{T}}_{{irr}}$: {P.N + 1} nodes, $h_d$={P.h_dense():.4f}", xlabel="t ∈ T", ylabel="x(t)")
        h = [s[0].h_dense() for s in sols[:i + 1]]; E = [s[2] for s in sols[:i + 1]]
        ax[1].loglog(h, E, "o-", color="#ff7675"); hh = np.array([sols[0][0].h_dense(), sols[-1][0].h_dense()])
        ax[1].loglog(hh, sols[0][2] * (hh / hh[0]) ** 2, "--", color="#74b9ff", lw=0.8, label="slope 2")
        ax[1].set(xlim=(5e-4, 0.08), ylim=(2e-10, 5e-6), title="dense refinement, six jumps fixed", xlabel="$h_d$", ylabel="max nodal error")
        ax[1].legend()
    save(FuncAnimation(fig, draw, frames=range(len(sols))), "refinement_irregular.gif", 1.5); plt.close(fig)


def gif_gallery():
    D = load_json("exp7_unusual")
    keys = ["C4", "Pab", "Rnd", "Harm", "Sqrt", "Qacc"]
    titles = {"C4": "Cantor prefractal C4 of [0,3]", "Pab": "P_{0.3,0.2} = U [0.5k, 0.5k+0.3]",
              "Rnd": "random irregular union (seed 2026)", "Harm": "harmonic numbers {H_n}, n<=400",
              "Sqrt": "sqrt(N0) on [0,20]", "Qacc": "0.9^N0 U {0}  (0 is right-dense)"}
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    def draw(i):
        ax.clear(); k = keys[i]; d = D[k]; t = np.array(d["profile"]["t"]); x = np.array(d["profile"]["x"])
        if "comps" in d:
            comps = [tuple(c) for c in d["comps"]]; cols = plot_on_scale(ax, comps, t, x, cmap="turbo", ms=5)
            scale_strip(ax, comps, x.min() - 0.1, colors=cols)
        else:
            if k == "Qacc":
                t = np.concatenate([[0], t]); x = np.concatenate([[1], x]); ax.set_xscale("symlog", linthresh=1e-3)
            ax.scatter(t, x, c=np.log10(np.maximum(t, 1e-6)) if k == "Qacc" else t, cmap="turbo", s=10)
        ax.set_title("x = f + ∫ e^{-(t-s)} sin x(s) Δs  on  " + titles[k]); ax.set_xlabel("t ∈ T"); ax.set_ylabel("x(t)")
    save(FuncAnimation(fig, draw, frames=range(len(keys))), "timescale_gallery.gif", 0.6); plt.close(fig)


def gif_family():
    K, Kx, L = kernel_exp_sin(1, 1); f = lambda t: 1 + 0.5 * np.sin(2 * np.asarray(t))
    nus = np.linspace(0, 1, 41); frames = []
    for nu in nus:
        T = ts_family(nu); prob = VolterraProblem(T, K, Kx, L, f=f)
        P = Partition.gap_conforming(T, h=2 ** -6) if nu > 0 else Partition(T, T.p.copy())
        frames.append((T, P.t, nystrom.solve(prob, P)))
    fig, ax = plt.subplots(figsize=(7.5, 3.4), layout="constrained")
    def draw(i):
        ax.clear(); T, t, X = frames[i]; cols = plot_on_scale(ax, T.comps, t, X, cmap="cool", ms=6)
        scale_strip(ax, T.comps, 0.9, colors=cols); ax.set_xlim(-0.1, 4.1); ax.set_ylim(0.85, 2.5)
        ax.set_title(f"family T_ν = ∪[k, k+ν] ∪ {{4}},  ν = {nus[i]:.3f}"); ax.set_xlabel("t ∈ T_ν"); ax.set_ylabel("x(t)")
    save(FuncAnimation(fig, draw, frames=range(len(frames))), "family_nu.gif", 6); plt.close(fig)


def gif_qacc():
    D = load_json("exp7_unusual"); tr = np.array(D["Qacc"]["profile"]["t"]); xr = np.array(D["Qacc"]["profile"]["x"])
    q = 0.9; fd = lambda s: 1 + 0.5 * np.sin(2 * s); ns = list(range(2, 121, 3)); frames = []
    for n in ns:
        tk = q ** np.arange(n, -1, -1.0); mu = np.concatenate([np.diff(tk), [0]]); X = np.empty(n + 1)
        for i in range(n + 1):
            X[i] = fd(tk[i]) + np.sum(mu[:i] * np.exp(-(tk[i] - tk[:i])) * np.sin(X[:i]))
        ref = xr[len(tr) - n - 1:]; frames.append((tk, X, float(np.max(abs(X - ref)))))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    def draw(i):
        for a in ax: a.clear()
        tk, X, e = frames[i]
        ax[0].scatter(tr, xr, s=5, color=(1, 1, 1, .25)); ax[0].scatter(tk, X, c=np.log10(tk), cmap="turbo", s=14)
        ax[0].set_xscale("symlog", linthresh=1e-3); ax[0].set(title=f"retained points q^{ns[i]},...,1", xlabel="t", ylabel="x(t)")
        ax[1].semilogy(ns[:i + 1], [fr[2] for fr in frames[:i + 1]], "o-", color="#fdcb6e")
        ax[1].set(xlim=(0, 122), ylim=(1e-6, 1), title="truncation error near the accumulation point 0", xlabel="n")
    save(FuncAnimation(fig, draw, frames=range(len(frames))), "accumulation_point.gif", 6); plt.close(fig)


def gif_training():
    prob, poly, exact, _ = irregular_problem(); T = prob.T; m = 8; rng = np.random.default_rng(0)
    P = Partition.gap_conforming(T, h=.025); op = GaussVolterraOperator(T, P.t, n_gauss=8)
    A = .18 * op.mask * op.w[None, :] * (1 + .07 * op.t[:, None] * op.s[None, :]); f = prob.f(op.t); x0 = float(np.arcsin(.2))
    theta = np.concatenate([[0.], rng.normal(0, .01, m), rng.uniform(-2, 2, m), rng.uniform(-1, 1, m)])
    snaps = []
    def res(th):
        yt, _ = network(op.t, th, m, T.b); ys, _ = network(op.s, th, m, T.b); return x0 + yt - f - A @ np.sin(x0 + ys)
    def jac(th):
        snaps.append(th.copy())
        yt, Jt = network(op.t, th, m, T.b); ys, Js = network(op.s, th, m, T.b)
        return Jt - A @ (np.cos(x0 + ys)[:, None] * Js)
    t0 = time.time()
    sol = least_squares(res, theta, jac=jac, method="trf", ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=200)
    LOG["training_seconds"] = time.time() - t0
    tt = np.concatenate([np.linspace(a, b, 120) if b > a else [a] for a, b in T.comps])
    hist = [float(np.max(abs(x0 + network(tt, th, m, T.b)[0] - exact(tt)))) for th in snaps]
    cols = component_colors(T.m, "spring")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    def draw(i):
        for a in ax: a.clear()
        v = x0 + network(tt, snaps[i], m, T.b)[0]
        plot_on_scale(ax[0], T.comps, tt, exact(tt), colors=[(1, 1, 1, .45)] * T.m, lw=3)
        plot_on_scale(ax[0], T.comps, tt, v, colors=cols, lw=1.6, ms=6)
        ax[0].set(ylim=(0.15, 0.45), title=f"tanh network (m=8) on $\\mathbb{{T}}_{{irr}}$, trust-region GN iteration {i}", xlabel="t ∈ T", ylabel="v_θ(t)")
        ax[1].semilogy(range(i + 1), hist[:i + 1], "o-", color="#55efc4", ms=3)
        ax[1].set(xlim=(-1, len(snaps)), ylim=(min(hist) / 3, max(hist) * 3), title="sampled max error on T", xlabel="iteration")
    save(FuncAnimation(fig, draw, frames=range(len(snaps))), "nn_training_irregular.gif", 5); plt.close(fig)
    LOG["training_iterations"] = len(snaps); LOG["training_final_error"] = hist[-1]


if __name__ == "__main__":
    for g in [gif_forward_substitution, gif_refinement_irregular, gif_gallery, gif_family, gif_qacc, gif_training]:
        g()
    save_json("gif_log", LOG); print(LOG)
