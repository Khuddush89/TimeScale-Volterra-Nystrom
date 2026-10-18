"""Time-scale-aware redraws of the paper figures (quadrature panel (a) and the irregular-scale figure).
Rule: curves only on dense components, markers on isolated points, gaps shown explicitly (hatched)."""
from common import *
import matplotlib.pyplot as plt
from tsvolterra.timescale import Partition
from tsvolterra import nystrom
from tsvolterra.tsplot import PAPER, plot_on_scale, component_colors, scale_strip
import exp2_quadrature as e2
from revision_checks import irregular_problem


def fig_quadrature():
    res = load_json("exp2_quadrature")
    e2.plot(res)                                 # panels (b),(c) in the original layout (overwritten below)
    plt.rcParams.update(PAPER)
    steps = [[(0, 1)], [(v, v) for v in np.linspace(0, 1, 9)], [(v, v) for v in [0, .1, .25, .45, .7, 1]],
             [(0, 0), (.15, .15), (.3, .3), (.45, 1)]]
    names = list(res["profiles"].keys())
    fig = plt.figure(figsize=(7.2, 4.6), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.3])
    for k, (name, comps) in enumerate(zip(names, steps)):
        ax = fig.add_subplot(gs[0, k]); d = res["profiles"][name]; u = np.array(d["u"]); g = np.array(d["g"])
        cols = component_colors(len(comps), "turbo")
        uu = np.linspace(0, 1, 200); ax.plot(uu, uu * (1 - uu) / 2, ":", color="0.55", lw=0.8)
        for (q, p) in [(comps[i][1], comps[i + 1][0]) for i in range(len(comps) - 1)]:     # gaps (q,p)
            gv = float(np.interp(p, u, g))
            ax.fill_between([q, p], [0, 0], [gv, gv], facecolor="none", edgecolor="0.6", hatch="////", lw=0)
            ax.plot([q, p], [gv, gv], color="0.45", lw=0.7)
        for (p, q), c in zip(comps, cols):
            if q > p:
                m = (u >= p) & (u <= q); ax.fill_between(u[m], 0, g[m], color=c, alpha=0.25, lw=0); ax.plot(u[m], g[m], color=c, lw=1.6)
            else:
                ax.plot([p], [np.interp(p, u, g)], "o", color=c, ms=4, mec="k", mew=0.4, zorder=5)
        scale_strip(ax, comps, 0.004, colors=cols)
        ax.set(ylim=(0, 0.17), xlim=(-0.02, 1.02), xticks=[0, .5, 1], yticks=[0, .05, .1]); ax.set_title(name, fontsize=7.2)
        ax.text(0.5, 0.152, r"$\kappa=%.4f$" % d["kappa"], ha="center", fontsize=7.2)
        if k == 0: ax.set_ylabel(r"$-\Pi(\pi^+(u))$")
        else: ax.set_yticklabels([])
        ax.set_xlabel(r"$u\in[0,1]$", labelpad=1)
    r = res["random"]; D, g3, kap = map(np.array, (r["D"], r["g3"], r["kappa"]))
    ax = fig.add_subplot(gs[1, 0:2]); ax.plot([0, 1], [0, 1], color="0.5", lw=0.8)
    sc = ax.scatter(1 - g3 / D ** 3, kap / (D ** 3 / 12), s=7, c=D, cmap="plasma", lw=0)
    cax = ax.inset_axes([0.06, 0.55, 0.03, 0.35]); fig.colorbar(sc, cax=cax, label=r"$\Delta$")
    ax.set(xlabel=r"$1-\sum_i g_i^3/\Delta^3$", ylabel=r"$\kappa/(\Delta^3/12)$", xlim=(0, 1.02), ylim=(0, 1.02))
    ax.set_title("(b) 1500 random steps: $\\kappa\\leq(\\Delta^3-\\sum g_i^3)/12$", fontsize=8, loc="left")
    ax = fig.add_subplot(gs[1, 2:4]); c = res["composite"]; C = ["#1f77ff", "#ff5f1f", "#00b386", "#d6249f"]
    ax.loglog(c["R"]["h"], c["R"]["err"], "o-", color=C[0], label=r"$\mathbb{T}=[0,2]$")
    ax.loglog(c["hyb"]["h"], c["hyb"]["err"], "s-", color=C[1], label=r"$\mathbb{T}_{\rm hyb}$, gap-conforming")
    ax.loglog(c["hyb_nc"]["h"], c["hyb_nc"]["err"], "^-", color=C[2], label=r"$\mathbb{T}_{\rm hyb}$, not conforming")
    H = np.array(c["q"]["H"]); e = np.array(c["q"]["err"])
    ax.loglog(H[1:], e[1:], "D-", color=C[3], label=r"$q^{\mathbb{N}_0}$, coarse $H$ (exact if $H=\mu$)")
    ax.set(xlabel=r"largest non-gap step $h$ (or $H$)", ylabel="quadrature error"); ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(c) composite rule", fontsize=8, loc="left")
    fig.suptitle("(a) Peano profiles on unit steps (coloured: $\\mathbb{T}$; hatched: gaps, where $\\pi^+$ is constant; dotted: $\\mathbb{T}=\\mathbb{R}$)", fontsize=8, x=0.02, ha="left")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})): fig.savefig(os.path.join(FIG, "fig_quadrature." + ext), **kw)
    plt.close(fig)


def fig_irregular():
    plt.rcParams.update(PAPER)
    prob, poly, x, _ = irregular_problem(); T = prob.T; d = load_json("revision_validation") if False else None
    rows = []
    for M in [1, 2, 4, 8, 16, 32, 64]:
        counts = [round((q - p) / .05) * M if q > p else 0 for p, q in T.comps]
        P = Partition.gap_conforming(T, counts=counts); X = nystrom.solve(prob, P)
        rows.append((P.h_dense(), P, X, float(np.max(abs(X - x(P.t))))))
    P, X = rows[1][1], rows[1][2]
    tt = np.concatenate([np.linspace(a, b, 200) if b > a else [a] for a, b in T.comps])
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6), layout="constrained")
    cols = plot_on_scale(ax[0], T.comps, tt, x(tt), cmap="turbo", lw=2.6, ms=7, label="exact $x$")
    ax[0].plot(P.t, X, "o", ms=2.6, mfc="white", mec="k", mew=0.5, label=f"Nystrom nodes ($h_d$={P.h_dense():.3f})")
    scale_strip(ax[0], T.comps, 0.19, colors=cols)
    ax[0].set(xlabel=r"$t\in\mathbb{T}_{\rm irr}$", ylabel=r"$x(t)$", title="(a) nonlinear solution on $\\mathbb{T}_{\\rm irr}$ (gaps left empty)"); ax[0].legend(fontsize=7, loc="upper left")
    tab = load_json("irregular_hybrid_metadata")
    h = np.array([r[0] for r in rows]); E = np.array([r[3] for r in rows])
    ax[1].loglog(h, E, "o-", color="#1f77ff", label="nodal error (measured)")
    ax[1].loglog(h, 3 * E[0] * (h / h[0]) ** 2, ":", color="0.4", label="slope 2")
    ax[1].set(xlabel=r"dense step $h_d$ (six jumps fixed)", ylabel="max nodal error", title="(b) refinement of the dense parts"); ax[1].legend(fontsize=7)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})): fig.savefig(os.path.join(FIG, "fig_irregular." + ext), **kw)
    plt.close(fig); return [(float(r[0]), r[1].N + 1, r[3]) for r in rows]


if __name__ == "__main__":
    fig_quadrature(); print(fig_irregular())
