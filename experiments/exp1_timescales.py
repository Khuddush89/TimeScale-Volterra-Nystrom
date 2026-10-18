"""Experiment 1 (Section 7.1): Figure 1, the time scales used in the experiments."""
from common import *
from tsvolterra.timescale import TimeScale, Partition
from tsvolterra.problems import ts_hybrid, ts_pendulum, ts_qscale, ts_family
from tsvolterra import plotting as pl
import matplotlib.pyplot as plt


def main():
    pl.setup()
    fig = plt.figure(figsize=(6.9, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    rows = [(r"$\mathbb{T}_{\rm hyb}$", ts_hybrid(), 0.125), (r"$\mathbb{T}_{\rm pend}$", ts_pendulum(), 0.25)]
    y = 2.0
    for name, T, h in rows:
        pl.draw_timescale(ax, T, y=y, lw=3.2, color=pl.C[0])
        P = Partition.gap_conforming(T, h)
        for tt, k in zip(P.t[:-1], P.kind):
            pass
        ax.plot(P.t, np.full(len(P.t), y - 0.3), "|", color=pl.INK2, mec=pl.INK2, ms=5, mew=0.6)
        for (q, p) in T.gaps():
            ax.annotate("", xy=(p, y + 0.22), xytext=(q, y + 0.22),
                        arrowprops=dict(arrowstyle="->", color=pl.C[1], lw=0.7, shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3,rad=-0.45"))
        ax.text(-0.25, y, name, ha="right", va="center", fontsize=8.5)
        y -= 1.2
    TQ = ts_qscale(1.002, 1.0, 3.0)
    ax.plot(TQ.p, np.full(len(TQ.p), y), "o", ms=1.3, color=pl.C[0], mew=0)
    Pc = Partition.coarse(TQ, 32)
    ax.plot(Pc.t, np.full(len(Pc.t), y - 0.3), "|", color=pl.INK2, mec=pl.INK2, ms=5, mew=0.6)
    ax.text(-0.25, y, r"$q^{\mathbb{N}_0}\cap[1,3]$", ha="right", va="center", fontsize=8.5)
    ax.text(3.12, y, "550 points", fontsize=6.6, color=pl.INK2, va="center")
    ax.set_xlim(-0.1, 6.2); ax.set_ylim(y - 0.7, 2.6)
    ax.set_yticks([]); ax.grid(False)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(r"$t$")
    ax.set_title(r"(a) $\mathbb{T}_{\rm hyb}$, $\mathbb{T}_{\rm pend}$ and $q^{\mathbb{N}_0}$ with partitions", loc="left", fontsize=8)
    # legend proxies
    ax.plot([], [], color=pl.C[0], lw=3, label="dense component")
    ax.plot([], [], "o", color=pl.C[0], ms=3.2, label="isolated point")
    ax.plot([], [], "|", color=pl.INK2, mec=pl.INK2, ms=5, label="partition node")
    ax.plot([], [], color=pl.C[1], lw=0.7, label=r"jump $t\mapsto\sigma(t)$")
    ax.legend(loc="lower right", fontsize=6.3, ncol=1, bbox_to_anchor=(1.03, 0.0), handlelength=1.4)
    ax = fig.add_subplot(gs[0, 1])
    nus = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for k, nu in enumerate(nus):
        T = ts_family(nu, 4)
        pl.draw_timescale(ax, T, y=len(nus) - k, lw=3.0, color=pl.C[0], ms=3.2)
        ax.text(-0.15, len(nus) - k, r"$\nu=%.1f$" % nu, ha="right", va="center", fontsize=7.5)
    ax.set_xlim(-0.05, 4.1); ax.set_ylim(0.3, len(nus) + 0.6)
    ax.set_yticks([]); ax.grid(False)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(r"$t$")
    ax.set_title(r"(b) $\mathbb{T}_\nu=\cup_{k=0}^{3}\,[k,k+\nu]\,\cup\{4\}$", loc="left", fontsize=8)
    pl.save(fig, os.path.join(FIG, "fig_timescales"))


if __name__ == "__main__":
    main(); print("ok")
