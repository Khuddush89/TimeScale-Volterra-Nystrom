"""Figures for Experiment 4: training dynamics, cost-accuracy, error profiles."""
from common import *
from tsvolterra.problems import ts_hybrid
from tsvolterra import plotting as pl
import matplotlib.pyplot as plt

T = ts_hybrid()


def load():
    cl = load_json("exp4_classical")
    nn = load_json("exp4_neural")
    try:
        lm = load_json("exp4_neural_lm")
    except FileNotFoundError:
        lm = dict(rows=[], profiles={})
    return cl, nn, lm


def fig_training(nn):
    pl.setup()
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.4), sharey=True)
    for ax, fam, title in zip(axs, ["mlp_D", "mlp_C", "mlpts_C"],
                              [r"(a) MLP, $\mathcal{L}_D$ ($h=2^{-4}$)", r"(b) MLP, $\mathcal{L}_C$", r"(c) MLP-TS, $\mathcal{L}_C$"]):
        h = nn["hists"][fam]
        it = np.array([r["it"] for r in h]); loss = np.sqrt(np.array([r["loss"] for r in h]))
        err = np.array([r["err"] for r in h]); cert = np.array([r["cert"] for r in h])
        ax.semilogy(it, cert, color=pl.C[1], lw=1.3, label=r"indicator $e_L(b,a)\,\sup|r_v|$")
        ax.semilogy(it, err, color=pl.C[0], lw=1.3, label=r"true error $\|v_\theta-x\|_\infty$")
        ax.semilogy(it, loss, color=pl.C[2], lw=1.1, label=r"training RMS residual $\sqrt{\mathcal{L}}$")
        ax.axvline(5000, color=pl.MUTED, lw=0.6)
        ax.text(5150, 6.0, "L-BFGS", fontsize=6.6, color=pl.INK2, va="top")
        ax.text(4850, 6.0, "Adam", fontsize=6.6, color=pl.INK2, va="top", ha="right")
        ax.set_title(title, loc="left"); ax.set_xlabel("iteration")
        ax.set_ylim(3e-8, 10)
    axs[0].set_ylabel("")
    axs[0].legend(loc="lower left", fontsize=6.0)
    fig.tight_layout(w_pad=0.5)
    pl.save(fig, os.path.join(FIG, "fig_training"))


def fig_pareto(cl, nn, lm):
    pl.setup()
    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    rows = cl["classical"]

    # grid methods: controlled timings (median of 7 serial repetitions) when available
    try:
        tc = load_json("exp4_timing_controlled")["rows"]
        TC = {(r["method"], r["lev"]): r["median"] for r in tc}
    except FileNotFoundError:
        TC = {}
    gname = {"nystrom": "Nystrom", "picard": "Picard", "richardson": "Richardson"}

    def tget(r):
        return TC.get((gname.get(r["family"]), r.get("lev")), r["time"])

    def line(fam, color, marker, label, key="time"):
        rr = sorted([r for r in rows if r["family"] == fam], key=lambda r: r[key])
        ax.loglog([tget(r) for r in rr], [max(r["err"], 1e-16) for r in rr], color=color, marker=marker,
                  ms=4.6, label=label, lw=1.2, mec="white", mew=0.5)
        return rr
    line("nystrom", pl.C[0], "o", r"Nystr$\ddot{\mathrm{o}}$m, forward substitution, $h=2^{-2},\dots,2^{-10}$", "lev")
    line("picard", pl.C[0], "^", r"Nystr$\ddot{\mathrm{o}}$m, Picard sweeps, $h=2^{-2},\dots,2^{-8}$", "lev")
    line("richardson", pl.C[0], "s", r"Richardson extrapolation, $h=2^{-2},\dots,2^{-9}$", "lev")
    line("spectral", pl.C[2], "D", r"TS-spectral collocation, degree $4,\dots,20$", "deg")
    allnn = nn["rows"] + lm["rows"]
    groups = [("mlp_D", "v", r"MLP $3\times32$, $\mathcal{L}_D$, Adam+L-BFGS"),
              ("mlp_C", "P", r"MLP $3\times32$, $\mathcal{L}_C$, Adam+L-BFGS"),
              ("mlpts_C", "X", r"MLP-TS $3\times32$, $\mathcal{L}_C$, Adam+L-BFGS"),
              ("mlp_C_alm", "*", r"MLP $2\times16$, $\mathcal{L}_C$, Adam+LM"),
              ("mlpts_C_alm", "h", r"MLP-TS $2\times16$, $\mathcal{L}_C$, Adam+LM"),
              ("rf", "p", r"random features, $m=64,128,256$, LM"),
              ("cheb", "d", r"Chebyshev network, degree $10,\dots,40$, LM")]
    msize = {"*": 8.0, "h": 6.0, "p": 6.0, "d": 6.0, "v": 6.0, "P": 6.4, "X": 6.4}
    for fam, mk, lab in groups:
        rr = [r for r in allnn if r["family"] == fam]
        if not rr:
            continue
        ax.loglog([r["time"] for r in rr], [r["err"] for r in rr], ls="none", marker=mk, ms=msize.get(mk, 5.6), color=pl.C[1],
                  mec="white", mew=0.5, label=lab)
    ax.set_xlabel("wall-clock time [s]"); ax.set_ylabel("maximal error")
    ax.set_ylim(1e-15, 2)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=6.7, handlelength=1.8,
              columnspacing=1.2, frameon=False)
    fig.tight_layout()
    pl.save(fig, os.path.join(FIG, "fig_pareto"))


def fig_profiles(cl=None, nn=None, lm=None):
    pl.setup()
    d = load_json("exp4_profiles")
    E = np.array(d["E"]); profs = d["profiles"]
    pick = [(r"Nystr$\ddot{\mathrm{o}}$m interpolant, $h=2^{-4}$", "Nystrom-pl h=2^-4"),
            (r"MLP, $\mathcal{L}_C$, Adam+L-BFGS", "MLP 3x32, loss C (seed 0)"),
            (r"MLP-TS, $\mathcal{L}_C$, Adam+L-BFGS", "MLP-TS 3x32, loss C (seed 0)"),
            (r"random features $m=256$, LM", "RF 256, loss C, LM (seed 0)"),
            (r"TS-spectral, degree 12", "TS-spectral deg 12")]
    fig, axs = plt.subplots(len(pick), 1, figsize=(6.9, 1.05 * len(pick) + 0.55), sharex=True)
    for ax, (lab, key) in zip(axs, pick):
        p = profs[key]
        e = np.maximum(np.array(p["err"]), 1e-17); env = np.array(p["env_maj"])
        for (q, pp) in T.gaps():
            ax.axvspan(q, pp, color=pl.GAP, lw=0, zorder=0)
        for (p_, q_) in T.comps:
            msk = (E >= p_ - 1e-12) & (E <= q_ + 1e-12)
            if q_ > p_:
                ax.semilogy(E[msk], e[msk], color=pl.C[0], lw=0.9)
                ax.semilogy(E[msk], env[msk], color=pl.C[1], lw=1.1)
            else:
                ax.semilogy(E[msk], e[msk], "o", color=pl.C[0], ms=3.6)
                ax.semilogy(E[msk], env[msk], "s", color=pl.C[1], ms=3.4)
        sel = E > 0.05
        lo = np.percentile(e[sel], 1) / 5; hi = np.max(env) * 6
        ax.set_ylim(lo, hi)
        ax.text(3.03, hi / 1.6, lab, fontsize=7.0, va="top", ha="left")
        ax.tick_params(axis="y", labelsize=6.8)
    axs[-1].set_xlabel(r"$t\in\mathbb{T}_{\rm hyb}$ (grey bands: gaps; markers: isolated points)")
    axs[0].plot([], [], color=pl.C[0], label=r"error $|v(t)-x(t)|$")
    axs[0].plot([], [], color=pl.C[1], label=r"sampled indicator $u_\ell(t)\,\sup_{s\leq t}|r_v(s)|$")
    axs[0].legend(loc="lower left", fontsize=6.6, ncol=2, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout(h_pad=0.25)
    pl.save(fig, os.path.join(FIG, "fig_profiles"))


def fig_overfit():
    d = load_json("exp4_overfit")["rows"]
    pl.setup()
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.3), sharey=True)
    for ax, tag, title in zip(axs, ["coarse", "fine"], ["(a) 57 collocation points", "(b) 107 collocation points"]):
        rr = sorted([r for r in d if r["family"] == "cheb_" + tag], key=lambda r: r["deg"])
        deg = [r["deg"] for r in rr]
        ax.semilogy(deg, [r["eta_maj"] for r in rr], color=pl.C[1], marker="s", label=r"indicator $\eta_{\rm maj}$ (on $\mathcal{E}$)")
        ax.semilogy(deg, [r["err"] for r in rr], color=pl.C[0], marker="o", label=r"true error on $\mathcal{E}$")
        ax.semilogy(deg, [max(r["train_rms"], 1e-16) for r in rr], color=pl.C[2], marker="^", label=r"training RMS residual on $\mathcal{C}$")
        ax.set_xlabel("degree of the Chebyshev network"); ax.set_title(title, loc="left")
        ax.set_ylim(1e-16, 30)
    axs[0].legend(loc="lower left", fontsize=6.4)
    fig.tight_layout(w_pad=0.6)
    pl.save(fig, os.path.join(FIG, "fig_overfit"))


if __name__ == "__main__":
    cl, nn, lm = load()
    fig_training(nn)
    fig_pareto(cl, nn, lm)
    fig_overfit()
    try:
        fig_profiles()
    except FileNotFoundError:
        print("exp4_profiles.json missing: run exp4_profiles.py first")
    print("figures written")
